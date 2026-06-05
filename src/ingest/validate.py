from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pandas.api import types as ptypes

from src.config import DATA_DIR, PROCESSED_DIR

# Where malformed rows are parked (one file per season) and where the report goes.
QUARANTINE_DIR = DATA_DIR / "quarantine"
REPORT_PATH = PROCESSED_DIR / "validation_report.json"

# Required canonical columns and the dtype family each must belong to.
# Validation runs on the canonical-shaped frame, so dtypes are already normalised.
REQUIRED_COLUMNS = {
    "match_id": "string",
    "date": "datetime",
    "season": "integer",
    "home_team": "string",
    "away_team": "string",
    "home_goals": "integer",
    "away_goals": "integer",
    "result": "string",
}

# Maps each dtype family to the pandas type-check it must satisfy.
_DTYPE_CHECKS = {
    "string": lambda s: ptypes.is_string_dtype(s) or ptypes.is_object_dtype(s),
    "datetime": ptypes.is_datetime64_any_dtype,
    "integer": ptypes.is_integer_dtype,
}


def _check_schema(df: pd.DataFrame) -> dict:
    """Check required columns are present and have an acceptable dtype.

    Missing columns are a structural failure and raise. Dtype mismatches are
    recorded (not fatal) so the report can surface them without blocking ingest.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"validate_matches: missing required columns: {missing}")

    dtype_mismatches = {}
    for col, family in REQUIRED_COLUMNS.items():
        check = _DTYPE_CHECKS[family]
        if not check(df[col]):
            dtype_mismatches[col] = str(df[col].dtype)

    return {"missing_columns": missing, "dtype_mismatches": dtype_mismatches}


def _malformed_reasons(df: pd.DataFrame) -> pd.Series:
    """Label each row with its first malformed reason, or '' if the row is clean.

    A row is malformed if it is missing any required value: goals, date, or team
    names. These rows carry no usable label for the classifier, so they are
    quarantined rather than allowed into the canonical dataset.
    """
    reason = pd.Series("", index=df.index, dtype="object")

    empty_team = (
        df["home_team"].isna()
        | df["away_team"].isna()
        | (df["home_team"].astype("string").str.strip() == "")
        | (df["away_team"].astype("string").str.strip() == "")
    )
    missing_goals = df["home_goals"].isna() | df["away_goals"].isna()
    missing_date = df["date"].isna()

    # Assign in priority order; the first matching reason wins per row.
    reason[empty_team] = "missing_teams"
    reason[missing_date & (reason == "")] = "missing_date"
    reason[missing_goals & (reason == "")] = "missing_goals"
    return reason


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a parquet file atomically (temp-then-replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _atomic_write_json(payload: dict, path: Path) -> None:
    """Write a JSON file atomically (temp-then-replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def validate_matches(
    df: pd.DataFrame,
    *,
    quarantine_dir: Path = QUARANTINE_DIR,
    report_path: Path = REPORT_PATH,
    write: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Validate a canonical-shaped frame: check schema, quarantine malformed rows, report.

    Steps:
      1. Schema check — required columns present (raises if not) + dtype recorded.
      2. Flag malformed rows (missing goals / date / teams) and split them out.
      3. Quarantine the dropped rows to data/quarantine/dropped_<season>.parquet.
      4. Write a data-quality report to data/processed/validation_report.json.

    Pass `write=False` (or override the paths) in tests to avoid touching real artifacts.

    Returns:
        clean  — rows that passed validation (canonical training data)
        report — the data-quality report dict (also written to report_path)
    """
    schema = _check_schema(df)

    reasons = _malformed_reasons(df)
    is_bad = reasons != ""
    clean = df[~is_bad].copy().reset_index(drop=True)
    dropped = df[is_bad].copy()

    # Quarantine dropped rows, one parquet per season (NaN season -> 'unknown').
    if write and not dropped.empty:
        quarantine_dir = Path(quarantine_dir)
        for season, group in dropped.groupby(dropped["season"].astype("object"), dropna=False):
            label = "unknown" if pd.isna(season) else int(season)
            _atomic_write_parquet(group, quarantine_dir / f"dropped_{label}.parquet")

    # Per-season kept/dropped tallies for the report.
    by_season: dict[str, dict[str, int]] = {}
    for season, group in df.groupby(df["season"].astype("object"), dropna=False):
        label = "unknown" if pd.isna(season) else str(int(season))
        bad_count = int((reasons[group.index] != "").sum())
        by_season[label] = {"kept": len(group) - bad_count, "dropped": bad_count}

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema": schema,
        "total_rows": int(len(df)),
        "rows_kept": int(len(clean)),
        "rows_dropped": int(len(dropped)),
        "dropped_by_reason": {
            reason: int(count)
            for reason, count in reasons[is_bad].value_counts().items()
        },
        "by_season": by_season,
    }

    if write:
        _atomic_write_json(report, Path(report_path))

    return clean, report
