from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Literal

import pandas as pd

from src.config import DATA_DIR

# Input file (created by unify_raw.py)
RAW_PATH = DATA_DIR / "raw" / "matches_all.parquet"

# Output file (canonical dataset)
OUT_DIR = DATA_DIR / "processed"
OUT_PATH = OUT_DIR / "matches_canonical.parquet"

# Locked column order for the canonical dataset
CANONICAL_COLUMNS = [
    "match_id",
    "date",
    "season",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
]


def infer_season(date_utc: pd.Timestamp) -> int:
    # Premier League style season start year:
    # Aug-Dec -> season = same year
    # Jan-May -> season = previous year
    if pd.isna(date_utc):
        return pd.NA  # type: ignore[return-value]
    return int(date_utc.year if date_utc.month >= 8 else date_utc.year - 1)


def stable_match_id(
    season: int, date_utc: pd.Timestamp, home_team: str, away_team: str
) -> str:
    # Make a deterministic ID using season + date + teams
    if not pd.isna(date_utc):
        date_str = pd.Timestamp(date_utc).tz_convert("UTC").isoformat()
    else:
        date_str = ""

    key = f"{season}|{date_str}|{home_team.strip().lower()}|{away_team.strip().lower()}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def compute_result(home_goals, away_goals) -> Literal["HW", "D", "AW"] | None:
    # Convert goals to result label:
    # HW = home win, D = draw, AW = away win
    # Return None if goals are missing
    if pd.isna(home_goals) or pd.isna(away_goals):
        return None

    hg = int(home_goals)
    ag = int(away_goals)

    if hg > ag:
        return "HW"
    if hg == ag:
        return "D"
    return "AW"


def build_canonical(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    # Check the unified raw file exists
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {raw_path}. Please run unify_raw first."
        )

    raw = pd.read_parquet(raw_path)

    # Check required raw columns exist
    required_raw_cols = ["date", "home_team", "away_team", "home_goals", "away_goals"]
    missing = [c for c in required_raw_cols if c not in raw.columns]
    if missing:
        raise ValueError(f"Raw data missing required columns: {missing}")

    # Keep only required columns first
    df = raw[required_raw_cols].copy()

    # Normalize datatypes
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce").astype("Int64")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce").astype("Int64")
    df["home_team"] = df["home_team"].astype("string").fillna("")
    df["away_team"] = df["away_team"].astype("string").fillna("")

    # Create season column
    df["season"] = df["date"].apply(infer_season).astype("Int64")

    # Create stable match_id
    df["match_id"] = df.apply(
        lambda row: stable_match_id(
            int(row["season"]) if not pd.isna(row["season"]) else -1,
            row["date"],
            str(row["home_team"]),
            str(row["away_team"]),
        ),
        axis=1,
    ).astype("string")

    # Create result label (vectorised – avoids row-wise apply & Pylance overload issues)
    hg = df["home_goals"]
    ag = df["away_goals"]
    goals_known = hg.notna() & ag.notna()
    df["result"] = pd.array([pd.NA] * len(df), dtype="string")
    df.loc[goals_known & (hg > ag), "result"] = "HW"
    df.loc[goals_known & (hg == ag), "result"] = "D"
    df.loc[goals_known & (hg < ag), "result"] = "AW"

    # Force locked column order
    df = df[CANONICAL_COLUMNS].copy()

    # Sort rows for consistency
    df = df.sort_values(
        ["season", "date", "home_team", "away_team"],
        kind="mergesort",
    ).reset_index(drop=True)

    return df


def main() -> None:
    # No arguments needed yet, but keep argparse for later extensibility
    parser = argparse.ArgumentParser(
        description="Build canonical matches dataset from raw unified data."
    )
    parser.parse_args()

    df = build_canonical()

    # Create output directory if needed
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Safe overwrite: write temp then replace
    tmp_path = OUT_DIR / ".matches_canonical.parquet.tmp"
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(OUT_PATH)

    print(f"Canonical dataset saved to {OUT_PATH} with {len(df)} records.")


if __name__ == "__main__":
    main()
