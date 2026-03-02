from __future__ import annotations

import argparse
import pandas as pd

from src.config import RAW_DIR

REQUIRED_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "pull_timestamp",
    "data_source",
]


def unify_raw() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Load all season parquet files, excluding the unified output + temp files
    files = sorted(RAW_DIR.glob("matches_*.parquet"))
    files = [
        p
        for p in files
        if p.name != "matches_all.parquet" and not p.name.startswith(".")
    ]

    if not files:
        raise FileNotFoundError(
            f"No season parquet files found in {RAW_DIR}. Expected matches_<season>.parquet"
        )

    dfs: list[pd.DataFrame] = []
    for p in files:
        df = pd.read_parquet(p)

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{p.name} missing required columns: {missing}")

        df = df[REQUIRED_COLUMNS].copy()

        # Normalize types to keep a consistent unified schema
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce").astype(
            "Int64"
        )
        df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce").astype(
            "Int64"
        )
        df["home_team"] = df["home_team"].astype("string")
        df["away_team"] = df["away_team"].astype("string")
        df["pull_timestamp"] = df["pull_timestamp"].astype("string")
        df["data_source"] = df["data_source"].astype("string")

        dfs.append(df)

    unified = pd.concat(dfs, ignore_index=True)

    # Sort by date ascending
    unified = unified.sort_values("date", ascending=True, kind="mergesort").reset_index(
        drop=True
    )

    # Safe overwrite: write temp then replace
    out_path = RAW_DIR / "matches_all.parquet"
    tmp_path = RAW_DIR / ".matches_all.parquet.tmp"
    unified.to_parquet(tmp_path, index=False)
    tmp_path.replace(out_path)

    return unified


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unify raw season match files into one dataset."
    )
    parser.parse_args()

    df = unify_raw()

    # Basic validations
    out_path = RAW_DIR / "matches_all.parquet"
    if not out_path.exists():
        raise RuntimeError("matches_all.parquet was not created")

    if list(df.columns) != REQUIRED_COLUMNS:
        raise RuntimeError("Unified schema is not the expected stable schema")

    if not df["date"].is_monotonic_increasing:
        raise RuntimeError("Unified dataset is not sorted by date ascending")

    print(f"Created {out_path} with {len(df):,} rows from {len(df):,} total rows.")


if __name__ == "__main__":
    main()
