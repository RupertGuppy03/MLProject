from __future__ import annotations

# This script pulls match data for a given Premier League season using the FootballDataClient,
# normalizes it into a consistent DataFrame format, and saves it as a CSV file in the raw data directory.

import argparse
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd

from src.config import PL_COMPETITION_CODE, RAW_DIR
from src.services.football_data_client import FootballDataClient

REQUIRED_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "pull_timestamp",
    "data_source",
]


def _normalize_matches(raw_json: Dict[str, Any]) -> pd.DataFrame:
    matches: List[Dict[str, Any]] = raw_json.get("matches", [])
    rows: List[Dict[str, Any]] = []

    for m in matches:
        utc_date = m.get("utcDate")
        home = (m.get("homeTeam") or {}).get("name")
        away = (m.get("awayTeam") or {}).get("name")

        score = m.get("score") or {}
        full_time = score.get("fullTime") or {}
        home_goals = full_time.get("home")
        away_goals = full_time.get("away")

        rows.append(
            {
                "date": utc_date,
                "home_team": home,
                "away_team": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
            }
        )

    df = pd.DataFrame(rows)

    # Parse date to timezone-aware datetime (UTC)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

    # Nullable integer goals
    for col in ["home_goals", "away_goals"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df


def pull_season(season: int, overwrite: bool = True) -> pd.DataFrame:
    client = FootballDataClient()
    raw = client.get_matches(competition_code=PL_COMPETITION_CODE, season=season)
    df = _normalize_matches(raw)

    # Metadata columns required by acceptance tests
    df["pull_timestamp"] = datetime.now(timezone.utc).isoformat()
    df["data_source"] = "football-data.org"

    # Stable schema: ensure all required columns exist and correct order
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[REQUIRED_COLUMNS].copy()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"matches_{season}.parquet"

    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"{out_path} already exists. Re-run with overwrite enabled."
        )

    # Safe overwrite: write temp then replace
    tmp_path = RAW_DIR / f".matches_{season}.parquet.tmp"
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(out_path)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull Premier League matches data")
    parser.add_argument(
        "--season", type=int, required=True, help="Season year (e.g. 2023)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing file if it exists",
    )
    args = parser.parse_args()

    # default behavior: overwrite is False unless provided
    pull_season(args.season, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
