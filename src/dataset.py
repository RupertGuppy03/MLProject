# src/dataset.py
"""
Dataset ingestion for the project.

We keep this separate from features/models so that:
- data collection is reproducible
- raw snapshots are saved before any transformations (important for ML projects)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv

from src.services.football_data_api import FootballDataAPI

RAW_DIR = Path("data/raw")


def fetch_pl_matches(date_from: str, date_to: str) -> pd.DataFrame:
    """
    Fetch Premier League matches from football-data.org between date_from and date_to.

    date_from / date_to must be 'YYYY-MM-DD'
    Returns a pandas DataFrame with one row per match.
    """
    load_dotenv()  # loads .env into environment variables

    base_url = os.getenv("FOOTBALL_API_BASE_URL", "https://api.football-data.org/v4")
    api_key = os.getenv("FOOTBALL_API_KEY", "")

    api = FootballDataAPI(base_url=base_url, api_key=api_key)

    # football-data.org supports competition code 'PL' for Premier League
    payload = api.get(
        "competitions/PL/matches",
        params={"dateFrom": date_from, "dateTo": date_to},
    )

    matches = payload.get("matches", [])
    return normalize_matches(matches)


def normalize_matches(matches: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert raw match JSON objects into a flat DataFrame.

    Why we flatten:
    - APIs return nested JSON; ML needs tabular data
    - We keep only fields we care about for modeling
    """
    rows = []
    for m in matches:
        score = m.get("score") or {}
        ft = score.get("fullTime") or {}

        rows.append(
            {
                "match_id": m.get("id"),
                "utc_date": m.get("utcDate"),
                "status": m.get("status"),
                "matchday": m.get("matchday"),
                "season_start": (m.get("season") or {}).get("startDate"),
                "season_end": (m.get("season") or {}).get("endDate"),
                "home_team": (m.get("homeTeam") or {}).get("name"),
                "away_team": (m.get("awayTeam") or {}).get("name"),
                "home_goals": ft.get("home"),
                "away_goals": ft.get("away"),
                # winner is useful for quick checks; later we’ll compute target label ourselves
                "winner": score.get("winner"),  # HOME_TEAM / AWAY_TEAM / DRAW / None
            }
        )

    df = pd.DataFrame(rows)

    # Convert utc_date string -> actual datetime (so we can sort, split, roll features)
    df["utc_date"] = pd.to_datetime(df["utc_date"], utc=True, errors="coerce")
    return df


def save_raw_matches(df: pd.DataFrame, prefix: str = "pl_matches") -> Path:
    """
    Save a raw immutable snapshot to data/raw/ with a timestamp in the filename.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RAW_DIR / f"{prefix}_{ts}.parquet"

    # Parquet is fast + preserves types better than CSV
    df.to_parquet(path, index=False)
    return path


def log_sanity_stats(df: pd.DataFrame) -> None:
    """
    Print quick sanity checks so we know the ingestion worked.
    """
    print("Rows:", len(df))
    print("Unique match_id:", df["match_id"].nunique())
    print("Date range:", df["utc_date"].min(), "->", df["utc_date"].max())
    print("Status counts (top):")
    print(df["status"].value_counts().head(10))
    print("Winner counts (including NaN):")
    print(df["winner"].value_counts(dropna=False))
