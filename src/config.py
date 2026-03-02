# src/config.py
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

# Premier League competition code (Football-Data)
PL_COMPETITION_CODE = "PL"

# Football-Data settings (read from .env)
FOOTBALL_DATA_BASE_URL = os.getenv(
    "FOOTBALL_DATA_BASE_URL", "https://api.football-data.org/v4"
)

# Name of the env var that holds your Football-Data API token
FOOTBALL_DATA_API_TOKEN_ENV = "FOOTBALL_DATA_API_TOKEN"
