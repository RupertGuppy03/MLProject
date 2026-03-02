# src/config.py
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

# Premier League competition code
PL_COMPETITION_CODE = "PL"

# Read base URL from .env, fallback to default
FOOTBALL_DATA_BASE_URL = os.getenv(
    "FOOTBALL_API_BASE_URL", "https://api.football-data.org/v4"
)

# Tell the client which env var contains the API key
FOOTBALL_DATA_API_TOKEN_ENV = "FOOTBALL_API_KEY"
