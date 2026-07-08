# src/config.py
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Premier League competition code (Football-Data)
PL_COMPETITION_CODE = "PL"

# Football-Data settings (read from .env)
FOOTBALL_DATA_BASE_URL = os.getenv(
    "FOOTBALL_DATA_BASE_URL", "https://api.football-data.org/v4"
)

# Name of the env var that holds your Football-Data API token
FOOTBALL_DATA_API_TOKEN_ENV = "FOOTBALL_DATA_API_TOKEN"

# --- Current-season roster (drives the selectable teams in the app) ---
# The 20 clubs that competed in the 2025/26 Premier League. The app is based on this completed
# season until the first 2026/27 data pull, at which point this list rolls over. All 20 are present
# in the dataset with real data, so this is purely a selection filter (no new-team imputation).
# The five dataset teams NOT in 25/26 (Ipswich, Leicester, Luton, Sheffield United, Southampton)
# stay in the training data as historical opponents but are not selectable.
CURRENT_SEASON = "2025/26"

CURRENT_SEASON_TEAMS = [
    "AFC Bournemouth",
    "Arsenal FC",
    "Aston Villa FC",
    "Brentford FC",
    "Brighton & Hove Albion FC",
    "Burnley FC",
    "Chelsea FC",
    "Crystal Palace FC",
    "Everton FC",
    "Fulham FC",
    "Leeds United FC",
    "Liverpool FC",
    "Manchester City FC",
    "Manchester United FC",
    "Newcastle United FC",
    "Nottingham Forest FC",
    "Sunderland AFC",
    "Tottenham Hotspur FC",
    "West Ham United FC",
    "Wolverhampton Wanderers FC",
]
