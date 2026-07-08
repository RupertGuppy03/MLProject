"""Thin client for the FastAPI backend, plus small pure helpers.

Kept separate from app.py so the logic here is unit-testable without importing Streamlit
(importing the app would run st.* calls at module load). The dashboard talks to the API only
through these functions.
"""

from __future__ import annotations

import os

import requests

# Base URL of the FastAPI service. Defaults to localhost; the Local-network story makes this
# fully configurable (with CORS) for access from other devices.
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Timeout (seconds) for backend calls so the UI fails fast if the API is down.
_TIMEOUT = 10


def get_teams(base_url: str = API_BASE_URL) -> list[str]:
    """Return the selectable teams from GET /teams."""
    resp = requests.get(f"{base_url}/teams", timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["teams"]


def get_metadata(base_url: str = API_BASE_URL) -> dict:
    """Return the data/artifact freshness dates from GET /metadata."""
    resp = requests.get(f"{base_url}/metadata", timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_elos(base_url: str = API_BASE_URL) -> dict:
    """Return current Elo ratings for every current-season team from GET /elos."""
    resp = requests.get(f"{base_url}/elos", timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def predict(
    home_team: str,
    away_team: str,
    as_of_date: str | None = None,
    base_url: str = API_BASE_URL,
) -> dict:
    """Call POST /predict and return the response payload.

    A 400 from the API (unknown/duplicate team) is surfaced as a ValueError carrying the API's
    `detail` message so the UI can display it directly.
    """
    payload = {"home_team": home_team, "away_team": away_team, "as_of_date": as_of_date}
    resp = requests.post(f"{base_url}/predict", json=payload, timeout=_TIMEOUT)
    if resp.status_code == 400:
        raise ValueError(resp.json().get("detail", "Invalid request."))
    resp.raise_for_status()
    return resp.json()


def selection_error(home_team: str, away_team: str) -> str | None:
    """Return an error message if the team selection is invalid, else None.

    Guards against picking the same team on both sides (checked client-side before calling the
    API so the user gets instant feedback).
    """
    if home_team == away_team:
        return "Please select two different teams."
    return None


def format_outcome(outcome: str, home_team: str, away_team: str) -> str:
    """Map the API's predicted_outcome code to a human-readable label."""
    return {
        "home_win": f"{home_team} win",
        "draw": "Draw",
        "away_win": f"{away_team} win",
    }.get(outcome, outcome)
