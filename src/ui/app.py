"""Streamlit dashboard for the Premier League match predictor.

Basic version: pick a home and away team, run the prediction, and see outcome probabilities,
implied odds, and the predicted outcome. The rich match-context/SHAP panel is a later story.

Run:  streamlit run src/ui/app.py   (with the API running: uvicorn src.api.main:app --reload)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit runs this file as a standalone script, so the project root isn't on sys.path.
# Add it so `import src...` resolves (mirrors how uvicorn/pytest put the repo root on the path).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import requests  # noqa: E402
import streamlit as st  # noqa: E402

from src.ui.api_client import (  # noqa: E402
    format_outcome,
    get_teams,
    predict,
    selection_error,
)

st.set_page_config(page_title="PL Match Predictor", page_icon="⚽", layout="centered")

st.title("Premier League Match Predictor")
st.caption("Select two teams to get win / draw / win probabilities and implied odds.")

# Load the selectable teams from the API. If the API is unreachable, guide the user to start it.
try:
    teams = get_teams()
except requests.RequestException:
    st.error(
        "Cannot reach the API. Start it in another terminal with:\n\n"
        "`uvicorn src.api.main:app --reload`"
    )
    st.stop()

# Team selection: default the away side to a different team so the form is valid by default.
col_home, col_away = st.columns(2)
home_team = col_home.selectbox("Home team", teams, index=0)
away_team = col_away.selectbox("Away team", teams, index=1 if len(teams) > 1 else 0)

if st.button("Run Prediction", type="primary"):
    # Client-side guard: prevent selecting the same team on both sides (Acc Test 1).
    error = selection_error(home_team, away_team)
    if error:
        st.error(error)
        st.stop()

    try:
        result = predict(home_team, away_team)
    except ValueError as exc:  # 400 from the API — show its message
        st.error(str(exc))
        st.stop()
    except requests.RequestException:
        st.error(
            "Cannot reach the API. Start it with: `uvicorn src.api.main:app --reload`"
        )
        st.stop()

    probs = result["probabilities"]
    odds = result["implied_odds"]

    # Predicted outcome headline.
    st.subheader(f"Predicted: {format_outcome(result['predicted_outcome'], home_team, away_team)}")

    # Three outcome cards: probability as a percentage + implied odds beneath.
    cols = st.columns(3)
    cards = [
        (cols[0], f"{home_team} win", probs["p_home"], odds["home"]),
        (cols[1], "Draw", probs["p_draw"], odds["draw"]),
        (cols[2], f"{away_team} win", probs["p_away"], odds["away"]),
    ]
    for col, label, prob, odd in cards:
        col.metric(label, f"{prob * 100:.1f}%")
        col.caption(f"Implied odds: {odd:.2f}" if odd is not None else "Implied odds: —")
