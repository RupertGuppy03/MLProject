"""Streamlit dashboard for the Premier League match predictor.

Pick a home and away team, run the prediction, and see outcome probabilities, implied odds, and a
rich explanation panel: recent form, Elo trajectory, rolling goals, home/away splits, a team radar,
head-to-head, model confidence, and per-prediction SHAP contributions — all from the API.

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
    get_teams,
    predict,
    selection_error,
)
from src.ui.charts import (  # noqa: E402
    elo_trajectory_chart,
    radar_figure,
    rolling_goals_chart,
    shap_chart,
    venue_splits_chart,
)
from src.ui.team_colors import get_team_color  # noqa: E402

# Neutral amber used for the Draw outcome (matches the reference mock).
DRAW_COLOR = "#EAB308"
# Form-chip colours (W/D/L).
FORM_COLORS = {"W": "#22C55E", "D": "#EAB308", "L": "#EF4444"}


def form_chips(sequence: list[str]) -> str:
    """Render a W/D/L sequence as coloured chips (HTML)."""
    chips = [
        f"<span style='background:{FORM_COLORS[r]}22;color:{FORM_COLORS[r]};"
        f"padding:4px 10px;border-radius:6px;font-weight:700;margin-right:4px'>{r}</span>"
        for r in sequence
    ]
    return "".join(chips) or "—"


def colored_name(team: str, color: str | None = None) -> str:
    """Render a team name (or label) as bold HTML in its brand colour."""
    color = color or get_team_color(team)
    return f"<span style='color:{color};font-weight:700'>{team}</span>"


st.set_page_config(page_title="PL Match Predictor", page_icon="⚽", layout="wide")

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
# Streamlit can't colour the dropdown options themselves, so show the selected team's name in its
# brand colour just beneath each selector for instant feedback.
col_home, col_away = st.columns(2)
home_team = col_home.selectbox("Home team", teams, index=0)
col_home.markdown(colored_name(home_team), unsafe_allow_html=True)
away_team = col_away.selectbox("Away team", teams, index=1 if len(teams) > 1 else 0)
col_away.markdown(colored_name(away_team), unsafe_allow_html=True)

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

    # Predicted-outcome headline, with the winning team's name in its brand colour.
    outcome = result["predicted_outcome"]
    if outcome == "home_win":
        headline = f"Predicted: {colored_name(home_team)} win"
    elif outcome == "away_win":
        headline = f"Predicted: {colored_name(away_team)} win"
    else:
        headline = f"Predicted: {colored_name('Draw', DRAW_COLOR)}"
    st.markdown(f"<h3>{headline}</h3>", unsafe_allow_html=True)

    # Three outcome cards: coloured label, probability as a percentage, implied odds beneath.
    cols = st.columns(3)
    cards = [
        (cols[0], f"{home_team} win", get_team_color(home_team), probs["p_home"], odds["home"]),
        (cols[1], "Draw", DRAW_COLOR, probs["p_draw"], odds["draw"]),
        (cols[2], f"{away_team} win", get_team_color(away_team), probs["p_away"], odds["away"]),
    ]
    for col, label, color, prob, odd in cards:
        col.markdown(colored_name(label, color), unsafe_allow_html=True)
        col.markdown(
            f"<div style='font-size:2.4rem;font-weight:700'>{prob * 100:.1f}%</div>",
            unsafe_allow_html=True,
        )
        col.caption(f"Implied odds: {odd:.2f}" if odd is not None else "Implied odds: —")

    # ---- Explanation panel (all data from the API context/explanation) ----
    context = result["context"]
    explanation = result["explanation"]
    home_block, away_block = context["home"], context["away"]

    st.divider()

    # Model confidence = the highest outcome probability.
    max_prob = max(probs["p_home"], probs["p_draw"], probs["p_away"])
    st.markdown("**Model confidence**")
    st.progress(int(round(max_prob * 100)), text=f"{max_prob * 100:.0f}% on the favoured outcome")

    # Last-5 form for each team.
    fcol1, fcol2 = st.columns(2)
    for col, team, block in ((fcol1, home_team, home_block), (fcol2, away_team, away_block)):
        col.markdown(f"Last 5 · {colored_name(team)}", unsafe_allow_html=True)
        col.markdown(form_chips(block["form"]), unsafe_allow_html=True)
        col.caption(
            f"PPG {block['ppg']:.1f} · Position #{block['league_position']} "
            f"· Clean sheets {block['clean_sheets']}"
        )

    # Elo trajectory and rolling goals.
    st.subheader("Elo rating trajectory")
    st.altair_chart(elo_trajectory_chart(context, home_team, away_team), use_container_width=True)

    st.subheader("Goals scored vs conceded — rolling 5-match")
    st.altair_chart(rolling_goals_chart(context, home_team, away_team), use_container_width=True)

    # Splits + radar side by side.
    scol, rcol = st.columns(2)
    scol.subheader("Home / away splits")
    scol.altair_chart(venue_splits_chart(context, home_team, away_team), use_container_width=True)
    rcol.subheader("Team profile radar")
    rcol.pyplot(radar_figure(context, home_team, away_team))

    # Head-to-head record.
    st.subheader("Head-to-head record")
    h2h = context["head_to_head"]
    h1, h2, h3 = st.columns(3)
    h1.metric(f"{home_team} wins", h2h["home_wins"])
    h2.metric("Draws", h2h["draws"])
    h3.metric(f"{away_team} wins", h2h["away_wins"])
    st.caption(f"{h2h['total']} previous meetings")

    # Per-prediction SHAP contributions.
    st.subheader("Why this prediction — top SHAP contributors")
    st.altair_chart(shap_chart(explanation), use_container_width=True)
    st.caption(
        f"Green pushes toward the predicted outcome "
        f"({result['predicted_outcome'].replace('_', ' ')}), red away from it."
    )
