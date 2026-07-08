"""About page: what the project is, how it works, and how to read the dashboard."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("About this project")

    st.markdown(
        """
**Premier League Match Predictor** estimates win / draw / win probabilities for any fixture
between two current-season clubs, with implied odds and a full explanation of *why* the model
favoured an outcome. It's an end-to-end machine-learning portfolio project: data pipeline →
model → API → dashboard.
        """
    )

    st.subheader("How it works")
    st.markdown(
        """
- **Model:** a tuned **Random Forest** classifier. It's the main model; Elo and logistic
  regression were built as baselines.
- **Features (leakage-safe):** sequential **Elo** ratings, rolling 5-match form (win rate, goals
  for/against, points-per-game, clean sheets), home/away venue splits, days rest, and pre-match
  league position. Every feature is computed using only matches *before* the one being predicted.
- **Explanations:** per-prediction **SHAP** values from the Random Forest show the top features
  driving each result.
- **Serving:** a **FastAPI** backend builds the one-row feature vector and serves the model; this
  **Streamlit** dashboard calls it.
        """
    )

    st.subheader("Data")
    st.markdown(
        """
Match data comes from **Football-Data.org** (seasons 2023–2025). The app is based on the completed
**2025/26** season; when the first 2026/27 data pull arrives, the roster and features roll over to
the new season.
        """
    )

    st.subheader("Reading the dashboard")
    st.markdown(
        """
- **Probability cards** — the three outcome probabilities and implied odds; the model's pick is
  flagged.
- **Elo rating history** — each team's Elo entering its recent matches.
- **Rolling goals** — 5-match goals scored (solid) vs conceded (dashed).
- **Home / away splits** and **team radar** — attack/defence profiles at a glance.
- **Head-to-head** — the sides' historical record.
- **SHAP contributors** — green pushes toward the predicted outcome, red away from it.
        """
    )

    st.caption(
        "Club crests are the trademarks of their respective clubs and are used here for a "
        "non-commercial, educational portfolio project."
    )
