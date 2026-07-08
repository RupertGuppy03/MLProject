"""About page: what the project is, how it works, and how to read the dashboard."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("About this project")

    st.markdown(
        """
**Premier League Match Predictor** takes any two clubs from the current season and estimates the
chance of a home win, draw, or away win, along with the betting odds those probabilities imply and a
breakdown of *why* the model leaned the way it did.

It's a portfolio project built to show a complete machine-learning system end-to-end: from pulling
raw data, turning it into useful signals, training and testing a model for classification, serving it
through an API, presenting it in this dashboard, and finally deploying it where it's accessible to
anyone.
        """
    )

    st.subheader("How it works")
    st.markdown(
        """
The app runs as a pipeline, each stage feeding the next: **data → features → model → API →
dashboard**.

**The model.** The main predictor is a tuned **Random Forest** — a model that combines hundreds of
decision trees and averages their votes to reach a prediction. Two simpler models were also built as
benchmarks to make sure the Random Forest was actually adding value: an **Elo** rating system (the
same idea used in chess rankings) and a **logistic regression** model. The Random Forest was chosen
because it predicts well and can explain its reasoning, achieving the lowest log loss and Brier
scores — key metrics for a model's reliability.

**The features.** Many features were engineered to bring new data to the model and improve accuracy.
These are the signals the model learns from, and they're all built to be **leakage-safe**: every
number is calculated using only matches played *before* the one being predicted. Nothing about the
match itself, or anything after it, ever sneaks into the prediction. This is the single most
important rule in the whole project — a model that accidentally peeks at future results may look
accurate in testing but will fall apart in the real world. Some of the key features:
        """
    )
    st.markdown(
        """
- **Elo rating** — a running strength score for each team that rises after wins and falls after
  losses.
- **Recent form** — rolling stats over each team's last five matches: win rate, goals scored and
  conceded, points per game, and clean sheets.
- **Home/away splits** — how each team performs specifically at home versus away, since many sides
  are much stronger on their own ground.
- **Days of rest** — how long since each team last played, a rough proxy for fatigue.
        """
    )
    st.markdown(
        """
**The explanations.** For every prediction, the app uses **SHAP** — a technique that breaks a single
prediction down into the exact contribution of each feature — to show which signals pushed the
result toward the predicted outcome and which pushed against it. So instead of a black-box answer,
you see the actual reasoning.

**Serving it.** A **FastAPI** backend builds the feature vector for the chosen fixture and runs it
through the model, returning the probabilities, odds, explanation, and all the supporting match
context. This **Streamlit** dashboard calls that API and renders everything you see.
        """
    )

    st.subheader("The data")
    st.markdown(
        """
Match data comes from **Football-Data.org**, covering the 2023 to 2026 seasons. The app is currently
based on the completed **2025/26** season, so every prediction reflects how teams actually performed
across that full campaign. When the first 2026/27 data arrives, the roster and features roll over to
the new season, and the model is automatically retrained on the new data.
        """
    )

    st.subheader("Reading the dashboard")
    st.markdown(
        """
Each panel answers a different question about the fixture:

- **Probability cards** — the three outcome chances and their implied odds, with the model's pick
  highlighted. Implied odds are just the probability expressed the way a bookmaker would (a 50%
  chance shows as odds of 2.00).
- **Elo rating history** — each team's strength score over recent matches, so you can see who's
  trending up and who's sliding.
- **Rolling goals** — goals scored (solid line) versus conceded (dashed line) over the last five
  matches, a quick read on whether a team is attacking or defending strongly.
- **Home/away splits and team radar** — attack and defence profiles side by side, so you can compare
  the two teams' strengths and weaknesses at a glance.
- **Head-to-head** — the historical record between these two specific clubs.
- **SHAP contributors** — the features that drove this prediction: green bars pushed toward the
  predicted outcome, red bars pushed away from it.
        """
    )

    st.markdown(
        """
Overall, this project represents the full lifecycle of a modern ML project — from gathering the data
from an API, to training the model and displaying the results, to deployment.

For further details, or if you'd like to run this project locally, check out my GitHub under
**ML-Project**: https://github.com/RupertGuppy03/MLProject

If you have any questions, feel free to contact me at **rupertguppy03@gmail.com**.
        """
    )

    
