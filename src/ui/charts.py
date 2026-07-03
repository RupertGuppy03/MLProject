"""Chart builders for the dashboard explanation panel.

Pure functions that turn the API's `context` / `explanation` payload into Altair charts and a
Matplotlib radar figure. Kept out of app.py so the chart logic is importable and app.py stays a
thin layout. Every chart reads real values from the payload — nothing is hardcoded. Team colours
come from `get_team_color`.
"""

from __future__ import annotations

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ui.team_colors import get_team_color

# Form / SHAP accent colours.
WIN_COLOR = "#22C55E"
LOSS_COLOR = "#EF4444"


def _team_scale(home: str, away: str) -> alt.Scale:
    """Altair colour scale mapping each team to its brand colour."""
    return alt.Scale(domain=[home, away], range=[get_team_color(home), get_team_color(away)])


def _right_aligned_x(n: int) -> list[int]:
    """Positions ending at 0 (latest), so different-length series align on the most recent match."""
    return list(range(-(n - 1), 1))


def elo_trajectory_chart(context: dict, home: str, away: str) -> alt.Chart:
    """Two Elo lines (one per team) over their recent matches."""
    rows = []
    for side, team in (("home", home), ("away", away)):
        traj = context[side]["elo_trajectory"]
        for x, elo in zip(_right_aligned_x(len(traj)), traj):
            rows.append({"match": x, "team": team, "elo": elo})
    df = pd.DataFrame(rows)
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("match:Q", title="Recent matches (0 = latest)"),
            y=alt.Y("elo:Q", title="Elo", scale=alt.Scale(zero=False)),
            color=alt.Color("team:N", scale=_team_scale(home, away), title=None),
        )
        .properties(height=220)
    )


def rolling_goals_chart(context: dict, home: str, away: str) -> alt.Chart:
    """Rolling-5 goals scored (solid) vs conceded (dashed), per team."""
    rows = []
    for side, team in (("home", home), ("away", away)):
        scored = context[side]["rolling_goals_scored"]
        conceded = context[side]["rolling_goals_conceded"]
        xs = _right_aligned_x(len(scored))
        for x, v in zip(xs, scored):
            rows.append({"match": x, "team": team, "metric": "Scored", "goals": v})
        for x, v in zip(_right_aligned_x(len(conceded)), conceded):
            rows.append({"match": x, "team": team, "metric": "Conceded", "goals": v})
    df = pd.DataFrame(rows)
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("match:Q", title="Recent matches (0 = latest)"),
            y=alt.Y("goals:Q", title="Rolling-5 goals"),
            color=alt.Color("team:N", scale=_team_scale(home, away), title=None),
            strokeDash=alt.StrokeDash("metric:N", title=None),
        )
        .properties(height=220)
    )


def venue_splits_chart(context: dict, home: str, away: str) -> alt.Chart:
    """Grouped bars comparing home/away attack & defence for both teams."""
    labels = {
        "home_goals_for": "Home Att",
        "home_goals_against": "Home Def",
        "away_goals_for": "Away Att",
        "away_goals_against": "Away Def",
    }
    rows = []
    for side, team in (("home", home), ("away", away)):
        splits = context[side]["venue_splits"]
        for key, label in labels.items():
            rows.append({"metric": label, "team": team, "value": splits[key]})
    df = pd.DataFrame(rows)
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("metric:N", title=None, sort=list(labels.values())),
            y=alt.Y("value:Q", title="Rolling-5 goals"),
            color=alt.Color("team:N", scale=_team_scale(home, away), title=None),
            xOffset="team:N",
        )
        .properties(height=240)
    )


def shap_chart(explanation: dict) -> alt.Chart:
    """Horizontal bars of the top signed SHAP contributions (green = toward, red = away)."""
    df = pd.DataFrame(explanation["top_features"])
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("contribution:Q", title="SHAP contribution"),
            y=alt.Y("feature:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum.contribution > 0, alt.value(WIN_COLOR), alt.value(LOSS_COLOR)
            ),
        )
        .properties(height=220)
    )


def radar_figure(context: dict, home: str, away: str):
    """Matplotlib polar radar comparing both teams across six normalised axes."""
    axes = ["Attack", "Defence", "Form", "Elo", "Clean Sheets", "H2H"]
    h2h = context["head_to_head"]
    total = max(h2h["total"], 1)

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def team_values(side: str, wins: int) -> list[float]:
        block = context[side]
        return [
            min(1.0, _mean(block["rolling_goals_scored"]) / 3),          # Attack
            max(0.0, 1 - min(1.0, _mean(block["rolling_goals_conceded"]) / 3)),  # Defence
            min(1.0, block["ppg"] / 3),                                   # Form
            min(1.0, max(0.0, (block["current_elo"] - 1400) / 400)),      # Elo
            min(1.0, block["clean_sheets"] / 10),                         # Clean sheets
            wins / total,                                                 # H2H
        ]

    angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={"polar": True})
    for side, team, wins in (("home", home, h2h["home_wins"]), ("away", away, h2h["away_wins"])):
        vals = team_values(side, wins)
        vals += vals[:1]
        color = get_team_color(team)
        ax.plot(angles, vals, color=color, linewidth=2, label=team)
        ax.fill(angles, vals, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes, fontsize=8)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=7)
    fig.tight_layout()
    return fig
