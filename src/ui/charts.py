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
import pydeck as pdk

from src.ui.stadiums import get_stadium
from src.ui.team_colors import get_team_color
from src.ui.team_display import display_name

# Form / SHAP accent colours.
WIN_COLOR = "#22C55E"
LOSS_COLOR = "#EF4444"

# Friendly names for the cryptic model feature columns shown in the SHAP panel.
FEATURE_LABELS: dict[str, str] = {
    "elo_diff": "Elo difference",
    "elo_home_pre": "Home Elo",
    "elo_away_pre": "Away Elo",
    "home_league_position": "Home league position",
    "away_league_position": "Away league position",
    "home_position_diff": "League position gap",
    "home_rolling_points_per_game": "Home recent PPG",
    "away_rolling_points_per_game": "Away recent PPG",
    "home_rolling_win_rate": "Home recent win rate",
    "away_rolling_win_rate": "Away recent win rate",
    "home_rolling_goals_for_avg": "Home recent goals scored",
    "away_rolling_goals_for_avg": "Away recent goals scored",
    "home_rolling_goals_against_avg": "Home recent goals conceded",
    "away_rolling_goals_against_avg": "Away recent goals conceded",
}

# Force integer x-axis ticks (a match count can't be fractional).
_MATCH_AXIS = alt.Axis(tickMinStep=1, format="d")


def _prettify(feature: str) -> str:
    """Fallback label for any feature not in FEATURE_LABELS."""
    text = feature.replace("home_", "Home ").replace("away_", "Away ")
    text = text.replace("rolling_", "").replace("_", " ")
    return text[:1].upper() + text[1:]


def _team_scale(home: str, away: str) -> alt.Scale:
    """Colour scale keyed by display name, coloured by the canonical team's brand colour."""
    return alt.Scale(
        domain=[display_name(home), display_name(away)],
        range=[get_team_color(home), get_team_color(away)],
    )


def _right_aligned_x(n: int) -> list[int]:
    """Positions ending at 0 (latest), so different-length series align on the most recent match."""
    return list(range(-(n - 1), 1))


def elo_history_chart(context: dict, home: str, away: str) -> alt.Chart:
    """Two Elo lines (one per team) over their recent matches."""
    rows = []
    for side, team in (("home", home), ("away", away)):
        traj = context[side]["elo_trajectory"]
        for x, elo in zip(_right_aligned_x(len(traj)), traj):
            rows.append({"match": x, "team": display_name(team), "elo": elo})
    df = pd.DataFrame(rows)
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("match:Q", title="Recent matches (0 = latest)", axis=_MATCH_AXIS),
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
        for x, v in zip(_right_aligned_x(len(scored)), scored):
            rows.append({"match": x, "team": display_name(team), "metric": "Scored", "goals": v})
        for x, v in zip(_right_aligned_x(len(conceded)), conceded):
            rows.append({"match": x, "team": display_name(team), "metric": "Conceded", "goals": v})
    df = pd.DataFrame(rows)
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("match:Q", title="Recent matches (0 = latest)", axis=_MATCH_AXIS),
            y=alt.Y("goals:Q", title="Rolling-5 goals"),
            color=alt.Color("team:N", scale=_team_scale(home, away), title=None),
            strokeDash=alt.StrokeDash(
                "metric:N",
                # Pin patterns: Scored = solid, Conceded = dashed (don't let Altair auto-order).
                scale=alt.Scale(domain=["Scored", "Conceded"], range=[[1, 0], [6, 4]]),
                # Hide from the legend — explained by a caption under the chart instead.
                legend=None,
            ),
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
            rows.append({"metric": label, "team": display_name(team), "value": splits[key]})
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
    df["label"] = df["feature"].map(lambda f: FEATURE_LABELS.get(f, _prettify(f)))
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("contribution:Q", title="SHAP contribution"),
            y=alt.Y("label:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum.contribution > 0, alt.value(WIN_COLOR), alt.value(LOSS_COLOR)
            ),
        )
        .properties(height=220)
    )


def radar_figure(context: dict, home: str, away: str, dark: bool = True):
    """Matplotlib polar radar comparing both teams across six normalised axes.

    `dark` controls the label/grid colours and a transparent background so the figure blends into
    whichever Streamlit theme is active.
    """
    axes = ["Attack", "Defence", "Form", "Elo", "Clean Sheets", "H2H"]
    h2h = context["head_to_head"]
    total = max(h2h["total"], 1)

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def team_values(side: str, wins: int) -> list[float]:
        block = context[side]
        return [
            min(1.0, _mean(block["rolling_goals_scored"]) / 3),  # Attack
            max(0.0, 1 - min(1.0, _mean(block["rolling_goals_conceded"]) / 3)),  # Defence
            min(1.0, block["ppg"] / 3),  # Form
            min(1.0, max(0.0, (block["current_elo"] - 1400) / 400)),  # Elo
            min(1.0, block["clean_sheets"] / 10),  # Clean sheets
            wins / total,  # H2H
        ]

    # Colours chosen for contrast against the active Streamlit theme.
    fg = "#E5E7EB" if dark else "#1F2937"  # axis/legend text
    grid = "#6B7280" if dark else "#9CA3AF"  # spokes and rings

    angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={"polar": True})
    # Transparent so whichever theme background shows through.
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    for side, team, wins in (("home", home, h2h["home_wins"]), ("away", away, h2h["away_wins"])):
        vals = team_values(side, wins)
        vals += vals[:1]
        color = get_team_color(team)
        ax.plot(angles, vals, color=color, linewidth=2, label=display_name(team))
        ax.fill(angles, vals, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes, fontsize=8, color=fg)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.tick_params(colors=fg)
    ax.grid(color=grid, alpha=0.6)
    ax.spines["polar"].set_color(grid)
    legend = ax.legend(
        loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=7, framealpha=0
    )
    for text in legend.get_texts():
        text.set_color(fg)
    fig.tight_layout()
    return fig


# --- Stadium map ---

# Elo bar-height tuning: subtract a baseline so bars start near the ground, then a large scale so
# the ~1400–1700 spread reads as tens of km of height on a country-wide 3D map.
_ELO_BASELINE = 1350.0
_ELEVATION_SCALE = 150
_GREY = [130, 130, 130, 120]  # de-emphasised colour for non-selected clubs


def _hex_to_rgb(hex_color: str) -> list[int]:
    """Convert a '#RRGGBB' brand colour to an [r, g, b] list for pydeck."""
    h = hex_color.lstrip("#")
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)]


def stadium_map(elos: dict, home: str, away: str) -> pdk.Deck:
    """3D map of Premier League stadiums as hexagonal bars, height driven by current Elo.

    `elos` is a mapping of canonical team name -> current Elo (from GET /elos). The selected home
    and away teams are drawn in their brand colours; every other club is greyed out. The view is
    centred on the home team's stadium.
    """
    selected = {home, away}
    rows = []
    for team, elo in elos.items():
        venue = get_stadium(team)
        if venue is None:  # no coordinates for this club -> skip it
            continue
        if team in selected:
            color = _hex_to_rgb(get_team_color(team)) + [255]
        else:
            color = _GREY
        rows.append(
            {
                "team": display_name(team),
                "stadium": venue["stadium"],
                "lat": venue["lat"],
                "lon": venue["lon"],
                "elo": round(elo),
                "elevation": max(elo - _ELO_BASELINE, 20.0),
                "color": color,
            }
        )
    df = pd.DataFrame(rows)

    # Hexagonal columns (disk_resolution=6) so each stadium is one bar at a controlled height.
    columns = pdk.Layer(
        "ColumnLayer",
        data=df,
        get_position=["lon", "lat"],
        get_elevation="elevation",
        elevation_scale=_ELEVATION_SCALE,
        radius=4000,
        disk_resolution=6,
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
        extruded=True,
    )
    # Stadium-name labels sitting at each bar's base.
    labels = pdk.Layer(
        "TextLayer",
        data=df,
        get_position=["lon", "lat"],
        get_text="stadium",
        get_size=12,
        get_color=[230, 230, 230, 220],
        get_alignment_baseline="'top'",
    )

    # Centre on the home stadium; fall back to a Midlands-ish centre if it has no coords.
    home_venue = get_stadium(home) or {"lat": 52.9, "lon": -1.5}
    view = pdk.ViewState(
        latitude=home_venue["lat"],
        longitude=home_venue["lon"],
        zoom=5.2,
        pitch=45,
        bearing=0,
    )

    return pdk.Deck(
        layers=[columns, labels],
        initial_view_state=view,
        map_style="dark_no_labels",
        tooltip={"text": "{team}\n{stadium}\nElo {elo}"},
    )
