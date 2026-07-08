"""Match-context builder for the explanation panel.

Assembles the real, leakage-safe data behind every chart in the dashboard mock: current Elo and
Elo trajectory, last-5 form, rolling goals scored/conceded, home/away venue splits, points-per-game,
league position, and head-to-head record. Everything is computed from matches strictly on or before
the as-of date, reusing the same feature functions as training/inference (no re-implemented maths,
no hardcoded values).
"""

from __future__ import annotations

import pandas as pd

from src.api.inference_features import _load_canonical, get_valid_teams
from src.features.elo import compute_elo_features
from src.features.rolling import (
    add_extended_rolling_features,
    add_league_position,
    add_rolling_features,
    add_venue_splits,
    build_team_match_history,
)

# How many recent matches to show in the trajectory charts / form sequence.
TRAJECTORY_LEN = 10
FORM_LEN = 5


def _clean(value: float) -> float:
    """JSON-safe float: NaN -> 0.0, rounded."""
    return 0.0 if pd.isna(value) else round(float(value), 3)


def _team_rows(history: pd.DataFrame, team: str) -> pd.DataFrame:
    """A team's own long-form rows, chronological."""
    return history[history["team"] == team].sort_values(["date", "match_id"])


def _elo_trajectory(elo_df: pd.DataFrame, team: str, n: int = TRAJECTORY_LEN) -> list[float]:
    """Pre-match Elo entering each of the team's last n matches (read from elo.py output)."""
    sub = elo_df[
        (elo_df["home_team"] == team) | (elo_df["away_team"] == team)
    ].sort_values(["date", "match_id"])
    vals = [
        float(r["elo_home_pre"] if r["home_team"] == team else r["elo_away_pre"])
        for _, r in sub.iterrows()
    ]
    return [round(v, 1) for v in vals[-n:]]


def _form(hist_team: pd.DataFrame, n: int = FORM_LEN) -> list[str]:
    """Last n results as W/D/L (chronological)."""
    tail = hist_team.tail(n)
    return ["W" if r["win"] else ("D" if r["draw"] else "L") for _, r in tail.iterrows()]


def _team_block(
    team: str,
    elo_df: pd.DataFrame,
    state,
    hist_full: pd.DataFrame,
    position: int,
) -> dict:
    """Build one team's context block."""
    rows = _team_rows(hist_full, team)
    last = rows.iloc[-1]
    recent = rows.tail(TRAJECTORY_LEN)

    return {
        "team": team,
        "current_elo": round(float(state.get(team)), 1),
        "elo_trajectory": _elo_trajectory(elo_df, team),
        "form": _form(rows),
        "rolling_goals_scored": [_clean(v) for v in recent["rolling_goals_for_avg"]],
        "rolling_goals_conceded": [_clean(v) for v in recent["rolling_goals_against_avg"]],
        "venue_splits": {
            "home_goals_for": _clean(last["rolling_goals_for_home"]),
            "home_goals_against": _clean(last["rolling_goals_against_home"]),
            "away_goals_for": _clean(last["rolling_goals_for_away"]),
            "away_goals_against": _clean(last["rolling_goals_against_away"]),
        },
        "ppg": _clean(last["rolling_points_per_game"]),
        "clean_sheets": int(recent["clean_sheet"].sum()),
        "league_position": position,
    }


def _head_to_head(history: pd.DataFrame, home: str, away: str) -> dict:
    """Historical record between the two teams, from the current home team's perspective."""
    mask = (
        (history["home_team"] == home) & (history["away_team"] == away)
    ) | ((history["home_team"] == away) & (history["away_team"] == home))
    meetings = history[mask]

    home_wins = draws = away_wins = 0
    for _, r in meetings.iterrows():
        if pd.isna(r["home_goals"]) or pd.isna(r["away_goals"]):
            continue
        # Re-orient goals so gf/ga are always for the *current* home team.
        if r["home_team"] == home:
            gf, ga = r["home_goals"], r["away_goals"]
        else:
            gf, ga = r["away_goals"], r["home_goals"]
        if gf > ga:
            home_wins += 1
        elif gf == ga:
            draws += 1
        else:
            away_wins += 1

    return {
        "home_wins": home_wins,
        "draws": draws,
        "away_wins": away_wins,
        "total": home_wins + draws + away_wins,
    }


def _latest_position(pos_df: pd.DataFrame, team: str) -> int:
    """The team's league position at its most recent match (pre-match snapshot)."""
    sub = pos_df[
        (pos_df["home_team"] == team) | (pos_df["away_team"] == team)
    ].sort_values(["date", "match_id"])
    if sub.empty:
        return 0
    last = sub.iloc[-1]
    col = "home_league_position" if last["home_team"] == team else "away_league_position"
    return int(last[col])


def build_match_context(
    home_team: str,
    away_team: str,
    as_of_date: str | pd.Timestamp | None = None,
    matches: pd.DataFrame | None = None,
) -> dict:
    """Build the full match-context dictionary for a fixture (leakage-safe, prior to as_of_date)."""
    df = _load_canonical(matches)

    valid = get_valid_teams(df)
    unknown = [t for t in (home_team, away_team) if t not in valid]
    if unknown:
        raise ValueError(f"Unknown team(s): {unknown}. Valid teams: {valid}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    as_of = df["date"].max() if as_of_date is None else pd.to_datetime(as_of_date)
    history = df[df["date"] <= as_of].reset_index(drop=True)
    if history.empty:
        raise ValueError(f"No matches on or before as_of_date {as_of.date()}.")

    # Elo (current + trajectory) from elo.py — not recomputed independently.
    elo_df, state = compute_elo_features(history)

    # Rolling form / goals / venue splits / ppg / clean sheets (same functions as training).
    hist_full = build_team_match_history(history)
    hist_full = add_rolling_features(hist_full)
    hist_full = add_extended_rolling_features(hist_full)
    hist_full = add_venue_splits(hist_full)

    pos_df = add_league_position(history)

    return {
        "as_of_date": as_of.date().isoformat(),
        "home": _team_block(
            home_team, elo_df, state, hist_full, _latest_position(pos_df, home_team)
        ),
        "away": _team_block(
            away_team, elo_df, state, hist_full, _latest_position(pos_df, away_team)
        ),
        "head_to_head": _head_to_head(history, home_team, away_team),
    }
