from __future__ import annotations

import pandas as pd

from src.features.rolling import (
    add_days_rest,
    add_extended_rolling_features,
    add_new_team_imputation,
    add_rolling_features,
    add_venue_splits,
    build_match_level_features,
    build_team_match_history,
)


def _build_match_features(matches: pd.DataFrame) -> pd.DataFrame:
    history = build_team_match_history(matches)
    history = add_rolling_features(history, window_size=2)
    history = add_venue_splits(history, window_size=2)
    history = add_extended_rolling_features(history, window_size=2)
    history = add_days_rest(history)
    history = add_new_team_imputation(history)
    return build_match_level_features(history)


def test_home_new_team_gets_flagged() -> None:
    matches = pd.DataFrame(
        {
            "match_id": [1, 2],
            "date": ["2024-01-01", "2024-01-08"],
            "home_team": ["VeteranFC", "PromotedFC"],
            "away_team": ["AwayFC", "VeteranFC"],
            "home_goals": [2, 1],
            "away_goals": [0, 0],
        }
    )

    match_features = _build_match_features(matches)
    promoted_row = match_features[match_features["match_id"] == 2].iloc[0]

    assert promoted_row["home_team"] == "PromotedFC"
    assert promoted_row["home_is_new_team"] == 1
    assert promoted_row["away_is_new_team"] == 0


def test_away_new_team_gets_flagged() -> None:
    matches = pd.DataFrame(
        {
            "match_id": [1, 2],
            "date": ["2024-01-01", "2024-01-08"],
            "home_team": ["VeteranFC", "VeteranFC"],
            "away_team": ["AwayFC", "PromotedFC"],
            "home_goals": [2, 1],
            "away_goals": [0, 0],
        }
    )

    match_features = _build_match_features(matches)
    promoted_row = match_features[match_features["match_id"] == 2].iloc[0]

    assert promoted_row["away_team"] == "PromotedFC"
    assert promoted_row["home_is_new_team"] == 0
    assert promoted_row["away_is_new_team"] == 1


def test_cold_start_defaults_fill_rolling_columns_and_days_rest() -> None:
    matches = pd.DataFrame(
        {
            "match_id": [1],
            "date": ["2024-01-01"],
            "home_team": ["PromotedHome"],
            "away_team": ["PromotedAway"],
            "home_goals": [1],
            "away_goals": [0],
        }
    )

    match_features = _build_match_features(matches)
    row = match_features.iloc[0]

    assert row["home_is_new_team"] == 1
    assert row["away_is_new_team"] == 1
    assert row["home_days_rest"] == 7
    assert row["away_days_rest"] == 7

    rolling_cols = [
        "home_rolling_win_rate",
        "home_rolling_goals_for_avg",
        "home_rolling_goals_against_avg",
        "home_rolling_win_rate_home",
        "home_rolling_win_rate_away",
        "home_rolling_goals_for_home",
        "home_rolling_goals_for_away",
        "home_rolling_goals_against_home",
        "home_rolling_goals_against_away",
        "home_rolling_points_per_game",
        "home_rolling_goal_difference_per_game",
        "home_rolling_clean_sheet_rate",
        "away_rolling_win_rate",
        "away_rolling_goals_for_avg",
        "away_rolling_goals_against_avg",
        "away_rolling_win_rate_home",
        "away_rolling_win_rate_away",
        "away_rolling_goals_for_home",
        "away_rolling_goals_for_away",
        "away_rolling_goals_against_home",
        "away_rolling_goals_against_away",
        "away_rolling_points_per_game",
        "away_rolling_goal_difference_per_game",
        "away_rolling_clean_sheet_rate",
    ]

    for col in rolling_cols:
        assert pd.notna(row[col]), f"{col} should be filled for cold-start teams"


def test_imputation_preserves_existing_non_nan_values() -> None:
    df = pd.DataFrame(
        {
            "team": ["VeteranFC"],
            "date": [pd.Timestamp("2024-01-01")],
            "match_id": [1],
            "is_home": [1],
            "days_rest": [pd.NA],
            "rolling_win_rate": [0.73],
            "rolling_goals_for_avg": [1.9],
            "rolling_goals_against_avg": [0.8],
            "rolling_win_rate_home": [0.81],
            "rolling_win_rate_away": [0.24],
            "rolling_goals_for_home": [2.1],
            "rolling_goals_for_away": [0.7],
            "rolling_goals_against_home": [0.6],
            "rolling_goals_against_away": [1.4],
            "rolling_points_per_game": [2.2],
            "rolling_goal_difference_per_game": [1.3],
            "rolling_clean_sheet_rate": [0.5],
        }
    )

    result = add_new_team_imputation(df)

    assert result.loc[0, "days_rest"] == 7
    assert result.loc[0, "is_new_team"] == 1
    assert result.loc[0, "rolling_win_rate"] == 0.73
    assert result.loc[0, "rolling_goals_for_avg"] == 1.9
    assert result.loc[0, "rolling_goals_against_avg"] == 0.8
    assert result.loc[0, "rolling_win_rate_home"] == 0.81
    assert result.loc[0, "rolling_win_rate_away"] == 0.24
    assert result.loc[0, "rolling_goals_for_home"] == 2.1
    assert result.loc[0, "rolling_goals_for_away"] == 0.7
    assert result.loc[0, "rolling_goals_against_home"] == 0.6
    assert result.loc[0, "rolling_goals_against_away"] == 1.4
    assert result.loc[0, "rolling_points_per_game"] == 2.2
    assert result.loc[0, "rolling_goal_difference_per_game"] == 1.3
    assert result.loc[0, "rolling_clean_sheet_rate"] == 0.5
