from __future__ import annotations

import math

import pandas as pd
import pytest

from src.features.elo import (
    DEFAULT_HOME_ADVANTAGE,
    DEFAULT_K_FACTOR,
    DEFAULT_REGRESSION_FACTOR,
    DEFAULT_REGRESSION_TARGET,
    DEFAULT_STARTING_ELO,
    EloState,
    compute_elo_features,
    load_state,
    save_state,
)


def _matches(rows: list[dict]) -> pd.DataFrame:
    """Tiny helper: build a canonical-shaped frame for tests."""
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


# Acc Test 1 + DoD: leakage safety
def test_pre_match_elo_independent_of_current_match_result():
    """Changing match M's own result must not change M's pre-match Elo features."""
    base = [
        {"match_id": "m1", "date": "2024-08-10", "season": 2024,
         "home_team": "Arsenal", "away_team": "Chelsea", "result": "HW"},
        {"match_id": "m2", "date": "2024-08-17", "season": 2024,
         "home_team": "Liverpool", "away_team": "Arsenal", "result": "HW"},
    ]
    with_hw, _ = compute_elo_features(_matches(base))

    flipped = [dict(r) for r in base]
    flipped[1]["result"] = "AW"
    with_aw, _ = compute_elo_features(_matches(flipped))

    # Match 2's pre-match Elo cannot depend on match 2's own outcome.
    row_hw = with_hw[with_hw["match_id"] == "m2"].iloc[0]
    row_aw = with_aw[with_aw["match_id"] == "m2"].iloc[0]
    assert row_hw["elo_home_pre"] == row_aw["elo_home_pre"]
    assert row_hw["elo_away_pre"] == row_aw["elo_away_pre"]


# Acc Test 1 + DoD: updates occur AFTER feature computation
def test_pre_match_elo_for_next_match_reflects_prior_match_update():
    rows = [
        {"match_id": "m1", "date": "2024-08-10", "season": 2024,
         "home_team": "Arsenal", "away_team": "Chelsea", "result": "HW"},
        {"match_id": "m2", "date": "2024-08-17", "season": 2024,
         "home_team": "Arsenal", "away_team": "Liverpool", "result": "D"},
    ]
    df, _ = compute_elo_features(_matches(rows))
    m1 = df[df["match_id"] == "m1"].iloc[0]
    m2 = df[df["match_id"] == "m2"].iloc[0]

    # Arsenal won at home in m1, so by m2 their Elo must have risen above starting.
    assert m1["elo_home_pre"] == DEFAULT_STARTING_ELO
    assert m2["elo_home_pre"] > DEFAULT_STARTING_ELO


# Acc Test 2: elo_diff column present
def test_elo_diff_column_equals_home_minus_away():
    rows = [
        {"match_id": "m1", "date": "2024-08-10", "season": 2024,
         "home_team": "Arsenal", "away_team": "Chelsea", "result": "HW"},
    ]
    df, _ = compute_elo_features(_matches(rows))
    assert "elo_diff" in df.columns
    row = df.iloc[0]
    assert row["elo_diff"] == row["elo_home_pre"] - row["elo_away_pre"]


def test_starting_elo_for_new_team_is_1450():
    state = EloState()
    assert state.get("Newcastle") == DEFAULT_STARTING_ELO


def test_k_factor_drives_post_match_update_magnitude():
    """After one match between equal-rated teams, Elo change must equal K * (actual - expected)."""
    state = EloState()
    elo_home = state.get("A")
    elo_away = state.get("B")
    expected_home = state.expected_home(elo_home, elo_away)

    state.update("A", "B", "HW")

    expected_new_a = elo_home + DEFAULT_K_FACTOR * (1.0 - expected_home)
    expected_new_b = elo_away + DEFAULT_K_FACTOR * (0.0 - (1.0 - expected_home))
    assert math.isclose(state.ratings["A"], expected_new_a, rel_tol=1e-9)
    assert math.isclose(state.ratings["B"], expected_new_b, rel_tol=1e-9)


def test_home_advantage_makes_equal_teams_favourite_at_home():
    state = EloState()
    p_home = state.expected_home(DEFAULT_STARTING_ELO, DEFAULT_STARTING_ELO)
    assert p_home > 0.5
    # With +60 home advantage and equal ratings, P(home win) ~ 0.586.
    expected = 1.0 / (1.0 + 10 ** (-DEFAULT_HOME_ADVANTAGE / 400.0))
    assert math.isclose(p_home, expected, rel_tol=1e-9)


def test_season_regression_pulls_toward_target_at_first_match_of_new_season():
    rows = [
        {"match_id": "m1", "date": "2024-08-10", "season": 2024,
         "home_team": "Arsenal", "away_team": "Chelsea", "result": "HW"},
        {"match_id": "m2", "date": "2025-08-09", "season": 2025,
         "home_team": "Arsenal", "away_team": "Chelsea", "result": "D"},
    ]
    df, _ = compute_elo_features(_matches(rows))

    m1 = df[df["match_id"] == "m1"].iloc[0]
    m2 = df[df["match_id"] == "m2"].iloc[0]

    # Capture Arsenal's post-m1 Elo by replaying the single update manually.
    replay = EloState()
    replay.regress_for_season("Arsenal", 2024)
    replay.regress_for_season("Chelsea", 2024)
    replay.update("Arsenal", "Chelsea", "HW")
    post_m1_arsenal = replay.ratings["Arsenal"]

    expected_m2_arsenal = post_m1_arsenal + DEFAULT_REGRESSION_FACTOR * (
        DEFAULT_REGRESSION_TARGET - post_m1_arsenal
    )
    assert math.isclose(m2["elo_home_pre"], expected_m2_arsenal, rel_tol=1e-9)
    # And m1's pre-match Elo was still starting Elo (no prior season).
    assert m1["elo_home_pre"] == DEFAULT_STARTING_ELO


def test_no_regression_within_same_season():
    rows = [
        {"match_id": "m1", "date": "2024-08-10", "season": 2024,
         "home_team": "Arsenal", "away_team": "Chelsea", "result": "HW"},
        {"match_id": "m2", "date": "2024-09-01", "season": 2024,
         "home_team": "Arsenal", "away_team": "Liverpool", "result": "HW"},
    ]
    df, state = compute_elo_features(_matches(rows))

    # Independently apply just m1's update — m2's pre-match Elo for Arsenal
    # should match exactly (no regression between m1 and m2).
    replay = EloState()
    replay.regress_for_season("Arsenal", 2024)
    replay.regress_for_season("Chelsea", 2024)
    replay.update("Arsenal", "Chelsea", "HW")

    m2 = df[df["match_id"] == "m2"].iloc[0]
    assert math.isclose(m2["elo_home_pre"], replay.ratings["Arsenal"], rel_tol=1e-9)
    assert state.last_season["Arsenal"] == 2024


def test_unplayed_match_produces_features_but_does_not_update_state():
    rows = [
        {"match_id": "m1", "date": "2024-08-10", "season": 2024,
         "home_team": "Arsenal", "away_team": "Chelsea", "result": None},
        {"match_id": "m2", "date": "2024-08-17", "season": 2024,
         "home_team": "Arsenal", "away_team": "Liverpool", "result": None},
    ]
    df, state = compute_elo_features(_matches(rows))

    assert df["elo_home_pre"].notna().all()
    assert df["elo_diff"].notna().all()
    # No play means no updates: every team still at starting Elo.
    for team in ("Arsenal", "Chelsea", "Liverpool"):
        assert state.get(team) == DEFAULT_STARTING_ELO


def test_save_and_load_round_trip_preserves_state(tmp_path):
    rows = [
        {"match_id": "m1", "date": "2024-08-10", "season": 2024,
         "home_team": "Arsenal", "away_team": "Chelsea", "result": "HW"},
        {"match_id": "m2", "date": "2025-08-09", "season": 2025,
         "home_team": "Liverpool", "away_team": "Arsenal", "result": "AW"},
    ]
    _, state = compute_elo_features(_matches(rows))

    path = tmp_path / "current_elo.json"
    save_state(state, path, as_of_date="2025-08-09")
    loaded = load_state(path)

    assert loaded.ratings == state.ratings
    assert loaded.last_season == state.last_season
    assert loaded.k_factor == DEFAULT_K_FACTOR
    assert loaded.home_advantage == DEFAULT_HOME_ADVANTAGE
    assert loaded.starting_elo == DEFAULT_STARTING_ELO
    assert loaded.regression_target == DEFAULT_REGRESSION_TARGET
    assert loaded.regression_factor == DEFAULT_REGRESSION_FACTOR


def test_missing_required_columns_raises():
    bad = pd.DataFrame({"match_id": ["m1"], "date": ["2024-08-10"]})
    with pytest.raises(ValueError, match="missing columns"):
        compute_elo_features(bad)
