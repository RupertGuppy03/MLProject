import pandas as pd
import pytest
from src.features.rolling import (
    build_team_match_history,
    add_rolling_features,
    add_venue_splits,
    add_extended_rolling_features,
    add_days_rest,
    add_new_team_imputation,
    build_match_level_features,
)


class TestVenueSplitsLeakageSafety:
    """Test that venue-split rolling features are leakage-safe and distinct."""

    def test_venue_splits_differ_for_team_with_different_home_away_form(self):
        """
        Verify that rolling_win_rate_home and rolling_win_rate_away differ
        for a team with different home vs away win rates.

        TeamA: wins at home (matches 1, 3), loses away (matches 2, 4)
        """
        matches = pd.DataFrame(
            {
                "match_id": [1, 2, 3, 4, 5],
                "date": [
                    "2024-01-01",
                    "2024-01-08",
                    "2024-01-15",
                    "2024-01-22",
                    "2024-01-29",
                ],
                "home_team": ["TeamA", "TeamB", "TeamA", "TeamC", "TeamA"],
                "away_team": ["TeamB", "TeamA", "TeamC", "TeamA", "TeamD"],
                "home_goals": [2, 1, 3, 2, 1],
                "away_goals": [0, 0, 1, 0, 1],
            }
        )

        history = build_team_match_history(matches)
        history = add_rolling_features(history, window_size=2)
        history = add_venue_splits(history, window_size=2)
        history = add_extended_rolling_features(history, window_size=2)
        history = add_days_rest(history)
        history = add_new_team_imputation(history)

        match_features = build_match_level_features(history)

        # Match 5: TeamA at home with prior history
        row_match5 = match_features[match_features["match_id"] == 5].iloc[0]

        # TeamA should have different home vs away rolling win rates
        # because: wins matches 1 and 3 (both at home), loses matches 2 and 4 (both away)
        assert "home_rolling_win_rate_home" in match_features.columns
        assert "home_rolling_win_rate_away" in match_features.columns
        assert (
            row_match5["home_rolling_win_rate_home"]
            != row_match5["home_rolling_win_rate_away"]
        ), "Home and away venue splits should differ for TeamA"

    def test_first_match_is_new_team(self):
        """
        Verify that the first match for any team has is_new_team = 1
        and rolling features are imputed safely.
        """
        matches = pd.DataFrame(
            {
                "match_id": [1],
                "date": ["2024-01-01"],
                "home_team": ["NewTeamA"],
                "away_team": ["NewTeamB"],
                "home_goals": [1],
                "away_goals": [0],
            }
        )

        history = build_team_match_history(matches)
        history = add_rolling_features(history, window_size=5)
        history = add_venue_splits(history, window_size=5)
        history = add_extended_rolling_features(history, window_size=5)
        history = add_days_rest(history)
        history = add_new_team_imputation(history)

        match_features = build_match_level_features(history)

        # Both teams are new
        assert match_features.iloc[0]["home_is_new_team"] == 1
        assert match_features.iloc[0]["away_is_new_team"] == 1

        # Rolling features should be imputed, not NaN
        rolling_cols = [
            "home_rolling_win_rate_home",
            "home_rolling_win_rate_away",
            "away_rolling_win_rate_home",
            "away_rolling_win_rate_away",
        ]
        for col in rolling_cols:
            assert not pd.isna(match_features.iloc[0][col]), f"{col} should not be NaN"

    def test_no_data_leakage_shift_by_one(self):
        """
        Verify that rolling features use only PRIOR matches (shift(1)).

        TeamA wins match 1 at home (2-0).
        At match 2 (away), rolling_win_rate_away should still be NaN or imputed
        (no prior away match yet), not include match 2 itself.
        """
        matches = pd.DataFrame(
            {
                "match_id": [1, 2],
                "date": ["2024-01-01", "2024-01-08"],
                "home_team": ["TeamA", "TeamB"],
                "away_team": ["TeamB", "TeamA"],
                "home_goals": [2, 1],
                "away_goals": [0, 3],
            }
        )

        history = build_team_match_history(matches)
        history = add_rolling_features(history, window_size=2)
        history = add_venue_splits(history, window_size=2)
        history = add_days_rest(history)
        history = add_new_team_imputation(history)

        # Check history before match_level conversion
        teamA_match2 = history[
            (history["team"] == "TeamA") & (history["match_id"] == 2)
        ].iloc[0]

        # Match 2 is TeamA away, they lost 1-3.
        # Prior rolling_win_rate_away should NOT include this loss (shift(1) excludes current).
        # Since match 1 was home, rolling_win_rate_away at match 2 should be NaN or a fallback.
        assert teamA_match2["rolling_win_rate_away"] in [0.0, 0.5], (
            "rolling_win_rate_away at match 2 should not include match 2 itself (leakage)"
        )

    def test_venue_splits_columns_created(self):
        """Verify all venue-split columns are created."""
        matches = pd.DataFrame(
            {
                "match_id": [1, 2],
                "date": ["2024-01-01", "2024-01-08"],
                "home_team": ["A", "B"],
                "away_team": ["B", "A"],
                "home_goals": [1, 1],
                "away_goals": [0, 0],
            }
        )

        history = build_team_match_history(matches)
        history = add_rolling_features(history, window_size=2)
        history = add_venue_splits(history, window_size=2)
        history = add_extended_rolling_features(history, window_size=2)
        history = add_days_rest(history)
        history = add_new_team_imputation(history)
        match_features = build_match_level_features(history)

        expected_cols = [
            "home_rolling_win_rate_home",
            "home_rolling_win_rate_away",
            "home_rolling_goals_for_home",
            "home_rolling_goals_for_away",
            "home_rolling_goals_against_home",
            "home_rolling_goals_against_away",
            "away_rolling_win_rate_home",
            "away_rolling_win_rate_away",
            "away_rolling_goals_for_home",
            "away_rolling_goals_for_away",
            "away_rolling_goals_against_home",
            "away_rolling_goals_against_away",
        ]
        for col in expected_cols:
            assert col in match_features.columns, f"Missing column: {col}"

    def test_extended_rolling_features_created(self):
        """Verify extended rolling features (points, goal_diff, clean_sheet) are created."""
        matches = pd.DataFrame(
            {
                "match_id": [1, 2],
                "date": ["2024-01-01", "2024-01-08"],
                "home_team": ["A", "B"],
                "away_team": ["B", "A"],
                "home_goals": [1, 1],
                "away_goals": [0, 0],
            }
        )

        history = build_team_match_history(matches)
        history = add_rolling_features(history, window_size=2)
        history = add_venue_splits(history, window_size=2)
        history = add_extended_rolling_features(history, window_size=2)
        history = add_days_rest(history)
        history = add_new_team_imputation(history)
        match_features = build_match_level_features(history)

        expected_cols = [
            "home_rolling_points_per_game",
            "home_rolling_goal_difference_per_game",
            "home_rolling_clean_sheet_rate",
            "away_rolling_points_per_game",
            "away_rolling_goal_difference_per_game",
            "away_rolling_clean_sheet_rate",
        ]
        for col in expected_cols:
            assert col in match_features.columns, f"Missing extended column: {col}"

    def test_window_size_validation(self):
        """Verify window_size < 1 raises ValueError."""
        matches = pd.DataFrame(
            {
                "match_id": [1],
                "date": ["2024-01-01"],
                "home_team": ["A"],
                "away_team": ["B"],
                "home_goals": [1],
                "away_goals": [0],
            }
        )

        history = build_team_match_history(matches)

        with pytest.raises(ValueError, match="window_size must be >= 1"):
            add_rolling_features(history, window_size=0)

        with pytest.raises(ValueError, match="window_size must be >= 1"):
            add_rolling_features(history, window_size=-1)

    def test_missing_columns_validation(self):
        """Verify missing columns raise ValueError with clear message."""
        bad_df = pd.DataFrame({"wrong_col": [1, 2]})

        with pytest.raises(ValueError, match="missing columns"):
            build_team_match_history(bad_df)

    def test_days_rest_calculated_correctly(self):
        """Verify days_rest is calculated as difference between consecutive match dates."""
        matches = pd.DataFrame(
            {
                "match_id": [1, 2, 3],
                "date": ["2024-01-01", "2024-01-08", "2024-01-15"],
                "home_team": ["TeamA", "TeamA", "TeamA"],
                "away_team": ["TeamB", "TeamB", "TeamB"],
                "home_goals": [1, 1, 1],
                "away_goals": [0, 0, 0],
            }
        )

        history = build_team_match_history(matches)
        history = add_rolling_features(history, window_size=2)
        history = add_venue_splits(history, window_size=2)
        history = add_days_rest(history)

        # First match should have NaN (no prior match)
        first = history[history["match_id"] == 1].iloc[0]
        assert pd.isna(first["days_rest"])

        # Second match: 8 days after first
        second = history[history["match_id"] == 2].iloc[0]
        assert second["days_rest"] == 7  # 2024-01-08 - 2024-01-01 = 7 days

        # Third match: 7 days after second
        third = history[history["match_id"] == 3].iloc[0]
        assert third["days_rest"] == 7  # 2024-01-15 - 2024-01-08 = 7 days

    def test_imputation_fills_nans_safely(self):
        """
        Verify that add_new_team_imputation fills NaNs with sensible defaults
        when league average is also NaN (cold-start edge case).
        """
        matches = pd.DataFrame(
            {
                "match_id": [1],
                "date": ["2024-01-01"],
                "home_team": ["X"],
                "away_team": ["Y"],
                "home_goals": [1],
                "away_goals": [0],
            }
        )

        history = build_team_match_history(matches)
        history = add_rolling_features(history, window_size=5)
        history = add_venue_splits(history, window_size=5)
        history = add_extended_rolling_features(history, window_size=5)
        history = add_days_rest(history)
        history = add_new_team_imputation(history)

        # No NaNs should remain in rolling numeric columns
        rolling_cols = [col for col in history.columns if col.startswith("rolling_")]
        for col in rolling_cols:
            assert not history[col].isna().any(), (
                f"{col} should have no NaN after imputation"
            )

    def test_match_level_merge_correctness(self):
        """
        Verify that match-level features merge correctly on match_id, home_team, away_team.
        """
        matches = pd.DataFrame(
            {
                "match_id": [1, 2],
                "date": ["2024-01-01", "2024-01-08"],
                "home_team": ["A", "B"],
                "away_team": ["B", "A"],
                "home_goals": [1, 0],
                "away_goals": [0, 1],
            }
        )

        history = build_team_match_history(matches)
        history = add_rolling_features(history, window_size=2)
        history = add_venue_splits(history, window_size=2)
        history = add_extended_rolling_features(history, window_size=2)
        history = add_days_rest(history)
        history = add_new_team_imputation(history)
        match_features = build_match_level_features(history)

        # Should have one row per match
        assert len(match_features) == 2

        # Match 1: A home, B away
        match1 = match_features[match_features["match_id"] == 1].iloc[0]
        assert match1["home_team"] == "A"
        assert match1["away_team"] == "B"

        # Match 2: B home, A away
        match2 = match_features[match_features["match_id"] == 2].iloc[0]
        assert match2["home_team"] == "B"
        assert match2["away_team"] == "A"

    def test_all_required_columns_in_output(self):
        """Verify the final match-level output has all expected columns."""
        matches = pd.DataFrame(
            {
                "match_id": [1, 2],
                "date": ["2024-01-01", "2024-01-08"],
                "home_team": ["A", "B"],
                "away_team": ["B", "A"],
                "home_goals": [1, 1],
                "away_goals": [0, 0],
            }
        )

        history = build_team_match_history(matches)
        history = add_rolling_features(history, window_size=2)
        history = add_venue_splits(history, window_size=2)
        history = add_extended_rolling_features(history, window_size=2)
        history = add_days_rest(history)
        history = add_new_team_imputation(history)
        match_features = build_match_level_features(history)

        required_base = ["match_id", "date", "home_team", "away_team"]
        required_features = [
            "home_rolling_win_rate",
            "away_rolling_win_rate",
            "home_days_rest",
            "away_days_rest",
            "home_is_new_team",
            "away_is_new_team",
        ]

        for col in required_base + required_features:
            assert col in match_features.columns, f"Missing required column: {col}"

    def test_no_duplicate_rows_in_output(self):
        """Verify no duplicate rows in the final output."""
        matches = pd.DataFrame(
            {
                "match_id": [1, 2, 3],
                "date": ["2024-01-01", "2024-01-08", "2024-01-15"],
                "home_team": ["A", "B", "A"],
                "away_team": ["B", "C", "C"],
                "home_goals": [1, 1, 2],
                "away_goals": [0, 0, 1],
            }
        )

        history = build_team_match_history(matches)
        history = add_rolling_features(history, window_size=2)
        history = add_venue_splits(history, window_size=2)
        history = add_extended_rolling_features(history, window_size=2)
        history = add_days_rest(history)
        history = add_new_team_imputation(history)
        match_features = build_match_level_features(history)

        assert len(match_features) == len(match_features.drop_duplicates()), (
            "Output should have no duplicate rows"
        )
