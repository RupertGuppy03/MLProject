import pandas as pd

from src.features.rolling import add_league_position, DEFAULT_LEAGUE_POSITION


class TestLeaguePosition:
    """Tests for the leakage-safe, season-resetting league position feature."""

    def test_gw1_defaults_to_tenth(self):
        """Every team on the opening matchday has no prior matches, so all default to 10th."""
        matches = pd.DataFrame(
            {
                "match_id": [1, 2],
                "season": [2024, 2024],
                "date": ["2024-08-10", "2024-08-10"],
                "home_team": ["A", "C"],
                "away_team": ["B", "D"],
                "home_goals": [2, 1],
                "away_goals": [0, 0],
            }
        )

        result = add_league_position(matches)

        # All four teams are playing their first match of the season.
        assert (result["home_league_position"] == DEFAULT_LEAGUE_POSITION).all()
        assert (result["away_league_position"] == DEFAULT_LEAGUE_POSITION).all()

    def test_position_excludes_current_match(self):
        """A team that wins match M must still show its PRE-match position at M (no leakage)."""
        matches = pd.DataFrame(
            {
                "match_id": [1, 2],
                "season": [2024, 2024],
                "date": ["2024-08-10", "2024-08-17"],
                "home_team": ["A", "A"],
                "away_team": ["B", "C"],
                "home_goals": [3, 1],
                "away_goals": [0, 0],
            }
        )

        result = add_league_position(matches)

        # Match 1 is A's first match: position is the default, NOT influenced by the 3-0 win.
        match1 = result[result["match_id"] == 1].iloc[0]
        assert match1["home_league_position"] == DEFAULT_LEAGUE_POSITION

        # Match 2: A's win in match 1 now counts, so A is ranked above teams it leads.
        match2 = result[result["match_id"] == 2].iloc[0]
        assert match2["home_league_position"] == 1

    def test_cumulative_points_ranking(self):
        """Side with more cumulative points ranks lower-numbered (points -> GD -> GF)."""
        matches = pd.DataFrame(
            {
                "match_id": [1, 2, 3],
                "season": [2024, 2024, 2024],
                "date": ["2024-08-10", "2024-08-10", "2024-08-17"],
                # Matchday 1: A beats B 2-0, C beats D 1-0.
                "home_team": ["A", "C", "A"],
                "away_team": ["B", "D", "C"],
                "home_goals": [2, 1, 1],
                "away_goals": [0, 0, 0],
            }
        )

        result = add_league_position(matches)

        # Before match 3: A has 3 pts (GD +2), C has 3 pts (GD +1) -> A above C on GD.
        match3 = result[result["match_id"] == 3].iloc[0]
        assert match3["home_league_position"] == 1  # A
        assert match3["away_league_position"] == 2  # C

    def test_position_resets_each_season(self):
        """Positions reset to the default at the start of a new season."""
        matches = pd.DataFrame(
            {
                "match_id": [1, 2, 3],
                "season": [2024, 2024, 2025],
                "date": ["2024-08-10", "2024-08-17", "2025-08-09"],
                "home_team": ["A", "A", "A"],
                "away_team": ["B", "C", "B"],
                "home_goals": [3, 2, 0],
                "away_goals": [0, 0, 0],
            }
        )

        result = add_league_position(matches)

        # Match 3 opens season 2025: A's strong 2024 form is wiped, back to the default.
        match3 = result[result["match_id"] == 3].iloc[0]
        assert match3["home_league_position"] == DEFAULT_LEAGUE_POSITION
        assert match3["away_league_position"] == DEFAULT_LEAGUE_POSITION

    def test_columns_present(self):
        """All three league-position columns must be created."""
        matches = pd.DataFrame(
            {
                "match_id": [1],
                "season": [2024],
                "date": ["2024-08-10"],
                "home_team": ["A"],
                "away_team": ["B"],
                "home_goals": [1],
                "away_goals": [0],
            }
        )

        result = add_league_position(matches)

        for col in [
            "home_league_position",
            "away_league_position",
            "home_position_diff",
        ]:
            assert col in result.columns, f"Missing column: {col}"

    def test_position_diff_formula(self):
        """home_position_diff must equal away_league_position - home_league_position."""
        matches = pd.DataFrame(
            {
                "match_id": [1, 2, 3],
                "season": [2024, 2024, 2024],
                "date": ["2024-08-10", "2024-08-10", "2024-08-17"],
                "home_team": ["A", "C", "A"],
                "away_team": ["B", "D", "C"],
                "home_goals": [2, 1, 1],
                "away_goals": [0, 0, 0],
            }
        )

        result = add_league_position(matches)

        expected = result["away_league_position"] - result["home_league_position"]
        assert (result["home_position_diff"] == expected).all()
