import altair as alt
import matplotlib
import pydeck as pdk

matplotlib.use("Agg")  # headless backend for tests

from src.ui.charts import (  # noqa: E402
    elo_history_chart,
    radar_figure,
    rolling_goals_chart,
    shap_chart,
    stadium_map,
    venue_splits_chart,
)


def _team_block() -> dict:
    return {
        "team": "T",
        "current_elo": 1600.0,
        "elo_trajectory": [1550.0, 1560.0, 1575.0],
        "form": ["W", "D", "L"],
        "rolling_goals_scored": [1.4, 1.6, 1.8],
        "rolling_goals_conceded": [1.0, 0.8, 1.2],
        "venue_splits": {
            "home_goals_for": 1.8,
            "home_goals_against": 0.9,
            "away_goals_for": 1.2,
            "away_goals_against": 1.3,
        },
        "ppg": 1.7,
        "clean_sheets": 4,
        "league_position": 6,
    }


def _context() -> dict:
    return {
        "as_of_date": "2026-05-24",
        "home": {**_team_block(), "team": "Arsenal FC"},
        "away": {**_team_block(), "team": "Chelsea FC"},
        "head_to_head": {"home_wins": 2, "draws": 1, "away_wins": 1, "total": 4},
    }


def _explanation() -> dict:
    return {
        "predicted_class": "home_win",
        "top_features": [
            {"feature": "elo_diff", "contribution": 0.05},
            {"feature": "home_league_position", "contribution": -0.02},
        ],
    }


class TestCharts:
    def test_altair_charts_build(self):
        ctx = _context()
        for builder in (elo_history_chart, rolling_goals_chart, venue_splits_chart):
            assert isinstance(builder(ctx, "Arsenal FC", "Chelsea FC"), alt.Chart)
        assert isinstance(shap_chart(_explanation()), alt.Chart)

    def test_radar_builds_for_both_themes(self):
        ctx = _context()
        for dark in (True, False):
            fig = radar_figure(ctx, "Arsenal FC", "Chelsea FC", dark=dark)
            assert fig is not None

    def test_stadium_map_builds(self):
        elos = {
            "Arsenal FC": 1694.0,
            "Chelsea FC": 1560.0,
            "Liverpool FC": 1650.0,
            "Wolverhampton Wanderers FC": 1470.0,
        }
        deck = stadium_map(elos, "Arsenal FC", "Chelsea FC")
        assert isinstance(deck, pdk.Deck)
