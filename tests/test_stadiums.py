"""Stadium coordinate coverage for the dashboard 3D map."""

from src.config import CURRENT_SEASON_TEAMS
from src.ui.stadiums import get_stadium


class TestStadiums:
    def test_every_current_season_team_has_a_stadium(self):
        """Every selectable team resolves to a stadium with name + coordinates."""
        missing = [t for t in CURRENT_SEASON_TEAMS if get_stadium(t) is None]
        assert not missing, f"no stadium coordinates for: {missing}"
        for team in CURRENT_SEASON_TEAMS:
            venue = get_stadium(team)
            assert venue["stadium"]
            assert isinstance(venue["lat"], float)
            assert isinstance(venue["lon"], float)

    def test_unknown_team_returns_none(self):
        assert get_stadium("Nonexistent FC") is None
