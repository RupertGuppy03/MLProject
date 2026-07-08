from src.config import CURRENT_SEASON_TEAMS
from src.ui.team_logos import get_team_logo


class TestTeamLogos:
    def test_every_roster_team_has_a_logo_file(self):
        """Each selectable 25/26 team resolves to an existing crest file."""
        missing = [t for t in CURRENT_SEASON_TEAMS if get_team_logo(t) is None]
        assert not missing, f"Missing logo files for: {missing}"

    def test_unknown_team_returns_none(self):
        assert get_team_logo("Nonexistent FC") is None
