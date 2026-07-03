from src.api.inference_features import get_valid_teams
from src.ui.team_colors import DEFAULT_COLOR, TEAM_COLORS, get_team_color


class TestTeamColors:
    """Team brand-colour mapping used by the dashboard."""

    def test_known_team_returns_exact_hex(self):
        assert get_team_color("Arsenal FC") == "#EF0107"
        assert get_team_color("Liverpool FC") == "#C8102E"
        assert get_team_color("Chelsea FC") == "#034694"

    def test_unknown_team_returns_default(self):
        assert get_team_color("Nonexistent FC") == DEFAULT_COLOR

    def test_all_values_are_hex(self):
        """Every mapped colour is a 6-digit hex string."""
        for color in TEAM_COLORS.values():
            assert color.startswith("#") and len(color) == 7

    def test_every_dataset_team_has_a_colour(self):
        """Coverage: no team in the canonical dataset falls back to the default grey."""
        missing = [t for t in get_valid_teams() if t not in TEAM_COLORS]
        assert not missing, f"Teams missing a brand colour: {missing}"
