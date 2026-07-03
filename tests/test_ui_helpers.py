from src.ui.api_client import format_outcome, selection_error


class TestSelectionError:
    """Client-side guard for the team selection (Acc Test 1)."""

    def test_same_team_returns_message(self):
        msg = selection_error("Arsenal FC", "Arsenal FC")
        assert msg is not None
        assert "different" in msg.lower()

    def test_different_teams_returns_none(self):
        assert selection_error("Arsenal FC", "Chelsea FC") is None


class TestFormatOutcome:
    """Mapping the API's predicted_outcome code to a readable label."""

    def test_home_win(self):
        assert format_outcome("home_win", "Arsenal FC", "Chelsea FC") == "Arsenal FC win"

    def test_away_win(self):
        assert format_outcome("away_win", "Arsenal FC", "Chelsea FC") == "Chelsea FC win"

    def test_draw(self):
        assert format_outcome("draw", "Arsenal FC", "Chelsea FC") == "Draw"
