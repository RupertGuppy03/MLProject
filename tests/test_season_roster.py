from fastapi.testclient import TestClient

from src.api.main import app
from src.config import CURRENT_SEASON_TEAMS

client = TestClient(app)

# Teams in the dataset but NOT in the 2025/26 season — must not be selectable.
NOT_IN_SEASON = [
    "Ipswich Town FC",
    "Leicester City FC",
    "Luton Town FC",
    "Sheffield United FC",
    "Southampton FC",
]


class TestSeasonRoster:
    """The /teams endpoint reflects the 25/26 roster, not the full historical dataset."""

    def test_teams_is_the_roster(self):
        teams = client.get("/teams").json()["teams"]
        assert teams == sorted(CURRENT_SEASON_TEAMS)
        assert len(teams) == 20

    def test_season_teams_present(self):
        teams = client.get("/teams").json()["teams"]
        for team in ["Arsenal FC", "Wolverhampton Wanderers FC", "Sunderland AFC"]:
            assert team in teams

    def test_out_of_season_teams_absent(self):
        teams = client.get("/teams").json()["teams"]
        for team in NOT_IN_SEASON:
            assert team not in teams, f"{team} should not be selectable in 25/26"

    def test_every_selectable_team_predicts(self):
        """Every roster team is valid for prediction (they're all in the dataset)."""
        teams = client.get("/teams").json()["teams"]
        resp = client.post(
            "/predict", json={"home_team": teams[0], "away_team": teams[1]}
        )
        assert resp.status_code == 200
