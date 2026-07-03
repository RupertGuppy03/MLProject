from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestTeamsEndpoint:
    """Acceptance tests for GET /teams."""

    def test_teams_sorted_unique_nonempty(self):
        """Acc Test 1: /teams returns a non-empty, sorted, de-duplicated list."""
        resp = client.get("/teams")
        assert resp.status_code == 200

        teams = resp.json()["teams"]
        assert teams, "team list must be non-empty"
        assert teams == sorted(teams), "team list must be sorted"
        assert len(teams) == len(set(teams)), "team list must have no duplicates"

    def test_teams_consistent_with_predict(self):
        """Acc Test 2: any team returned by /teams is accepted by /predict."""
        teams = client.get("/teams").json()["teams"]
        home, away = teams[0], teams[1]

        resp = client.post("/predict", json={"home_team": home, "away_team": away})
        assert resp.status_code == 200
