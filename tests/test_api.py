import pytest
from fastapi.testclient import TestClient

from src.api.main import app

# Integration tests: they exercise the real served artifacts (chosen_model.pkl) and the
# canonical dataset, so team names are pulled from /teams rather than hardcoded.
client = TestClient(app)


def _two_real_teams() -> tuple[str, str]:
    teams = client.get("/teams").json()["teams"]
    assert len(teams) >= 2, "need at least two teams to test /predict"
    return teams[0], teams[1]


class TestHealth:
    def test_health_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestMetadata:
    def test_metadata_returns_dates(self):
        """DoD: /metadata returns last_updated_date and data_through_date."""
        resp = client.get("/metadata")
        assert resp.status_code == 200
        body = resp.json()
        assert body["last_updated_date"]
        assert body["data_through_date"]


class TestPredict:
    def test_predict_returns_probabilities(self):
        """Acc Test 1: p_home/p_draw/p_away present and sum to 1 within tolerance."""
        home, away = _two_real_teams()
        resp = client.post("/predict", json={"home_team": home, "away_team": away})
        assert resp.status_code == 200

        probs = resp.json()["probabilities"]
        total = probs["p_home"] + probs["p_draw"] + probs["p_away"]
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_predict_returns_implied_odds(self):
        """Acc Test 2: implied odds are approximately 1 / probability."""
        home, away = _two_real_teams()
        resp = client.post("/predict", json={"home_team": home, "away_team": away})
        assert resp.status_code == 200

        body = resp.json()
        probs = body["probabilities"]
        odds = body["implied_odds"]
        for outcome, prob_key in [("home", "p_home"), ("draw", "p_draw"), ("away", "p_away")]:
            p = probs[prob_key]
            if p > 0:
                # Odds are rounded to 2dp, so allow a small absolute tolerance.
                assert odds[outcome] == pytest.approx(1.0 / p, abs=0.01)

    def test_predict_unknown_team_returns_400(self):
        """Acc Test 3: unknown team -> 400 with a message listing valid teams."""
        home, _ = _two_real_teams()
        resp = client.post(
            "/predict", json={"home_team": home, "away_team": "Nonexistent FC"}
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "Nonexistent FC" in detail
        assert home in detail  # the valid-teams list is surfaced to the caller

    def test_predict_same_team_returns_400(self):
        """Selecting the same team for both sides is rejected."""
        home, _ = _two_real_teams()
        resp = client.post("/predict", json={"home_team": home, "away_team": home})
        assert resp.status_code == 400
