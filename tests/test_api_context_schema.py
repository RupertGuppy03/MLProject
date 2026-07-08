from fastapi.testclient import TestClient

from src.api.main import app

# Integration: /predict against the real artifacts + canonical dataset. Verifies the response
# carries every field the dashboard consumes (Acc Test: rich match context + SHAP).
client = TestClient(app)

TEAM_BLOCK_FIELDS = {
    "team",
    "current_elo",
    "elo_trajectory",
    "form",
    "rolling_goals_scored",
    "rolling_goals_conceded",
    "venue_splits",
    "ppg",
    "clean_sheets",
    "league_position",
}


def _predict_two_teams():
    teams = client.get("/teams").json()["teams"]
    resp = client.post("/predict", json={"home_team": teams[0], "away_team": teams[1]})
    assert resp.status_code == 200
    return resp.json()


class TestContextSchema:
    def test_response_has_context_and_explanation(self):
        body = _predict_two_teams()
        assert "context" in body
        assert "explanation" in body

    def test_team_blocks_have_all_fields(self):
        ctx = _predict_two_teams()["context"]
        for side in ("home", "away"):
            assert TEAM_BLOCK_FIELDS <= set(ctx[side].keys()), f"missing fields in {side}"
        # Venue splits carry the four home/away attack/defence numbers.
        assert set(ctx["home"]["venue_splits"].keys()) == {
            "home_goals_for",
            "home_goals_against",
            "away_goals_for",
            "away_goals_against",
        }

    def test_head_to_head_present(self):
        h2h = _predict_two_teams()["context"]["head_to_head"]
        assert set(h2h.keys()) == {"home_wins", "draws", "away_wins", "total"}
        assert h2h["total"] == h2h["home_wins"] + h2h["draws"] + h2h["away_wins"]

    def test_shap_top_five_signed(self):
        explanation = _predict_two_teams()["explanation"]
        feats = explanation["top_features"]
        assert len(feats) == 5
        for f in feats:
            assert "feature" in f and "contribution" in f
            assert isinstance(f["contribution"], float)

    def test_trajectories_are_non_empty(self):
        ctx = _predict_two_teams()["context"]
        for side in ("home", "away"):
            assert ctx[side]["elo_trajectory"], "elo trajectory should not be empty"
            assert ctx[side]["rolling_goals_scored"], "rolling goals should not be empty"
