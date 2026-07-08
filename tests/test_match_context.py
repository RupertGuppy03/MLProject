import pandas as pd

from src.api.match_context import build_match_context


def make_history() -> pd.DataFrame:
    """Synthetic season. A and B meet three times up to 2024-08-24 (A win, draw, B win),
    with two later A-B / filler matches that must be ignored when as_of = 2024-08-24."""
    return pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(1, 11)],
            "date": [
                "2024-08-03", "2024-08-03",
                "2024-08-10", "2024-08-10",
                "2024-08-17", "2024-08-17",
                "2024-08-24", "2024-08-24",
                "2024-08-31", "2024-08-31",  # after as_of
            ],
            "season": [2024] * 10,
            "home_team": ["A", "C", "A", "B", "B", "D", "A", "C", "A", "C"],
            "away_team": ["B", "D", "C", "D", "A", "C", "B", "D", "B", "D"],
            "home_goals": [2, 1, 1, 0, 1, 2, 0, 0, 3, 1],
            "away_goals": [0, 1, 0, 1, 1, 2, 1, 0, 0, 0],
            "result": ["HW", "D", "HW", "AW", "D", "D", "AW", "D", "HW", "HW"],
        }
    )


class TestMatchContext:
    def test_head_to_head_counts(self):
        """From A's perspective: one A win, one draw, one B win (matches up to as_of)."""
        ctx = build_match_context("A", "B", as_of_date="2024-08-24", matches=make_history())
        assert ctx["head_to_head"] == {
            "home_wins": 1,
            "draws": 1,
            "away_wins": 1,
            "total": 3,
        }

    def test_future_matches_do_not_leak(self):
        """Altering a match dated after as_of must not change the context."""
        history = make_history()
        as_of = "2024-08-24"

        baseline = build_match_context("A", "B", as_of_date=as_of, matches=history)

        tampered = history.copy()
        after = pd.to_datetime(tampered["date"]) > pd.to_datetime(as_of)
        tampered.loc[after, ["home_goals", "away_goals"]] = [9, 0]
        tampered.loc[after, "result"] = "HW"

        after_ctx = build_match_context("A", "B", as_of_date=as_of, matches=tampered)
        assert after_ctx == baseline

    def test_context_structure(self):
        ctx = build_match_context("A", "B", as_of_date="2024-08-24", matches=make_history())
        for side in ("home", "away"):
            block = ctx[side]
            assert block["team"] in {"A", "B"}
            assert block["elo_trajectory"], "trajectory should not be empty"
            assert len(block["form"]) <= 5
            assert set(block["venue_splits"].keys()) == {
                "home_goals_for",
                "home_goals_against",
                "away_goals_for",
                "away_goals_against",
            }
