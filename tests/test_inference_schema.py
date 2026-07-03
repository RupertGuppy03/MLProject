import json

import pandas as pd
import pytest

from src.features.build_features import SCHEMA_PATH
from src.api.inference_features import build_inference_features, get_valid_teams


def make_history() -> pd.DataFrame:
    """A small canonical-shaped season where every team has several prior matches,
    so rolling/Elo/position features are all populated for an A-vs-B inference row."""
    return pd.DataFrame(
        {
            "match_id": ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"],
            "date": [
                "2024-08-01",
                "2024-08-01",
                "2024-08-08",
                "2024-08-08",
                "2024-08-15",
                "2024-08-15",
                "2024-08-22",
                "2024-08-22",
            ],
            "season": [2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024],
            "home_team": ["A", "C", "A", "B", "A", "C", "B", "D"],
            "away_team": ["B", "D", "C", "D", "D", "B", "A", "C"],
            "home_goals": [2, 1, 1, 0, 3, 2, 1, 0],
            "away_goals": [0, 1, 0, 2, 1, 2, 1, 0],
            "result": ["HW", "D", "HW", "AW", "HW", "D", "D", "D"],
        }
    )


class TestInferenceFeatures:
    """Acceptance tests for the inference-time feature builder."""

    def test_default_as_of_uses_latest_date(self):
        """Acc Test 1: omitting as_of_date uses the latest match date in the dataset."""
        history = make_history()
        latest = pd.to_datetime(history["date"]).max()

        default_row = build_inference_features("A", "B", matches=history)
        explicit_row = build_inference_features(
            "A", "B", as_of_date=latest, matches=history
        )

        pd.testing.assert_frame_equal(default_row, explicit_row)

    def test_features_match_schema_exactly(self):
        """Acc Test 2: the one-row feature vector matches the locked schema exactly."""
        schema = json.loads(SCHEMA_PATH.read_text())
        expected_columns = schema["feature_columns"]
        expected_dtypes = schema["dtypes"]

        x_row = build_inference_features("A", "B", matches=make_history())

        assert len(x_row) == 1, "inference must produce exactly one row"
        assert list(x_row.columns) == expected_columns, "columns/order must match schema"
        for col, dtype in expected_dtypes.items():
            assert str(x_row[col].dtype) == dtype, f"dtype mismatch on {col}"

    def test_unknown_team_raises_with_valid_list(self):
        """An unknown team is rejected with a message listing valid teams."""
        with pytest.raises(ValueError) as excinfo:
            build_inference_features("A", "Nonexistent FC", matches=make_history())

        message = str(excinfo.value)
        assert "Nonexistent FC" in message
        # The error surfaces the valid teams so the caller (API) can echo them.
        assert "A" in message and "B" in message

    def test_future_matches_do_not_leak(self):
        """Leakage guard: data dated after as_of_date must not change the output row."""
        history = make_history()
        as_of = "2024-08-15"

        baseline = build_inference_features("A", "B", as_of_date=as_of, matches=history)

        # Mutate a match that falls AFTER the as-of date; it must be ignored.
        tampered = history.copy()
        after_mask = pd.to_datetime(tampered["date"]) > pd.to_datetime(as_of)
        tampered.loc[after_mask, ["home_goals", "away_goals"]] = [9, 0]
        tampered.loc[after_mask, "result"] = "HW"

        after = build_inference_features("A", "B", as_of_date=as_of, matches=tampered)

        pd.testing.assert_frame_equal(baseline, after)

    def test_get_valid_teams_sorted_unique_nonempty(self):
        """get_valid_teams returns a sorted, de-duplicated, non-empty list."""
        teams = get_valid_teams(make_history())

        assert teams == ["A", "B", "C", "D"]
        assert len(teams) == len(set(teams)), "no duplicates"
        assert teams == sorted(teams), "sorted"
        assert teams, "non-empty"
