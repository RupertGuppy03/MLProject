from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.baseline_elo import EloBaseline, run_elo_baseline


def _make_frames():
    """Small synthetic X/y/meta with the Elo columns the baseline consumes."""
    X = pd.DataFrame(
        {
            "elo_home_pre": [1500.0, 1700.0, 1300.0, 1500.0],
            "elo_away_pre": [1500.0, 1300.0, 1700.0, 1500.0],
        }
    )
    y = pd.Series(["HW", "HW", "AW", "D"])
    meta = pd.DataFrame(
        {
            "match_id": ["m1", "m2", "m3", "m4"],
            "date": ["2024-08-10", "2024-08-17", "2024-08-24", "2024-08-31"],
            "season": [2024, 2024, 2024, 2024],
        }
    )
    return X, y, meta


class TestEloBaseline:
    """Tests for the Elo-only baseline (Sprint 2)."""

    def test_probabilities_sum_to_one(self):
        """AT1: each match's three probabilities sum to 1."""
        X, y, _ = _make_frames()
        probs = EloBaseline().fit(y).predict_proba(X["elo_home_pre"], X["elo_away_pre"])
        row_sums = probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-9)

    def test_outputs_three_probability_columns(self):
        """AT1: output has exactly three columns (home, draw, away)."""
        X, y, _ = _make_frames()
        probs = EloBaseline().fit(y).predict_proba(X["elo_home_pre"], X["elo_away_pre"])
        assert probs.shape == (len(X), 3)
        assert (probs >= 0).all()

    def test_stronger_home_has_higher_p_home(self):
        """A higher home Elo should yield a higher home win probability."""
        model = EloBaseline(draw_rate=0.25)
        weak = model.predict_proba([1400.0], [1500.0])[0, 0]
        strong = model.predict_proba([1700.0], [1500.0])[0, 0]
        assert strong > weak

    def test_draw_rate_fitted_from_labels(self):
        """fit() sets draw_rate to the empirical frequency of draws in y."""
        y = pd.Series(["HW", "D", "AW", "D"])  # 50% draws
        model = EloBaseline().fit(y)
        assert model.draw_rate == 0.5

    def test_runs_across_dataset(self):
        """AT2: the runner produces one valid row per match with no NaNs."""
        X, y, meta = _make_frames()
        preds = run_elo_baseline(X, y, meta, save=False)
        assert len(preds) == len(X)
        assert not preds[["p_home", "p_draw", "p_away"]].isna().any().any()
        sums = preds[["p_home", "p_draw", "p_away"]].sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-9)

    def test_predictions_schema(self):
        """The predictions frame has the backtest-compatible columns."""
        X, y, meta = _make_frames()
        preds = run_elo_baseline(X, y, meta, save=False)
        expected = {
            "match_id",
            "date",
            "season",
            "model",
            "p_home",
            "p_draw",
            "p_away",
            "y_true",
        }
        assert expected.issubset(preds.columns)
        assert (preds["model"] == "baseline_elo").all()
