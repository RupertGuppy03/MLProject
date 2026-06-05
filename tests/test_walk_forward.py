from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.walk_forward import (
    _brier,
    _expanding_folds,
    _log_loss,
    elo_backtest_model,
    log_reg_backtest_model,
    walk_forward,
)


def _make_frames(n_per_season: int = 30):
    """3 seasons of synthetic matches with Elo columns + 2 features; all classes each season."""
    rng = np.random.default_rng(0)
    seasons = [2023, 2024, 2025]
    X_rows, y_vals, meta_rows = [], [], []
    mid = 0
    for s in seasons:
        base = pd.Timestamp(f"{s}-08-01")
        for i in range(n_per_season):
            label = ["HW", "D", "AW"][i % 3]
            eh, ea = 1500 + rng.normal(0, 50), 1500 + rng.normal(0, 50)
            if label == "HW":
                eh += 120
            elif label == "AW":
                ea += 120
            X_rows.append(
                {"elo_home_pre": eh, "elo_away_pre": ea, "elo_diff": eh - ea,
                 "f1": rng.normal(), "f2": rng.normal()}
            )
            y_vals.append(label)
            meta_rows.append({"match_id": f"m{mid}", "date": base + pd.Timedelta(days=3 * i), "season": s})
            mid += 1
    return pd.DataFrame(X_rows), pd.Series(y_vals, name="result"), pd.DataFrame(meta_rows)


class TestWalkForward:
    """Tests for the walk-forward backtesting engine (Sprint 2)."""

    def test_folds_no_leakage(self):
        """AT1: each fold trains strictly before it tests, with no overlap."""
        _, _, meta = _make_frames()
        meta = meta.copy()
        meta["date"] = pd.to_datetime(meta["date"])
        folds = _expanding_folds(meta, initial_train=30, step=10)
        assert len(folds) > 1
        for train_pos, test_pos in folds:
            assert set(train_pos).isdisjoint(set(test_pos))
            max_train = meta.iloc[train_pos]["date"].max()
            min_test = meta.iloc[test_pos]["date"].min()
            assert max_train < min_test

    def test_metrics_per_fold_and_overall(self):
        """AT2: metrics include per-fold rows and an overall row per model, with both metrics."""
        X, y, meta = _make_frames()
        _, metrics = walk_forward(
            [elo_backtest_model(), log_reg_backtest_model()],
            X, y, meta, initial_train=30, step=10, save=False,
        )
        for model in ["baseline_elo", "log_reg"]:
            md = metrics[metrics["model"] == model]
            assert (md["fold"] == "overall").sum() == 1
            assert (md["fold"] != "overall").sum() >= 1  # at least one per-fold row
        assert {"log_loss", "brier"}.issubset(metrics.columns)
        assert metrics["log_loss"].notna().all()

    def test_artifacts_saved(self, tmp_path):
        """AT3: predictions.parquet and metrics.csv are written."""
        X, y, meta = _make_frames()
        walk_forward(
            [elo_backtest_model()], X, y, meta,
            initial_train=30, step=10, save=True, out_dir=tmp_path,
        )
        assert (tmp_path / "predictions.parquet").exists()
        assert (tmp_path / "metrics.csv").exists()
        assert (tmp_path / "README.md").exists()

    def test_predictions_valid(self):
        """Every prediction row has valid probabilities summing to 1, no NaNs."""
        X, y, meta = _make_frames()
        preds, _ = walk_forward(
            [elo_backtest_model(), log_reg_backtest_model()],
            X, y, meta, initial_train=30, step=10, save=False,
        )
        probs = preds[["p_home", "p_draw", "p_away"]]
        assert not probs.isna().any().any()
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_both_models_present(self):
        """Both adapters appear in the predictions output."""
        X, y, meta = _make_frames()
        preds, _ = walk_forward(
            [elo_backtest_model(), log_reg_backtest_model()],
            X, y, meta, initial_train=30, step=10, save=False,
        )
        assert set(preds["model"].unique()) == {"baseline_elo", "log_reg"}

    def test_metric_helpers(self):
        """_log_loss and _brier match hand-computed values."""
        P = np.array([[0.7, 0.2, 0.1]])
        y_idx = np.array([0])  # actual = HW (column 0)
        assert np.isclose(_log_loss(P, y_idx), -np.log(0.7))
        assert np.isclose(_brier(P, y_idx), 0.3**2 + 0.2**2 + 0.1**2)
