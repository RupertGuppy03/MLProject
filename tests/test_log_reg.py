from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.log_reg import (
    LABEL_ORDER,
    load_model,
    predict_proba,
    save_model,
    train_log_reg,
)


def _make_frames():
    """A small, separable synthetic dataset with all three classes."""
    rng = np.random.default_rng(0)
    n = 60
    # Three feature columns whose means differ by class -> learnable.
    base = rng.normal(0, 1, size=(n, 3))
    y = pd.Series((["HW", "D", "AW"] * (n // 3)))
    shift = y.map({"HW": 1.5, "D": 0.0, "AW": -1.5}).to_numpy().reshape(-1, 1)
    X = pd.DataFrame(base + shift, columns=["f1", "f2", "f3"])
    return X, y


class TestLogReg:
    """Tests for the multinomial Logistic Regression baseline (Sprint 2)."""

    def test_trains_successfully(self):
        """AT1: training completes and returns a fitted pipeline with predict_proba."""
        X, y = _make_frames()
        model = train_log_reg(X, y)
        assert isinstance(model, Pipeline)
        assert hasattr(model, "predict_proba")

    def test_predict_proba_three_classes_sum_to_one(self):
        """AT2: predict_proba returns 3 columns per row summing to 1."""
        X, y = _make_frames()
        model = train_log_reg(X, y)
        proba = predict_proba(model, X)
        assert proba.shape == (len(X), 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_orders_home_draw_away(self):
        """The helper maps columns to [HW, D, AW] regardless of sklearn's class order."""
        X, y = _make_frames()
        model = train_log_reg(X, y)
        ordered = predict_proba(model, X)
        raw = model.predict_proba(X)
        classes = list(model.classes_)
        # Column for each label must match the raw column at that class's index.
        for i, label in enumerate(LABEL_ORDER):
            assert np.allclose(ordered[:, i], raw[:, classes.index(label)])

    def test_pipeline_includes_scaler(self):
        """Fair-comparison guard: the model scales features before the classifier."""
        X, y = _make_frames()
        model = train_log_reg(X, y)
        assert isinstance(model.named_steps["scaler"], StandardScaler)

    def test_save_and_load_round_trip(self, tmp_path):
        """DoD: the model can be saved and loaded with joblib, preserving predictions."""
        X, y = _make_frames()
        model = train_log_reg(X, y)
        path = tmp_path / "model_log_reg.pkl"
        save_model(model, path)
        loaded = load_model(path)
        assert np.allclose(predict_proba(loaded, X), predict_proba(model, X))
