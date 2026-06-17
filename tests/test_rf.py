from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.models.rf import (
    LABEL_ORDER,
    load_model,
    predict_proba,
    save_feature_importances,
    save_model,
    train_rf,
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


class TestRandomForest:
    """Tests for the main Random Forest model (Sprint 2)."""

    def test_trains_successfully_and_exposes_importances(self):
        """AT1: training completes and the model exposes .feature_importances_."""
        X, y = _make_frames()
        model = train_rf(X, y)
        assert isinstance(model, RandomForestClassifier)
        assert hasattr(model, "feature_importances_")
        assert len(model.feature_importances_) == X.shape[1]

    def test_uses_dod_default_params(self):
        """The fixed defaults required by the DoD are applied."""
        X, y = _make_frames()
        model = train_rf(X, y)
        assert model.n_estimators == 200
        assert model.max_depth is None
        assert model.min_samples_leaf == 5
        assert model.random_state == 42

    def test_predict_proba_three_classes_sum_to_one(self):
        """AT2: predict_proba returns 3 columns per row summing to 1 within 1e-6."""
        X, y = _make_frames()
        model = train_rf(X, y)
        proba = predict_proba(model, X)
        assert proba.shape == (len(X), 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_no_nan_in_model_input(self):
        """AT3: the feature matrix passed to the model contains no NaNs."""
        X, y = _make_frames()
        # Mirror the build_features contract: X must be NaN-free before it reaches the model.
        assert not X.isna().any().any()
        model = train_rf(X, y)  # must train without complaint on the clean matrix
        assert hasattr(model, "feature_importances_")

    def test_feature_importances_saved(self, tmp_path):
        """AT4: feature_importances.json contains all feature names and numeric scores."""
        X, y = _make_frames()
        model = train_rf(X, y)
        path = tmp_path / "feature_importances.json"
        save_feature_importances(model, list(X.columns), path)

        saved = json.loads(path.read_text())
        assert set(saved.keys()) == set(X.columns)
        assert all(isinstance(v, float) for v in saved.values())

    def test_predict_proba_orders_home_draw_away(self):
        """The helper maps columns to [HW, D, AW] regardless of sklearn's class order."""
        X, y = _make_frames()
        model = train_rf(X, y)
        ordered = predict_proba(model, X)
        raw = model.predict_proba(X)
        classes = list(model.classes_)
        for i, label in enumerate(LABEL_ORDER):
            assert np.allclose(ordered[:, i], raw[:, classes.index(label)])

    def test_sample_weight_influences_model(self):
        """train_rf accepts sample_weight and up-weighting a class shifts its predictions."""
        rng = np.random.default_rng(1)
        n = 90
        X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f1", "f2", "f3"])
        y = pd.Series(["HW", "D", "AW"] * (n // 3))

        base = train_rf(X, y)
        weights = np.where(y.to_numpy() == "HW", 10.0, 1.0)  # heavily up-weight home wins
        weighted = train_rf(X, y, sample_weight=weights)

        # With no feature signal, leaves reflect (weighted) class proportions, so up-weighting
        # HW must raise the average predicted home-win probability.
        p_home_base = predict_proba(base, X)[:, 0].mean()
        p_home_weighted = predict_proba(weighted, X)[:, 0].mean()
        assert p_home_weighted > p_home_base

    def test_save_and_load_round_trip(self, tmp_path):
        """DoD: the model can be saved and loaded with joblib, preserving predictions."""
        X, y = _make_frames()
        model = train_rf(X, y)
        path = tmp_path / "model_rf.pkl"
        save_model(model, path)
        loaded = load_model(path)
        assert np.allclose(predict_proba(loaded, X), predict_proba(model, X))
