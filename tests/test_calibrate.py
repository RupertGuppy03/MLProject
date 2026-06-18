from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

from src.models.calibrate import calibrate_rf, load_model, predict_proba, save_model
from src.models.rf import predict_proba as rf_predict_proba
from src.models.rf import train_rf


def _make_frames(n=210, shift=1.0, seed=0):
    """A time-ordered, moderately separable 3-class dataset."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, size=(n, 4))
    y = pd.Series(["HW", "D", "AW"] * (n // 3))
    bump = y.map({"HW": shift, "D": 0.0, "AW": -shift}).to_numpy().reshape(-1, 1)
    X = pd.DataFrame(base + bump, columns=["f1", "f2", "f3", "f4"])
    return X, y


def _brier(P, y):
    idx = {"HW": 0, "D": 1, "AW": 2}
    yi = pd.Series(y).map(idx).to_numpy()
    onehot = np.zeros_like(P)
    onehot[np.arange(len(P)), yi] = 1.0
    return float(np.mean(np.sum((P - onehot) ** 2, axis=1)))


class TestCalibrate:
    """Tests for time-safe probability calibration (Sprint 2)."""

    def test_uses_timeseriessplit_cv(self):
        """Leakage-safe: the calibrator's cv is a TimeSeriesSplit."""
        X, y = _make_frames()
        model = calibrate_rf(X, y, n_splits=3)
        assert isinstance(model, CalibratedClassifierCV)
        assert isinstance(model.cv, TimeSeriesSplit)

    def test_predict_proba_three_classes_sum_to_one(self):
        """predict_proba returns 3 columns summing to 1, in [HW, D, AW] order."""
        X, y = _make_frames()
        model = calibrate_rf(X, y, n_splits=3)
        proba = predict_proba(model, X)
        assert proba.shape == (len(X), 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_save_and_load_round_trip(self, tmp_path):
        """The calibrated model round-trips through joblib, preserving predictions."""
        X, y = _make_frames()
        model = calibrate_rf(X, y, n_splits=3)
        path = tmp_path / "model_calibrated.pkl"
        save_model(model, path)
        loaded = load_model(path)
        assert np.allclose(predict_proba(loaded, X), predict_proba(model, X))

    def test_calibration_improves_brier_on_overconfident_base(self):
        """AT1: calibration improves Brier when the base model is miscalibrated.

        Pure-noise data (no real signal) makes an unconstrained RF overfit and become
        overconfident, which Brier punishes. Sigmoid calibration should tame that and lower
        Brier. (Both models use the same overconfident params so only calibration differs.)
        """
        rng = np.random.default_rng(3)
        n = 300
        X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["f1", "f2", "f3", "f4"])
        y = pd.Series(["HW", "D", "AW"] * (n // 3))  # labels independent of X (no signal)

        split = 240  # train on the past, evaluate on the future
        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        y_tr, y_te = y.iloc[:split], y.iloc[split:]

        overconfident = {
            "n_estimators": 100, "max_depth": None, "min_samples_leaf": 1, "random_state": 42,
        }
        uncal = train_rf(X_tr, y_tr, params=overconfident)
        uncal_brier = _brier(rf_predict_proba(uncal, X_te), y_te)

        cal = calibrate_rf(X_tr, y_tr, params=overconfident, n_splits=3)
        cal_brier = _brier(predict_proba(cal, X_te), y_te)

        assert cal_brier <= uncal_brier + 1e-9
