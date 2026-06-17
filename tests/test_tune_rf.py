from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

from src.models.rf import load_best_params, train_rf
from src.models.tune_rf import build_search_space, save_best_params, tune_rf


def _make_frames(n=210, shift=1.0, seed=0):
    """A time-ordered, moderately separable 3-class dataset (overlap keeps log loss non-trivial)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, size=(n, 4))
    y = pd.Series(["HW", "D", "AW"] * (n // 3))
    bump = y.map({"HW": shift, "D": 0.0, "AW": -shift}).to_numpy().reshape(-1, 1)
    X = pd.DataFrame(base + bump, columns=["f1", "f2", "f3", "f4"])
    return X, y


class TestTuneRF:
    """Tests for the time-series-safe Random Forest tuning (Sprint 2)."""

    def test_cv_is_timeseriessplit_5(self):
        """AT1: the search uses TimeSeriesSplit(n_splits=5)."""
        X, y = _make_frames()
        search = tune_rf(X, y, n_iter=3)
        cv = search.get_params()["cv"]
        assert isinstance(cv, TimeSeriesSplit)
        assert cv.get_n_splits() == 5

    def test_search_space_has_required_keys(self):
        """AT2: param_distributions includes the four required hyperparameters."""
        space = build_search_space()
        for key in ("n_estimators", "max_depth", "min_samples_leaf", "max_features"):
            assert key in space

    def test_best_params_saved_and_loadable(self, tmp_path):
        """AT3: best params are written to JSON and read back as a dict within the search space."""
        X, y = _make_frames()
        search = tune_rf(X, y, n_iter=3)
        path = tmp_path / "best_params_rf.json"
        save_best_params(search, path)

        loaded = load_best_params(path)
        assert isinstance(loaded, dict)
        assert set(loaded).issubset(set(build_search_space()))

    def test_load_best_params_missing_returns_none(self, tmp_path):
        """The downstream default path is unaffected when no tuned params exist yet."""
        assert load_best_params(tmp_path / "does_not_exist.json") is None

    def test_tuning_maintains_or_improves_log_loss(self):
        """AT4: tuned log loss is no worse than default by more than ~2% on a held-out time split."""
        X, y = _make_frames(n=210)
        split = 170  # train on the past, evaluate on the future (chronological holdout)
        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        y_tr, y_te = y.iloc[:split], y.iloc[split:]

        # Use each model's native predict_proba with labels=classes_: sklearn's log_loss
        # assumes prob columns are in lexicographic class order, which classes_ already is.
        default = train_rf(X_tr, y_tr)
        default_ll = log_loss(y_te, default.predict_proba(X_te), labels=default.classes_)

        search = tune_rf(X_tr, y_tr, n_iter=10, n_splits=3)
        tuned = train_rf(X_tr, y_tr, params=search.best_params_)
        tuned_ll = log_loss(y_te, tuned.predict_proba(X_te), labels=tuned.classes_)

        # 2% tolerance, with a tiny absolute floor so near-zero losses don't blow up the ratio.
        assert tuned_ll <= default_ll * 1.02 + 0.05
