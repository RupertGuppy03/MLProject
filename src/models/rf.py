from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import ARTIFACTS_DIR
from src.features.build_features import build_features

# Where the trained model and its feature importances are persisted.
MODEL_PATH = ARTIFACTS_DIR / "model_rf.pkl"
IMPORTANCES_PATH = ARTIFACTS_DIR / "feature_importances.json"
# Tuned hyperparameters written by src/models/tune_rf.py (optional — may not exist yet).
BEST_PARAMS_PATH = ARTIFACTS_DIR / "best_params_rf.json"

# Canonical home/draw/away order, shared with the other models so every model's
# probabilities line up column-for-column in the comparison report.
LABEL_ORDER = ["HW", "D", "AW"]

# Default hyperparameters (fixed per the DoD). Tuning is a separate story (tune_rf.py).
# No StandardScaler is needed: a Random Forest splits on thresholds, so it is
# scale-invariant — this is the deliberate difference from the logistic regression baseline.
DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 5,
    "random_state": 42,
    "n_jobs": -1,
}


def train_rf(X: pd.DataFrame, y, params: dict | None = None) -> RandomForestClassifier:
    """Train the main Random Forest classifier on the feature matrix.

    Merges any overrides onto the fixed defaults, fits, and returns the model. The model
    exposes `.feature_importances_` for explainability downstream (SHAP / diagnostics).
    """
    clf_params = {**DEFAULT_PARAMS, **(params or {})}
    model = RandomForestClassifier(**clf_params)
    model.fit(X, y)
    return model


def predict_proba(model: RandomForestClassifier, X: pd.DataFrame) -> np.ndarray:
    """Return an (n, 3) array of [p_home, p_draw, p_away] for each row.

    sklearn orders probability columns by `classes_` (alphabetical: AW, D, HW); we reorder
    into the canonical home/draw/away layout used across all models.
    """
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    order = [classes.index(label) for label in LABEL_ORDER]
    return proba[:, order]


def save_model(model: RandomForestClassifier, path: Path = MODEL_PATH) -> None:
    """Persist the trained model with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path = MODEL_PATH) -> RandomForestClassifier:
    """Load a model previously saved with save_model."""
    return joblib.load(Path(path))


def load_best_params(path: Path = BEST_PARAMS_PATH) -> dict | None:
    """Return tuned hyperparameters from tune_rf.py, or None if tuning hasn't run yet.

    Downstream callers opt in by passing the result to train_rf(X, y, params=...), so the
    default-RF path is unaffected when no tuned params exist.
    """
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_feature_importances(
    model: RandomForestClassifier,
    feature_names: list[str],
    path: Path = IMPORTANCES_PATH,
) -> dict[str, float]:
    """Write feature names + importance scores to JSON, sorted high-to-low.

    This is the persistent artifact the follow-up notebook reads back to investigate which
    features matter; it is evidence for that analysis, not a feature-selection decision.
    Atomic write (tmp + replace) so a crash never leaves a half-written file.
    """
    importances = {
        name: float(score)
        for name, score in zip(feature_names, model.feature_importances_)
    }
    # Present most-important first for readability.
    importances = dict(
        sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(importances, indent=2))
    tmp.replace(path)
    return importances


def main() -> None:
    """Train on the full feature matrix, save the model + importances, print sanity log loss."""
    from sklearn.metrics import log_loss

    X, y, _ = build_features(save_schema=False)

    # Leakage-safe guard: the model must never see NaNs (build_features already imputes,
    # but assert it here so a regression upstream fails loudly at train time).
    assert not X.isna().any().any(), "Feature matrix X contains NaN values."

    model = train_rf(X, y)
    save_model(model)
    save_feature_importances(model, list(X.columns))

    # In-sample only — optimistic by definition. The real comparison number comes from the
    # leakage-safe walk-forward backtest, not this fit-then-predict-on-the-same-data check.
    ll = log_loss(y, model.predict_proba(X), labels=model.classes_)
    print(
        f"Random Forest trained on X={X.shape}. "
        f"In-sample (optimistic) log loss: {ll:.4f}. Class order: {LABEL_ORDER}. "
        f"Saved model -> {MODEL_PATH.name}, importances -> {IMPORTANCES_PATH.name}."
    )


if __name__ == "__main__":
    main()
