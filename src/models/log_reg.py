from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ARTIFACTS_DIR
from src.features.build_features import build_features

# Where the trained pipeline is persisted.
MODEL_PATH = ARTIFACTS_DIR / "model_log_reg.pkl"

# Canonical home/draw/away order, shared with the Elo baseline so every model's
# probabilities line up column-for-column in the comparison report.
LABEL_ORDER = ["HW", "D", "AW"]

# Default hyperparameters for the logistic regression step.
DEFAULT_PARAMS = {"C": 1.0, "max_iter": 1000, "random_state": 42}


def train_log_reg(X: pd.DataFrame, y, params: dict | None = None) -> Pipeline:
    """Train a multinomial logistic regression on the feature matrix.

    Wrapped in a Pipeline with StandardScaler because logistic regression is scale-sensitive;
    fusing the scaler to the model keeps inference and save/load correct with no separate
    preprocessing step. On scikit-learn 1.8 the default lbfgs solver is multinomial for
    multiclass y, so no explicit (deprecated) multi_class argument is needed.
    """
    clf_params = {**DEFAULT_PARAMS, **(params or {})}
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**clf_params)),
        ]
    )
    model.fit(X, y)
    return model


def predict_proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return an (n, 3) array of [p_home, p_draw, p_away] for each row.

    sklearn orders probability columns by `classes_` (alphabetical: AW, D, HW); we reorder
    into the canonical home/draw/away layout used across all models.
    """
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    order = [classes.index(label) for label in LABEL_ORDER]
    return proba[:, order]


def save_model(model: Pipeline, path: Path = MODEL_PATH) -> None:
    """Persist the trained pipeline with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    """Load a pipeline previously saved with save_model."""
    return joblib.load(Path(path))


def main() -> None:
    """Train on the full feature matrix and print an in-sample sanity log loss."""
    from sklearn.metrics import log_loss

    X, y, _ = build_features(save_schema=False)
    model = train_log_reg(X, y)

    # In-sample only — optimistic by definition. The real comparison number comes from the
    # leakage-safe walk-forward backtest, not this fit-then-predict-on-the-same-data check.
    # Use the pipeline's native predict_proba: its columns already align with model.classes_,
    # which is what log_loss expects (sklearn orders labels lexicographically).
    ll = log_loss(y, model.predict_proba(X), labels=model.classes_)
    print(
        f"Logistic Regression trained on X={X.shape}. "
        f"In-sample (optimistic) log loss: {ll:.4f}. Class order: {LABEL_ORDER}."
    )


if __name__ == "__main__":
    main()
