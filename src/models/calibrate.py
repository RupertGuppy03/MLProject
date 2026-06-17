from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

from src.config import ARTIFACTS_DIR, PROJECT_ROOT
from src.features.build_features import build_features
from src.models.rf import DEFAULT_PARAMS, load_best_params

# Where the calibrated model is persisted.
MODEL_PATH = ARTIFACTS_DIR / "model_calibrated.pkl"
CURVE_PATH = PROJECT_ROOT / "reports" / "calibration_curve.png"

# Canonical home/draw/away order, shared with the other models.
LABEL_ORDER = ["HW", "D", "AW"]


def calibrate_rf(
    X: pd.DataFrame,
    y,
    params: dict | None = None,
    n_splits: int = 5,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    """Wrap the tuned Random Forest in a time-safe probability calibrator.

    Uses CalibratedClassifierCV with cv=TimeSeriesSplit so the calibrator is fit on folds
    strictly after the base model's training folds — no leakage. method='sigmoid' (Platt) is
    chosen over isotonic because isotonic overfits on this small (~3-season) dataset. Caller
    must pass X/y already sorted chronologically.
    """
    rf_params = params or load_best_params() or DEFAULT_PARAMS
    base = RandomForestClassifier(**rf_params)
    calibrated = CalibratedClassifierCV(
        estimator=base, cv=TimeSeriesSplit(n_splits=n_splits), method=method
    )
    calibrated.fit(X, y)
    return calibrated


def predict_proba(model: CalibratedClassifierCV, X: pd.DataFrame) -> np.ndarray:
    """Return an (n, 3) array of [p_home, p_draw, p_away], reordered from sklearn's class order."""
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    order = [classes.index(label) for label in LABEL_ORDER]
    return proba[:, order]


def save_model(model: CalibratedClassifierCV, path: Path = MODEL_PATH) -> None:
    """Persist the calibrated model with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path = MODEL_PATH) -> CalibratedClassifierCV:
    """Load a calibrated model previously saved with save_model."""
    return joblib.load(Path(path))


def _save_calibration_curve(path: Path = CURVE_PATH) -> None:
    """Plot per-class reliability before (rf_tuned) vs after (rf_calibrated) and save the PNG.

    Uses the leakage-safe walk-forward backtest for predictions, so the curve reflects honest
    out-of-sample behaviour rather than an optimistic in-sample fit.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    from src.backtest.walk_forward import (
        rf_calibrated_backtest_model,
        rf_tuned_backtest_model,
        walk_forward,
    )

    preds, _ = walk_forward(
        [rf_tuned_backtest_model(), rf_calibrated_backtest_model()], save=False
    )

    palette = {"HW": "#2a9d8f", "D": "#e9c46a", "AW": "#e76f51"}
    names = {"HW": "Home win", "D": "Draw", "AW": "Away win"}
    prob_cols = {"HW": "p_home", "D": "p_draw", "AW": "p_away"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, label in zip(axes, LABEL_ORDER):
        for model_name, style in [("rf_tuned", "--"), ("rf_calibrated", "o-")]:
            d = preds[preds["model"] == model_name]
            y_bin = (d["y_true"] == label).astype(int).to_numpy()
            frac_pos, mean_pred = calibration_curve(
                y_bin, d[prob_cols[label]].to_numpy(), n_bins=10, strategy="quantile"
            )
            ax.plot(mean_pred, frac_pos, style, label=model_name, color=palette[label],
                    alpha=0.6 if model_name == "rf_tuned" else 1.0)
        ax.plot([0, 1], [0, 1], "k:", alpha=0.6)
        ax.set_title(names[label])
        ax.set_xlabel("Mean predicted prob")
    axes[0].set_ylabel("Observed frequency")
    axes[0].legend()
    fig.suptitle("Reliability — uncalibrated (dashed) vs calibrated (solid)")
    plt.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Fit + save the calibrated model and write the calibration curve."""
    import numpy as np

    X, y, meta = build_features(save_schema=False)
    order = np.argsort(
        meta[["date", "match_id"]].apply(tuple, axis=1).to_numpy(), kind="stable"
    )
    X, y = X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True)

    model = calibrate_rf(X, y)
    save_model(model)
    _save_calibration_curve()

    print(
        f"Calibrated RF (sigmoid, TimeSeriesSplit) saved -> {MODEL_PATH.name}. "
        f"Calibration curve -> {CURVE_PATH.relative_to(PROJECT_ROOT)}."
    )


if __name__ == "__main__":
    main()
