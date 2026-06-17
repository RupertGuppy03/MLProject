from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.features.build_features import build_features
from src.models.baseline_elo import EloBaseline
from src.models.log_reg import predict_proba as log_reg_predict_proba
from src.models.log_reg import train_log_reg
from src.models.rf import BEST_PARAMS_PATH as RF_BEST_PARAMS_PATH
from src.models.rf import load_best_params as rf_load_best_params
from src.models.rf import predict_proba as rf_predict_proba
from src.models.rf import train_rf

# Canonical home/draw/away column order used for every model's probabilities.
LABEL_ORDER = ["HW", "D", "AW"]
_LABEL_IDX = {label: i for i, label in enumerate(LABEL_ORDER)}

REPORTS_DIR = PROJECT_ROOT / "reports"
PREDICTIONS_PATH = REPORTS_DIR / "predictions.parquet"
METRICS_PATH = REPORTS_DIR / "metrics.csv"
README_PATH = REPORTS_DIR / "README.md"

_EPS = 1e-15


@dataclass
class BacktestModel:
    """A uniform adapter so any model plugs into the backtest.

    fit(X_train, y_train) returns a fitted state; predict_proba(state, X_test) returns an
    (n, 3) array of probabilities in [HW, D, AW] order.
    """

    name: str
    fit: Callable[[pd.DataFrame, pd.Series], Any]
    predict_proba: Callable[[Any, pd.DataFrame], np.ndarray]


def elo_backtest_model() -> BacktestModel:
    """Elo baseline adapter: fit the draw rate on the train labels, predict from Elo columns."""
    def fit(_X_train, y_train):
        return EloBaseline().fit(y_train)  # Elo baseline fits only the draw rate from labels

    def predict(state, X_test):
        return state.predict_proba(X_test["elo_home_pre"], X_test["elo_away_pre"])

    return BacktestModel("baseline_elo", fit, predict)


def log_reg_backtest_model() -> BacktestModel:
    """Logistic Regression adapter (scaler fit on the train split only — no leakage)."""
    def fit(X_train, y_train):
        return train_log_reg(X_train, y_train)

    def predict(state, X_test):
        return log_reg_predict_proba(state, X_test)

    return BacktestModel("log_reg", fit, predict)


def rf_backtest_model() -> BacktestModel:
    """Random Forest adapter (the main model) — trained on the train split of each fold."""
    def fit(X_train, y_train):
        return train_rf(X_train, y_train)

    def predict(state, X_test):
        return rf_predict_proba(state, X_test)

    return BacktestModel("rf", fit, predict)


def rf_tuned_backtest_model() -> BacktestModel:
    """Tuned Random Forest adapter — uses the params saved by tune_rf.py (best_params_rf.json)."""
    def fit(X_train, y_train):
        return train_rf(X_train, y_train, params=rf_load_best_params())

    def predict(state, X_test):
        return rf_predict_proba(state, X_test)

    return BacktestModel("rf_tuned", fit, predict)


def _expanding_folds(
    meta: pd.DataFrame, initial_train: int, step: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds over date-ordered rows.

    The first fold trains on `initial_train` rows and tests the next `step`; each later fold
    grows the training set by one block. Returns (train_positions, test_positions) pairs into
    the date-sorted order — strictly past-trains-on, future-tests (leakage-safe).
    """
    order = np.argsort(
        meta[["date", "match_id"]].apply(tuple, axis=1).to_numpy(), kind="stable"
    )
    n = len(order)
    folds = []
    start = initial_train
    while start < n:
        train_pos = order[:start]
        test_pos = order[start : start + step]
        if len(test_pos) > 0:
            folds.append((train_pos, test_pos))
        start += step
    return folds


def _y_to_idx(y) -> np.ndarray:
    """Map HW/D/AW labels to their [HW, D, AW] column index."""
    return np.array([_LABEL_IDX[str(v)] for v in np.asarray(y)])


def _log_loss(P: np.ndarray, y_idx: np.ndarray) -> float:
    """Multiclass log loss with the [HW, D, AW] column convention."""
    picked = P[np.arange(len(P)), y_idx]
    return float(-np.mean(np.log(np.clip(picked, _EPS, None))))


def _brier(P: np.ndarray, y_idx: np.ndarray) -> float:
    """Multiclass Brier score: mean of sum_k (p_k - onehot_k)^2."""
    onehot = np.zeros_like(P)
    onehot[np.arange(len(P)), y_idx] = 1.0
    return float(np.mean(np.sum((P - onehot) ** 2, axis=1)))


def walk_forward(
    models: list[BacktestModel],
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    meta: pd.DataFrame | None = None,
    initial_train: int | None = None,
    step: int = 10,
    save: bool = True,
    out_dir: Path = REPORTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run an expanding-window walk-forward backtest for each model.

    Returns (predictions_df, metrics_df). When frames are None they are loaded from
    build_features(). Metrics (log loss + Brier) are reported per fold and overall per model.
    """
    if X is None or y is None or meta is None:
        X, y, meta = build_features(save_schema=False)

    meta = meta.copy().reset_index(drop=True)
    meta["date"] = pd.to_datetime(meta["date"])
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y), name="result").reset_index(drop=True)

    # Default the initial training window to the first season, so no fold is data-starved.
    if initial_train is None:
        first_season = sorted(meta["season"].unique())[0]
        initial_train = int((meta["season"] == first_season).sum())

    folds = _expanding_folds(meta, initial_train, step)

    pred_rows = []
    for model in models:
        for fold_id, (train_pos, test_pos) in enumerate(folds):
            state = model.fit(X.iloc[train_pos], y.iloc[train_pos])
            proba = model.predict_proba(state, X.iloc[test_pos])
            block = meta.iloc[test_pos][["match_id", "date", "season"]].copy()
            block["fold"] = fold_id
            block["model"] = model.name
            block["p_home"] = proba[:, 0]
            block["p_draw"] = proba[:, 1]
            block["p_away"] = proba[:, 2]
            block["y_true"] = y.iloc[test_pos].to_numpy()
            pred_rows.append(block)

    predictions = pd.concat(pred_rows, ignore_index=True)
    metrics = _compute_metrics(predictions)

    if save:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_parquet(predictions, out_dir / PREDICTIONS_PATH.name)
        _atomic_write_text(metrics.to_csv(index=False), out_dir / METRICS_PATH.name)
        _atomic_write_text(_render_readme(metrics, predictions), out_dir / README_PATH.name)

    return predictions, metrics


def _compute_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-(model, fold) and overall log loss + Brier."""
    rows = []
    prob_cols = ["p_home", "p_draw", "p_away"]
    for model_name, model_df in predictions.groupby("model", sort=False):
        # Per fold.
        for fold_id, fold_df in model_df.groupby("fold", sort=True):
            P = fold_df[prob_cols].to_numpy()
            yi = _y_to_idx(fold_df["y_true"])
            rows.append(
                {
                    "model": model_name,
                    "fold": str(fold_id),
                    "n": len(fold_df),
                    "log_loss": _log_loss(P, yi),
                    "brier": _brier(P, yi),
                }
            )
        # Overall (pooled across folds).
        P = model_df[prob_cols].to_numpy()
        yi = _y_to_idx(model_df["y_true"])
        rows.append(
            {
                "model": model_name,
                "fold": "overall",
                "n": len(model_df),
                "log_loss": _log_loss(P, yi),
                "brier": _brier(P, yi),
            }
        )
    return pd.DataFrame(rows)


def _render_readme(metrics: pd.DataFrame, predictions: pd.DataFrame) -> str:
    """Build the reports/README.md summary: overall comparison + per-season breakdown."""
    overall = metrics[metrics["fold"] == "overall"].sort_values("log_loss")
    lines = [
        "# Backtest reports",
        "",
        "Walk-forward (expanding-window) backtest. Lower is better for both metrics.",
        "",
        "## Overall (pooled across folds)",
        "",
        "| Model | n | Log loss | Brier |",
        "|---|---|---|---|",
    ]
    for _, r in overall.iterrows():
        lines.append(f"| {r['model']} | {r['n']} | {r['log_loss']:.4f} | {r['brier']:.4f} |")

    # Per-season pooled breakdown for interpretability.
    lines += ["", "## By test season", "", "| Model | Season | n | Log loss | Brier |", "|---|---|---|---|---|"]
    prob_cols = ["p_home", "p_draw", "p_away"]
    for model_name, mdf in predictions.groupby("model", sort=False):
        for season, sdf in mdf.groupby("season", sort=True):
            P = sdf[prob_cols].to_numpy()
            yi = _y_to_idx(sdf["y_true"])
            lines.append(
                f"| {model_name} | {season} | {len(sdf)} | {_log_loss(P, yi):.4f} | {_brier(P, yi):.4f} |"
            )
    lines.append("")
    return "\n".join(lines)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _atomic_write_text(text: str, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def main() -> None:
    """Run the backtest for the current baselines and print the overall comparison."""
    models = [elo_backtest_model(), log_reg_backtest_model(), rf_backtest_model()]
    # Include the tuned RF only once tuning has produced its params, so this still runs before then.
    if RF_BEST_PARAMS_PATH.exists():
        models.append(rf_tuned_backtest_model())
    _, metrics = walk_forward(models)
    overall = metrics[metrics["fold"] == "overall"].sort_values("log_loss")
    print("Walk-forward backtest — overall (lower is better):")
    for _, r in overall.iterrows():
        print(f"  {r['model']:<14} log_loss={r['log_loss']:.4f}  brier={r['brier']:.4f}  (n={r['n']})")
    print(f"Artifacts written to {REPORTS_DIR}/ (predictions.parquet, metrics.csv, README.md).")


if __name__ == "__main__":
    main()
