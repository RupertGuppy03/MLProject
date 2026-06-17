from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from src.config import ARTIFACTS_DIR
from src.features.build_features import build_features
from src.models.rf import DEFAULT_PARAMS

# Where the tuned hyperparameters are persisted (consumed by src/models/rf.load_best_params).
BEST_PARAMS_PATH = ARTIFACTS_DIR / "best_params_rf.json"


def build_search_space() -> dict:
    """RandomizedSearchCV param distributions for the Random Forest.

    Expanded beyond the original story grid on the strength of the diagnostics notebook: the
    max_depth sweep showed shallow trees (~4) beat None/10/20/30, and larger min_samples_leaf
    helps regularise a small, noisy dataset. CV picks the winner from this wider grid.
    """
    return {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6, 8, 12, 20, None],
        "min_samples_leaf": [1, 3, 5, 10, 20, 50],
        "max_features": ["sqrt", "log2", 0.5],
    }


def tune_rf(
    X: pd.DataFrame,
    y,
    n_iter: int = 50,
    random_state: int = 42,
    n_splits: int = 5,
) -> RandomizedSearchCV:
    """Time-series-safe randomized hyperparameter search, scored on negative log loss.

    Uses TimeSeriesSplit so every fold trains on the past and validates on the future — no
    leakage. The caller must pass X/y already sorted chronologically (main() does this).
    Returns the fitted search; read `.best_params_` / `.best_score_` off it.
    """
    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=random_state, n_jobs=-1),
        param_distributions=build_search_space(),
        n_iter=n_iter,
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="neg_log_loss",
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X, y)
    return search


def save_best_params(search: RandomizedSearchCV, path: Path = BEST_PARAMS_PATH) -> dict:
    """Atomically write the search's best_params_ to JSON; return them."""
    params = search.best_params_
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(params, indent=2))
    tmp.replace(path)
    return params


def _sort_by_date(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame):
    """Return X/y reordered chronologically by (date, match_id) for time-safe CV.

    Mirrors the ordering used by the walk-forward backtest (walk_forward._expanding_folds).
    """
    order = np.argsort(
        meta[["date", "match_id"]].apply(tuple, axis=1).to_numpy(), kind="stable"
    )
    return X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True)


def main() -> None:
    """Tune on the full (date-sorted) feature matrix, save best params, print tuned vs default."""
    from sklearn.model_selection import cross_val_score

    X, y, meta = build_features(save_schema=False)
    X, y = _sort_by_date(X, y, meta)

    search = tune_rf(X, y)
    params = save_best_params(search)

    # Context: compare tuned CV log loss against the current default params under the same CV.
    cv = TimeSeriesSplit(n_splits=5)
    default_clf = RandomForestClassifier(**DEFAULT_PARAMS)
    default_score = cross_val_score(default_clf, X, y, cv=cv, scoring="neg_log_loss").mean()

    print(f"Best params: {params}")
    print(
        f"CV log loss (lower is better) — tuned: {-search.best_score_:.4f}  "
        f"default: {-default_score:.4f}. Saved -> {BEST_PARAMS_PATH.name}."
    )


if __name__ == "__main__":
    main()
