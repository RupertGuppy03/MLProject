from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.features.build_features import build_features
from src.features.elo import DEFAULT_HOME_ADVANTAGE, EloState

# Where the comparison-ready predictions file is written.
PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "predictions_baseline_elo.parquet"

# Fallback draw rate when no labels are available to fit from (~PL long-run average).
BASE_DRAW_RATE = 0.25

# Tiny floor so clamped tail probabilities stay positive before renormalising.
_PROB_FLOOR = 1e-6


class EloBaseline:
    """A simple Elo-only benchmark that turns pre-match Elo into 3-way probabilities.

    Elo natively gives a 2-way expected score `e = p_home + 0.5*p_draw`. We split this into
    home/draw/away using a fixed draw rate `d`:
        p_draw = d,  p_home = e - 0.5*d,  p_away = (1 - e) - 0.5*d
    The locked Elo win formula (incl. +60 home advantage) is reused from EloState, so there is
    no duplicate Elo implementation.
    """

    def __init__(
        self,
        draw_rate: float | None = None,
        home_advantage: float = DEFAULT_HOME_ADVANTAGE,
    ) -> None:
        self.draw_rate = draw_rate
        # Held only to reuse expected_home with the locked home-advantage offset.
        self._elo = EloState(home_advantage=home_advantage)

    def fit(self, y) -> "EloBaseline":
        """Estimate the draw rate from training labels (skipped if one was given)."""
        if self.draw_rate is None:
            self.draw_rate = float((pd.Series(y) == "D").mean())
        return self

    def predict_proba(self, elo_home_pre, elo_away_pre) -> np.ndarray:
        """Return an (n, 3) array of [p_home, p_draw, p_away] for each match."""
        d = self.draw_rate if self.draw_rate is not None else BASE_DRAW_RATE
        eh = np.asarray(elo_home_pre, dtype=float)
        ea = np.asarray(elo_away_pre, dtype=float)

        # Reuse the locked Elo expected-score formula per row.
        e = np.array([self._elo.expected_home(h, a) for h, a in zip(eh, ea)])

        p_home = e - 0.5 * d
        p_draw = np.full_like(e, d)
        p_away = (1.0 - e) - 0.5 * d

        probs = np.column_stack([p_home, p_draw, p_away])
        # Extreme mismatches can push a tail slightly negative — clamp then renormalise.
        probs = np.clip(probs, _PROB_FLOOR, None)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs


def run_elo_baseline(
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    meta: pd.DataFrame | None = None,
    draw_rate: float | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Run the Elo baseline across the dataset and return a tidy predictions frame.

    When X/y/meta are None they are loaded from build_features(). The output is one row per
    match in the backtest-compatible long format that later models and the comparison report
    concatenate: match_id, date, season, model, p_home, p_draw, p_away, y_true.
    """
    if X is None or y is None or meta is None:
        X, y, meta = build_features(save_schema=False)

    model = EloBaseline(draw_rate=draw_rate).fit(y)
    probs = model.predict_proba(X["elo_home_pre"], X["elo_away_pre"])

    preds = meta[["match_id", "date", "season"]].copy().reset_index(drop=True)
    preds["model"] = "baseline_elo"
    preds["p_home"] = probs[:, 0]
    preds["p_draw"] = probs[:, 1]
    preds["p_away"] = probs[:, 2]
    preds["y_true"] = np.asarray(y)

    if save:
        _atomic_write_parquet(preds, PREDICTIONS_PATH)

    return preds


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a parquet file atomically (temp-then-replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def main() -> None:
    """Run the baseline on the full dataset and report a quick summary."""
    preds = run_elo_baseline()
    fitted = float(preds["p_draw"].iloc[0])
    means = preds[["p_home", "p_draw", "p_away"]].mean()
    print(
        f"Elo baseline: {len(preds)} predictions written to {PREDICTIONS_PATH}.\n"
        f"Fitted draw rate: {fitted:.3f} | mean probs "
        f"home={means['p_home']:.3f} draw={means['p_draw']:.3f} away={means['p_away']:.3f}"
    )


if __name__ == "__main__":
    main()
