from __future__ import annotations

import numpy as np
import pandas as pd


def recency_weights(
    dates,
    half_life_days: float = 365.0,
    reference_date=None,
) -> np.ndarray:
    """Exponential sample weights that decay with match age (newest = 1.0).

    weight = 0.5 ** (age_days / half_life_days), where age_days is how far each match sits
    before the reference date. A match exactly `half_life_days` older than the reference gets
    half the weight of the most recent one; older matches decay further.

    Leakage-safe by construction: the reference defaults to the most recent date in `dates`
    (i.e. the latest training match), so weights depend only on the training window — never on
    anything in the future. Callers pass the training dates of a fold, not the whole dataset.

    Args:
        dates: datetime-like Series/array, one entry per training row.
        half_life_days: age (in days) at which a match's weight halves.
        reference_date: the "now" anchor; defaults to max(dates).

    Returns:
        Float ndarray of weights, aligned row-for-row with `dates`.
    """
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    if reference_date is None:
        reference_date = dates.max()
    reference_date = pd.Timestamp(reference_date)

    age_days = (reference_date - dates).dt.total_seconds() / 86400.0
    weights = np.power(0.5, age_days.to_numpy() / half_life_days)
    return weights.astype(float)
