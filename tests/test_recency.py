from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.recency import recency_weights


def _dates(days_ago):
    """Build a date Series at the given ages (days before a fixed 'now')."""
    now = pd.Timestamp("2025-06-01")
    return pd.Series([now - pd.Timedelta(days=d) for d in days_ago])


class TestRecencyWeights:
    """Tests for exponential recency sample weights (Sprint 2)."""

    def test_half_life_halves_weight(self):
        """A match exactly half_life_days older than the newest has half its weight."""
        dates = _dates([0, 365])  # newest, then one half-life older
        w = recency_weights(dates, half_life_days=365.0)
        assert np.isclose(w[0], 1.0)
        assert np.isclose(w[1], 0.5, atol=1e-9)

    def test_monotonic_decreasing_with_age(self):
        """Older matches get smaller weights; all weights are strictly positive."""
        dates = _dates([0, 100, 400, 900])
        w = recency_weights(dates, half_life_days=365.0)
        assert np.all(w > 0)
        assert np.all(np.diff(w) < 0)  # strictly decreasing as age increases

    def test_newest_is_max_and_reference_defaults_to_latest(self):
        """With no reference_date, the most recent match anchors at weight 1.0 (no future info)."""
        dates = _dates([900, 0, 365])  # unordered input
        w = recency_weights(dates, half_life_days=365.0)
        assert np.isclose(w.max(), 1.0)
        assert w.argmax() == 1  # the 0-days-ago row
        assert len(w) == len(dates)

    def test_explicit_reference_date(self):
        """An explicit reference shifts all ages; a match at the reference weighs 1.0."""
        dates = _dates([0, 365])
        ref = pd.Timestamp("2025-06-01")  # equals the newest date
        w = recency_weights(dates, half_life_days=365.0, reference_date=ref)
        assert np.isclose(w[0], 1.0)
        assert np.isclose(w[1], 0.5)
