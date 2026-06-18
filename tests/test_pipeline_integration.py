from __future__ import annotations

import json

import pandas as pd
import pytest

from src.config import ARTIFACTS_DIR
from src.features.build_features import build_features
from src.ingest.build_canonical import build_canonical
from src.ingest.unify_raw import unify_raw

# Outcome / raw-goal columns that must never leak into the feature matrix.
LEAKAGE_COLUMNS = {"result", "home_goals", "away_goals"}


@pytest.fixture(scope="class")
def pipeline_outputs():
    """Run the full pipeline once against the real data/raw files: ingest -> canonical -> features.

    Deterministic, so regenerating matches_all/matches_canonical in place is idempotent.
    Returns the (X, y, meta) the model/API consume.
    """
    unify_raw()  # data/raw/matches_*.parquet -> matches_all.parquet
    build_canonical()  # matches_all.parquet -> matches_canonical.parquet (+ validation)
    X, y, meta = build_features(save_schema=False)
    return X, y, meta


class TestPipelineIntegration:
    """Sprint 2: single end-to-end integration test over the composed pipeline."""

    def test_pipeline_runs_end_to_end(self, pipeline_outputs):
        """AT1: the full chain completes without error and yields non-empty outputs."""
        X, y, meta = pipeline_outputs
        assert isinstance(X, pd.DataFrame) and len(X) > 0
        assert isinstance(meta, pd.DataFrame) and len(meta) > 0
        assert len(y) > 0

    def test_output_shape_and_columns(self, pipeline_outputs):
        """AT2: expected columns present, row counts consistent, no NaN, no leakage columns."""
        X, y, meta = pipeline_outputs

        # All and only the locked-schema feature columns, in order.
        schema = json.loads((ARTIFACTS_DIR / "feature_schema.json").read_text())
        assert list(X.columns) == schema["feature_columns"]

        # Row counts consistent across X / y / meta.
        assert len(X) == len(y) == len(meta)

        # No missing values reach the model.
        assert not X.isna().any().any()

        # Outcome / raw-goal columns never leak into X.
        assert LEAKAGE_COLUMNS.isdisjoint(X.columns)
