from __future__ import annotations

import json

import joblib
import numpy as np

from src.features.build_features import build_features
from src.models.rf import predict_proba
from src.models.save_artifacts import save_final_artifacts

# Small, fast forest so the test trains quickly while still exercising the real save path.
FAST_PARAMS = {"n_estimators": 20, "random_state": 42}


class TestSaveArtifacts:
    """Sprint 2: save final artifacts + schema contract."""

    def test_artifacts_saved(self, tmp_path):
        """AT1: running the pipeline writes chosen_model.pkl and feature_schema.json."""
        summary = save_final_artifacts(artifacts_dir=tmp_path, params=FAST_PARAMS)
        assert (tmp_path / "chosen_model.pkl").exists()
        assert (tmp_path / "feature_schema.json").exists()
        # The one command regenerates every served artifact, not just the model + schema.
        assert (tmp_path / "current_elo.json").exists()
        assert (tmp_path / "feature_importances.json").exists()
        assert summary["n_features"] > 0

    def test_artifacts_loadable_and_predict(self, tmp_path):
        """AT2: artifacts load and a sample prediction yields valid probabilities."""
        save_final_artifacts(artifacts_dir=tmp_path, params=FAST_PARAMS)
        model = joblib.load(tmp_path / "chosen_model.pkl")

        X, _, _ = build_features(save_schema=False)
        proba = predict_proba(model, X.head(5))
        assert proba.shape == (5, 3)
        assert np.all(proba >= 0) and np.all(proba <= 1)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_schema_includes_columns_and_preprocessing(self, tmp_path):
        """DoD: schema lists feature columns and documents preprocessing details."""
        save_final_artifacts(artifacts_dir=tmp_path, params=FAST_PARAMS)
        schema = json.loads((tmp_path / "feature_schema.json").read_text())
        assert schema["feature_columns"], "feature_columns missing"
        assert "preprocessing" in schema
        assert "scaling" in schema["preprocessing"]
        assert "imputation" in schema["preprocessing"]
