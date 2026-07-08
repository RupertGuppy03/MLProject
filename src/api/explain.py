"""Per-prediction SHAP explanations.

Computes the feature contributions behind a single prediction with shap.TreeExplainer, run on the
served Random Forest (`chosen_model.pkl`). That model IS the plain RF (not a calibrated wrapper),
so the SHAP values explain exactly the same model that produces the served probabilities — there is
no calibrated/pre-calibration conflation to worry about here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

# TreeExplainer construction isn't free, so cache one per model instance.
_explainer = None
_explainer_model_id: int | None = None


def _get_explainer(model) -> shap.TreeExplainer:
    global _explainer, _explainer_model_id
    if _explainer is None or _explainer_model_id != id(model):
        _explainer = shap.TreeExplainer(model)
        _explainer_model_id = id(model)
    return _explainer


def _class_contributions(shap_values, class_index: int) -> np.ndarray:
    """Return the (n_features,) SHAP contributions for one class of a single-row input.

    Handles both shap output layouts: a 3D array (n_samples, n_features, n_classes) and the older
    list-of-arrays (one (n_samples, n_features) per class).
    """
    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_index])[0]
    arr = np.asarray(shap_values)
    if arr.ndim == 3:  # (n_samples, n_features, n_classes)
        return arr[0, :, class_index]
    if arr.ndim == 2:  # (n_samples, n_features) — single output
        return arr[0]
    raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")


def top_feature_contributions(
    model, X: pd.DataFrame, predicted_label: str, k: int = 5
) -> list[dict[str, float | str]]:
    """Top-k features driving the prediction for the predicted class.

    Args:
        model: the served RandomForest.
        X: one-row inference feature matrix (columns match the training schema).
        predicted_label: the predicted class in model terms (one of model.classes_, e.g. "HW").
        k: how many top features to return.

    Returns:
        List of {"feature", "contribution"} sorted by absolute contribution, with signed values
        (positive pushes toward the predicted class, negative away from it).
    """
    explainer = _get_explainer(model)
    class_index = list(model.classes_).index(predicted_label)
    contribs = _class_contributions(explainer.shap_values(X), class_index)

    features = list(X.columns)
    order = np.argsort(np.abs(contribs))[::-1][:k]
    return [
        {"feature": features[i], "contribution": round(float(contribs[i]), 4)}
        for i in order
    ]
