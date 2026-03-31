"""SHAP-based model explainability for individual predictions.

Uses a ``TreeExplainer`` for tree-based models (Random Forest, Gradient
Boosting) and a ``LinearExplainer`` for Logistic Regression.  The module
degrades gracefully when SHAP is not installed and always returns the
top-N most influential features in a stable dict structure so callers
don't need to handle raw SHAP arrays.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils import get_logger

LOGGER = get_logger(__name__)

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    shap = None  # type: ignore[assignment]
    SHAP_AVAILABLE = False


# Mapping from sklearn class name → SHAP explainer factory.
_TREE_MODELS = {"RandomForestClassifier", "GradientBoostingClassifier", "XGBClassifier"}
_LINEAR_MODELS = {"LogisticRegression", "LinearSVC", "SGDClassifier"}


def _get_preprocessed(pipeline: Any, X: pd.DataFrame) -> np.ndarray:
    """Return the numeric array after applying the pipeline preprocessor."""
    if not hasattr(pipeline, "named_steps"):
        raise TypeError("Expected a fitted sklearn Pipeline with named_steps.")
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        return X.values
    return preprocessor.transform(X)


def _get_feature_names(pipeline: Any, original_columns: list[str]) -> list[str]:
    """Retrieve output feature names from the fitted preprocessor."""
    if not hasattr(pipeline, "named_steps"):
        return original_columns
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        try:
            return [str(n) for n in preprocessor.get_feature_names_out(original_columns)]
        except Exception:
            pass
    return original_columns


def _build_explainer(pipeline: Any, X_transformed: np.ndarray) -> Any:
    """Instantiate the most appropriate SHAP explainer for the pipeline model."""
    if not hasattr(pipeline, "named_steps"):
        raise TypeError("Expected a fitted sklearn Pipeline with named_steps.")
    model = pipeline.named_steps.get("model")
    if model is None:
        raise ValueError("Pipeline must contain a step named 'model'.")

    model_class = type(model).__name__

    if model_class in _TREE_MODELS:
        return shap.TreeExplainer(model)

    if model_class in _LINEAR_MODELS:
        masker = shap.maskers.Independent(X_transformed)
        return shap.LinearExplainer(model, masker)

    # Fallback: KernelExplainer works with any predict_proba interface.
    LOGGER.warning(
        "No native SHAP explainer for %s — using KernelExplainer (slower).",
        model_class,
    )
    background = shap.kmeans(X_transformed, min(50, len(X_transformed)))
    return shap.KernelExplainer(pipeline.predict_proba, background)


def explain_prediction(
    pipeline: Any,
    record: pd.DataFrame,
    top_n: int = 10,
) -> dict[str, Any]:
    """Compute SHAP values for a single record and return top contributing features.

    Parameters
    ----------
    pipeline:
        A fitted sklearn Pipeline with ``preprocessor`` and ``model`` steps.
    record:
        A single-row DataFrame containing the raw (pre-transform) feature values.
    top_n:
        Number of top features to return, ordered by absolute SHAP value.

    Returns
    -------
    dict with keys:
        ``top_features`` — list of ``{feature, shap_value, direction}`` dicts
        ``base_value``   — expected model output (log-odds or probability)
        ``shap_available`` — bool
    """
    if not SHAP_AVAILABLE:
        return {
            "top_features": [],
            "base_value": None,
            "shap_available": False,
            "reason": "shap_not_installed",
        }

    if not hasattr(pipeline, "named_steps"):
        return {
            "top_features": [],
            "base_value": None,
            "shap_available": False,
            "reason": "invalid_pipeline",
        }

    original_columns = record.columns.tolist()
    X_transformed = _get_preprocessed(pipeline, record)
    feature_names = _get_feature_names(pipeline, original_columns)

    try:
        explainer = _build_explainer(pipeline, X_transformed)
        shap_values = explainer.shap_values(X_transformed)
    except Exception as exc:
        LOGGER.warning("SHAP explanation failed: %s", exc)
        return {
            "top_features": [],
            "base_value": None,
            "shap_available": False,
            "reason": str(exc),
        }

    # For binary classifiers, shap_values may be [neg_class, pos_class] list.
    if isinstance(shap_values, list) and len(shap_values) == 2:
        values = shap_values[1][0]  # Churn = positive class
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        values = shap_values[0]
    else:
        values = np.asarray(shap_values).flatten()

    # Align length in case feature extraction differs.
    n = min(len(values), len(feature_names))
    values = values[:n]
    names = feature_names[:n]

    # Sort by absolute magnitude and take top_n.
    order = np.argsort(np.abs(values))[::-1][:top_n]
    top_features = [
        {
            "feature": names[i],
            "shap_value": round(float(values[i]), 6),
            "direction": "increases_churn" if values[i] > 0 else "decreases_churn",
        }
        for i in order
    ]

    base_value = None
    if hasattr(explainer, "expected_value"):
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            base_value = float(ev[1]) if len(ev) > 1 else float(ev[0])
        else:
            base_value = float(ev)

    return {
        "top_features": top_features,
        "base_value": base_value,
        "shap_available": True,
    }
