"""Feature engineering helpers for model artifact introspection."""

from __future__ import annotations

from typing import Any


def extract_transformed_feature_names(
    trained_pipeline: Any, original_feature_names: list[str]
) -> list[str]:
    """Return transformed feature names from a fitted preprocessing pipeline."""
    preprocessor = trained_pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        return original_feature_names

    if hasattr(preprocessor, "get_feature_names_out"):
        return preprocessor.get_feature_names_out(original_feature_names).tolist()

    return original_feature_names
