"""Prediction helpers for saved churn model pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import MODEL_PATH
from src.utils import load_saved_model


def load_model(model_path: Path = MODEL_PATH) -> Any:
    """Load persisted model pipeline from disk."""
    return load_saved_model(model_path)


def _to_dataframe(data: pd.DataFrame | Mapping[str, Any]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, Mapping):
        return pd.DataFrame([dict(data)])
    raise TypeError("Input must be a pandas DataFrame or a dict-like record.")


def predict_batch(
    data: pd.DataFrame | Mapping[str, Any],
    model: Any | None = None,
    model_path: Path = MODEL_PATH,
) -> dict[str, list[float | int | str] | None]:
    """Predict churn classes for one or many records."""
    pipeline = model if model is not None else load_model(model_path)
    features = _to_dataframe(data)

    raw_predictions = pipeline.predict(features)
    predicted_labels = ["Churn" if int(value) == 1 else "No Churn" for value in raw_predictions]

    churn_probabilities: list[float] | None = None
    if hasattr(pipeline, "predict_proba"):
        churn_probabilities = pipeline.predict_proba(features)[:, 1].tolist()

    return {
        "predicted_class": [int(value) for value in raw_predictions],
        "predicted_label": predicted_labels,
        "churn_probability": churn_probabilities,
    }


def predict_single(
    record: pd.DataFrame | Mapping[str, Any],
    model: Any | None = None,
    model_path: Path = MODEL_PATH,
) -> dict[str, float | int | str | None]:
    """Predict churn for a single record and return formatted output."""
    result = predict_batch(record, model=model, model_path=model_path)

    probability = None
    if result["churn_probability"] is not None:
        probability = float(result["churn_probability"][0])

    return {
        "predicted_class": int(result["predicted_class"][0]),
        "predicted_label": str(result["predicted_label"][0]),
        "churn_probability": probability,
    }
