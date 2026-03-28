"""Integration tests that exercise the full trained artifact.

These tests are automatically skipped when ``models/best_model.joblib`` does
not exist so that CI passes during the unit-test job (which runs before
training).  Run them locally after ``python -m src.train``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

MODEL_PATH = Path("models/best_model.joblib")
requires_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="No trained model found — run `python -m src.train` first.",
)

# ── Full churn record for the real pipeline ───────────────────────────────────

FULL_RECORD: dict = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 24,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 55.0,
    "TotalCharges": 1320.0,
}


# ── Integration tests ─────────────────────────────────────────────────────────


@requires_model
def test_predict_single_returns_valid_output() -> None:
    from src.predict import predict_single

    result = predict_single(FULL_RECORD)
    assert isinstance(result, dict)
    assert result["predicted_class"] in (0, 1)
    assert result["predicted_label"] in ("Churn", "No Churn")


@requires_model
def test_predict_single_probability_is_in_range() -> None:
    from src.predict import predict_single

    result = predict_single(FULL_RECORD)
    prob = result["churn_probability"]
    assert prob is not None
    assert 0.0 <= prob <= 1.0


@requires_model
def test_predict_batch_matches_input_length() -> None:
    from src.predict import predict_batch

    df = pd.DataFrame([FULL_RECORD, FULL_RECORD, FULL_RECORD])
    result = predict_batch(df)
    assert len(result["predicted_class"]) == 3
    assert len(result["predicted_label"]) == 3
    assert result["churn_probability"] is not None
    assert len(result["churn_probability"]) == 3


@requires_model
def test_model_metadata_file_exists() -> None:
    meta_path = Path("models/model_metadata.json")
    assert meta_path.exists(), "model_metadata.json missing after training."


@requires_model
def test_model_metrics_contain_required_keys() -> None:
    from src.utils import load_json

    metrics = load_json(Path("models/model_metrics.json"))
    assert isinstance(metrics, dict)
    for model_name, m in metrics.items():
        for key in ("accuracy", "f1", "precision", "recall"):
            assert key in m, f"Key '{key}' missing in metrics for {model_name}"
