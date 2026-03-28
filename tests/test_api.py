"""Tests for FastAPI prediction endpoints.

The ``client`` fixture injects the session-scoped ``trained_pipeline``
(from conftest.py) into the API module so these tests run without
a trained model file on disk and without hitting the real model loader.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from api.main import app

# ── Shared sample payload ─────────────────────────────────────────────────────

SAMPLE_PAYLOAD: dict = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 60.0,
    "TotalCharges": 720.0,
}


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def client(trained_pipeline: Pipeline):
    """Return a TestClient with the fixture pipeline injected as the model."""
    with patch("api.main.load_model", return_value=trained_pipeline):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture()
def client_no_model():
    """Return a TestClient where model loading always fails."""
    with patch("api.main.load_model", side_effect=FileNotFoundError("no model")):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ── Health endpoint ───────────────────────────────────────────────────────────


def test_health_returns_200_when_model_loaded(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "version" in data


def test_health_returns_model_not_loaded_when_missing(client_no_model) -> None:
    response = client_no_model.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


# ── Single predict endpoint ───────────────────────────────────────────────────


def test_predict_returns_200(client) -> None:
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200


def test_predict_response_has_expected_keys(client) -> None:
    data = client.post("/predict", json=SAMPLE_PAYLOAD).json()
    assert "predicted_class" in data
    assert "predicted_label" in data
    assert "churn_probability" in data


def test_predict_class_is_binary(client) -> None:
    data = client.post("/predict", json=SAMPLE_PAYLOAD).json()
    assert data["predicted_class"] in (0, 1)


def test_predict_label_is_valid(client) -> None:
    data = client.post("/predict", json=SAMPLE_PAYLOAD).json()
    assert data["predicted_label"] in ("Churn", "No Churn")


def test_predict_probability_in_range(client) -> None:
    data = client.post("/predict", json=SAMPLE_PAYLOAD).json()
    prob = data["churn_probability"]
    if prob is not None:
        assert 0.0 <= prob <= 1.0


def test_predict_invalid_gender_returns_422(client) -> None:
    bad = {**SAMPLE_PAYLOAD, "gender": "Unknown"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_negative_tenure_returns_422(client) -> None:
    bad = {**SAMPLE_PAYLOAD, "tenure": -5}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_503_when_no_model(client_no_model) -> None:
    response = client_no_model.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 503


# ── Batch predict endpoint ────────────────────────────────────────────────────


def test_batch_predict_returns_correct_count(client) -> None:
    response = client.post("/predict/batch", json=[SAMPLE_PAYLOAD, SAMPLE_PAYLOAD])
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2


def test_batch_predict_each_item_has_required_keys(client) -> None:
    data = client.post("/predict/batch", json=[SAMPLE_PAYLOAD]).json()
    item = data["predictions"][0]
    assert "predicted_class" in item
    assert "predicted_label" in item


def test_batch_predict_empty_list_returns_422(client) -> None:
    response = client.post("/predict/batch", json=[])
    assert response.status_code == 422
