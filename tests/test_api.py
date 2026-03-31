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

import api.main as api_main
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
        api_main.RATE_LIMIT_BUCKETS.clear()
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture()
def client_no_model():
    """Return a TestClient where model loading always fails."""
    with patch("api.main.load_model", side_effect=FileNotFoundError("no model")):
        api_main.RATE_LIMIT_BUCKETS.clear()
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
    assert data["rate_limit_backend"] in {"memory", "redis"}


def test_health_returns_model_not_loaded_when_missing(client_no_model) -> None:
    response = client_no_model.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


# ── Single predict endpoint ───────────────────────────────────────────────────


def test_predict_returns_200(client) -> None:
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200
    assert "x-request-id" in response.headers


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


def test_predict_requires_api_key_when_enabled(client, monkeypatch) -> None:
    monkeypatch.setattr(api_main, "API_KEY", "secret-token")
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 401

    authorized = client.post(
        "/predict",
        json=SAMPLE_PAYLOAD,
        headers={"x-api-key": "secret-token"},
    )
    assert authorized.status_code == 200


def test_predict_rate_limit_returns_429(client, monkeypatch) -> None:
    monkeypatch.setattr(api_main, "API_RATE_LIMIT", 1)
    api_main.RATE_LIMIT_BUCKETS.clear()
    monkeypatch.setattr(api_main, "REDIS_RATE_LIMITER", None)

    first = client.post("/predict", json=SAMPLE_PAYLOAD)
    second = client.post("/predict", json=SAMPLE_PAYLOAD)

    assert first.status_code == 200
    assert second.status_code == 429


def test_predict_rate_limit_can_use_redis_backend(client, monkeypatch) -> None:
    class FakeRedisLimiter:
        def __init__(self) -> None:
            self.calls = 0

        def is_rate_limited(self, client_id: str, limit: int, window_seconds: int):
            self.calls += 1
            return (self.calls > 1, 42 if self.calls > 1 else None)

    fake = FakeRedisLimiter()
    monkeypatch.setattr(api_main, "REDIS_RATE_LIMITER", fake)
    monkeypatch.setattr(api_main, "API_RATE_LIMIT", 1)

    first = client.post("/predict", json=SAMPLE_PAYLOAD)
    second = client.post("/predict", json=SAMPLE_PAYLOAD)

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["Retry-After"] == "42"


def test_predict_falls_back_to_memory_when_redis_errors(client, monkeypatch) -> None:
    class BrokenRedisLimiter:
        def is_rate_limited(self, client_id: str, limit: int, window_seconds: int):
            raise RuntimeError("redis down")

    monkeypatch.setattr(api_main, "REDIS_RATE_LIMITER", BrokenRedisLimiter())
    monkeypatch.setattr(api_main, "API_RATE_LIMIT", 1)
    api_main.RATE_LIMIT_BUCKETS.clear()

    first = client.post("/predict", json=SAMPLE_PAYLOAD)
    second = client.post("/predict", json=SAMPLE_PAYLOAD)

    assert first.status_code == 200
    assert second.status_code == 429


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


def test_drift_endpoint_returns_report_structure(client, monkeypatch) -> None:
    monkeypatch.setattr(
        api_main,
        "generate_drift_report",
        lambda: {
            "status": "stable",
            "current_rows": 12,
            "numeric_drift": {"tenure": {"drift_detected": False}},
            "categorical_drift": {},
        },
    )

    response = client.get("/drift")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "stable"
    assert payload["current_rows"] == 12


def test_drift_endpoint_returns_404_when_baseline_missing(client, monkeypatch) -> None:
    def _raise_missing() -> dict:
        raise FileNotFoundError("missing baseline")

    monkeypatch.setattr(api_main, "generate_drift_report", _raise_missing)
    response = client.get("/drift")
    assert response.status_code == 404


@pytest.fixture(autouse=True)
def reset_api_guards(monkeypatch):
    monkeypatch.setattr(api_main, "API_KEY", "")
    monkeypatch.setattr(api_main, "API_RATE_LIMIT", 60)
    monkeypatch.setattr(api_main, "API_RATE_LIMIT_WINDOW_SECONDS", 60)
    monkeypatch.setattr(api_main, "REDIS_RATE_LIMITER", None)
    api_main.RATE_LIMIT_BUCKETS.clear()
