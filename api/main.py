"""FastAPI REST service for the Telecom Churn Prediction pipeline.

Endpoints
---------
GET  /health          — API liveness + model readiness check
GET  /metrics         — Evaluation metrics for all trained models
POST /predict         — Single-record churn prediction
POST /predict/batch   — Batch churn prediction (list of records)

Run locally
-----------
::

    uvicorn api.main:app --reload --port 8000

Interactive docs available at http://localhost:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.config import METRICS_PATH, MODEL_METADATA_PATH
from src.monitoring import log_prediction
from src.predict import load_model, predict_batch, predict_single
from src.schemas import (
    BatchPredictionResponse,
    CustomerRecord,
    HealthResponse,
    ModelMetricsResponse,
    PredictionResponse,
)
from src.utils import load_json

LOGGER = logging.getLogger(__name__)

# Module-level model cache populated during startup.
_MODEL: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model into memory when the server starts."""
    global _MODEL
    try:
        _MODEL = load_model()
        LOGGER.info("Trained model loaded successfully on startup.")
    except FileNotFoundError:
        LOGGER.warning(
            "Model artefact not found at startup. "
            "Train the model first: python -m src.train"
        )
    yield
    _MODEL = None


app = FastAPI(
    title="Telecom Churn Prediction API",
    description=(
        "REST API for predicting customer churn probability "
        "from Telco account features using a pre-trained ML pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _require_model() -> Any:
    """Return the loaded model or raise 503 if not available."""
    if _MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model not loaded. Train the model first: "
                "python -m src.train"
            ),
        )
    return _MODEL


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Liveness and model-readiness check",
)
def health_check() -> HealthResponse:
    """Returns 200 with ``model_loaded=true`` when the model is ready."""
    return HealthResponse(status="healthy", model_loaded=_MODEL is not None)


@app.get(
    "/metrics",
    response_model=ModelMetricsResponse,
    tags=["model"],
    summary="Evaluation metrics for all trained models",
)
def get_metrics() -> ModelMetricsResponse:
    """Returns per-model hold-out and cross-validation metrics."""
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics file not found. Train the model first.",
        )
    metrics = load_json(METRICS_PATH)

    best_model: str | None = None
    if MODEL_METADATA_PATH.exists():
        meta = load_json(MODEL_METADATA_PATH)
        if isinstance(meta, dict):
            best_model = meta.get("best_model")

    return ModelMetricsResponse(metrics=metrics, best_model=best_model)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
    summary="Predict churn for a single customer",
)
def predict_one(record: CustomerRecord) -> PredictionResponse:
    """Returns a churn class label and probability for one customer record."""
    model = _require_model()
    input_df = pd.DataFrame([record.model_dump()])

    try:
        result = predict_single(input_df, model=model)
    except Exception as exc:
        LOGGER.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    log_prediction(record.model_dump(), result, source="api")
    return PredictionResponse(**result)


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["prediction"],
    summary="Predict churn for multiple customers",
)
def predict_many(records: list[CustomerRecord]) -> BatchPredictionResponse:
    """Returns predictions for a list of customer records in one request."""
    if not records:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Batch must contain at least one record.",
        )

    model = _require_model()
    input_df = pd.DataFrame([r.model_dump() for r in records])

    try:
        results = predict_batch(input_df, model=model)
    except Exception as exc:
        LOGGER.error("Batch prediction failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    predictions = [
        PredictionResponse(
            predicted_class=results["predicted_class"][i],
            predicted_label=results["predicted_label"][i],
            churn_probability=(
                results["churn_probability"][i]
                if results["churn_probability"] is not None
                else None
            ),
        )
        for i in range(len(records))
    ]
    return BatchPredictionResponse(predictions=predictions, count=len(predictions))
