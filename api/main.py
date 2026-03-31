"""FastAPI REST service for the Telecom Churn Prediction pipeline.

Endpoints
GET  /health          — API liveness + model readiness check
GET  /metrics         — Evaluation metrics for all trained models
POST /predict         — Single-record churn prediction
POST /predict/batch   — Batch churn prediction (list of records)
POST /explain         — SHAP feature-importance for a single prediction
GET  /versions        — List all saved model versions

Run locally
-----------
::

    uvicorn api.main:app --reload --port 8000

Interactive docs available at http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from src.config import (
    API_KEY,
    API_RATE_LIMIT,
    API_RATE_LIMIT_WINDOW_SECONDS,
    METRICS_PATH,
    MODEL_METADATA_PATH,
    REDIS_URL,
)
from src.drift_monitoring import generate_drift_report
from src.monitoring import log_prediction
from src.predict import load_model, predict_batch, predict_single
from src.rate_limiting import InMemoryRateLimiter, build_redis_rate_limiter
from src.schemas import (
    BatchPredictionResponse,
    CustomerRecord,
    DriftReportResponse,
    ExplainResponse,
    HealthResponse,
    ModelMetricsResponse,
    ModelVersionsResponse,
    PredictionResponse,
)
from src.utils import load_json

LOGGER = logging.getLogger(__name__)

# Module-level model cache populated during startup.
_MODEL: Any = None
RATE_LIMIT_BUCKETS: dict[str, deque[float]] = defaultdict(deque)
MEMORY_RATE_LIMITER = InMemoryRateLimiter(RATE_LIMIT_BUCKETS)
REDIS_RATE_LIMITER = build_redis_rate_limiter(REDIS_URL)


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


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    """Attach a request ID and enforce lightweight per-IP rate limiting."""
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.request_id = request_id

    if request.method == "POST" and request.url.path.startswith("/predict"):
        client_host = request.client.host if request.client else "unknown"
        limited, retry_after, backend = _enforce_rate_limit(client_host)
        request.state.rate_limit_backend = backend
        if limited:
            return Response(
                content="Rate limit exceeded.",
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={
                    "x-request-id": request_id,
                    "Retry-After": str(retry_after or 1),
                },
            )
    else:
        request.state.rate_limit_backend = _rate_limit_backend_name()

    start_time = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    response.headers["x-request-id"] = request_id
    LOGGER.info(
        "request_id=%s method=%s path=%s status_code=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


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


def _require_api_key(request: Request) -> None:
    """Enforce API key auth for inference endpoints when configured."""
    if not API_KEY:
        return

    provided_key = request.headers.get("x-api-key", "")
    if provided_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized. Provide a valid x-api-key header.",
        )


def _rate_limit_backend_name() -> str:
    """Return the currently active rate limiting backend name."""
    return "redis" if REDIS_RATE_LIMITER is not None else "memory"


def _enforce_rate_limit(client_host: str) -> tuple[bool, int | None, str]:
    """Enforce per-client rate limiting using Redis when available."""
    if REDIS_RATE_LIMITER is not None:
        try:
            limited, retry_after = REDIS_RATE_LIMITER.is_rate_limited(
                client_host,
                API_RATE_LIMIT,
                API_RATE_LIMIT_WINDOW_SECONDS,
            )
            return limited, retry_after, "redis"
        except Exception as exc:
            LOGGER.warning("Redis rate limit check failed (%s). Falling back to memory.", exc)

    limited, retry_after = MEMORY_RATE_LIMITER.is_rate_limited(
        client_host,
        API_RATE_LIMIT,
        API_RATE_LIMIT_WINDOW_SECONDS,
    )
    return limited, retry_after, "memory"


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Liveness and model-readiness check",
)
def health_check() -> HealthResponse:
    """Returns 200 with ``model_loaded=true`` when the model is ready."""
    return HealthResponse(
        status="healthy",
        model_loaded=_MODEL is not None,
        rate_limit_backend=_rate_limit_backend_name(),
    )


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


@app.get(
    "/drift",
    response_model=DriftReportResponse,
    tags=["monitoring"],
    summary="Compare live inference traffic against the training baseline",
)
def get_drift_report() -> DriftReportResponse:
    """Generate a drift summary using the saved baseline and prediction log."""
    try:
        report = generate_drift_report()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    return DriftReportResponse(**report)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
    summary="Predict churn for a single customer",
)
def predict_one(record: CustomerRecord, request: Request) -> PredictionResponse:
    """Returns a churn class label and probability for one customer record."""
    _require_api_key(request)
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

    log_prediction(
        record.model_dump(),
        result,
        source="api",
        request_id=getattr(request.state, "request_id", None),
    )
    return PredictionResponse(**result)


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["prediction"],
    summary="Predict churn for multiple customers",
)
def predict_many(records: list[CustomerRecord], request: Request) -> BatchPredictionResponse:
    """Returns predictions for a list of customer records in one request."""
    _require_api_key(request)
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
    for index, prediction in enumerate(predictions):
        log_prediction(
            records[index].model_dump(),
            prediction.model_dump(),
            source="api-batch",
            request_id=getattr(request.state, "request_id", None),
        )
    return BatchPredictionResponse(predictions=predictions, count=len(predictions))


@app.post(
    "/explain",
    response_model=ExplainResponse,
    tags=["prediction"],
    summary="SHAP feature importance for a single customer prediction",
)
def explain_one(record: CustomerRecord, request: Request) -> ExplainResponse:
    """Returns the top SHAP feature contributions alongside the churn prediction."""
    from src.explainability import explain_prediction  # noqa: PLC0415

    _require_api_key(request)
    model = _require_model()
    input_df = pd.DataFrame([record.model_dump()])

    try:
        result = predict_single(input_df, model=model)
        explanation = explain_prediction(model, input_df)
    except Exception as exc:
        LOGGER.error("Explain failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    return ExplainResponse(
        predicted_class=result["predicted_class"],
        predicted_label=result["predicted_label"],
        churn_probability=result.get("churn_probability"),
        top_features=explanation["top_features"],
        base_value=explanation["base_value"],
        shap_available=explanation["shap_available"],
    )


@app.get(
    "/versions",
    response_model=ModelVersionsResponse,
    tags=["model"],
    summary="List all saved model versions",
)
def get_versions() -> ModelVersionsResponse:
    """Returns the version registry with all trained model snapshots."""
    from src.model_versioning import list_versions  # noqa: PLC0415

    try:
        data = list_versions()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Version registry not found. Train the model first.",
        ) from exc

    return ModelVersionsResponse(
        versions=data.get("versions", []),
        active_version=data.get("active_version"),
    )
