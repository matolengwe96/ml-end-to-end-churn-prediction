"""Pydantic schemas for request and response validation.

These schemas are consumed by the FastAPI layer to strictly validate all
input before passing a DataFrame to the prediction pipeline.  They are
kept here (inside the ``src`` package) so they can be imported by both the
API and by tests without creating a circular dependency.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CustomerRecord(BaseModel):
    """Validated input schema for a single Telco customer record."""

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=72, description="Customer tenure in months.")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0.0, description="Monthly bill amount in USD.")
    TotalCharges: float = Field(ge=0.0, description="Total charges to date in USD.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
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
                "MonthlyCharges": 60.0,
                "TotalCharges": 720.0,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Output schema for a single-record prediction result."""

    predicted_class: int
    predicted_label: str
    churn_probability: float | None = None


class BatchPredictionResponse(BaseModel):
    """Output schema for a list of prediction results."""

    predictions: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    """API liveness and readiness response."""

    status: str
    model_loaded: bool
    version: str = "1.0.0"


class ModelMetricsResponse(BaseModel):
    """Evaluation metrics for all trained models."""

    metrics: dict
    best_model: str | None = None


class DriftReportResponse(BaseModel):
    """High-level training-vs-inference drift status."""

    status: str
    current_rows: int
    numeric_drift: dict
    categorical_drift: dict


class ExplainFeature(BaseModel):
    """SHAP contribution of a single feature to the prediction."""

    feature: str
    shap_value: float
    direction: str


class ExplainResponse(BaseModel):
    """SHAP explanation for a single prediction."""

    predicted_class: int
    predicted_label: str
    churn_probability: float | None = None
    top_features: list[ExplainFeature]
    base_value: float | None = None
    shap_available: bool


class ModelVersion(BaseModel):
    """Metadata for a single saved model version."""

    version_id: str
    model_name: str
    trained_at_utc: str
    f1_score: float | None = None
    roc_auc: float | None = None
    artifact_path: str


class ModelVersionsResponse(BaseModel):
    """Registry of all saved model versions."""

    versions: list[ModelVersion]
    active_version: str | None = None
