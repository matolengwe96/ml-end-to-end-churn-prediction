"""Shared pytest fixtures used across all test modules."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_preprocessing import build_preprocessor

# ── Synthetic dataset (full Telco columns) ────────────────────────────────────


@pytest.fixture(scope="session")
def sample_features() -> pd.DataFrame:
    """Return a small but representative feature DataFrame."""
    return pd.DataFrame(
        {
            "gender": ["Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 0, 1, 1, 0, 1],
            "Partner": ["Yes", "No", "Yes", "No", "No", "Yes", "Yes", "No"],
            "Dependents": ["No", "No", "Yes", "No", "Yes", "No", "No", "Yes"],
            "tenure": [1, 24, 12, 3, 45, 8, 30, 5],
            "PhoneService": ["Yes", "Yes", "No", "Yes", "Yes", "No", "Yes", "Yes"],
            "MultipleLines": [
                "No",
                "Yes",
                "No phone service",
                "No",
                "Yes",
                "No phone service",
                "Yes",
                "No",
            ],
            "InternetService": [
                "DSL",
                "Fiber optic",
                "DSL",
                "No",
                "Fiber optic",
                "DSL",
                "DSL",
                "No",
            ],
            "OnlineSecurity": [
                "No",
                "No",
                "Yes",
                "No internet service",
                "Yes",
                "No",
                "Yes",
                "No internet service",
            ],
            "OnlineBackup": [
                "Yes",
                "No",
                "Yes",
                "No internet service",
                "No",
                "Yes",
                "No",
                "No internet service",
            ],
            "DeviceProtection": [
                "No",
                "Yes",
                "No",
                "No internet service",
                "Yes",
                "No",
                "Yes",
                "No internet service",
            ],
            "TechSupport": [
                "No",
                "No",
                "Yes",
                "No internet service",
                "Yes",
                "No",
                "No",
                "No internet service",
            ],
            "StreamingTV": [
                "No",
                "Yes",
                "No",
                "No internet service",
                "Yes",
                "No",
                "Yes",
                "No internet service",
            ],
            "StreamingMovies": [
                "No",
                "Yes",
                "No",
                "No internet service",
                "No",
                "Yes",
                "Yes",
                "No internet service",
            ],
            "Contract": [
                "Month-to-month",
                "One year",
                "Two year",
                "Month-to-month",
                "One year",
                "Two year",
                "Month-to-month",
                "One year",
            ],
            "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            "MonthlyCharges": [29.0, 90.0, 45.0, 20.0, 110.0, 55.0, 75.0, 25.0],
            "TotalCharges": [29.0, 2100.0, 560.0, 70.0, 4800.0, 430.0, 2200.0, 110.0],
        }
    )


@pytest.fixture(scope="session")
def sample_target() -> pd.Series:
    """Return binary target labels matching sample_features."""
    return pd.Series([0, 1, 0, 1, 0, 1, 0, 1], name="Churn")


@pytest.fixture(scope="session")
def trained_pipeline(sample_features: pd.DataFrame, sample_target: pd.Series) -> Pipeline:
    """Fit a minimal LR pipeline on synthetic data — reused across all tests."""
    preprocessor, _, _ = build_preprocessor(sample_features)
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )
    pipeline.fit(sample_features, sample_target)
    return pipeline
