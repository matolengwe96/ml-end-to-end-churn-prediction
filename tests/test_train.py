"""Tests for training and evaluation pipeline."""

from __future__ import annotations

import pandas as pd

from src.data_preprocessing import build_preprocessor
from src.evaluate import compare_models, evaluate_model


def make_synthetic_data() -> pd.DataFrame:
    """Create lightweight synthetic churn-like data for tests."""
    return pd.DataFrame(
        {
            "gender": ["Female", "Male", "Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 0, 1, 1],
            "Partner": ["Yes", "No", "Yes", "No", "No", "Yes"],
            "Dependents": ["No", "No", "Yes", "No", "Yes", "No"],
            "tenure": [1, 24, 12, 3, 45, 8],
            "PhoneService": ["Yes", "Yes", "No", "Yes", "Yes", "No"],
            "MultipleLines": [
                "No",
                "Yes",
                "No phone service",
                "No",
                "Yes",
                "No phone service",
            ],
            "InternetService": ["DSL", "Fiber optic", "DSL", "No", "Fiber optic", "DSL"],
            "OnlineSecurity": ["No", "No", "Yes", "No internet service", "Yes", "No"],
            "MonthlyCharges": [29.0, 90.0, 45.0, 20.0, 110.0, 55.0],
            "TotalCharges": [29.0, 2100.0, 560.0, 70.0, 4800.0, 430.0],
            "Churn": [0, 1, 0, 1, 0, 1],
        }
    )


def test_build_preprocessor_detects_feature_types() -> None:
    df = make_synthetic_data().drop(columns=["Churn"])
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(df)

    assert preprocessor is not None
    assert "tenure" in numeric_cols
    assert "gender" in categorical_cols


def test_evaluate_model_returns_expected_keys() -> None:
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    y_proba = [0.1, 0.9, 0.4, 0.2]

    metrics = evaluate_model(y_true=y_true, y_pred=y_pred, y_proba=y_proba)
    expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc"}

    assert expected_keys.issubset(metrics.keys())
    assert metrics["f1"] is not None


def test_compare_models_prefers_higher_f1_then_auc() -> None:
    results = {
        "ModelA": {"f1": 0.75, "roc_auc": 0.81},
        "ModelB": {"f1": 0.75, "roc_auc": 0.84},
    }
    best_name, best_metrics = compare_models(results)

    assert best_name == "ModelB"
    assert best_metrics["roc_auc"] == 0.84
