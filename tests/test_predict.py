"""Tests for prediction helper functions."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_preprocessing import build_preprocessor
from src.predict import predict_batch, predict_single


def make_train_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
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
            "InternetService": ["DSL", "Fiber optic", "DSL", "No", "Fiber optic", "DSL", "DSL", "No"],
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
            "MonthlyCharges": [29.0, 90.0, 45.0, 20.0, 110.0, 55.0, 75.0, 25.0],
            "TotalCharges": [29.0, 2100.0, 560.0, 70.0, 4800.0, 430.0, 2200.0, 110.0],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    return X, y


def fit_test_pipeline() -> Pipeline:
    X, y = make_train_data()
    preprocessor, _, _ = build_preprocessor(X)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=500)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def test_predict_single_returns_expected_keys() -> None:
    model = fit_test_pipeline()
    record = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 10,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "MonthlyCharges": 60.0,
        "TotalCharges": 500.0,
    }

    output = predict_single(record, model=model)

    assert set(output.keys()) == {"predicted_class", "predicted_label", "churn_probability"}
    assert output["predicted_label"] in {"Churn", "No Churn"}


def test_predict_batch_dataframe_shape() -> None:
    model = fit_test_pipeline()
    X, _ = make_train_data()
    batch_output = predict_batch(X.head(3), model=model)

    assert len(batch_output["predicted_class"]) == 3
    assert len(batch_output["predicted_label"]) == 3
    assert batch_output["churn_probability"] is not None


def test_predict_single_raises_on_missing_required_feature() -> None:
    model = fit_test_pipeline()
    invalid_record = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
    }

    with pytest.raises(ValueError, match="Missing required input features"):
        predict_single(invalid_record, model=model)
