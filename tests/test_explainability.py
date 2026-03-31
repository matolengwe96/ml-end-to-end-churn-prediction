"""Tests for src.explainability — SHAP-based feature importance."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.explainability import explain_prediction


@pytest.fixture()
def single_record(sample_features: pd.DataFrame) -> pd.DataFrame:
    """Return a one-row DataFrame from the shared fixture."""
    return sample_features.iloc[:1].copy()


class TestExplainPrediction:
    def test_returns_expected_keys(self, trained_pipeline: Pipeline, single_record: pd.DataFrame):
        result = explain_prediction(trained_pipeline, single_record)
        assert "top_features" in result
        assert "base_value" in result
        assert "shap_available" in result

    def test_shap_available(self, trained_pipeline: Pipeline, single_record: pd.DataFrame):
        result = explain_prediction(trained_pipeline, single_record)
        assert result["shap_available"] is True

    def test_top_features_is_list(self, trained_pipeline: Pipeline, single_record: pd.DataFrame):
        result = explain_prediction(trained_pipeline, single_record)
        assert isinstance(result["top_features"], list)

    def test_top_features_have_required_keys(
        self, trained_pipeline: Pipeline, single_record: pd.DataFrame
    ):
        result = explain_prediction(trained_pipeline, single_record)
        for feat in result["top_features"]:
            assert "feature" in feat
            assert "shap_value" in feat
            assert "direction" in feat

    def test_direction_values_are_valid(
        self, trained_pipeline: Pipeline, single_record: pd.DataFrame
    ):
        result = explain_prediction(trained_pipeline, single_record)
        for feat in result["top_features"]:
            assert feat["direction"] in {"increases_churn", "decreases_churn"}

    def test_top_n_limits_result_length(
        self, trained_pipeline: Pipeline, single_record: pd.DataFrame
    ):
        result = explain_prediction(trained_pipeline, single_record, top_n=3)
        assert len(result["top_features"]) <= 3

    def test_base_value_is_float_or_none(
        self, trained_pipeline: Pipeline, single_record: pd.DataFrame
    ):
        result = explain_prediction(trained_pipeline, single_record)
        assert result["base_value"] is None or isinstance(result["base_value"], float)

    def test_shap_values_are_floats(
        self, trained_pipeline: Pipeline, single_record: pd.DataFrame
    ):
        result = explain_prediction(trained_pipeline, single_record)
        for feat in result["top_features"]:
            assert isinstance(feat["shap_value"], float)

    def test_degraded_gracefully_with_non_pipeline(self, single_record: pd.DataFrame):
        """A plain dict (not a pipeline) should return shap_available=False."""
        result = explain_prediction({}, single_record)  # type: ignore[arg-type]
        assert result["shap_available"] is False
        assert result["top_features"] == []
