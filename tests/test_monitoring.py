"""Tests for drift monitoring and experiment-tracking-adjacent utilities."""

from __future__ import annotations

import json

import src.monitoring as monitoring
from src.drift_monitoring import (
    build_reference_profile,
    load_logged_prediction_inputs,
    summarize_drift,
)


def test_build_reference_profile_includes_numeric_and_categorical(sample_features) -> None:
    profile = build_reference_profile(sample_features)

    assert profile["row_count"] == len(sample_features)
    assert "tenure" in profile["numeric"]
    assert "gender" in profile["categorical"]


def test_load_logged_prediction_inputs_reads_raw_input_records(tmp_path) -> None:
    log_path = tmp_path / "predictions.jsonl"
    entries = [
        {
            "input_record": {"tenure": 12, "gender": "Female"},
            "predicted_class": 0,
        },
        {
            "input_record": {"tenure": 36, "gender": "Male"},
            "predicted_class": 1,
        },
    ]
    log_path.write_text("\n".join(json.dumps(item) for item in entries), encoding="utf-8")

    loaded = load_logged_prediction_inputs(log_path)

    assert list(loaded.columns) == ["tenure", "gender"]
    assert len(loaded) == 2


def test_summarize_drift_flags_large_numeric_shift(sample_features) -> None:
    reference = build_reference_profile(sample_features)
    shifted = sample_features.copy()
    shifted["tenure"] = shifted["tenure"] + 100

    report = summarize_drift(reference, shifted, numeric_threshold=0.5)

    assert report["status"] == "drift_detected"
    assert report["numeric_drift"]["tenure"]["drift_detected"] is True


def test_log_prediction_rotates_when_size_limit_exceeded(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(monitoring, "PREDICTION_LOG_PATH", log_path)
    monkeypatch.setattr(monitoring, "PREDICTION_LOG_MAX_BYTES", 250)
    monkeypatch.setattr(monitoring, "PREDICTION_LOG_BACKUP_COUNT", 2)

    record = {"tenure": 12, "gender": "Female", "Contract": "Month-to-month"}
    prediction = {
        "predicted_class": 0,
        "predicted_label": "No Churn",
        "churn_probability": 0.12,
    }

    monitoring.log_prediction(record, prediction, source="api")
    monitoring.log_prediction(record, prediction, source="api")
    monitoring.log_prediction(record, prediction, source="api")

    assert log_path.exists()
    assert (tmp_path / "predictions.1.jsonl").exists()


def test_load_logged_prediction_inputs_reads_rotated_logs(tmp_path) -> None:
    current_path = tmp_path / "predictions.jsonl"
    rotated_path = tmp_path / "predictions.1.jsonl"

    rotated_entries = [
        {"input_record": {"tenure": 12, "gender": "Female"}},
    ]
    current_entries = [
        {"input_record": {"tenure": 36, "gender": "Male"}},
    ]

    rotated_path.write_text(
        "\n".join(json.dumps(item) for item in rotated_entries),
        encoding="utf-8",
    )
    current_path.write_text(
        "\n".join(json.dumps(item) for item in current_entries),
        encoding="utf-8",
    )

    loaded = load_logged_prediction_inputs(current_path)

    assert len(loaded) == 2
    assert set(loaded["tenure"].tolist()) == {12, 36}