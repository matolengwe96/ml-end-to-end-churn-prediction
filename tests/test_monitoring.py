"""Tests for drift monitoring and experiment-tracking-adjacent utilities."""

from __future__ import annotations

import json

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