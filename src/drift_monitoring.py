"""Reference-profile and drift monitoring utilities for inference data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PREDICTION_LOG_PATH, TRAINING_BASELINE_PATH
from src.utils import get_logger, load_json, save_json

LOGGER = get_logger(__name__)


def _prediction_log_files(log_path: Path) -> list[Path]:
	"""Return the current prediction log plus any rotated siblings."""
	current = [log_path] if log_path.exists() else []
	rotated = sorted(
		path for path in log_path.parent.glob(f"{log_path.stem}.*{log_path.suffix}") if path.is_file()
	)
	return rotated + current


def build_reference_profile(features: pd.DataFrame) -> dict[str, Any]:
	"""Create a lightweight training-data profile for later drift checks."""
	numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
	categorical_columns = features.select_dtypes(exclude=["number"]).columns.tolist()

	profile: dict[str, Any] = {
		"row_count": int(len(features)),
		"numeric": {},
		"categorical": {},
	}

	for column in numeric_columns:
		series = features[column].astype(float)
		profile["numeric"][column] = {
			"mean": float(series.mean()),
			"std": float(series.std() or 0.0),
		}

	for column in categorical_columns:
		frequencies = features[column].astype(str).value_counts(normalize=True)
		profile["categorical"][column] = frequencies.to_dict()

	return profile


def save_reference_profile(profile: dict[str, Any], output_path: Path = TRAINING_BASELINE_PATH) -> None:
	"""Persist the training reference profile."""
	save_json(profile, output_path)


def load_reference_profile(input_path: Path = TRAINING_BASELINE_PATH) -> dict[str, Any]:
	"""Load the saved training reference profile."""
	data = load_json(input_path)
	if not isinstance(data, dict):
		raise ValueError("Training baseline profile must be a JSON object.")
	return data


def load_logged_prediction_inputs(log_path: Path = PREDICTION_LOG_PATH) -> pd.DataFrame:
	"""Read raw inference inputs from the JSONL prediction log."""
	log_files = _prediction_log_files(log_path)
	if not log_files:
		return pd.DataFrame()

	records: list[dict[str, Any]] = []
	for path in log_files:
		with path.open("r", encoding="utf-8") as handle:
			for line in handle:
				line = line.strip()
				if not line:
					continue
				payload = json.loads(line)
				record = payload.get("input_record")
				if isinstance(record, dict):
					records.append(record)

	if not records:
		return pd.DataFrame()
	return pd.DataFrame(records)


def summarize_drift(
	reference_profile: dict[str, Any],
	current_features: pd.DataFrame,
	numeric_threshold: float = 0.25,
	categorical_threshold: float = 0.15,
) -> dict[str, Any]:
	"""Compare current inference data with the training baseline."""
	if current_features.empty:
		return {
			"status": "insufficient_data",
			"current_rows": 0,
			"numeric_drift": {},
			"categorical_drift": {},
		}

	numeric_drift: dict[str, Any] = {}
	for column, stats in reference_profile.get("numeric", {}).items():
		if column not in current_features.columns:
			continue
		current_mean = float(pd.to_numeric(current_features[column], errors="coerce").mean())
		reference_mean = float(stats.get("mean", 0.0))
		reference_std = float(stats.get("std", 0.0))
		shift = abs(current_mean - reference_mean)
		shift_ratio = shift / reference_std if reference_std > 0 else shift
		numeric_drift[column] = {
			"reference_mean": reference_mean,
			"current_mean": current_mean,
			"shift_ratio": round(float(shift_ratio), 4),
			"drift_detected": bool(shift_ratio > numeric_threshold),
		}

	categorical_drift: dict[str, Any] = {}
	for column, reference_distribution in reference_profile.get("categorical", {}).items():
		if column not in current_features.columns:
			continue
		current_distribution = (
			current_features[column].astype(str).value_counts(normalize=True).to_dict()
		)
		all_categories = set(reference_distribution) | set(current_distribution)
		max_delta = max(
			abs(float(reference_distribution.get(category, 0.0)) - float(current_distribution.get(category, 0.0)))
			for category in all_categories
		) if all_categories else 0.0
		categorical_drift[column] = {
			"max_share_delta": round(float(max_delta), 4),
			"drift_detected": bool(max_delta > categorical_threshold),
		}

	overall_drift = any(item["drift_detected"] for item in numeric_drift.values()) or any(
		item["drift_detected"] for item in categorical_drift.values()
	)
	return {
		"status": "drift_detected" if overall_drift else "stable",
		"current_rows": int(len(current_features)),
		"numeric_drift": numeric_drift,
		"categorical_drift": categorical_drift,
	}


def generate_drift_report(
	baseline_path: Path = TRAINING_BASELINE_PATH,
	log_path: Path = PREDICTION_LOG_PATH,
) -> dict[str, Any]:
	"""Load artifacts and return a drift report for recent inference traffic."""
	reference_profile = load_reference_profile(baseline_path)
	current_features = load_logged_prediction_inputs(log_path)
	report = summarize_drift(reference_profile, current_features)
	LOGGER.info("Generated drift report with status=%s", report.get("status"))
	return report