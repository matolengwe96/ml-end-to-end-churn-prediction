"""Centralized project configuration and runtime settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
	"""Runtime settings with environment variable overrides."""

	project_root: Path
	data_dir: Path
	raw_data_path: Path
	processed_data_dir: Path
	model_dir: Path
	model_path: Path
	metrics_path: Path
	feature_columns_path: Path
	model_metadata_path: Path
	reports_dir: Path
	figures_dir: Path
	random_state: int
	test_size: float


def _build_settings() -> Settings:
	project_root = Path(__file__).resolve().parents[1]

	data_dir = project_root / "data"
	model_dir = project_root / "models"
	reports_dir = project_root / "reports"

	random_state = int(os.getenv("CHURN_RANDOM_STATE", "42"))
	test_size = float(os.getenv("CHURN_TEST_SIZE", "0.2"))

	if not 0.0 < test_size < 1.0:
		raise ValueError("CHURN_TEST_SIZE must be between 0 and 1 (exclusive).")

	return Settings(
		project_root=project_root,
		data_dir=data_dir,
		raw_data_path=Path(os.getenv("CHURN_RAW_DATA_PATH", str(data_dir / "raw" / "churn.csv"))),
		processed_data_dir=data_dir / "processed",
		model_dir=model_dir,
		model_path=Path(os.getenv("CHURN_MODEL_PATH", str(model_dir / "best_model.joblib"))),
		metrics_path=Path(os.getenv("CHURN_METRICS_PATH", str(model_dir / "model_metrics.json"))),
		feature_columns_path=Path(
			os.getenv("CHURN_FEATURE_COLUMNS_PATH", str(model_dir / "feature_columns.json"))
		),
		model_metadata_path=Path(
			os.getenv("CHURN_MODEL_METADATA_PATH", str(model_dir / "model_metadata.json"))
		),
		reports_dir=reports_dir,
		figures_dir=reports_dir / "figures",
		random_state=random_state,
		test_size=test_size,
	)


SETTINGS = _build_settings()

# Backward-compatible constants used by existing modules.
PROJECT_ROOT = SETTINGS.project_root
DATA_DIR = SETTINGS.data_dir
RAW_DATA_PATH = SETTINGS.raw_data_path
PROCESSED_DATA_DIR = SETTINGS.processed_data_dir
MODEL_DIR = SETTINGS.model_dir
MODEL_PATH = SETTINGS.model_path
METRICS_PATH = SETTINGS.metrics_path
FEATURE_COLUMNS_PATH = SETTINGS.feature_columns_path
MODEL_METADATA_PATH = SETTINGS.model_metadata_path
REPORTS_DIR = SETTINGS.reports_dir
FIGURES_DIR = SETTINGS.figures_dir
RANDOM_STATE = SETTINGS.random_state
TEST_SIZE = SETTINGS.test_size
