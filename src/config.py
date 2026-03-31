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
	logs_dir: Path
	model_dir: Path
	model_path: Path
	metrics_path: Path
	feature_columns_path: Path
	model_metadata_path: Path
	training_baseline_path: Path
	prediction_log_path: Path
	prediction_log_max_bytes: int
	prediction_log_backup_count: int
	reports_dir: Path
	figures_dir: Path
	random_state: int
	test_size: float
	mlflow_tracking_uri: str
	mlflow_experiment_name: str
	mlflow_enabled: bool
	mlflow_log_model: bool
	api_key: str
	api_rate_limit: int
	api_rate_limit_window_seconds: int


def _build_settings() -> Settings:
	project_root = Path(__file__).resolve().parents[1]

	data_dir = project_root / "data"
	logs_dir = project_root / "logs"
	model_dir = project_root / "models"
	reports_dir = project_root / "reports"

	random_state = int(os.getenv("CHURN_RANDOM_STATE", "42"))
	test_size = float(os.getenv("CHURN_TEST_SIZE", "0.2"))
	mlflow_enabled = os.getenv("CHURN_ENABLE_MLFLOW", "1").strip().lower() in {"1", "true", "yes"}
	mlflow_log_model = os.getenv("CHURN_MLFLOW_LOG_MODEL", "0").strip().lower() in {"1", "true", "yes"}
	prediction_log_max_bytes = int(os.getenv("CHURN_PREDICTION_LOG_MAX_BYTES", "1048576"))
	prediction_log_backup_count = int(os.getenv("CHURN_PREDICTION_LOG_BACKUP_COUNT", "5"))
	api_rate_limit = int(os.getenv("CHURN_API_RATE_LIMIT", "60"))
	api_rate_limit_window_seconds = int(os.getenv("CHURN_API_RATE_LIMIT_WINDOW_SECONDS", "60"))

	if not 0.0 < test_size < 1.0:
		raise ValueError("CHURN_TEST_SIZE must be between 0 and 1 (exclusive).")

	return Settings(
		project_root=project_root,
		data_dir=data_dir,
		raw_data_path=Path(os.getenv("CHURN_RAW_DATA_PATH", str(data_dir / "raw" / "churn.csv"))),
		processed_data_dir=data_dir / "processed",
		logs_dir=logs_dir,
		model_dir=model_dir,
		model_path=Path(os.getenv("CHURN_MODEL_PATH", str(model_dir / "best_model.joblib"))),
		metrics_path=Path(os.getenv("CHURN_METRICS_PATH", str(model_dir / "model_metrics.json"))),
		feature_columns_path=Path(
			os.getenv("CHURN_FEATURE_COLUMNS_PATH", str(model_dir / "feature_columns.json"))
		),
		model_metadata_path=Path(
			os.getenv("CHURN_MODEL_METADATA_PATH", str(model_dir / "model_metadata.json"))
		),
		training_baseline_path=Path(
			os.getenv("CHURN_TRAINING_BASELINE_PATH", str(model_dir / "training_baseline.json"))
		),
		prediction_log_path=Path(
			os.getenv("CHURN_PREDICTION_LOG_PATH", str(logs_dir / "predictions.jsonl"))
		),
		prediction_log_max_bytes=prediction_log_max_bytes,
		prediction_log_backup_count=prediction_log_backup_count,
		reports_dir=reports_dir,
		figures_dir=reports_dir / "figures",
		random_state=random_state,
		test_size=test_size,
		mlflow_tracking_uri=os.getenv(
			"CHURN_MLFLOW_TRACKING_URI", str(project_root / "mlruns")
		),
		mlflow_experiment_name=os.getenv("CHURN_MLFLOW_EXPERIMENT_NAME", "telecom-churn"),
		mlflow_enabled=mlflow_enabled,
		mlflow_log_model=mlflow_log_model,
		api_key=os.getenv("CHURN_API_KEY", "").strip(),
		api_rate_limit=api_rate_limit,
		api_rate_limit_window_seconds=api_rate_limit_window_seconds,
	)


SETTINGS = _build_settings()

# Backward-compatible constants used by existing modules.
PROJECT_ROOT = SETTINGS.project_root
DATA_DIR = SETTINGS.data_dir
RAW_DATA_PATH = SETTINGS.raw_data_path
PROCESSED_DATA_DIR = SETTINGS.processed_data_dir
LOGS_DIR = SETTINGS.logs_dir
MODEL_DIR = SETTINGS.model_dir
MODEL_PATH = SETTINGS.model_path
METRICS_PATH = SETTINGS.metrics_path
FEATURE_COLUMNS_PATH = SETTINGS.feature_columns_path
MODEL_METADATA_PATH = SETTINGS.model_metadata_path
TRAINING_BASELINE_PATH = SETTINGS.training_baseline_path
PREDICTION_LOG_PATH = SETTINGS.prediction_log_path
PREDICTION_LOG_MAX_BYTES = SETTINGS.prediction_log_max_bytes
PREDICTION_LOG_BACKUP_COUNT = SETTINGS.prediction_log_backup_count
REPORTS_DIR = SETTINGS.reports_dir
FIGURES_DIR = SETTINGS.figures_dir
RANDOM_STATE = SETTINGS.random_state
TEST_SIZE = SETTINGS.test_size
MLFLOW_TRACKING_URI = SETTINGS.mlflow_tracking_uri
MLFLOW_EXPERIMENT_NAME = SETTINGS.mlflow_experiment_name
MLFLOW_ENABLED = SETTINGS.mlflow_enabled
MLFLOW_LOG_MODEL = SETTINGS.mlflow_log_model
API_KEY = SETTINGS.api_key
API_RATE_LIMIT = SETTINGS.api_rate_limit
API_RATE_LIMIT_WINDOW_SECONDS = SETTINGS.api_rate_limit_window_seconds
