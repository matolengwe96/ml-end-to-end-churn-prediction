"""Optional MLflow experiment tracking helpers.

The training workflow uses this module to log model comparison runs,
artifacts, and the final selected pipeline. The integration is designed to
degrade gracefully when MLflow is not installed or tracking is disabled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import MLFLOW_ENABLED, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from src.utils import get_logger

LOGGER = get_logger(__name__)

try:
	import mlflow
	import mlflow.sklearn

	MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional dependency
	mlflow = None
	MLFLOW_AVAILABLE = False


def log_training_run(
	best_pipeline: Any,
	best_model_name: str,
	metrics_by_model: dict[str, dict[str, float | None]],
	metadata: dict[str, Any],
	artifact_paths: dict[str, str],
) -> dict[str, Any]:
	"""Log a training run to MLflow when enabled and available."""
	if not MLFLOW_ENABLED:
		return {"enabled": False, "reason": "disabled_by_config"}

	if not MLFLOW_AVAILABLE:
		LOGGER.warning("MLflow tracking requested but mlflow is not installed.")
		return {"enabled": False, "reason": "mlflow_not_installed"}

	# MLflow requires a URI scheme.  Convert bare/relative paths to file:// URIs
	# so it works identically on Windows, macOS, and Linux.
	raw_uri = MLFLOW_TRACKING_URI
	if raw_uri and not raw_uri.startswith(("http://", "https://", "file://", "databricks")):
		tracking_uri = Path(raw_uri).resolve().as_uri()
	else:
		tracking_uri = raw_uri

	mlflow.set_tracking_uri(tracking_uri)
	mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

	with mlflow.start_run(run_name=f"train-{best_model_name}") as run:
		mlflow.log_params(
			{
				"best_model": best_model_name,
				"random_state": metadata.get("random_state"),
				"test_size": metadata.get("test_size"),
				"dataset_rows": metadata.get("dataset_rows"),
				"feature_count": metadata.get("feature_count"),
			}
		)

		for model_name, metric_values in metrics_by_model.items():
			for metric_name, metric_value in metric_values.items():
				if metric_value is None:
					continue
				mlflow.log_metric(
					f"{model_name}.{metric_name}",
					float(metric_value),
				)

		for _, artifact_path in artifact_paths.items():
			path = Path(artifact_path)
			if path.exists():
				mlflow.log_artifact(str(path), artifact_path="project_artifacts")

		mlflow.sklearn.log_model(best_pipeline, artifact_path="best_model")
		run_id = run.info.run_id

	LOGGER.info("Logged training run to MLflow: %s", run_id)
	return {
		"enabled": True,
		"tracking_uri": tracking_uri,
		"experiment_name": MLFLOW_EXPERIMENT_NAME,
		"run_id": run_id,
	}