"""Prediction monitoring — appends inference events to a JSONL audit log.

Each call to :func:`log_prediction` writes one JSON line containing the
timestamp, source service, input feature count, predicted class, and
churn probability.  The file grows indefinitely and can be consumed by any
downstream log aggregator (e.g. Splunk, Datadog, or a simple pandas script).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import (
    PREDICTION_LOG_BACKUP_COUNT,
    PREDICTION_LOG_MAX_BYTES,
    PREDICTION_LOG_PATH,
)

LOGGER = logging.getLogger(__name__)

LOG_PATH: Path = PREDICTION_LOG_PATH


def _rotation_path(log_path: Path, index: int) -> Path:
    """Return a rotated log filename like ``predictions.1.jsonl``."""
    return log_path.with_name(f"{log_path.stem}.{index}{log_path.suffix}")


def _rotate_prediction_logs(log_path: Path, backup_count: int) -> None:
    """Rotate the current prediction log in-place, keeping a bounded history."""
    if not log_path.exists():
        return

    if backup_count <= 0:
        log_path.unlink(missing_ok=True)
        return

    oldest_backup = _rotation_path(log_path, backup_count)
    if oldest_backup.exists():
        oldest_backup.unlink()

    for index in range(backup_count - 1, 0, -1):
        source = _rotation_path(log_path, index)
        target = _rotation_path(log_path, index + 1)
        if source.exists():
            source.replace(target)

    log_path.replace(_rotation_path(log_path, 1))


def log_prediction(
    record: dict[str, Any],
    prediction: dict[str, Any],
    source: str = "api",
    request_id: str | None = None,
) -> None:
    """Append a single prediction event to the JSONL prediction log.

    Parameters
    ----------
    record:
        The raw input dict (before DataFrame conversion).
    prediction:
        The output dict from :func:`src.predict.predict_single`.
    source:
        Identifier for the calling service (e.g. ``"api"``, ``"app"``).
    """
    PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    event: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "request_id": request_id,
        "input_feature_count": len(record),
        "input_record": record,
        "predicted_class": prediction.get("predicted_class"),
        "predicted_label": prediction.get("predicted_label"),
        "churn_probability": prediction.get("churn_probability"),
    }

    serialized_event = json.dumps(event) + "\n"

    try:
        if (
            PREDICTION_LOG_MAX_BYTES > 0
            and PREDICTION_LOG_PATH.exists()
            and PREDICTION_LOG_PATH.stat().st_size + len(serialized_event.encode("utf-8"))
            > PREDICTION_LOG_MAX_BYTES
        ):
            _rotate_prediction_logs(PREDICTION_LOG_PATH, PREDICTION_LOG_BACKUP_COUNT)

        with PREDICTION_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(serialized_event)
    except OSError as exc:
        LOGGER.warning("Failed to write prediction log entry: %s", exc)
