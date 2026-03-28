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

from src.config import PROJECT_ROOT

LOGGER = logging.getLogger(__name__)

LOG_PATH: Path = PROJECT_ROOT / "logs" / "predictions.jsonl"


def log_prediction(
    record: dict[str, Any],
    prediction: dict[str, Any],
    source: str = "api",
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
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    event: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "input_feature_count": len(record),
        "predicted_class": prediction.get("predicted_class"),
        "churn_probability": prediction.get("churn_probability"),
    }

    try:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")
    except OSError as exc:
        LOGGER.warning("Failed to write prediction log entry: %s", exc)
