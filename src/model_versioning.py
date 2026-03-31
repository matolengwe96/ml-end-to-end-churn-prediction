"""Model versioning registry.

Every time ``save_versioned_model`` is called (at the end of training),
the fitted pipeline is written to a timestamped file under ``models/versions/``
and a lightweight JSON registry is updated at ``models/version_registry.json``.

The ``get_active_version`` helper reads ``models/best_model.joblib`` to
determine the currently-deployed artifact, and the registry provides a
full audit trail of past training runs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import MODEL_DIR, MODEL_PATH
from src.utils import ensure_directory, get_logger, load_json, save_json, save_model

LOGGER = get_logger(__name__)

VERSIONS_DIR: Path = MODEL_DIR / "versions"
REGISTRY_PATH: Path = MODEL_DIR / "version_registry.json"


def _load_registry() -> dict[str, Any]:
    if REGISTRY_PATH.exists():
        data = load_json(REGISTRY_PATH)
        if isinstance(data, dict):
            return data
    return {"versions": [], "active_version": None}


def save_versioned_model(
    pipeline: Any,
    model_name: str,
    metrics: dict[str, float | None],
) -> str:
    """Save a versioned copy of the pipeline and update the registry.

    Parameters
    ----------
    pipeline:
        The fitted sklearn Pipeline to persist.
    model_name:
        Human-readable name of the best model class (e.g. ``LogisticRegression``).
    metrics:
        Hold-out metrics dict for the best model (used for registry metadata).

    Returns
    -------
    version_id:
        Timestamp-based ID, e.g. ``20260328_175900``.
    """
    ensure_directory(VERSIONS_DIR)
    version_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    versioned_path = VERSIONS_DIR / f"{version_id}_{model_name}.joblib"

    save_model(pipeline, versioned_path)

    registry = _load_registry()
    entry: dict[str, Any] = {
        "version_id": version_id,
        "model_name": model_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "f1_score": metrics.get("f1"),
        "roc_auc": metrics.get("roc_auc"),
        "artifact_path": str(versioned_path),
    }
    registry["versions"].append(entry)
    registry["active_version"] = version_id

    save_json(registry, REGISTRY_PATH)
    LOGGER.info("Saved model version %s → %s", version_id, versioned_path)
    return version_id


def list_versions() -> dict[str, Any]:
    """Return the full contents of the version registry."""
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Version registry not found at {REGISTRY_PATH}. "
            "Train the model first: python -m src.train"
        )
    return _load_registry()


def load_version(version_id: str) -> Any:
    """Load a specific versioned model artifact by version ID.

    Raises
    ------
    FileNotFoundError
        If no entry with the given ``version_id`` exists.
    """
    import joblib

    registry = _load_registry()
    for entry in registry.get("versions", []):
        if entry["version_id"] == version_id:
            path = Path(entry["artifact_path"])
            if not path.exists():
                raise FileNotFoundError(
                    f"Versioned model file missing: {path}"
                )
            return joblib.load(path)
    raise FileNotFoundError(
        f"Version '{version_id}' not found in registry. "
        f"Available: {[v['version_id'] for v in registry.get('versions', [])]}"
    )


def promote_version(version_id: str) -> None:
    """Copy a versioned model to ``models/best_model.joblib`` and mark it active.

    This allows manual promotion of a previous version without re-training.
    """
    import shutil

    registry = _load_registry()
    for entry in registry.get("versions", []):
        if entry["version_id"] == version_id:
            src = Path(entry["artifact_path"])
            if not src.exists():
                raise FileNotFoundError(f"Artifact not found: {src}")
            shutil.copy2(src, MODEL_PATH)
            registry["active_version"] = version_id
            save_json(registry, REGISTRY_PATH)
            LOGGER.info("Promoted version %s to active model.", version_id)
            return
    raise FileNotFoundError(f"Version '{version_id}' not found in registry.")
