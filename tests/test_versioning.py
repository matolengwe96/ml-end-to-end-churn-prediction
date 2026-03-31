"""Tests for src.model_versioning — versioned model save/load/promote."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Helpers: patch the versioning module to use a temp directory in every test.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _use_temp_versions_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect version storage to a temporary directory for isolation."""
    import src.model_versioning as mv  # noqa: PLC0415

    monkeypatch.setattr(mv, "VERSIONS_DIR", tmp_path / "versions")
    monkeypatch.setattr(mv, "REGISTRY_PATH", tmp_path / "version_registry.json")
    monkeypatch.setattr(mv, "MODEL_PATH", tmp_path / "best_model.joblib")
    (tmp_path / "versions").mkdir(parents=True, exist_ok=True)


@pytest.fixture()
def minimal_pipeline(sample_features, sample_target) -> Pipeline:
    """Return a tiny fitted LR pipeline."""
    from src.data_preprocessing import build_preprocessor  # noqa: PLC0415

    preprocessor, _, _ = build_preprocessor(sample_features)
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=100))]
    )
    pipeline.fit(sample_features, sample_target)
    return pipeline


@pytest.fixture()
def sample_metrics() -> dict:
    return {"f1": 0.72, "accuracy": 0.80, "roc_auc": 0.85}


class TestSaveVersionedModel:
    def test_returns_version_id(self, minimal_pipeline, sample_metrics):
        import src.model_versioning as mv  # noqa: PLC0415

        version_id = mv.save_versioned_model(minimal_pipeline, "LR", sample_metrics)
        assert isinstance(version_id, str)
        assert len(version_id) > 0

    def test_creates_joblib_file(self, minimal_pipeline, sample_metrics):
        import src.model_versioning as mv  # noqa: PLC0415

        mv.save_versioned_model(minimal_pipeline, "LR", sample_metrics)
        versions_dir = mv.VERSIONS_DIR
        saved_files = list(versions_dir.glob("*.joblib"))
        assert len(saved_files) == 1

    def test_creates_registry_file(self, minimal_pipeline, sample_metrics):
        import src.model_versioning as mv  # noqa: PLC0415

        mv.save_versioned_model(minimal_pipeline, "LR", sample_metrics)
        assert mv.REGISTRY_PATH.exists()

    def test_registry_contains_version(self, minimal_pipeline, sample_metrics):
        import src.model_versioning as mv  # noqa: PLC0415

        version_id = mv.save_versioned_model(minimal_pipeline, "LR", sample_metrics)
        registry = json.loads(mv.REGISTRY_PATH.read_text())
        version_ids = [v["version_id"] for v in registry.get("versions", [])]
        assert version_id in version_ids

    def test_multiple_saves_accumulate(self, minimal_pipeline, sample_metrics):
        import src.model_versioning as mv  # noqa: PLC0415

        mv.save_versioned_model(minimal_pipeline, "LR", sample_metrics)
        mv.save_versioned_model(minimal_pipeline, "RF", {**sample_metrics, "f1": 0.75})
        registry = json.loads(mv.REGISTRY_PATH.read_text())
        assert len(registry["versions"]) == 2


class TestListVersions:
    def test_returns_dict_with_versions_key(self, minimal_pipeline, sample_metrics):
        import src.model_versioning as mv  # noqa: PLC0415

        mv.save_versioned_model(minimal_pipeline, "LR", sample_metrics)
        result = mv.list_versions()
        assert "versions" in result

    def test_raises_when_no_registry(self):
        import src.model_versioning as mv  # noqa: PLC0415

        with pytest.raises(FileNotFoundError):
            mv.list_versions()


class TestLoadVersion:
    def test_loads_saved_pipeline(self, minimal_pipeline, sample_metrics, sample_features):
        import src.model_versioning as mv  # noqa: PLC0415

        version_id = mv.save_versioned_model(minimal_pipeline, "LR", sample_metrics)
        loaded = mv.load_version(version_id)
        # The loaded pipeline should produce predictions without error.
        preds = loaded.predict(sample_features)
        assert len(preds) == len(sample_features)

    def test_raises_for_missing_version(self):
        import src.model_versioning as mv  # noqa: PLC0415

        with pytest.raises(FileNotFoundError):
            mv.load_version("nonexistent_version_id")


class TestPromoteVersion:
    def test_promotes_to_best_model(self, minimal_pipeline, sample_metrics):
        import src.model_versioning as mv  # noqa: PLC0415

        version_id = mv.save_versioned_model(minimal_pipeline, "LR", sample_metrics)
        mv.promote_version(version_id)
        assert mv.MODEL_PATH.exists()

    def test_raises_for_missing_version(self):
        import src.model_versioning as mv  # noqa: PLC0415

        with pytest.raises(FileNotFoundError):
            mv.promote_version("nonexistent")
