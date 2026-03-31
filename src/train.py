"""Training entry point for telecom churn prediction models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    FEATURE_COLUMNS_PATH,
    METRICS_PATH,
    MLFLOW_ENABLED,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    TRAINING_BASELINE_PATH,
)
from src.data_preprocessing import (
    build_preprocessor,
    clean_data,
    load_raw_data,
    split_features_target,
)
from src.drift_monitoring import build_reference_profile, save_reference_profile
from src.evaluate import compare_models, evaluate_trained_model, format_training_summary
from src.experiment_tracking import log_training_run
from src.feature_engineering import extract_transformed_feature_names
from src.model_versioning import save_versioned_model
from src.utils import get_logger, print_section, save_json, save_model, setup_logging

LOGGER = get_logger(__name__)


def build_model_candidates() -> dict[str, object]:
    """Define candidate models for churn classification."""
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def _tune_estimator(estimator: object, X_train: pd.DataFrame, y_train: pd.Series) -> object:
    """Wrap estimator in RandomizedSearchCV and return best fitted estimator."""
    param_grids: dict[str, dict[str, list]] = {
        "LogisticRegression": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "solver": ["lbfgs", "saga"],
            "penalty": ["l2"],
        },
        "RandomForestClassifier": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        },
        "GradientBoostingClassifier": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.7, 0.8, 1.0],
        },
    }
    name = type(estimator).__name__
    grid = param_grids.get(name, {})
    if not grid:
        return estimator
    rscv = RandomizedSearchCV(
        estimator,
        param_distributions=grid,
        n_iter=20,
        scoring="f1",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    rscv.fit(X_train, y_train)
    LOGGER.info(
        "Tuned %s — best params: %s  best CV-F1: %.4f",
        name,
        rscv.best_params_,
        rscv.best_score_,
    )
    return rscv.best_estimator_


def train_and_compare_models(X: pd.DataFrame, y: pd.Series, tune: bool = False):
    """Train model pipelines and compare performance on held-out test data."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor, _, _ = build_preprocessor(X_train)
    candidates = build_model_candidates()

    trained_models: dict[str, Pipeline] = {}
    metrics_by_model: dict[str, dict[str, float | None]] = {}

    for model_name, estimator in candidates.items():
        if tune:
            LOGGER.info("Tuning %s ...", model_name)
            estimator = _tune_estimator(estimator, X_train, y_train)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        metrics_by_model[model_name] = evaluate_trained_model(pipeline, X_test, y_test)

        # 5-fold CV on the training partition for stability estimate.
        cv_pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", clone(estimator)),
            ]
        )
        cv_scores = cross_val_score(cv_pipeline, X_train, y_train, cv=5, scoring="f1")
        metrics_by_model[model_name]["cv_f1_mean"] = round(float(cv_scores.mean()), 4)
        metrics_by_model[model_name]["cv_f1_std"] = round(float(cv_scores.std()), 4)

        trained_models[model_name] = pipeline

    best_model_name, _ = compare_models(metrics_by_model)
    best_pipeline = trained_models[best_model_name]

    return best_model_name, best_pipeline, metrics_by_model, X.columns.tolist(), X_train.shape[0], X_test.shape[0]


def run_training(data_path: str | None = None, tune: bool = False) -> dict[str, object]:
    """Main training workflow used by CLI and external callers."""
    if data_path:
        raw_df = load_raw_data(data_path=Path(data_path))
    else:
        raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)
    X, y = split_features_target(cleaned_df)

    (
        best_model_name,
        best_pipeline,
        metrics_by_model,
        feature_names,
        train_rows,
        test_rows,
    ) = train_and_compare_models(X, y, tune=tune)

    save_model(best_pipeline, MODEL_PATH)
    save_json(metrics_by_model, METRICS_PATH)

    transformed_feature_names = extract_transformed_feature_names(best_pipeline, feature_names)
    save_json({"columns": transformed_feature_names}, FEATURE_COLUMNS_PATH)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": int(len(cleaned_df)),
        "train_rows": int(train_rows),
        "test_rows": int(test_rows),
        "feature_count": int(len(feature_names)),
        "feature_names": feature_names,
        "best_model": best_model_name,
        "selection_metric": "f1",
        "secondary_metric": "roc_auc",
        "random_state": RANDOM_STATE,
        "training_baseline_path": str(TRAINING_BASELINE_PATH),
        "test_size": TEST_SIZE,
    }
    # Save versioned copy alongside the canonical best_model.joblib.
    version_id = save_versioned_model(
        best_pipeline,
        best_model_name,
        metrics_by_model[best_model_name],
    )
    metadata["version_id"] = version_id

    reference_profile = build_reference_profile(X)
    save_reference_profile(reference_profile, TRAINING_BASELINE_PATH)

    artifact_paths = {
        "model_path": str(MODEL_PATH),
        "metrics_path": str(METRICS_PATH),
        "feature_columns_path": str(FEATURE_COLUMNS_PATH),
        "training_baseline_path": str(TRAINING_BASELINE_PATH),
    }

    mlflow_status = log_training_run(
        best_pipeline=best_pipeline,
        best_model_name=best_model_name,
        metrics_by_model=metrics_by_model,
        metadata=metadata,
        artifact_paths=artifact_paths,
    )
    metadata["mlflow"] = mlflow_status
    save_json(metadata, MODEL_METADATA_PATH)

    return {
        "best_model_name": best_model_name,
        "metrics": metrics_by_model,
        "training_baseline_path": str(TRAINING_BASELINE_PATH),
        "model_path": str(MODEL_PATH),
        "metrics_path": str(METRICS_PATH),
        "feature_columns_path": str(FEATURE_COLUMNS_PATH),
        "metadata_path": str(MODEL_METADATA_PATH),
        "mlflow_enabled": MLFLOW_ENABLED,
        "version_id": version_id,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training script."""
    parser = argparse.ArgumentParser(description="Train churn prediction models.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional override for input CSV path.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help="Enable RandomizedSearchCV hyperparameter tuning (slower, better results).",
    )
    return parser.parse_args()


def main() -> None:
    """Run training and print a concise summary."""
    setup_logging()
    args = parse_args()
    LOGGER.info("Starting training run")
    result = run_training(data_path=args.data_path, tune=args.tune)
    print_section("Training Complete")
    print(f"Best model: {result['best_model_name']}")
    print(f"Saved model: {result['model_path']}")
    print(f"Saved metrics: {result['metrics_path']}")
    print(f"Saved feature columns: {result['feature_columns_path']}")
    print(f"Saved metadata: {result['metadata_path']}")
    print(f"Saved training baseline: {result['training_baseline_path']}")
    print()
    print(format_training_summary(result["metrics"]))


if __name__ == "__main__":
    main()
