"""Training entry point for telecom churn prediction models."""

from __future__ import annotations

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    FEATURE_COLUMNS_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data_preprocessing import (
    build_preprocessor,
    clean_data,
    load_raw_data,
    split_features_target,
)
from src.evaluate import compare_models, evaluate_trained_model, format_training_summary
from src.feature_engineering import extract_transformed_feature_names
from src.utils import print_section, save_json, save_model


def build_model_candidates() -> dict[str, object]:
    """Define candidate models for churn classification."""
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def train_and_compare_models(X, y):
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
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        metrics_by_model[model_name] = evaluate_trained_model(pipeline, X_test, y_test)
        trained_models[model_name] = pipeline

    best_model_name, _ = compare_models(metrics_by_model)
    best_pipeline = trained_models[best_model_name]

    return best_model_name, best_pipeline, metrics_by_model, X.columns.tolist()


def run_training() -> dict[str, object]:
    """Main training workflow used by CLI and external callers."""
    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)
    X, y = split_features_target(cleaned_df)

    best_model_name, best_pipeline, metrics_by_model, feature_names = train_and_compare_models(X, y)

    save_model(best_pipeline, MODEL_PATH)
    save_json(metrics_by_model, METRICS_PATH)

    transformed_feature_names = extract_transformed_feature_names(best_pipeline, feature_names)
    save_json({"columns": transformed_feature_names}, FEATURE_COLUMNS_PATH)

    return {
        "best_model_name": best_model_name,
        "metrics": metrics_by_model,
        "model_path": str(MODEL_PATH),
        "metrics_path": str(METRICS_PATH),
        "feature_columns_path": str(FEATURE_COLUMNS_PATH),
    }


def main() -> None:
    """Run training and print a concise summary."""
    result = run_training()
    print_section("Training Complete")
    print(f"Best model: {result['best_model_name']}")
    print(f"Saved model: {result['model_path']}")
    print(f"Saved metrics: {result['metrics_path']}")
    print(f"Saved feature columns: {result['feature_columns_path']}")
    print()
    print(format_training_summary(result["metrics"]))


if __name__ == "__main__":
    main()
