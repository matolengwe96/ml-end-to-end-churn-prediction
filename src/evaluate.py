"""Reusable evaluation routines for binary classifiers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Compute common binary classification metrics."""
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None,
    }

    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update(
        {
            "true_negatives": float(tn),
            "false_positives": float(fp),
            "false_negatives": float(fn),
            "true_positives": float(tp),
        }
    )

    return metrics


def compare_models(results: dict[str, dict[str, float | None]]) -> tuple[str, dict[str, float | None]]:
    """Select best model by F1, with ROC-AUC as tie-breaker."""
    if not results:
        raise ValueError("No model results provided for comparison.")

    def ranking_key(item: tuple[str, dict[str, float | None]]) -> tuple[float, float]:
        _, metric_values = item
        f1_value = float(metric_values.get("f1") or 0.0)
        auc_value = float(metric_values.get("roc_auc") or -1.0)
        return f1_value, auc_value

    best_name, best_metrics = max(results.items(), key=ranking_key)
    return best_name, best_metrics


def format_training_summary(results: dict[str, dict[str, float | None]]) -> str:
    """Generate compact plain-text summary of model metrics."""
    lines = ["Model Performance Summary", "-" * 28]
    for model_name, metrics in results.items():
        roc_auc_value = metrics.get("roc_auc")
        roc_auc_text = f"{float(roc_auc_value):.3f}" if roc_auc_value is not None else "N/A"
        lines.append(
            
                f"{model_name:<24} "
                f"Acc={metrics['accuracy']:.3f} "
                f"Prec={metrics['precision']:.3f} "
                f"Rec={metrics['recall']:.3f} "
                f"F1={metrics['f1']:.3f} "
                f"ROC_AUC={roc_auc_text}"
            
        )
    return "\n".join(lines)


def evaluate_trained_model(model: Any, X_test: Any, y_test: Any) -> dict[str, float | None]:
    """Evaluate a fitted model object on test data."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    return evaluate_model(y_true=y_test, y_pred=y_pred, y_proba=y_proba)
