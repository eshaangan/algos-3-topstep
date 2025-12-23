"""
Metrics for classification with optional sample weights.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
)


def compute_metrics(y_true, y_prob, threshold=0.5, sample_weight=None):
    """Compute classification metrics with optional sample weights."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(
            y_true, y_pred, sample_weight=sample_weight
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred, sample_weight=sample_weight
        ),
        "log_loss": log_loss(
            y_true,
            np.vstack([1 - y_prob, y_prob]).T,
            labels=[0, 1],
            sample_weight=sample_weight,
        ),
        "brier": brier_score_loss(
            y_true, y_prob, sample_weight=sample_weight
        ),
        "precision": precision_score(
            y_true, y_pred, sample_weight=sample_weight, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, sample_weight=sample_weight, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, sample_weight=sample_weight, zero_division=0
        ),
    }

    if len(np.unique(y_true)) < 2:
        metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = roc_auc_score(
            y_true, y_prob, sample_weight=sample_weight
        )

    return metrics


def compute_multiclass_metrics(y_true, y_proba, labels, sample_weight=None):
    """Compute multiclass classification metrics with optional sample weights."""
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    if y_proba.ndim != 2:
        raise ValueError("y_proba must be 2D for multiclass metrics")
    y_pred = np.argmax(y_proba, axis=1).astype(int)
    labels = list(labels)

    return {
        "accuracy": accuracy_score(
            y_true, y_pred, sample_weight=sample_weight
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred, sample_weight=sample_weight
        ),
        "log_loss": log_loss(
            y_true,
            y_proba,
            labels=labels,
            sample_weight=sample_weight,
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            sample_weight=sample_weight,
            zero_division=0,
        ),
        "f1_weighted": f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="weighted",
            sample_weight=sample_weight,
            zero_division=0,
        ),
    }
