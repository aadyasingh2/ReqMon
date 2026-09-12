"""Metric Library defining supported metrics and task domain mappings for ML monitoring."""

from typing import Dict, List

# Master list of all supported evaluation metrics
SUPPORTED_METRICS: List[str] = [
    "recall",
    "precision",
    "f1",
    "fnr",
    "fpr",
    "accuracy",
    "auc",
]

# Task domain specific metric mappings
BINARY_CLASSIFICATION_METRICS: List[str] = [
    "recall",
    "precision",
    "f1",
    "fnr",
    "fpr",
    "accuracy",
    "auc",
]

TASK_METRIC_MAP: Dict[str, List[str]] = {
    "binary_classification": BINARY_CLASSIFICATION_METRICS,
}
