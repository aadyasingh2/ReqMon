"""Monitoring Runtime module for evaluating production model performance against requirement thresholds."""

import json
import os
from typing import Any, Dict, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from ramma_nlp.explain import check_violation
from ramma_nlp.schema import InterpretedRequirement


def generate_monitor_config(
    requirement: Union[InterpretedRequirement, Dict[str, Any]],
    calibrated_threshold: float | None = None,
    baseline_performance: float | None = None,
    config_dir: str = "configs",
) -> Dict[str, Any]:
    """Generates a monitor configuration dictionary with dual thresholds (business & operational).

    Args:
        requirement: Dict or InterpretedRequirement specifying metric, operator, threshold, etc.
        calibrated_threshold: Optional empirical threshold from threshold calibration engine.
        baseline_performance: Optional empirical mean performance on reference baseline dataset.
        config_dir: Directory where YAML config files are saved.

    Returns:
        Dict specifying dual threshold monitor configuration.
    """
    if isinstance(requirement, InterpretedRequirement):
        req_dict = requirement.model_dump()
    else:
        req_dict = dict(requirement)

    biz_threshold = float(req_dict.get("business_requirement_threshold", req_dict.get("threshold", 0.50)))
    operator = str(req_dict.get("operator", ">="))

    if calibrated_threshold is not None:
        op_threshold = float(calibrated_threshold)
    else:
        op_threshold = biz_threshold

    # Evaluate whether reference baseline performance already satisfied the business requirement
    eval_perf = baseline_performance if baseline_performance is not None else op_threshold
    requirement_met_by_baseline = not check_violation(operator, biz_threshold, eval_perf)

    config = {
        "metric": str(req_dict.get("metric", "recall")),
        "operator": operator,
        "business_requirement_threshold": round(biz_threshold, 4),
        "operational_baseline_threshold": round(op_threshold, 4),
        "requirement_met_by_baseline": requirement_met_by_baseline,
        "severity": str(req_dict.get("severity", "medium")),
        "raw_requirement": str(req_dict.get("raw_requirement", "")),
    }

    # Save configuration YAML file to configs/
    os.makedirs(config_dir, exist_ok=True)
    yaml_file = os.path.join(config_dir, "monitor_config.yaml")

    yaml_content = f"""# Generated RAMMA Monitor Configuration
metric: "{config['metric']}"
operator: "{config['operator']}"
business_requirement_threshold: {config['business_requirement_threshold']}
operational_baseline_threshold: {config['operational_baseline_threshold']}
requirement_met_by_baseline: {config['requirement_met_by_baseline']}
severity: "{config['severity']}"
raw_requirement: "{config['raw_requirement']}"
"""

    with open(yaml_file, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    return config


def compute_dataset_metric(
    y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray | None, metric_name: str
) -> float:
    """Computes specified evaluation metric value on ground truth and predictions."""
    m = metric_name.strip().lower()
    if m == "recall":
        return float(recall_score(y_true, y_pred, zero_division=0))
    elif m == "precision":
        return float(precision_score(y_true, y_pred, zero_division=0))
    elif m == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    elif m == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    elif m == "auc":
        if y_proba is not None:
            return float(roc_auc_score(y_true, y_proba))
        return float(roc_auc_score(y_true, y_pred))
    elif m == "fnr":
        rec = recall_score(y_true, y_pred, zero_division=0)
        return float(1.0 - rec)
    elif m == "fpr":
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        return 0.0
    else:
        raise ValueError(f"Unsupported metric: '{metric_name}'")


def run_monitor(config: Dict[str, Any], production_df: pd.DataFrame, model_path: str) -> Dict[str, Any]:
    """Executes a monitoring check against production data evaluating dual threshold conditions."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    if production_df.empty:
        raise ValueError("Production DataFrame is empty.")

    target_col = None
    for col in ["is_fraud", "target", "label"]:
        if col in production_df.columns:
            target_col = col
            break

    if not target_col:
        raise ValueError("Production dataset must contain a target column ('is_fraud', 'target', or 'label').")

    X_prod = production_df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y_true = production_df[target_col]

    clf = joblib.load(model_path)
    y_pred = clf.predict(X_prod)

    y_proba = None
    if hasattr(clf, "predict_proba"):
        try:
            probas = clf.predict_proba(X_prod)
            if probas.shape[1] > 1:
                y_proba = probas[:, 1]
        except Exception:
            pass

    metric_name = config.get("metric", "recall")
    observed_value = compute_dataset_metric(y_true, y_pred, y_proba, metric_name)

    operator = config.get("operator", ">=")
    biz_thresh = float(config.get("business_requirement_threshold", config.get("threshold", 0.50)))
    op_thresh = float(config.get("operational_baseline_threshold", config.get("threshold", 0.50)))

    biz_violation = check_violation(operator, biz_thresh, observed_value)
    drift_violation = check_violation(operator, op_thresh, observed_value)

    return {
        "observed_value": round(observed_value, 4),
        "business_requirement_violation": biz_violation,
        "operational_drift_violation": drift_violation,
        "requirement_met_by_baseline": config.get("requirement_met_by_baseline", False),
        "is_violation": biz_violation or drift_violation,
        "config": config,
    }
