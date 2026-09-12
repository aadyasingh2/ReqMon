"""Integration tests for monitoring runtime engine (monitoring_runtime.py)."""

import os
import pandas as pd
import pytest
from data.prepare_dataset import prepare_datasets_and_train_baseline
from ramma_backend.monitoring_runtime import run_monitor


@pytest.fixture(scope="module", autouse=True)
def ensure_datasets_and_model():
    """Ensure baseline dataset files and model artifact are prepared before tests run."""
    model_path = "models/baseline_classifier.pkl"
    clean_path = "data/production_clean.csv"
    drifted_path = "data/production_drifted.csv"

    if not (os.path.exists(model_path) and os.path.exists(clean_path) and os.path.exists(drifted_path)):
        prepare_datasets_and_train_baseline()


def test_run_monitor_clean_dataset_no_violation():
    """Test running monitor on clean production dataset (should NOT violate requirement >= 0.60)."""
    model_path = "models/baseline_classifier.pkl"
    clean_df = pd.read_csv("data/production_clean.csv")

    config = {
        "metric": "recall",
        "operator": ">=",
        "business_requirement_threshold": 0.60,
        "operational_baseline_threshold": 0.60,
    }

    result = run_monitor(config, clean_df, model_path)

    assert isinstance(result, dict)
    assert result["config"] == config
    assert result["observed_value"] >= 0.60
    assert result["business_requirement_violation"] is False
    assert result["operational_drift_violation"] is False
    assert result["is_violation"] is False


def test_run_monitor_drifted_dataset_should_violate():
    """Test running monitor on drifted production dataset (SHOULD violate requirement >= 0.85)."""
    model_path = "models/baseline_classifier.pkl"
    drifted_df = pd.read_csv("data/production_drifted.csv")

    config = {
        "metric": "recall",
        "operator": ">=",
        "business_requirement_threshold": 0.85,
        "operational_baseline_threshold": 0.85,
    }

    result = run_monitor(config, drifted_df, model_path)

    assert isinstance(result, dict)
    assert result["config"] == config
    assert result["observed_value"] < 0.85
    assert result["business_requirement_violation"] is True
    assert result["operational_drift_violation"] is True
    assert result["is_violation"] is True


def test_run_monitor_missing_model_file_raises_error():
    """Test that specifying a non-existent model path raises a FileNotFoundError."""
    clean_df = pd.read_csv("data/production_clean.csv")
    config = {"metric": "recall", "operator": ">=", "business_requirement_threshold": 0.60, "operational_baseline_threshold": 0.60}

    with pytest.raises(FileNotFoundError) as exc_info:
        run_monitor(config, clean_df, "models/non_existent_model.pkl")

    assert "Model file not found" in str(exc_info.value)
