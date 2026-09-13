"""Tests for multi-requirement batch evaluation in CLI and monitoring runtime."""

import pandas as pd
import pytest
from cli import compute_reference_metric_values, run_demo
from ramma_backend.model_adapter import LocalPickleModelAdapter
from ramma_backend.monitoring_runtime import generate_monitor_config
from ramma_backend.threshold_engine import calibrate_threshold
from ramma_nlp.interpreter import interpret_requirement


def test_multi_requirement_batch_execution(capsys):
    reqs = [
        "Recall must remain above 93%",
        "Precision must remain above 80%",
        "FNR must stay below 15%",
    ]

    run_demo(requirements=reqs, model_source="local")
    captured = capsys.readouterr().out

    # Verify all 3 requirements are processed independently
    assert "EVALUATING REQUIREMENT 1/3: \"Recall must remain above 93%\"" in captured
    assert "EVALUATING REQUIREMENT 2/3: \"Precision must remain above 80%\"" in captured
    assert "EVALUATING REQUIREMENT 3/3: \"FNR must stay below 15%\"" in captured

    assert "RECALL" in captured
    assert "PRECISION" in captured
    assert "FNR" in captured

    assert "END OF DEMO EXECUTION" in captured


def test_distinct_per_metric_threshold_calibration():
    """Regression test ensuring recall, precision, and fnr each receive distinct, metric-appropriate thresholds."""
    ref_df = pd.read_csv("data/reference.csv")
    adapter = LocalPickleModelAdapter("models/baseline_classifier.pkl")

    req_recall = interpret_requirement("Recall must remain above 93%")
    req_precision = interpret_requirement("Precision must remain above 80%")
    req_fnr = interpret_requirement("FNR must stay below 15%")

    vals_recall = compute_reference_metric_values(ref_df, req_recall.metric, adapter)
    calib_recall = calibrate_threshold(vals_recall, method="percentile", operator=req_recall.operator)
    config_recall = generate_monitor_config(req_recall, calibrated_threshold=calib_recall["calibrated_threshold"])

    vals_precision = compute_reference_metric_values(ref_df, req_precision.metric, adapter)
    calib_precision = calibrate_threshold(vals_precision, method="percentile", operator=req_precision.operator)
    config_precision = generate_monitor_config(req_precision, calibrated_threshold=calib_precision["calibrated_threshold"])

    vals_fnr = compute_reference_metric_values(ref_df, req_fnr.metric, adapter)
    calib_fnr = calibrate_threshold(vals_fnr, method="percentile", operator=req_fnr.operator)
    config_fnr = generate_monitor_config(req_fnr, calibrated_threshold=calib_fnr["calibrated_threshold"])

    t_recall = config_recall["operational_baseline_threshold"]
    t_precision = config_precision["operational_baseline_threshold"]
    t_fnr = config_fnr["operational_baseline_threshold"]

    # Assert all three thresholds are distinct and not equal to each other
    assert t_recall != t_precision, f"Recall threshold {t_recall} should not equal Precision threshold {t_precision}"
    assert t_recall != t_fnr, f"Recall threshold {t_recall} should not equal FNR threshold {t_fnr}"
    assert t_precision != t_fnr, f"Precision threshold {t_precision} should not equal FNR threshold {t_fnr}"

    # Sane range assertions per metric
    assert 0.85 <= t_recall <= 0.98, f"Recall threshold {t_recall} out of expected range [0.85, 0.98]"
    assert 0.65 <= t_precision <= 0.85, f"Precision threshold {t_precision} out of expected range [0.65, 0.85]"
    assert 0.00 <= t_fnr <= 0.20, f"FNR threshold {t_fnr} out of expected range [0.00, 0.20]"
