"""Unit tests for interactive model selection menu and threshold override functionality in cli.py."""

from unittest.mock import patch
import pytest
from cli import prompt_model_selection, run_demo
from ramma_backend.model_adapter import LocalPickleModelAdapter, RemoteEndpointModelAdapter, get_model_adapter
from ramma_backend.monitoring_runtime import generate_monitor_config
from ramma_nlp.interpreter import interpret_requirement


def test_interactive_model_selection_options():
    """Confirms that selecting each of the 3 menu options resolves to the expected adapter source and type."""
    # Option 1: RandomForest (local file)
    with patch("builtins.input", return_value="1"):
        src1, exp1 = prompt_model_selection()
        adapter1 = get_model_adapter(src1)
        assert isinstance(adapter1, LocalPickleModelAdapter)
        assert "models/baseline_classifier.pkl" in src1
        assert "RandomForest" in exp1

    # Option 2: LogisticRegression (local file)
    with patch("builtins.input", return_value="2"):
        src2, exp2 = prompt_model_selection()
        adapter2 = get_model_adapter(src2)
        assert isinstance(adapter2, LocalPickleModelAdapter)
        assert "models/alt_classifier_logreg.pkl" in src2
        assert "LogisticRegression" in exp2

    # Option 3: Remote HTTP model endpoint
    with patch("builtins.input", side_effect=["3", "http://localhost:5000"]):
        src3, exp3 = prompt_model_selection()
        adapter3 = get_model_adapter(src3)
        assert isinstance(adapter3, RemoteEndpointModelAdapter)
        assert "remote:http://localhost:5000" in src3
        assert "HTTP POST to http://localhost:5000/predict" in exp3


def test_manual_threshold_override_in_config():
    """Confirms that providing a manual threshold override sets threshold_source='user_override' and updates threshold."""
    req = interpret_requirement("Recall must remain above 93%")
    custom_threshold = 0.85

    config = generate_monitor_config(
        req,
        calibrated_threshold=custom_threshold,
        baseline_performance=0.90,
        threshold_source="user_override",
    )

    assert config["operational_baseline_threshold"] == 0.85
    assert config["threshold_source"] == "user_override"


def test_interactive_demo_threshold_override_flow(capsys):
    """Integration test simulating user selecting Option 2 and typing a custom threshold override."""
    # Mock inputs:
    # 1. Model selection: Option 2 (Logistic Regression)
    # 2. Requirement input: "Recall must remain above 93%"
    # 3. Requirement input: "done"
    # 4. Threshold override: "0.85"
    inputs = ["2", "Recall must remain above 93%", "done", "0.85"]
    with patch("builtins.input", side_effect=inputs):
        run_demo(interactive_model=True)

    captured = capsys.readouterr().out

    assert "Selected: Local LogisticRegression Classifier" in captured
    assert "User override applied: operational baseline threshold set to 85.00% (threshold_source: user_override)" in captured
    assert "Threshold Source:                 user_override" in captured
