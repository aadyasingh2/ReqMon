"""Unit tests for violation detection and explanation logic in explain.py."""

from unittest.mock import patch
import pytest
from ramma_nlp.explain import check_violation, explain_violation
from ramma_nlp.schema import InterpretedRequirement


@pytest.mark.parametrize(
    "operator,threshold,observed,expected_violation",
    [
        # >= operator
        (">=", 0.90, 0.95, False),
        (">=", 0.90, 0.90, False),
        (">=", 0.90, 0.85, True),
        # <= operator
        ("<=", 0.05, 0.02, False),
        ("<=", 0.05, 0.05, False),
        ("<=", 0.05, 0.08, True),
        # > operator
        (">", 0.80, 0.85, False),
        (">", 0.80, 0.80, True),
        (">", 0.80, 0.75, True),
        # < operator
        ("<", 0.10, 0.05, False),
        ("<", 0.10, 0.10, True),
        ("<", 0.10, 0.15, True),
        # == operator
        ("==", 0.90, 0.90, False),
        ("==", 0.90, 0.89, True),
        ("==", 0.90, 0.91, True),
    ],
)
def test_check_violation_operators(operator, threshold, observed, expected_violation):
    """Test deterministic Python logic for all comparison operators."""
    result = check_violation(operator, threshold, observed)
    assert result == expected_violation


@patch("ramma_nlp.explain.generate_llm_explanation")
@pytest.mark.parametrize(
    "operator,threshold,observed,expected_violation",
    [
        (">=", 0.90, 0.85, True),
        ("<=", 0.05, 0.08, True),
        (">", 0.80, 0.75, True),
        ("<", 0.10, 0.15, True),
        ("==", 0.90, 0.85, True),
    ],
)
def test_explain_violation_dict_config_each_operator(
    mock_llm, operator, threshold, observed, expected_violation
):
    """Test explain_violation with dict config for each operator type independent of LLM call."""
    mock_llm.return_value = (
        f"Requirement violated: recall was expected to be {operator} {threshold}, but observed value is {observed}."
    )

    config = {
        "metric": "recall",
        "operator": operator,
        "business_requirement_threshold": threshold,
        "operational_baseline_threshold": threshold,
        "severity": "high",
    }

    result = explain_violation(config, observed)

    assert result["is_violation"] == expected_violation
    assert result["business_requirement_violation"] == expected_violation
    assert result["operational_drift_violation"] == expected_violation
    assert result["observed_value"] == observed
    assert result["config"] == config
    assert mock_llm.called


def test_explain_violation_returns_structured_dict_with_pydantic():
    """Test that explain_violation works seamlessly with InterpretedRequirement object."""
    req = InterpretedRequirement(
        metric="recall",
        operator=">=",
        threshold=0.93,
        severity="high",
        is_ambiguous=False,
        raw_requirement="recall must remain above 93%",
    )

    # Violated case
    res_violated = explain_violation(req, 0.88)
    assert res_violated["is_violation"] is True
    assert res_violated["business_requirement_violation"] is True
    assert res_violated["observed_value"] == 0.88
    assert res_violated["requirement"] == req
    assert "Business Requirement" in res_violated["explanation"] or "recall" in res_violated["explanation"]

    # Non-violated case
    res_ok = explain_violation(req, 0.95)
    assert res_ok["is_violation"] is False
    assert res_ok["business_requirement_violation"] is False
    assert res_ok["operational_drift_violation"] is False
    assert res_ok["observed_value"] == 0.95
    assert "No violation" in res_ok["explanation"]
