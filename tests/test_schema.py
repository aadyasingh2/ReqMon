"""Unit tests for InterpretedRequirement schema validation."""

import pytest
from pydantic import ValidationError
from ramma_nlp.schema import InterpretedRequirement


def test_valid_interpreted_requirement():
    """Test creating a valid InterpretedRequirement object."""
    req = InterpretedRequirement(
        metric="recall",
        operator=">=",
        threshold=0.93,
        severity="high",
        is_ambiguous=False,
        ambiguity_reason=None,
        raw_requirement="Recall must remain above 93%",
    )
    assert req.metric == "recall"
    assert req.operator == ">="
    assert req.threshold == 0.93
    assert req.severity == "high"
    assert req.is_ambiguous is False
    assert req.ambiguity_reason is None
    assert req.raw_requirement == "Recall must remain above 93%"


def test_invalid_metric_rejection():
    """Test that an invalid metric raises a ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        InterpretedRequirement(
            metric="custom_loss",
            operator=">=",
            threshold=0.90,
            severity="medium",
            is_ambiguous=False,
            raw_requirement="Loss must be above 90%",
        )
    assert "Invalid metric" in str(exc_info.value)


def test_invalid_threshold_rejection():
    """Test that thresholds outside 0-1 range raise a ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        InterpretedRequirement(
            metric="precision",
            operator=">=",
            threshold=95.0,  # invalid (> 1.0)
            severity="high",
            is_ambiguous=False,
            raw_requirement="Precision must be 95%",
        )
    assert "Threshold 95.0 must be between 0.0 and 1.0" in str(exc_info.value)

    with pytest.raises(ValidationError):
        InterpretedRequirement(
            metric="precision",
            operator=">=",
            threshold=-0.1,  # invalid (< 0.0)
            severity="high",
            is_ambiguous=False,
            raw_requirement="Precision must be positive",
        )


def test_ambiguous_requirement_valid():
    """Test an ambiguous requirement representation."""
    req = InterpretedRequirement(
        metric="accuracy",
        operator=">=",
        threshold=0.0,
        severity="low",
        is_ambiguous=True,
        ambiguity_reason="The requirement is subjective and does not state a target metric or threshold.",
        raw_requirement="The model should behave responsibly",
    )
    assert req.is_ambiguous is True
    assert req.ambiguity_reason is not None
    assert "subjective" in req.ambiguity_reason


def test_invalid_operator_rejection():
    """Test that an invalid operator raises a ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        InterpretedRequirement(
            metric="f1",
            operator="around",
            threshold=0.8,
            severity="medium",
            is_ambiguous=False,
            raw_requirement="F1 around 80%",
        )
    assert "Invalid operator" in str(exc_info.value)
