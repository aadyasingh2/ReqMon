"""Integration tests for requirement interpreter endpoint and function."""

import json
import os
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from ramma_nlp.interpreter import app, interpret_requirement
from ramma_nlp.schema import InterpretedRequirement

client = TestClient(app)


def mock_gemini_response(json_data: dict | str) -> MagicMock:
    mock_resp = MagicMock()
    if isinstance(json_data, str):
        mock_resp.text = json_data
    else:
        mock_resp.text = json.dumps(json_data)
    return mock_resp


@patch("ramma_nlp.interpreter.get_gemini_model")
def test_interpret_clear_requirement(mock_get_model):
    """Test clear requirement: 'recall must remain above 93%'."""
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_gemini_response(
        {
            "metric": "recall",
            "operator": ">=",
            "threshold": 0.93,
            "severity": "high",
            "is_ambiguous": False,
            "ambiguity_reason": None,
            "raw_requirement": "recall must remain above 93%",
        }
    )
    mock_get_model.return_value = mock_model

    text = "recall must remain above 93%"
    result = interpret_requirement(text)

    assert result.metric == "recall"
    assert result.operator == ">="
    assert result.threshold == 0.93
    assert result.is_ambiguous is False
    assert result.raw_requirement == text


@patch("ramma_nlp.interpreter.get_gemini_model")
def test_interpret_borderline_requirement(mock_get_model):
    """Test borderline requirement: 'false positive rate should be low'."""
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_gemini_response(
        {
            "metric": "fpr",
            "operator": "<=",
            "threshold": 0.05,
            "severity": "medium",
            "is_ambiguous": True,
            "ambiguity_reason": "Term 'low' does not specify an explicit numerical threshold.",
            "raw_requirement": "false positive rate should be low",
        }
    )
    mock_get_model.return_value = mock_model

    text = "false positive rate should be low"
    result = interpret_requirement(text)

    assert result.metric == "fpr"
    assert result.is_ambiguous is True
    assert result.ambiguity_reason is not None


@patch("ramma_nlp.interpreter.get_gemini_model")
def test_interpret_ambiguous_requirement(mock_get_model):
    """Test ambiguous requirement: 'the model should behave responsibly'."""
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_gemini_response(
        {
            "metric": "accuracy",
            "operator": ">=",
            "threshold": 0.0,
            "severity": "low",
            "is_ambiguous": True,
            "ambiguity_reason": "Requirement is vague and does not define a measurable metric or threshold.",
            "raw_requirement": "the model should behave responsibly",
        }
    )
    mock_get_model.return_value = mock_model

    text = "the model should behave responsibly"
    response = client.post("/interpret", json={"text": text})

    assert response.status_code == 200
    data = response.json()
    assert data["is_ambiguous"] is True
    assert "vague" in data["ambiguity_reason"].lower() or "measurable" in data["ambiguity_reason"].lower()
