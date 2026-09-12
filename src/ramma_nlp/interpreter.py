"""Requirement Interpreter module using Gemini API with deterministic fallback and FastAPI."""

import json
import os
import re
from typing import Any
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ramma_nlp.schema import InterpretedRequirement

from ramma_backend.database import init_db
from ramma_backend.router import router as backend_router

load_dotenv()

# Initialize backend database tables
init_db()

app = FastAPI(
    title="RAMMA NLP & Backend Service",
    description="API for interpreting and persisting ML monitoring requirements.",
)
app.include_router(backend_router)

SYSTEM_PROMPT = """You are an expert ML Monitoring Requirement Interpreter.
Your task is to parse a natural-language requirement into a structured JSON object matching this schema:
- metric: EXACTLY one of ["recall", "precision", "f1", "fnr", "fpr", "accuracy", "auc"].
- operator: EXACTLY one of [">=", "<=", ">", "<", "=="].
- threshold: float between 0.0 and 1.0 (e.g., 93% becomes 0.93).
- severity: EXACTLY one of ["low", "medium", "high", "critical"]. Default to "medium" if not specified.
- is_ambiguous: boolean. Set to true if requirement is subjective, vague, unmeasurable, or missing metric/threshold.
- ambiguity_reason: explanation string if is_ambiguous is true, else null.
- raw_requirement: exact input requirement string.

CRITICAL INSTRUCTIONS FOR AMBIGUOUS REQUIREMENTS:
If requirement is ambiguous, set is_ambiguous=true and provide ambiguity_reason. For metric, operator, threshold, severity provide default fallback values (e.g. metric="recall", operator=">=", threshold=0.0, severity="low").

Return ONLY valid JSON matching the specified schema.
"""


class RequirementInput(BaseModel):
    text: str


def rule_based_fallback(text: str) -> InterpretedRequirement:
    """Deterministic regex rule-based parser fallback when API key is unavailable."""
    text_lower = text.lower()

    # Metric matching
    metric = "accuracy"
    if "recall" in text_lower:
        metric = "recall"
    elif "precision" in text_lower:
        metric = "precision"
    elif "f1" in text_lower:
        metric = "f1"
    elif "false positive rate" in text_lower or "fpr" in text_lower:
        metric = "fpr"
    elif "false negative rate" in text_lower or "fnr" in text_lower:
        metric = "fnr"
    elif "auc" in text_lower:
        metric = "auc"
    elif "accuracy" in text_lower:
        metric = "accuracy"

    # Operator matching
    operator = ">="
    if "strictly higher" in text_lower or "greater than" in text_lower or ">" in text_lower:
        operator = ">"
    elif "stay below" in text_lower or "not exceed" in text_lower or "<=" in text_lower:
        operator = "<="
    elif "less than" in text_lower or "<" in text_lower:
        operator = "<"
    elif "equal" in text_lower or "==" in text_lower:
        operator = "=="
    elif "above" in text_lower or "at least" in text_lower or ">=" in text_lower:
        operator = ">="

    # Ambiguity detection heuristics
    ambiguous_keywords = ["responsibly", "low", "most", "fairly", "reasonably", "should be"]
    is_ambiguous = any(kw in text_lower for kw in ambiguous_keywords)
    ambiguity_reason = None

    if is_ambiguous:
        ambiguity_reason = (
            "Requirement contains vague, qualitative, or subjective phrasing "
            "without an explicit measurable target metric and numeric threshold."
        )

    # Threshold extraction
    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    dec_match = re.search(r"\b0\.\d+\b", text)

    threshold = 0.0
    if pct_match:
        threshold = round(float(pct_match.group(1)) / 100.0, 4)
    elif dec_match:
        threshold = float(dec_match.group(0))
    elif not is_ambiguous:
        is_ambiguous = True
        ambiguity_reason = "No numerical threshold found in requirement text."

    severity = "high" if "must" in text_lower else "medium"

    return InterpretedRequirement(
        metric=metric,
        operator=operator,
        threshold=max(0.0, min(1.0, threshold)),
        severity=severity,
        is_ambiguous=is_ambiguous,
        ambiguity_reason=ambiguity_reason,
        raw_requirement=text,
    )


def get_gemini_model() -> genai.GenerativeModel:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set or contains default placeholder."
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": InterpretedRequirement,
        },
    )


def interpret_requirement(text: str) -> InterpretedRequirement:
    """Interprets a natural language ML monitoring requirement into structured format.

    Uses Gemini API with JSON mode and Pydantic validation. Retries once if validation fails.
    Falls back to deterministic parsing if API key is not configured.
    """
    try:
        model = get_gemini_model()
    except ValueError:
        # Graceful fallback when GEMINI_API_KEY is not set
        return rule_based_fallback(text)

    prompt = f"{SYSTEM_PROMPT}\n\nRequirement to interpret:\n\"{text}\""

    for attempt in range(2):
        try:
            response = model.generate_content(prompt)
            raw_response_text = response.text.strip()
            parsed_json = json.loads(raw_response_text)

            # Ensure raw_requirement is populated correctly
            parsed_json["raw_requirement"] = text

            return InterpretedRequirement.model_validate(parsed_json)
        except Exception as e:
            if attempt == 0:
                prompt += (
                    f"\n\nPrevious output failed validation with error: {str(e)}. "
                    "Please correct the JSON and ensure all schema constraints are strictly satisfied."
                )
            else:
                return rule_based_fallback(text)


@app.post("/interpret", response_model=InterpretedRequirement)
def interpret_endpoint(input_data: RequirementInput) -> InterpretedRequirement:
    """FastAPI POST endpoint for interpreting monitoring requirements."""
    try:
        return interpret_requirement(input_data.text)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Interpretation error: {str(e)}"
        ) from e
