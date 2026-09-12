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
- threshold: float between 0.0 and 1.0 (e.g., 93%, 93 percent, or 15 percent becomes 0.93 or 0.15).
- severity: EXACTLY one of ["low", "medium", "high", "critical"]. Default to "medium" if not specified.
- is_ambiguous: boolean. Set to true ONLY if the requirement is genuinely vague, qualitative, unmeasurable, or lacks a specific numeric threshold/metric.
- ambiguity_reason: explanation string if is_ambiguous is true, else null.
- raw_requirement: exact input requirement string.

CRITICAL INSTRUCTIONS ON AMBIGUITY, PHRASING, AND THRESHOLD EXTRACTION:
1. Spelled-out words like "percent", "per cent", or "pct" (e.g., "15 percent", "90 percent") are EXACTLY EQUIVALENT to the "%" symbol. They are fully precise, quantitative, and MUST NOT be flagged as ambiguous (is_ambiguous MUST be false).
2. Interrogative sentence structures or question phrasing (e.g., "Is my FNR less than 15 percent?", "Is my recall greater than 85%?") DO NOT imply ambiguity. Sentence mood (question vs statement) is irrelevant. If a specific metric, comparison operator, and numeric threshold are present, set is_ambiguous to false.
3. Subjective or qualitative phrasing WITHOUT an explicit numeric threshold (e.g., "Is my accuracy high enough?", "Is my precision good?", "What is my F1 score?", "Is my FNR low?") ARE ambiguous (is_ambiguous MUST be true).
4. NEVER silently default threshold to 0.0 for unambiguous requirements (is_ambiguous=false). You MUST extract the actual numeric threshold from the input text scaled to [0.0, 1.0]. If threshold extraction fails or no numeric threshold exists, set is_ambiguous=true with a clear ambiguity_reason.

FEW-SHOT EXAMPLES:
Input: "Recall must remain above 93%"
Output: {"metric": "recall", "operator": ">=", "threshold": 0.93, "severity": "high", "is_ambiguous": false, "ambiguity_reason": null, "raw_requirement": "Recall must remain above 93%"}

Input: "Is my FNR less than 15 percent?"
Output: {"metric": "fnr", "operator": "<", "threshold": 0.15, "severity": "medium", "is_ambiguous": false, "ambiguity_reason": null, "raw_requirement": "Is my FNR less than 15 percent?"}

Input: "Is my precision greater than 90 percent?"
Output: {"metric": "precision", "operator": ">", "threshold": 0.90, "severity": "medium", "is_ambiguous": false, "ambiguity_reason": null, "raw_requirement": "Is my precision greater than 90 percent?"}

Input: "Is my accuracy greater than 95 percent?"
Output: {"metric": "accuracy", "operator": ">", "threshold": 0.95, "severity": "medium", "is_ambiguous": false, "ambiguity_reason": null, "raw_requirement": "Is my accuracy greater than 95 percent?"}

Input: "Recall needs to be above than 85 percent"
Output: {"metric": "recall", "operator": ">", "threshold": 0.85, "severity": "medium", "is_ambiguous": false, "ambiguity_reason": null, "raw_requirement": "Recall needs to be above than 85 percent"}

Input: "Is my FNR low?"
Output: {"metric": "fnr", "operator": ">=", "threshold": 0.0, "severity": "low", "is_ambiguous": true, "ambiguity_reason": "Qualitative term 'low' without an explicit numerical threshold.", "raw_requirement": "Is my FNR low?"}

Input: "Is my accuracy high enough?"
Output: {"metric": "accuracy", "operator": ">=", "threshold": 0.0, "severity": "low", "is_ambiguous": true, "ambiguity_reason": "Subjective phrasing 'high enough' without a specific numeric threshold.", "raw_requirement": "Is my accuracy high enough?"}

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
    if "more than" in text_lower or "greater than" in text_lower or "above than" in text_lower or "strictly higher" in text_lower or ">" in text_lower:
        operator = ">"
    elif ("less than" in text_lower or "below than" in text_lower or "smaller than" in text_lower
          or "under" in text_lower or "beneath" in text_lower or "<" in text_lower):
        operator = "<"
    elif "stay below" in text_lower or "not exceed" in text_lower or "at most" in text_lower or "<=" in text_lower:
        operator = "<="
    elif "equal" in text_lower or "==" in text_lower:
        operator = "=="
    elif "above" in text_lower or "at least" in text_lower or "minimum of" in text_lower or ">=" in text_lower:
        operator = ">="

    # Ambiguity detection heuristics
    vague_qualitative = ["responsibly", "low", "most", "fairly", "reasonably", "should be"]

    # Threshold extraction (supports %, percent, per cent, pct, decimals, and plain integers)
    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent|per\s*cent|pct)\b", text, re.IGNORECASE)
    dec_match = re.search(r"\b0\.\d+\b", text)
    num_match = re.search(r"\b(\d+(?:\.\d+)?)\b", text)

    threshold = None
    if pct_match:
        val = float(pct_match.group(1))
        threshold = round(val / 100.0, 4) if val > 1.0 else round(val, 4)
    elif dec_match:
        threshold = float(dec_match.group(0))
    elif num_match:
        val_str = num_match.group(1)
        if not re.search(r"\bf1\b", text_lower) or val_str != "1":
            val = float(val_str)
            threshold = round(val / 100.0, 4) if val > 1.0 else round(val, 4)

    is_ambiguous = False
    ambiguity_reason = None

    has_vague_kw = any(re.search(r"\b" + re.escape(kw) + r"\b", text_lower) for kw in vague_qualitative)

    if threshold is None or has_vague_kw or "enough" in text_lower or text_lower.startswith("what is"):
        if threshold is None or "enough" in text_lower or (has_vague_kw and not pct_match and not dec_match) or text_lower.startswith("what is"):
            if threshold is None:
                is_ambiguous = True
                ambiguity_reason = "No numerical threshold found in requirement text."
                threshold = 0.0
            elif "enough" in text_lower or (has_vague_kw and not pct_match and not dec_match) or text_lower.startswith("what is"):
                is_ambiguous = True
                ambiguity_reason = "Requirement contains vague, qualitative, or subjective phrasing without a specific numeric threshold."
                threshold = 0.0

    severity = "high" if "must" in text_lower else "medium"

    if not is_ambiguous and threshold == 0.0 and "0" not in text:
        raise ValueError(
            f"Threshold extraction failed for unambiguous requirement: '{text}'. Silently defaulting to 0.0 is disabled."
        )

    return InterpretedRequirement(
        metric=metric,
        operator=operator,
        threshold=threshold,
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
        model_name="gemini-2.5-flash",
        generation_config={
            "response_mime_type": "application/json",
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
