"""Schema definitions for RAMMA NLP requirement interpreter."""

from pydantic import BaseModel, Field, field_validator, model_validator
from ramma_nlp.metric_library import SUPPORTED_METRICS

ALLOWED_METRICS = set(SUPPORTED_METRICS)
ALLOWED_OPERATORS = {">=", "<=", ">", "<", "=="}
ALLOWED_SEVERITIES = {"low", "medium", "high", "critical"}


class InterpretedRequirement(BaseModel):
    """Pydantic model representing a structured ML monitoring requirement."""

    metric: str
    operator: str
    threshold: float
    severity: str
    is_ambiguous: bool
    ambiguity_reason: str | None = None
    raw_requirement: str

    @field_validator("metric")
    @classmethod
    def validate_metric(cls, v: str) -> str:
        v_lower = v.strip().lower()
        if v_lower not in ALLOWED_METRICS:
            raise ValueError(
                f"Invalid metric '{v}'. Metric must be one of: {sorted(ALLOWED_METRICS)}"
            )
        return v_lower

    @field_validator("operator")
    @classmethod
    def validate_operator(cls, v: str) -> str:
        v_clean = v.strip()
        if v_clean not in ALLOWED_OPERATORS:
            raise ValueError(
                f"Invalid operator '{v}'. Operator must be one of: {sorted(ALLOWED_OPERATORS)}"
            )
        return v_clean

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Threshold {v} must be between 0.0 and 1.0 inclusive.")
        return v

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        v_lower = v.strip().lower()
        if v_lower not in ALLOWED_SEVERITIES:
            raise ValueError(
                f"Invalid severity '{v}'. Severity must be one of: {sorted(ALLOWED_SEVERITIES)}"
            )
        return v_lower

    @model_validator(mode="after")
    def validate_non_ambiguous_threshold(self) -> "InterpretedRequirement":
        if not self.is_ambiguous and self.threshold == 0.0:
            text_lower = self.raw_requirement.lower() if self.raw_requirement else ""
            if "0" not in text_lower and "zero" not in text_lower:
                raise ValueError(
                    f"Threshold silently defaulted to 0.0 for unambiguous requirement '{self.raw_requirement}'. "
                    "Threshold extraction must extract a valid numerical value from the requirement text."
                )
        return self
