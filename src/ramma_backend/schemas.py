"""Pydantic schemas for RAMMA Backend API endpoints."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class RequirementCreate(BaseModel):
    """Payload for posting a new natural-language monitoring requirement."""

    text: str = Field(..., json_schema_extra={"example": "Recall must remain above 93%"})


class RequirementResponse(BaseModel):
    """Response model for stored ML monitoring requirement records."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    raw_text: str
    metric: str
    operator: str
    threshold: float
    severity: str
    is_ambiguous: bool
    ambiguity_reason: Optional[str] = None
    created_at: datetime
