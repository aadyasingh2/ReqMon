"""FastAPI router for backend requirement management endpoints."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ramma_backend.database import get_db
from ramma_backend.models import Requirement
from ramma_backend.schemas import RequirementCreate, RequirementResponse

router = APIRouter(tags=["Requirements"])


@router.post("/requirements", response_model=RequirementResponse, status_code=status.HTTP_201_CREATED)
def create_requirement(
    payload: RequirementCreate, db: Session = Depends(get_db)
) -> RequirementResponse:
    """Interprets a natural-language requirement via ramma_nlp and saves it to the database."""
    from ramma_nlp.interpreter import interpret_requirement

    try:
        interpreted = interpret_requirement(payload.text)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to interpret requirement: {str(e)}",
        ) from e

    db_req = Requirement(
        raw_text=interpreted.raw_requirement,
        metric=interpreted.metric,
        operator=interpreted.operator,
        threshold=interpreted.threshold,
        severity=interpreted.severity,
        is_ambiguous=interpreted.is_ambiguous,
        ambiguity_reason=interpreted.ambiguity_reason,
    )
    db.add(db_req)
    db.commit()
    db.refresh(db_req)

    return db_req


@router.get("/requirements/{id}", response_model=RequirementResponse)
def get_requirement(id: int, db: Session = Depends(get_db)) -> RequirementResponse:
    """Retrieves a single requirement by its database ID."""
    db_req = db.query(Requirement).filter(Requirement.id == id).first()
    if not db_req:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Requirement with id {id} not found",
        )
    return db_req


@router.get("/requirements", response_model=List[RequirementResponse])
def list_requirements(db: Session = Depends(get_db)) -> List[RequirementResponse]:
    """Lists all stored requirement records from the database."""
    return db.query(Requirement).order_by(Requirement.id.asc()).all()
