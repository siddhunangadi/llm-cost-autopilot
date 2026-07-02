from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.api.dependencies import SessionFactoryDep
from backend.database.models import VerificationRow
from backend.verification.judge import VerificationDimensions
from backend.verification.status import VerificationStatus

router = APIRouter()


class VerificationResult(BaseModel):
    request_id: str
    status: VerificationStatus
    score: float | None
    passed: bool | None
    confidence: float | None
    rationale: str | None
    dimensions: VerificationDimensions | None
    judge_model: str | None
    judge_prompt_version: str | None
    evaluation_duration_ms: int | None
    error_type: str | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


@router.get("/chat/{request_id}/verification", response_model=VerificationResult)
async def get_verification(request_id: str, session_factory: SessionFactoryDep) -> VerificationResult:
    with session_factory() as session:
        row = session.query(VerificationRow).filter_by(request_id=request_id).first()
        if row is None:
            raise HTTPException(status_code=404, detail=f"No verification found for '{request_id}'")

        return VerificationResult(
            request_id=row.request_id,
            status=VerificationStatus(row.status),
            score=row.score,
            passed=row.passed,
            confidence=row.confidence,
            rationale=row.rationale,
            dimensions=VerificationDimensions(**row.dimensions) if row.dimensions else None,
            judge_model=row.judge_model,
            judge_prompt_version=row.judge_prompt_version,
            evaluation_duration_ms=row.evaluation_duration_ms,
            error_type=row.error_type,
            error=row.error,
            created_at=row.created_at,
            started_at=row.started_at,
            completed_at=row.completed_at,
        )
