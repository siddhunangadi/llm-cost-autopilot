from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel

from backend.api.dependencies import LearningServiceDep, SessionFactoryDep
from backend.database.models import VerificationRow
from backend.learning.generator import RecommendationEvidence, RecommendationSource, Severity
from backend.learning.rules import RuleType
from backend.verification.status import VerificationStatus

router = APIRouter()


class LearningSummary(BaseModel):
    total_verified: int
    overall_pass_rate: float
    by_model: dict[str, float]
    by_strategy: dict[str, float]
    by_complexity: dict[str, float]


class FailureRecord(BaseModel):
    request_id: str
    routing_model: str
    routing_strategy: str
    routing_complexity: str
    score: float | None
    rationale: str | None
    created_at: datetime


class RecommendationResponse(BaseModel):
    signature: str
    rule_type: RuleType
    subject: str
    text: str
    evidence_confidence: float
    severity: Severity
    evidence: RecommendationEvidence
    status: str
    source: RecommendationSource
    created_at: datetime
    updated_at: datetime


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _group_pass_rate(rows: list[VerificationRow], key: str) -> dict[str, float]:
    grouped: dict[str, list[bool]] = {}
    for row in rows:
        grouped.setdefault(getattr(row, key), []).append(1.0 if row.passed else 0.0)
    return {name: _avg(outcomes) for name, outcomes in grouped.items()}


@router.get("/learning/summary", response_model=LearningSummary)
async def get_learning_summary(session_factory: SessionFactoryDep) -> LearningSummary:
    with session_factory() as session:
        rows = (
            session.query(VerificationRow)
            .filter_by(status=VerificationStatus.COMPLETED.value)
            .all()
        )

    return LearningSummary(
        total_verified=len(rows),
        overall_pass_rate=_avg([1.0 if r.passed else 0.0 for r in rows]),
        by_model=_group_pass_rate(rows, "routing_model"),
        by_strategy=_group_pass_rate(rows, "routing_strategy"),
        by_complexity=_group_pass_rate(rows, "routing_complexity"),
    )


@router.get("/learning/failures", response_model=list[FailureRecord])
async def get_learning_failures(session_factory: SessionFactoryDep) -> list[FailureRecord]:
    with session_factory() as session:
        rows = (
            session.query(VerificationRow)
            .filter_by(status=VerificationStatus.COMPLETED.value, passed=False)
            .order_by(VerificationRow.created_at.desc())
            .all()
        )
        return [
            FailureRecord(
                request_id=r.request_id, routing_model=r.routing_model,
                routing_strategy=r.routing_strategy, routing_complexity=r.routing_complexity,
                score=r.score, rationale=r.rationale, created_at=r.created_at,
            )
            for r in rows
        ]


@router.get("/learning/recommendations", response_model=list[RecommendationResponse])
async def get_learning_recommendations(
    learning_service: LearningServiceDep,
) -> list[RecommendationResponse]:
    rows = learning_service.refresh_recommendations()
    return [
        RecommendationResponse(
            signature=r.signature, rule_type=RuleType(r.rule_type), subject=r.subject,
            text=r.recommendation_text, evidence_confidence=r.evidence_confidence,
            severity=Severity(r.severity), evidence=RecommendationEvidence(**r.evidence),
            status=r.status, source=RecommendationSource(r.source),
            created_at=r.created_at, updated_at=r.updated_at,
        )
        for r in rows
    ]
