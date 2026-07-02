from fastapi import APIRouter
from pydantic import BaseModel

from backend.api.dependencies import SessionFactoryDep
from backend.database.models import VerificationRow
from backend.verification.status import VerificationStatus

router = APIRouter()


class QualityMetrics(BaseModel):
    total_verified: int
    average_score: float
    average_confidence: float
    pass_rate: float
    average_queue_delay_ms: float
    average_evaluation_duration_ms: float
    average_total_verification_ms: float
    verification_failure_count: int
    by_model: dict[str, float]
    by_strategy: dict[str, float]
    by_complexity: dict[str, float]


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _group_avg(rows: list[VerificationRow], key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(getattr(row, key), []).append(row.score)
    return {name: _avg(scores) for name, scores in grouped.items()}


@router.get("/metrics/quality", response_model=QualityMetrics)
async def get_quality_metrics(session_factory: SessionFactoryDep) -> QualityMetrics:
    with session_factory() as session:
        completed = (
            session.query(VerificationRow)
            .filter_by(status=VerificationStatus.COMPLETED.value)
            .all()
        )
        failure_count = (
            session.query(VerificationRow)
            .filter_by(status=VerificationStatus.FAILED.value)
            .count()
        )

    queue_delays = [
        (row.started_at - row.created_at).total_seconds() * 1000
        for row in completed
        if row.started_at is not None
    ]
    total_durations = [
        (row.completed_at - row.started_at).total_seconds() * 1000
        for row in completed
        if row.started_at is not None and row.completed_at is not None
    ]
    eval_durations = [
        row.evaluation_duration_ms for row in completed if row.evaluation_duration_ms is not None
    ]

    return QualityMetrics(
        total_verified=len(completed),
        average_score=_avg([row.score for row in completed]),
        average_confidence=_avg([row.confidence for row in completed if row.confidence is not None]),
        pass_rate=_avg([1.0 if row.passed else 0.0 for row in completed]),
        average_queue_delay_ms=_avg(queue_delays),
        average_evaluation_duration_ms=_avg(eval_durations),
        average_total_verification_ms=_avg(total_durations),
        verification_failure_count=failure_count,
        by_model=_group_avg(completed, "routing_model"),
        by_strategy=_group_avg(completed, "routing_strategy"),
        by_complexity=_group_avg(completed, "routing_complexity"),
    )
