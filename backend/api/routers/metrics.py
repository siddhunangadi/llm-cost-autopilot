from dataclasses import asdict

from fastapi import APIRouter
from pydantic import BaseModel

from backend.api.dependencies import SessionFactoryDep
from backend.services.dashboard_repository import DashboardRepository

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


@router.get("/metrics/quality", response_model=QualityMetrics)
async def get_quality_metrics(session_factory: SessionFactoryDep) -> QualityMetrics:
    repository = DashboardRepository(session_factory=session_factory)
    aggregation = repository.get_quality_aggregation()
    return QualityMetrics(**asdict(aggregation))
