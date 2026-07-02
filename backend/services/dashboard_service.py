import asyncio
from datetime import date, datetime, timezone

from pydantic import BaseModel

from backend.database.models import RecommendationRow
from backend.learning.generator import RecommendationEvidence, RecommendationSource, Severity
from backend.learning.rules import RuleType
from backend.learning.service import LearningService
from backend.providers.executor import ProviderExecutor
from backend.providers.manager import ProviderManager
from backend.services.dashboard_repository import DashboardRepository, TimeWindow


class ProviderDashboardStatus(BaseModel):
    availability: str
    circuit_state: str
    consecutive_failures: int


class CostBucket(BaseModel):
    date: date
    request_count: int
    total_cost: float
    average_cost: float


class FailoverSummary(BaseModel):
    total_failovers: int
    request_ids: list[str]


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


class DashboardOverview(BaseModel):
    generated_at: datetime
    providers: dict[str, ProviderDashboardStatus]
    quality: QualityMetrics
    cost_trend: list[CostBucket]
    failovers: FailoverSummary
    recommendations: list[RecommendationResponse]


class DashboardService:
    def __init__(
        self,
        provider_manager: ProviderManager,
        provider_executor: ProviderExecutor,
        learning_service: LearningService,
        dashboard_repository: DashboardRepository,
    ) -> None:
        self._provider_manager = provider_manager
        self._provider_executor = provider_executor
        self._learning_service = learning_service
        self._dashboard_repository = dashboard_repository

    async def get_overview(self, window: TimeWindow) -> DashboardOverview:
        (
            availability,
            circuits,
            quality_agg,
            cost_buckets,
            failover_data,
            recommendation_rows,
        ) = await asyncio.gather(
            asyncio.to_thread(self._provider_manager.list_providers),
            asyncio.to_thread(self._provider_executor.circuit_states),
            asyncio.to_thread(self._dashboard_repository.get_quality_aggregation),
            asyncio.to_thread(self._dashboard_repository.get_cost_trend, window),
            asyncio.to_thread(self._dashboard_repository.get_failover_summary, window),
            asyncio.to_thread(self._learning_service.get_recommendations),
        )

        return DashboardOverview(
            generated_at=datetime.now(timezone.utc),
            providers=self._merge_provider_status(availability, circuits),
            quality=QualityMetrics(
                total_verified=quality_agg.total_verified,
                average_score=quality_agg.average_score,
                average_confidence=quality_agg.average_confidence,
                pass_rate=quality_agg.pass_rate,
                average_queue_delay_ms=quality_agg.average_queue_delay_ms,
                average_evaluation_duration_ms=quality_agg.average_evaluation_duration_ms,
                average_total_verification_ms=quality_agg.average_total_verification_ms,
                verification_failure_count=quality_agg.verification_failure_count,
                by_model=quality_agg.by_model,
                by_strategy=quality_agg.by_strategy,
                by_complexity=quality_agg.by_complexity,
            ),
            cost_trend=[
                CostBucket(
                    date=b.date, request_count=b.request_count, total_cost=b.total_cost,
                    average_cost=b.total_cost / b.request_count if b.request_count else 0.0,
                )
                for b in cost_buckets
            ],
            failovers=FailoverSummary(
                total_failovers=len(failover_data.request_ids),
                request_ids=failover_data.request_ids,
            ),
            recommendations=[self._to_recommendation_response(r) for r in recommendation_rows],
        )

    async def get_overview_fragment(self, window: TimeWindow) -> dict:
        availability, circuits, quality_agg, cost_buckets, failover_today = await asyncio.gather(
            asyncio.to_thread(self._provider_manager.list_providers),
            asyncio.to_thread(self._provider_executor.circuit_states),
            asyncio.to_thread(self._dashboard_repository.get_quality_aggregation),
            asyncio.to_thread(self._dashboard_repository.get_cost_trend, window),
            asyncio.to_thread(self._dashboard_repository.get_failover_summary, TimeWindow(days=1)),
        )
        return {
            "total_requests": sum(b.request_count for b in cost_buckets),
            "total_cost": sum(b.total_cost for b in cost_buckets),
            "average_quality_score": quality_agg.average_score,
            "pass_rate": quality_agg.pass_rate,
            "active_providers": sum(1 for status in availability.values() if status == "available"),
            "open_circuits": sum(1 for c in circuits.values() if c.get("state") == "open"),
            "failovers_today": len(failover_today.request_ids),
        }

    async def get_provider_fragment(self) -> dict:
        availability, circuits = await asyncio.gather(
            asyncio.to_thread(self._provider_manager.list_providers),
            asyncio.to_thread(self._provider_executor.circuit_states),
        )
        return {"providers": self._merge_provider_status(availability, circuits)}

    async def get_circuit_fragment(self) -> dict:
        circuits = await asyncio.to_thread(self._provider_executor.circuit_states)
        return {"circuits": circuits}

    async def get_recent_requests_fragment(self, limit: int = 50) -> dict:
        requests = await asyncio.to_thread(self._dashboard_repository.get_recent_requests, limit)
        return {"requests": requests}

    async def get_dashboard_page(self, window: TimeWindow) -> dict:
        (
            overview, provider_data, circuit_data, recent_requests_data,
            cost_trend, quality_trend, cost_by_model, failover_events, recommendation_rows,
        ) = await asyncio.gather(
            self.get_overview_fragment(window),
            self.get_provider_fragment(),
            self.get_circuit_fragment(),
            self.get_recent_requests_fragment(),
            asyncio.to_thread(self._dashboard_repository.get_cost_trend, window),
            asyncio.to_thread(self._dashboard_repository.get_quality_trend, window),
            asyncio.to_thread(self._dashboard_repository.get_cost_by_model, window),
            asyncio.to_thread(self._dashboard_repository.get_failover_events, window),
            asyncio.to_thread(self._learning_service.get_recommendations),
        )
        return {
            "overview": overview,
            "providers": provider_data["providers"],
            "circuits": circuit_data["circuits"],
            "requests": recent_requests_data["requests"],
            "cost_trend": cost_trend,
            "quality_trend": quality_trend,
            "cost_by_model": cost_by_model,
            "failover_events": failover_events,
            "recommendations": [self._to_recommendation_response(r) for r in recommendation_rows],
        }

    def _merge_provider_status(
        self, availability: dict[str, str], circuits: dict[str, dict],
    ) -> dict[str, ProviderDashboardStatus]:
        result = {}
        for name, status in availability.items():
            circuit = circuits.get(name, {})
            result[name] = ProviderDashboardStatus(
                availability=status,
                circuit_state=circuit.get("state", "unknown"),
                consecutive_failures=circuit.get("consecutive_failures", 0),
            )
        return result

    def _to_recommendation_response(self, r: RecommendationRow) -> RecommendationResponse:
        return RecommendationResponse(
            signature=r.signature, rule_type=RuleType(r.rule_type), subject=r.subject,
            text=r.recommendation_text, evidence_confidence=r.evidence_confidence,
            severity=Severity(r.severity), evidence=RecommendationEvidence(**r.evidence),
            status=r.status, source=RecommendationSource(r.source),
            created_at=r.created_at, updated_at=r.updated_at,
        )
