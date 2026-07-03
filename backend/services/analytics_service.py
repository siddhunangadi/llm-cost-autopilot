import asyncio
from datetime import date, datetime, timezone

from pydantic import BaseModel

from backend.services.dashboard_repository import DashboardRepository, TimeWindow


class CostTrendPoint(BaseModel):
    date: date
    request_count: int
    total_cost: float
    average_cost: float


class QualityTrendPoint(BaseModel):
    date: date
    average_score: float
    pass_rate: float


class FailoverTrendPoint(BaseModel):
    date: date
    failover_count: int


class RoutingDistributionPoint(BaseModel):
    date: date
    model: str
    request_count: int


class RecommendationTrendPoint(BaseModel):
    date: date
    generated_count: int
    open_count: int


class AnalyticsReport(BaseModel):
    generated_at: datetime
    window_days: int
    cost_trend: list[CostTrendPoint]
    quality_trend: list[QualityTrendPoint]
    failover_trend: list[FailoverTrendPoint]
    routing_distribution: list[RoutingDistributionPoint]
    recommendation_trend: list[RecommendationTrendPoint]


class AnalyticsService:
    """Read-only historical analytics. Never writes: no recommendation
    refresh, no routing/learning mutation -- only reads DashboardRepository."""

    def __init__(self, dashboard_repository: DashboardRepository) -> None:
        self._dashboard_repository = dashboard_repository

    async def get_report(self, window: TimeWindow) -> AnalyticsReport:
        (
            cost_buckets, quality_buckets, failover_buckets,
            routing_buckets, recommendation_buckets,
        ) = await asyncio.gather(
            asyncio.to_thread(self._dashboard_repository.get_cost_trend, window),
            asyncio.to_thread(self._dashboard_repository.get_quality_trend, window),
            asyncio.to_thread(self._dashboard_repository.get_failover_trend, window),
            asyncio.to_thread(self._dashboard_repository.get_routing_distribution, window),
            asyncio.to_thread(self._dashboard_repository.get_recommendation_trend, window),
        )

        return AnalyticsReport(
            generated_at=datetime.now(timezone.utc),
            window_days=window.days,
            cost_trend=[
                CostTrendPoint(
                    date=b.date, request_count=b.request_count, total_cost=b.total_cost,
                    average_cost=b.total_cost / b.request_count if b.request_count else 0.0,
                )
                for b in cost_buckets
            ],
            quality_trend=[
                QualityTrendPoint(date=b.date, average_score=b.average_score, pass_rate=b.pass_rate)
                for b in quality_buckets
            ],
            failover_trend=[
                FailoverTrendPoint(date=b.date, failover_count=b.failover_count)
                for b in failover_buckets
            ],
            routing_distribution=[
                RoutingDistributionPoint(date=b.date, model=b.model, request_count=b.request_count)
                for b in routing_buckets
            ],
            recommendation_trend=[
                RecommendationTrendPoint(
                    date=b.date, generated_count=b.generated_count, open_count=b.open_count,
                )
                for b in recommendation_buckets
            ],
        )
