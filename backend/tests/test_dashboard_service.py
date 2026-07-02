from datetime import date, datetime, timezone

import pytest

from backend.database.models import RecommendationRow
from backend.services.dashboard_repository import CostBucketData, FailoverData, QualityAggregation, TimeWindow
from backend.services.dashboard_service import DashboardService


class _FakeProviderManager:
    def __init__(self):
        self.list_providers_calls = 0

    def list_providers(self):
        self.list_providers_calls += 1
        return {"openai": "available", "anthropic": "disabled", "ollama": "disabled"}


class _FakeProviderExecutor:
    def __init__(self):
        self.circuit_states_calls = 0

    def circuit_states(self):
        self.circuit_states_calls += 1
        return {
            "openai": {"state": "closed", "consecutive_failures": 0, "successes": 5, "failures": 1},
            "anthropic": {"state": "closed", "consecutive_failures": 0, "successes": 0, "failures": 0},
            "ollama": {"state": "closed", "consecutive_failures": 0, "successes": 0, "failures": 0},
        }


class _FakeLearningService:
    def __init__(self, rows):
        self._rows = rows
        self.get_recommendations_calls = 0

    def get_recommendations(self):
        self.get_recommendations_calls += 1
        return self._rows


class _FakeDashboardRepository:
    def __init__(self, quality, cost_buckets, failover_data):
        self._quality = quality
        self._cost_buckets = cost_buckets
        self._failover_data = failover_data
        self.get_quality_aggregation_calls = 0
        self.get_cost_trend_calls = 0
        self.get_failover_summary_calls = 0

    def get_quality_aggregation(self):
        self.get_quality_aggregation_calls += 1
        return self._quality

    def get_cost_trend(self, window):
        self.get_cost_trend_calls += 1
        return self._cost_buckets

    def get_failover_summary(self, window):
        self.get_failover_summary_calls += 1
        return self._failover_data


def _recommendation_row():
    now = datetime.now(timezone.utc)
    return RecommendationRow(
        signature="model_complexity:gpt-4o-mini:medium", rule_type="model_complexity",
        subject="gpt-4o-mini:medium", recommendation_text="text", evidence_confidence=0.6,
        severity="high", evidence={"sample_size": 20, "pass_rate": 0.35, "threshold": 0.6},
        status="new", source="verification", created_at=now, updated_at=now,
    )


def _quality_aggregation():
    return QualityAggregation(
        total_verified=10, average_score=0.8, average_confidence=0.7, pass_rate=0.9,
        average_queue_delay_ms=5.0, average_evaluation_duration_ms=100.0,
        average_total_verification_ms=105.0, verification_failure_count=1,
        by_model={"gpt-4o-mini": 0.8}, by_strategy={"balanced": 0.8}, by_complexity={"simple": 0.8},
    )


async def test_get_overview_merges_all_six_inputs():
    provider_manager = _FakeProviderManager()
    provider_executor = _FakeProviderExecutor()
    learning_service = _FakeLearningService([_recommendation_row()])
    repository = _FakeDashboardRepository(
        quality=_quality_aggregation(),
        cost_buckets=[CostBucketData(date=date(2026, 7, 1), request_count=2, total_cost=0.30)],
        failover_data=FailoverData(request_ids=["req-failover"]),
    )
    service = DashboardService(
        provider_manager=provider_manager, provider_executor=provider_executor,
        learning_service=learning_service, dashboard_repository=repository,
    )

    overview = await service.get_overview(TimeWindow(days=7))

    assert overview.providers["openai"].availability == "available"
    assert overview.providers["openai"].circuit_state == "closed"
    assert overview.providers["openai"].consecutive_failures == 0
    assert overview.quality.total_verified == 10
    assert overview.cost_trend[0].date == date(2026, 7, 1)
    assert overview.cost_trend[0].request_count == 2
    assert overview.cost_trend[0].total_cost == pytest.approx(0.30)
    assert overview.cost_trend[0].average_cost == pytest.approx(0.15)
    assert overview.failovers.total_failovers == 1
    assert overview.failovers.request_ids == ["req-failover"]
    assert len(overview.recommendations) == 1
    assert overview.recommendations[0].signature == "model_complexity:gpt-4o-mini:medium"


async def test_get_overview_sets_generated_at():
    before = datetime.now(timezone.utc)
    service = DashboardService(
        provider_manager=_FakeProviderManager(), provider_executor=_FakeProviderExecutor(),
        learning_service=_FakeLearningService([]),
        dashboard_repository=_FakeDashboardRepository(
            quality=_quality_aggregation(), cost_buckets=[], failover_data=FailoverData(request_ids=[]),
        ),
    )

    overview = await service.get_overview(TimeWindow(days=7))
    after = datetime.now(timezone.utc)

    assert before <= overview.generated_at <= after


async def test_get_overview_computes_average_cost_correctly():
    repository = _FakeDashboardRepository(
        quality=_quality_aggregation(),
        cost_buckets=[CostBucketData(date=date(2026, 7, 1), request_count=4, total_cost=1.00)],
        failover_data=FailoverData(request_ids=[]),
    )
    service = DashboardService(
        provider_manager=_FakeProviderManager(), provider_executor=_FakeProviderExecutor(),
        learning_service=_FakeLearningService([]), dashboard_repository=repository,
    )

    overview = await service.get_overview(TimeWindow(days=7))

    assert overview.cost_trend[0].average_cost == pytest.approx(0.25)


async def test_get_overview_calls_each_collaborator_exactly_once():
    provider_manager = _FakeProviderManager()
    provider_executor = _FakeProviderExecutor()
    learning_service = _FakeLearningService([])
    repository = _FakeDashboardRepository(
        quality=_quality_aggregation(), cost_buckets=[], failover_data=FailoverData(request_ids=[]),
    )
    service = DashboardService(
        provider_manager=provider_manager, provider_executor=provider_executor,
        learning_service=learning_service, dashboard_repository=repository,
    )

    await service.get_overview(TimeWindow(days=7))

    assert provider_manager.list_providers_calls == 1
    assert provider_executor.circuit_states_calls == 1
    assert learning_service.get_recommendations_calls == 1
    assert repository.get_quality_aggregation_calls == 1
    assert repository.get_cost_trend_calls == 1
    assert repository.get_failover_summary_calls == 1
