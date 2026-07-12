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


class _FakeModelRegistry:
    """No models registered by default -- _compute_savings then resolves
    baseline_model_id to None and savings collapses to zero, matching the
    pre-existing behavior these tests assert on."""

    def get_models(self):
        return []

    def estimate_cost(self, model_id, input_tokens, output_tokens):
        return 0.0


class _FakeModelSpec:
    def __init__(self, id, input_cost, output_cost):
        self.id = id
        self.input_cost = input_cost
        self.output_cost = output_cost


class _FakePricedModelRegistry:
    """Two models with distinct per-1M-token pricing, mirroring ModelSpec's
    shape (id/input_cost/output_cost) closely enough for _compute_savings'
    baseline-selection (max combined cost) and estimate_cost math."""

    def __init__(self, models):
        self._models = models

    def get_models(self):
        return self._models

    def estimate_cost(self, model_id, input_tokens, output_tokens):
        spec = next(m for m in self._models if m.id == model_id)
        return (input_tokens / 1_000_000) * spec.input_cost + (
            output_tokens / 1_000_000
        ) * spec.output_cost


class _FakeLearningService:
    def __init__(self, rows):
        self._rows = rows
        self.get_recommendations_calls = 0

    def get_recommendations(self):
        self.get_recommendations_calls += 1
        return self._rows


class _FakeDashboardRepository:
    def __init__(self, quality, cost_buckets, failover_data,
                 quality_trend=None, failover_events=None, recent_requests=None, cost_by_model=None,
                 token_totals=None):
        self._quality = quality
        self._cost_buckets = cost_buckets
        self._failover_data = failover_data
        self._quality_trend = quality_trend or []
        self._failover_events = failover_events or []
        self._recent_requests = recent_requests or []
        self._cost_by_model = cost_by_model or {}
        self._token_totals = token_totals or []
        self.get_quality_aggregation_calls = 0
        self.get_cost_trend_calls = 0
        self.get_failover_summary_calls = 0
        self.get_quality_trend_calls = 0
        self.get_failover_events_calls = 0
        self.get_recent_requests_calls = 0
        self.get_cost_by_model_calls = 0

    def get_quality_aggregation(self, window=None):
        self.get_quality_aggregation_calls += 1
        return self._quality

    def get_cost_trend(self, window):
        self.get_cost_trend_calls += 1
        return self._cost_buckets

    def get_failover_summary(self, window):
        self.get_failover_summary_calls += 1
        return self._failover_data

    def get_quality_trend(self, window):
        self.get_quality_trend_calls += 1
        return self._quality_trend

    def get_failover_events(self, window):
        self.get_failover_events_calls += 1
        return self._failover_events

    def get_recent_requests(self, limit=50):
        self.get_recent_requests_calls += 1
        return self._recent_requests

    def get_cost_by_model(self, window):
        self.get_cost_by_model_calls += 1
        return self._cost_by_model

    def get_token_totals(self, window):
        return self._token_totals


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
        model_registry=_FakeModelRegistry()
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
        model_registry=_FakeModelRegistry(),
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
        model_registry=_FakeModelRegistry()
    )

    overview = await service.get_overview(TimeWindow(days=7))

    assert overview.cost_trend[0].average_cost == pytest.approx(0.25)


class _FakeProviderExecutorMissingProvider(_FakeProviderExecutor):
    def circuit_states(self):
        self.circuit_states_calls += 1
        return {
            "openai": {"state": "closed", "consecutive_failures": 0, "successes": 5, "failures": 1},
            "anthropic": {"state": "closed", "consecutive_failures": 0, "successes": 0, "failures": 0},
        }


async def test_get_overview_handles_missing_circuit_state_without_raising():
    provider_manager = _FakeProviderManager()
    provider_executor = _FakeProviderExecutorMissingProvider()
    learning_service = _FakeLearningService([])
    repository = _FakeDashboardRepository(
        quality=_quality_aggregation(), cost_buckets=[], failover_data=FailoverData(request_ids=[]),
    )
    service = DashboardService(
        provider_manager=provider_manager, provider_executor=provider_executor,
        learning_service=learning_service, dashboard_repository=repository,
        model_registry=_FakeModelRegistry()
    )

    overview = await service.get_overview(TimeWindow(days=7))

    assert overview.providers["ollama"].circuit_state == "unknown"
    assert overview.providers["ollama"].consecutive_failures == 0


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
        model_registry=_FakeModelRegistry()
    )

    await service.get_overview(TimeWindow(days=7))

    assert provider_manager.list_providers_calls == 1
    assert provider_executor.circuit_states_calls == 1
    assert learning_service.get_recommendations_calls == 1
    assert repository.get_quality_aggregation_calls == 1
    assert repository.get_cost_trend_calls == 1
    assert repository.get_failover_summary_calls == 1


@pytest.mark.asyncio
async def test_get_overview_fragment_computes_aggregate_stats():
    quality = QualityAggregation(
        total_verified=10, average_score=0.8, average_confidence=0.7, pass_rate=0.9,
        average_queue_delay_ms=5.0, average_evaluation_duration_ms=50.0,
        average_total_verification_ms=60.0, verification_failure_count=1,
        by_model={}, by_strategy={}, by_complexity={},
    )
    cost_buckets = [
        CostBucketData(date=date(2026, 7, 3), request_count=4, total_cost=1.20),
    ]
    failover_data = FailoverData(request_ids=["req-1"])
    repository = _FakeDashboardRepository(quality, cost_buckets, failover_data)
    service = DashboardService(
        provider_manager=_FakeProviderManager(),
        provider_executor=_FakeProviderExecutor(),
        learning_service=_FakeLearningService([]),
        dashboard_repository=repository,
        model_registry=_FakeModelRegistry()
    )

    result = await service.get_overview_fragment(TimeWindow(days=7))

    assert result["total_requests"] == 4
    assert result["total_cost"] == pytest.approx(1.20)
    assert result["average_quality_score"] == pytest.approx(0.8)
    assert result["pass_rate"] == pytest.approx(0.9)
    assert result["active_providers"] == 1
    assert result["open_circuits"] == 0
    assert result["failovers_today"] == 1


@pytest.mark.asyncio
async def test_get_provider_fragment_returns_only_providers_key():
    repository = _FakeDashboardRepository(
        QualityAggregation(0, 0, 0, 0, 0, 0, 0, 0, {}, {}, {}), [], FailoverData([]),
    )
    service = DashboardService(
        provider_manager=_FakeProviderManager(),
        provider_executor=_FakeProviderExecutor(),
        learning_service=_FakeLearningService([]),
        dashboard_repository=repository,
        model_registry=_FakeModelRegistry()
    )

    result = await service.get_provider_fragment()

    assert set(result.keys()) == {"providers"}
    assert set(result["providers"].keys()) == {"openai", "anthropic", "ollama"}


@pytest.mark.asyncio
async def test_get_circuit_fragment_returns_only_circuits_key():
    repository = _FakeDashboardRepository(
        QualityAggregation(0, 0, 0, 0, 0, 0, 0, 0, {}, {}, {}), [], FailoverData([]),
    )
    service = DashboardService(
        provider_manager=_FakeProviderManager(),
        provider_executor=_FakeProviderExecutor(),
        learning_service=_FakeLearningService([]),
        dashboard_repository=repository,
        model_registry=_FakeModelRegistry()
    )

    result = await service.get_circuit_fragment()

    assert set(result.keys()) == {"circuits"}
    assert result["circuits"]["openai"]["state"] == "closed"


@pytest.mark.asyncio
async def test_get_recent_requests_fragment_delegates_to_repository():
    repository = _FakeDashboardRepository(
        QualityAggregation(0, 0, 0, 0, 0, 0, 0, 0, {}, {}, {}), [], FailoverData([]),
        recent_requests=["fake-row"],
    )
    service = DashboardService(
        provider_manager=_FakeProviderManager(),
        provider_executor=_FakeProviderExecutor(),
        learning_service=_FakeLearningService([]),
        dashboard_repository=repository,
        model_registry=_FakeModelRegistry()
    )

    result = await service.get_recent_requests_fragment()

    assert result == {"requests": ["fake-row"]}
    assert repository.get_recent_requests_calls == 1


@pytest.mark.asyncio
async def test_get_dashboard_page_assembles_all_sections():
    quality = QualityAggregation(
        total_verified=1, average_score=0.9, average_confidence=0.8, pass_rate=1.0,
        average_queue_delay_ms=0, average_evaluation_duration_ms=0,
        average_total_verification_ms=0, verification_failure_count=0,
        by_model={}, by_strategy={}, by_complexity={},
    )
    repository = _FakeDashboardRepository(
        quality, [], FailoverData([]),
        quality_trend=["trend-bucket"], failover_events=["failover-event"],
        recent_requests=["request-row"], cost_by_model={"gpt-4o": 1.0},
    )
    service = DashboardService(
        provider_manager=_FakeProviderManager(),
        provider_executor=_FakeProviderExecutor(),
        learning_service=_FakeLearningService([]),
        dashboard_repository=repository,
        model_registry=_FakeModelRegistry()
    )

    result = await service.get_dashboard_page(TimeWindow(days=7))

    assert set(result.keys()) == {
        "overview", "providers", "circuits", "requests",
        "cost_trend", "quality_trend", "cost_by_model", "failover_events", "recommendations",
    }
    assert result["quality_trend"] == ["trend-bucket"]
    assert result["failover_events"] == ["failover-event"]
    assert result["requests"] == ["request-row"]
    assert result["cost_by_model"] == {"gpt-4o": 1.0}


class _TokenTotals:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


async def test_get_overview_computes_savings_against_costliest_model_by_default():
    """_compute_savings must pick the model with the highest combined
    input+output cost as the baseline when none is configured, then
    compute what this traffic would have cost on that model using its
    actual token counts -- not just re-scale the actual spend."""
    model_registry = _FakePricedModelRegistry([
        _FakeModelSpec(id="cheap-model", input_cost=1.0, output_cost=1.0),
        _FakeModelSpec(id="gpt-4o", input_cost=5.0, output_cost=15.0),
    ])
    repository = _FakeDashboardRepository(
        quality=_quality_aggregation(),
        cost_buckets=[CostBucketData(date=date(2026, 7, 1), request_count=2, total_cost=0.05)],
        failover_data=FailoverData(request_ids=[]),
        token_totals=[_TokenTotals(1000, 500), _TokenTotals(2000, 1000)],
    )
    service = DashboardService(
        provider_manager=_FakeProviderManager(), provider_executor=_FakeProviderExecutor(),
        learning_service=_FakeLearningService([]), dashboard_repository=repository,
        model_registry=model_registry,
    )

    overview = await service.get_overview(TimeWindow(days=7))

    expected_baseline = (
        (1000 / 1_000_000) * 5.0 + (500 / 1_000_000) * 15.0
        + (2000 / 1_000_000) * 5.0 + (1000 / 1_000_000) * 15.0
    )
    assert overview.savings.baseline_model_id == "gpt-4o"
    assert overview.savings.actual_cost == pytest.approx(0.05)
    assert overview.savings.baseline_cost == pytest.approx(expected_baseline)
    assert overview.savings.savings_amount == pytest.approx(expected_baseline - 0.05)
    assert overview.savings.savings_percent == pytest.approx(
        (expected_baseline - 0.05) / expected_baseline
    )


async def test_get_overview_savings_uses_configured_baseline_model_id():
    model_registry = _FakePricedModelRegistry([
        _FakeModelSpec(id="cheap-model", input_cost=1.0, output_cost=1.0),
        _FakeModelSpec(id="gpt-4o", input_cost=5.0, output_cost=15.0),
    ])
    repository = _FakeDashboardRepository(
        quality=_quality_aggregation(),
        cost_buckets=[CostBucketData(date=date(2026, 7, 1), request_count=1, total_cost=0.01)],
        failover_data=FailoverData(request_ids=[]),
        token_totals=[_TokenTotals(1000, 1000)],
    )
    service = DashboardService(
        provider_manager=_FakeProviderManager(), provider_executor=_FakeProviderExecutor(),
        learning_service=_FakeLearningService([]), dashboard_repository=repository,
        model_registry=model_registry, baseline_model_id="cheap-model",
    )

    overview = await service.get_overview(TimeWindow(days=7))

    assert overview.savings.baseline_model_id == "cheap-model"
    assert overview.savings.baseline_cost == pytest.approx(0.002)


async def test_get_overview_savings_is_zero_when_no_models_registered():
    repository = _FakeDashboardRepository(
        quality=_quality_aggregation(),
        cost_buckets=[CostBucketData(date=date(2026, 7, 1), request_count=1, total_cost=0.01)],
        failover_data=FailoverData(request_ids=[]),
    )
    service = DashboardService(
        provider_manager=_FakeProviderManager(), provider_executor=_FakeProviderExecutor(),
        learning_service=_FakeLearningService([]), dashboard_repository=repository,
        model_registry=_FakeModelRegistry(),
    )

    overview = await service.get_overview(TimeWindow(days=7))

    assert overview.savings.baseline_model_id is None
    assert overview.savings.savings_amount == pytest.approx(0.0)
    assert overview.savings.savings_percent == pytest.approx(0.0)
