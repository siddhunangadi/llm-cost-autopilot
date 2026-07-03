from datetime import datetime, timezone

import pytest

from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.database.models import (
    RecommendationRow, RequestRow, ResponseRow, RoutingEventRow, VerificationRow,
)
from backend.services.analytics_service import AnalyticsService
from backend.services.dashboard_repository import DashboardRepository, TimeWindow
from backend.verification.status import VerificationStatus


def _make_service(tmp_path):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)
    repository = DashboardRepository(session_factory=session_factory)
    return AnalyticsService(dashboard_repository=repository), session_factory


@pytest.mark.asyncio
async def test_get_report_composes_all_five_trends(tmp_path):
    service, session_factory = _make_service(tmp_path)
    now = datetime.now(timezone.utc)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-1", prompt="hi", strategy="balanced"))
        session.add(RoutingEventRow(
            request_id="req-1", complexity="simple", confidence=0.9, selected_model="gpt-4o-mini",
            selected_strategy="balanced", estimated_cost=0.01, estimated_latency_ms=100,
            reasoning="[]", created_at=now,
        ))
        session.add(ResponseRow(request_id="req-1", response_text="ok", actual_cost=0.10, created_at=now))
        session.add(VerificationRow(
            request_id="req-1", status=VerificationStatus.COMPLETED.value,
            routing_model="gpt-4o-mini", routing_strategy="balanced", routing_complexity="simple",
            score=0.9, passed=True, confidence=0.8, created_at=now,
        ))
        session.add(RecommendationRow(
            signature="sig-1", rule_type="model_complexity", subject="gpt-4o-mini",
            recommendation_text="text", evidence_confidence=0.9, severity="low",
            evidence={}, status="new", source="rule_based", created_at=now,
        ))
        session.commit()

    report = await service.get_report(TimeWindow(days=7))

    assert report.window_days == 7
    assert len(report.cost_trend) == 1
    assert report.cost_trend[0].total_cost == pytest.approx(0.10)
    assert len(report.quality_trend) == 1
    assert report.quality_trend[0].pass_rate == pytest.approx(1.0)
    assert report.failover_trend == []
    assert len(report.routing_distribution) == 1
    assert report.routing_distribution[0].model == "gpt-4o-mini"
    assert report.routing_distribution[0].request_count == 1
    assert len(report.recommendation_trend) == 1
    assert report.recommendation_trend[0].generated_count == 1
    assert report.recommendation_trend[0].open_count == 1


@pytest.mark.asyncio
async def test_get_report_has_no_write_capable_dependencies(tmp_path):
    service, session_factory = _make_service(tmp_path)

    # AnalyticsService must hold only a DashboardRepository reference --
    # no LearningService/ProviderManager/ProviderExecutor dependency that
    # could be used to write, unlike DashboardService.
    assert not hasattr(service, "_learning_service")
    assert not hasattr(service, "_provider_manager")
    assert not hasattr(service, "_provider_executor")

    report = await service.get_report(TimeWindow(days=7))
    assert report.window_days == 7
