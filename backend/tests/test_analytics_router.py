from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.database.models import (
    RecommendationRow, RequestRow, ResponseRow, RoutingEventRow, VerificationRow,
)
from backend.verification.status import VerificationStatus


def _seed(session_factory):
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=30)
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

        session.add(RequestRow(request_id="req-old", prompt="hi", strategy="balanced"))
        session.add(ResponseRow(
            request_id="req-old", response_text="ok", actual_cost=99.0, created_at=old,
        ))
        session.commit()


def test_analytics_report_returns_expected_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()
    with TestClient(app) as client:
        _seed(app.state.session_factory)

        response = client.get("/v1/analytics/report")

        assert response.status_code == 200
        body = response.json()
        assert body["window_days"] == 30
        assert len(body["cost_trend"]) == 1
        assert len(body["routing_distribution"]) == 1
        assert len(body["recommendation_trend"]) == 1


def test_analytics_report_days_param_narrows_window(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()
    with TestClient(app) as client:
        _seed(app.state.session_factory)

        response = client.get("/v1/analytics/report?days=1")

        assert response.status_code == 200
        body = response.json()
        assert body["window_days"] == 1
        # today's cost is still in-window; the 30-day-old response is not
        assert sum(b["total_cost"] for b in body["cost_trend"]) == pytest.approx(0.10)


def test_analytics_page_renders_200_with_no_polling_attributes(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()
    with TestClient(app) as client:
        _seed(app.state.session_factory)

        response = client.get("/dashboard/analytics")

        assert response.status_code == 200
        assert "hx-trigger" not in response.text
        assert 'id="cost-trend-chart"' in response.text
        assert 'id="quality-trend-chart"' in response.text
        assert 'id="routing-distribution-chart"' in response.text
        assert 'id="recommendation-trend-chart"' in response.text
        # no failover was seeded, so that section renders its empty state
        assert "No failovers recorded." in response.text


def test_analytics_page_handles_empty_database(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/dashboard/analytics")

        assert response.status_code == 200
        assert "No cost data yet" in response.text
