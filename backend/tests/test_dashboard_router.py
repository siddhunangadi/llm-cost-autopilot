from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.database.models import RequestRow, ResponseRow, RoutingEventRow, VerificationRow
from backend.verification.status import VerificationStatus


def _seed(session_factory):
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=30)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-1", prompt="hi", strategy="balanced"))
        session.add(VerificationRow(
            request_id="req-1", status=VerificationStatus.COMPLETED.value,
            routing_model="gpt-4o-mini", routing_strategy="balanced", routing_complexity="simple",
            score=0.9, passed=True, confidence=0.8,
        ))
        session.add(ResponseRow(request_id="req-1", response_text="ok", actual_cost=0.10, created_at=now))

        session.add(RequestRow(request_id="req-old", prompt="hi", strategy="balanced"))
        session.add(ResponseRow(
            request_id="req-old", response_text="ok", actual_cost=99.0, created_at=old,
        ))
        session.commit()


def test_dashboard_overview_returns_expected_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()
    with TestClient(app) as client:
        _seed(app.state.session_factory)

        response = client.get("/v1/dashboard/overview")

        assert response.status_code == 200
        body = response.json()
        assert "generated_at" in body
        assert set(body["providers"].keys()) == {"openai", "anthropic", "ollama"}
        assert body["quality"]["total_verified"] == 1
        assert body["failovers"]["total_failovers"] == 0
        assert body["recommendations"] == []


def test_dashboard_overview_days_param_narrows_cost_trend(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()
    with TestClient(app) as client:
        _seed(app.state.session_factory)

        response_default = client.get("/v1/dashboard/overview")
        response_narrow = client.get("/v1/dashboard/overview?days=1")

        default_dates = {b["date"] for b in response_default.json()["cost_trend"]}
        narrow_dates = {b["date"] for b in response_narrow.json()["cost_trend"]}
        assert len(narrow_dates) <= len(default_dates)
        # the 30-day-old response must never appear within a 1-day window
        assert sum(
            b["total_cost"] for b in response_narrow.json()["cost_trend"]
        ) == pytest.approx(0.10)
