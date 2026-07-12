from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.database.models import (
    RecommendationRow, RequestRow, ResponseRow, RoutingEventRow, VerificationRow,
)
from backend.verification.status import VerificationStatus


def _seed(session_factory):
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
        session.commit()


def test_dashboard_page_renders_200_with_all_sections(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()
    with TestClient(app) as client:
        _seed(app.state.session_factory)

        response = client.get("/dashboard")

        assert response.status_code == 200
        body = response.text
        for marker in [
            "section-overview", "section-providers", "section-cost", "section-quality",
            "section-recommendations", "section-failovers", "section-circuits",
            "section-recent-requests",
        ]:
            assert marker in body


def test_dashboard_page_empty_database_shows_empty_states(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/empty.db")
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/dashboard")

        assert response.status_code == 200
        assert "No requests yet." in response.text
        assert "No verification results available." in response.text
        assert "No recommendations yet." in response.text
        assert "No failovers recorded." in response.text


def test_dashboard_fragment_overview_returns_partial(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()
    with TestClient(app) as client:
        _seed(app.state.session_factory)

        response = client.get("/dashboard/fragments/overview")

        assert response.status_code == 200
        assert "section-overview" in response.text
        assert "section-providers" not in response.text


def test_dashboard_fragment_unknown_section_returns_404(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/dashboard/fragments/nonexistent")

        assert response.status_code == 404


def test_dashboard_page_polled_sections_carry_htmx_polling_attributes(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()
    with TestClient(app) as client:
        _seed(app.state.session_factory)

        response = client.get("/dashboard")

        body = response.text
        for section in ["overview", "providers", "circuits", "recent-requests"]:
            assert f'hx-get="/dashboard/fragments/{section}"' in body
        assert 'hx-trigger="every 15s"' in body
        assert 'hx-swap="outerHTML"' in body


def test_dashboard_renders_cost_optimization_recommendation(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()
    with TestClient(app) as client:
        with app.state.session_factory() as session:
            session.add(RecommendationRow(
                signature="cost_optimization:gpt-4o:complex",
                rule_type="cost_optimization",
                subject="gpt-4o:complex",
                recommendation_text=(
                    "Current model 'gpt-4o' consistently meets the quality threshold for "
                    "'complex' prompts. A lower-cost model, 'gpt-4o-mini', also meets the "
                    "threshold. Estimated monthly savings: ~$21.00."
                ),
                evidence_confidence=0.6,
                severity="high",
                evidence={"sample_size": 20, "pass_rate": 0.9, "threshold": 0.7, "comparison": {
                    "current_model": "gpt-4o", "suggested_model": "gpt-4o-mini",
                    "current_pass_rate": 0.9, "suggested_pass_rate": 0.85,
                    "current_cost_per_request": 0.10, "suggested_cost_per_request": 0.03,
                    "estimated_monthly_savings": 21.00,
                }},
                status="new",
                source="cost_optimization",
            ))
            session.commit()

        response = client.get("/dashboard")

        assert response.status_code == 200
        assert "gpt-4o-mini" in response.text
        assert "$21.00" in response.text
        assert "$0.70/day" in response.text  # 21.00 / 30, the V1-spec headline figure
        assert "20 requests" in response.text
