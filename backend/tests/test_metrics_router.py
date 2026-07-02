import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.database.models import RequestRow, VerificationRow
from backend.verification.status import VerificationStatus


def _seed(session_factory):
    with session_factory() as session:
        for i, (model, strategy, complexity, score) in enumerate([
            ("gpt-4o-mini", "balanced", "simple", 0.9),
            ("gpt-4o-mini", "balanced", "simple", 0.5),
            ("gpt-4o", "quality", "complex", 0.95),
        ]):
            request_id = f"req-{i}"
            session.add(RequestRow(request_id=request_id, prompt="hi", strategy=strategy))
            session.add(VerificationRow(
                request_id=request_id, status=VerificationStatus.COMPLETED.value,
                routing_model=model, routing_strategy=strategy, routing_complexity=complexity,
                score=score, passed=score >= 0.7, confidence=0.8,
                evaluation_duration_ms=100,
            ))
        session.add(RequestRow(request_id="req-failed", prompt="hi", strategy="balanced"))
        session.add(VerificationRow(
            request_id="req-failed", status=VerificationStatus.FAILED.value,
            routing_model="gpt-4o-mini", routing_strategy="balanced", routing_complexity="simple",
            error_type="ValidationError", error="bad json",
        ))
        session.commit()


def test_quality_metrics_aggregates(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    app = create_app()
    with TestClient(app) as client:
        _seed(app.state.session_factory)

        response = client.get("/v1/metrics/quality")

        assert response.status_code == 200
        body = response.json()
        assert body["total_verified"] == 3
        assert body["verification_failure_count"] == 1
        assert body["pass_rate"] == pytest.approx(2 / 3)
        assert body["by_model"]["gpt-4o-mini"] == pytest.approx((0.9 + 0.5) / 2)
        assert body["by_model"]["gpt-4o"] == pytest.approx(0.95)
