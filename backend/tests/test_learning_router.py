import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.database.models import RequestRow, VerificationRow
from backend.verification.status import VerificationStatus


def _seed(session_factory):
    with session_factory() as session:
        for i in range(20):
            request_id = f"req-{i}"
            session.add(RequestRow(request_id=request_id, prompt="hi", strategy="balanced"))
            session.add(VerificationRow(
                request_id=request_id, status=VerificationStatus.COMPLETED.value,
                routing_model="gpt-4o-mini", routing_strategy="balanced",
                routing_complexity="medium", passed=(i < 7), score=0.4 if i >= 7 else 0.9,
                rationale="Incomplete answer." if i >= 7 else "Good answer.",
            ))
        session.commit()


def test_learning_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()
    with TestClient(app) as client:
        _seed(app.state.session_factory)

        response = client.get("/v1/learning/summary")

        assert response.status_code == 200
        body = response.json()
        assert body["total_verified"] == 20
        assert body["overall_pass_rate"] == pytest.approx(7 / 20)
        assert body["by_model"]["gpt-4o-mini"] == pytest.approx(7 / 20)


def test_learning_failures(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()
    with TestClient(app) as client:
        _seed(app.state.session_factory)

        response = client.get("/v1/learning/failures")

        assert response.status_code == 200
        body = response.json()
        assert len(body) == 13  # 20 - 7 passed
        assert all(r["routing_model"] == "gpt-4o-mini" for r in body)


def test_learning_recommendations_triggers_refresh(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()
    with TestClient(app) as client:
        _seed(app.state.session_factory)

        response = client.get("/v1/learning/recommendations")

        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["signature"] == "model_complexity:gpt-4o-mini:medium"
        assert body[0]["status"] == "new"
