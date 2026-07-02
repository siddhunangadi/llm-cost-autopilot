from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.database.models import RequestRow, VerificationRow
from backend.verification.status import VerificationStatus


def test_get_verification_returns_completed_result(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    app = create_app()
    with TestClient(app) as client:
        session_factory = app.state.session_factory
        with session_factory() as session:
            session.add(RequestRow(request_id="req-1", prompt="hi", strategy="balanced"))
            session.add(VerificationRow(
                request_id="req-1", status=VerificationStatus.COMPLETED.value,
                routing_model="gpt-4o-mini", routing_strategy="balanced", routing_complexity="simple",
                score=0.9, passed=True, confidence=0.85, rationale="Good.",
                dimensions={
                    "correctness": 0.9, "completeness": 0.9,
                    "instruction_following": 0.9, "format_adherence": 0.9,
                },
                judge_model="gpt-4o", judge_prompt_version="v1", evaluation_duration_ms=120,
                started_at=datetime.now(timezone.utc), completed_at=datetime.now(timezone.utc),
            ))
            session.commit()

        response = client.get("/v1/chat/req-1/verification")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "completed"
        assert body["score"] == 0.9
        assert body["dimensions"]["correctness"] == 0.9
        assert "raw_judge_response" not in body


def test_get_verification_404_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/v1/chat/does-not-exist/verification")
        assert response.status_code == 404
