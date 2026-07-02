import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app


def test_chat_then_verification_completes_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    judge_json = json.dumps({
        "correctness": 0.9, "completeness": 0.9, "instruction_following": 0.9,
        "format_adherence": 0.9, "confidence": 0.9, "rationale": "Good answer.",
    })

    app = create_app()
    with (
        patch(
            "backend.providers.openai_provider.OpenAIProvider.health_check",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "backend.providers.openai_provider.OpenAIProvider.generate",
            new=AsyncMock(side_effect=["The answer is 4.", judge_json]),
        ),
        TestClient(app) as client,
    ):
        chat_response = client.post(
            "/v1/chat", json={"prompt": "What is 2+2?", "strategy": "balanced"}
        )
        assert chat_response.status_code == 200
        request_id = chat_response.json()["request_id"]

        verification_response = client.get(f"/v1/chat/{request_id}/verification")
        assert verification_response.status_code == 200
        body = verification_response.json()
        assert body["status"] == "completed"
        assert body["score"] == pytest.approx(0.9)

        metrics_response = client.get("/v1/metrics/quality")
        assert metrics_response.status_code == 200
        assert metrics_response.json()["total_verified"] == 1
