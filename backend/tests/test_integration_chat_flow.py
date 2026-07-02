from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.providers.base import ProviderError


def test_full_chat_flow_end_to_end(monkeypatch, tmp_path, mocker):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/integration.db")

    mocker.patch(
        "backend.providers.openai_provider.OpenAIProvider.generate",
        new_callable=AsyncMock,
        return_value="Here are three fruits: apple, banana, cherry.",
    )
    mocker.patch(
        "backend.providers.openai_provider.OpenAIProvider.health_check",
        new_callable=AsyncMock,
        return_value=True,
    )

    app = create_app()
    with TestClient(app) as client:
        health_response = client.get("/v1/health")
        assert health_response.status_code == 200
        assert health_response.json()["providers"]["openai"] == "available"

        chat_response = client.post(
            "/v1/chat", json={"prompt": "List three fruits.", "strategy": "cost"}
        )

    assert chat_response.status_code == 200
    body = chat_response.json()
    assert body["response"] == "Here are three fruits: apple, banana, cherry."
    assert body["routing"]["selected_model"] == "gpt-4o-mini"
    assert body["routing"]["strategy"] == "cost"
    assert "request_id" in body


def test_full_chat_flow_returns_502_on_real_provider_error(monkeypatch, tmp_path, mocker):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/integration2.db")

    mocker.patch(
        "backend.providers.openai_provider.OpenAIProvider.generate",
        new_callable=AsyncMock,
        side_effect=ProviderError("simulated upstream failure"),
    )
    mocker.patch(
        "backend.providers.openai_provider.OpenAIProvider.health_check",
        new_callable=AsyncMock,
        return_value=True,
    )

    app = create_app()
    with TestClient(app) as client:
        response = client.post("/v1/chat", json={"prompt": "Hello.", "strategy": "cost"})

    assert response.status_code == 502
