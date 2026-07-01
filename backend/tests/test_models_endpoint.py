from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.dependencies import get_model_registry
from backend.api.routers.models import router as models_router
from backend.services.model_registry import ModelSpec


class _FakeModelRegistry:
    def get_models(self):
        return [
            ModelSpec(
                id="gpt-4o-mini",
                provider="openai",
                model="gpt-4o-mini",
                input_cost=0.15,
                output_cost=0.60,
                context_window=128000,
                max_output_tokens=16384,
                supports_streaming=True,
                supports_tools=True,
                supports_json=True,
                supports_vision=False,
                benchmark_score=0.82,
                average_latency_ms=450,
                available=True,
            )
        ]


def test_list_models_returns_full_spec():
    app = FastAPI()
    app.include_router(models_router, prefix="/v1")
    app.dependency_overrides[get_model_registry] = lambda: _FakeModelRegistry()

    client = TestClient(app)
    response = client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == "gpt-4o-mini"
    assert body[0]["provider"] == "openai"
    assert body[0]["model"] == "gpt-4o-mini"
    assert body[0]["input_cost"] == 0.15
    assert body[0]["output_cost"] == 0.60
    assert body[0]["context_window"] == 128000
    assert body[0]["max_output_tokens"] == 16384
    assert body[0]["supports_streaming"] is True
    assert body[0]["supports_tools"] is True
    assert body[0]["supports_json"] is True
    assert body[0]["supports_vision"] is False
    assert body[0]["benchmark_score"] == 0.82
    assert body[0]["average_latency_ms"] == 450
    assert body[0]["available"] is True
