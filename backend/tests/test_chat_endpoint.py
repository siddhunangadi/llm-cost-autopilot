from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.dependencies import get_chat_service
from backend.api.routers.chat import router as chat_router
from backend.chat.service import ChatResult
from backend.classifier.complexity_classifier import ComplexityTier
from backend.providers.base import ProviderError
from backend.providers.circuit_breaker import CircuitState
from backend.providers.executor import CircuitOpenError
from backend.routing.engine import NoEligibleModelError, RoutingDecision


class _FakeChatService:
    def __init__(self, result=None, exception=None):
        self._result = result
        self._exception = exception

    async def chat(self, prompt, strategy="balanced", background_tasks=None):
        if self._exception:
            raise self._exception
        return self._result


def _sample_result() -> ChatResult:
    return ChatResult(
        request_id="req-1",
        response="Here are three fruits: apple, banana, cherry.",
        routing=RoutingDecision(
            selected_model="mock-model", strategy="balanced", complexity=ComplexityTier.SIMPLE,
            confidence=0.66, estimated_cost=0.001, estimated_latency_ms=450.0,
            reasoning=[
                "Classified as simple.",
                "Strategy 'balanced' evaluated 1 eligible model(s).",
                "Selected 'mock-model'.",
            ],
        ),
    )


def test_chat_endpoint_returns_result():
    app = FastAPI()
    app.include_router(chat_router, prefix="/v1")
    app.dependency_overrides[get_chat_service] = lambda: _FakeChatService(result=_sample_result())

    client = TestClient(app)
    response = client.post("/v1/chat", json={"prompt": "List three fruits.", "strategy": "balanced"})

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "req-1"
    assert body["routing"]["selected_model"] == "mock-model"


def test_chat_endpoint_defaults_strategy_to_balanced():
    app = FastAPI()
    app.include_router(chat_router, prefix="/v1")
    app.dependency_overrides[get_chat_service] = lambda: _FakeChatService(result=_sample_result())

    client = TestClient(app)
    response = client.post("/v1/chat", json={"prompt": "List three fruits."})

    assert response.status_code == 200


def test_chat_endpoint_returns_503_for_no_eligible_model():
    app = FastAPI()
    app.include_router(chat_router, prefix="/v1")
    app.dependency_overrides[get_chat_service] = lambda: _FakeChatService(
        exception=NoEligibleModelError("no models available")
    )

    client = TestClient(app)
    response = client.post("/v1/chat", json={"prompt": "Hello."})

    assert response.status_code == 503


def test_chat_endpoint_returns_502_for_provider_error():
    app = FastAPI()
    app.include_router(chat_router, prefix="/v1")
    app.dependency_overrides[get_chat_service] = lambda: _FakeChatService(
        exception=ProviderError("upstream failure")
    )

    client = TestClient(app)
    response = client.post("/v1/chat", json={"prompt": "Hello."})

    assert response.status_code == 502


def test_chat_endpoint_returns_503_for_circuit_open():
    app = FastAPI()
    app.include_router(chat_router, prefix="/v1")
    exc = CircuitOpenError(
        provider="openai",
        state=CircuitState.OPEN,
        consecutive_failures=5,
        retry_after_seconds=12.4,
    )
    app.dependency_overrides[get_chat_service] = lambda: _FakeChatService(exception=exc)

    client = TestClient(app)
    response = client.post("/v1/chat", json={"prompt": "Hello."})

    assert response.status_code == 503
    assert response.headers["Retry-After"] == str(round(exc.retry_after_seconds))
