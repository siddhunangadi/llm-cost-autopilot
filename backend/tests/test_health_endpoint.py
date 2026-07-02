import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.dependencies import (
    get_app_start_time,
    get_app_version,
    get_model_registry,
    get_provider_executor,
    get_provider_manager,
    get_session_factory,
    get_settings,
)
from backend.api.routers.health import router as health_router
from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db


class _FakeProviderManager:
    def list_providers(self):
        return {"openai": "available", "anthropic": "disabled", "ollama": "disabled"}


class _FakeModelRegistry:
    def get_models(self):
        return [object(), object()]


class _FakeProviderExecutor:
    def circuit_states(self):
        return {
            "openai": {"state": "closed", "consecutive_failures": 0, "successes": 3, "failures": 0},
            "anthropic": {"state": "closed", "consecutive_failures": 0, "successes": 0, "failures": 0},
            "ollama": {"state": "closed", "consecutive_failures": 0, "successes": 0, "failures": 0},
        }


def test_health_endpoint_returns_expected_shape(tmp_path):
    app = FastAPI()
    app.include_router(health_router, prefix="/v1")

    settings = Settings(_env_file=None, environment="test", database_url=f"sqlite:///{tmp_path}/t.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_app_version] = lambda: "0.1.0"
    app.dependency_overrides[get_app_start_time] = lambda: time.time() - 10
    app.dependency_overrides[get_provider_manager] = lambda: _FakeProviderManager()
    app.dependency_overrides[get_model_registry] = lambda: _FakeModelRegistry()
    app.dependency_overrides[get_session_factory] = lambda: session_factory
    app.dependency_overrides[get_provider_executor] = lambda: _FakeProviderExecutor()

    client = TestClient(app)
    response = client.get("/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["version"] == "0.1.0"
    assert body["environment"] == "test"
    assert body["database"] == "healthy"
    assert body["providers"] == {
        "openai": "available",
        "anthropic": "disabled",
        "ollama": "disabled",
    }
    assert body["loaded_models"] == 2
    assert body["uptime_seconds"] >= 10


def test_health_endpoint_includes_circuit_states(tmp_path):
    app = FastAPI()
    app.include_router(health_router, prefix="/v1")

    settings = Settings(_env_file=None, environment="test", database_url=f"sqlite:///{tmp_path}/t2.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_app_version] = lambda: "0.5.0"
    app.dependency_overrides[get_app_start_time] = lambda: time.time() - 5
    app.dependency_overrides[get_provider_manager] = lambda: _FakeProviderManager()
    app.dependency_overrides[get_model_registry] = lambda: _FakeModelRegistry()
    app.dependency_overrides[get_session_factory] = lambda: session_factory
    app.dependency_overrides[get_provider_executor] = lambda: _FakeProviderExecutor()

    client = TestClient(app)
    response = client.get("/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["circuits"] == {
        "openai": {"state": "closed", "consecutive_failures": 0, "successes": 3, "failures": 0},
        "anthropic": {"state": "closed", "consecutive_failures": 0, "successes": 0, "failures": 0},
        "ollama": {"state": "closed", "consecutive_failures": 0, "successes": 0, "failures": 0},
    }
