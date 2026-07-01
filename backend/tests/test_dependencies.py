from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.dependencies import (
    AppStartTimeDep,
    AppVersionDep,
    EventBusDep,
    ModelRegistryDep,
    ProviderManagerDep,
    SessionFactoryDep,
    SettingsDep,
)
from backend.config.settings import Settings
from backend.events.bus import EventBus


def test_dependencies_read_from_app_state():
    app = FastAPI()
    app.state.settings = Settings(_env_file=None, environment="test")
    app.state.event_bus = EventBus()
    app.state.provider_manager = "fake-provider-manager"
    app.state.model_registry = "fake-model-registry"
    app.state.session_factory = "fake-session-factory"
    app.state.version = "0.1.0"
    app.state.start_time = 123.0

    @app.get("/probe")
    def probe(
        settings: SettingsDep,
        event_bus: EventBusDep,
        provider_manager: ProviderManagerDep,
        model_registry: ModelRegistryDep,
        session_factory: SessionFactoryDep,
        version: AppVersionDep,
        start_time: AppStartTimeDep,
    ):
        return {
            "environment": settings.environment,
            "event_bus_type": type(event_bus).__name__,
            "provider_manager": provider_manager,
            "model_registry": model_registry,
            "session_factory": session_factory,
            "version": version,
            "start_time": start_time,
        }

    client = TestClient(app)
    response = client.get("/probe")

    assert response.status_code == 200
    body = response.json()
    assert body["environment"] == "test"
    assert body["event_bus_type"] == "EventBus"
    assert body["provider_manager"] == "fake-provider-manager"
    assert body["model_registry"] == "fake-model-registry"
    assert body["session_factory"] == "fake-session-factory"
    assert body["version"] == "0.1.0"
    assert body["start_time"] == 123.0
