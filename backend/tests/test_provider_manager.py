import io
import json
import logging

import pytest

from cryptography.fernet import Fernet

from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.providers.base import BaseProvider
from backend.providers.factory import ProviderFactory
from backend.providers.manager import ProviderManager
from backend.providers.mock_provider import MockProvider
from backend.providers.openai_provider import OpenAIProvider
from backend.services.credential_store import CredentialStore
from backend.telemetry.logging import JsonFormatter


def _make_factory():
    factory = ProviderFactory()
    factory.register("mock", MockProvider)
    factory.register("openai", OpenAIProvider)
    return factory


def _make_credential_store(tmp_path, **settings_kwargs):
    settings = Settings(
        _env_file=None, database_url=f"sqlite:///{tmp_path}/test.db",
        provider_credential_encryption_key=Fernet.generate_key().decode(),
        **settings_kwargs,
    )
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)
    return CredentialStore(session_factory=session_factory, settings=settings)


class _BrokenProvider(BaseProvider):
    def __init__(self, credential):
        raise RuntimeError("bad config")

    @property
    def name(self) -> str:
        return "broken"

    async def generate(self, prompt, model, **kwargs):
        return ""

    async def stream(self, prompt, model, **kwargs):
        yield ""

    async def health_check(self):
        return False

    def count_tokens(self, text):
        return 0

    def estimate_cost(self, input_tokens, output_tokens, input_cost, output_cost):
        return 0.0


def test_mock_provider_always_available(tmp_path):
    credential_store = _make_credential_store(tmp_path)
    manager = ProviderManager(_make_factory(), credential_store)

    assert manager.is_provider_available("mock") is True
    assert isinstance(manager.get_provider("mock"), MockProvider)


def test_openai_disabled_without_key(tmp_path):
    credential_store = _make_credential_store(tmp_path, openai_api_key=None)
    manager = ProviderManager(_make_factory(), credential_store)

    assert manager.is_provider_available("openai") is False
    with pytest.raises(KeyError):
        manager.get_provider("openai")


def test_openai_available_with_key(tmp_path):
    credential_store = _make_credential_store(tmp_path, openai_api_key="sk-test")
    manager = ProviderManager(_make_factory(), credential_store)

    assert manager.is_provider_available("openai") is True
    assert isinstance(manager.get_provider("openai"), OpenAIProvider)


def test_list_providers_covers_known_providers(tmp_path):
    credential_store = _make_credential_store(tmp_path, openai_api_key="sk-test")
    manager = ProviderManager(_make_factory(), credential_store)

    assert manager.list_providers() == {
        "openai": "available",
        "anthropic": "disabled",
        "ollama": "disabled",
    }


def test_optional_provider_initialization_failure_is_recorded_as_unavailable(tmp_path):
    factory = ProviderFactory()
    factory.register("mock", MockProvider)
    factory.register("openai", _BrokenProvider)

    credential_store = _make_credential_store(tmp_path, openai_api_key="sk-test")
    manager = ProviderManager(factory, credential_store)

    assert manager.is_provider_available("mock") is True
    assert manager.is_provider_available("openai") is False
    assert manager.list_providers()["openai"] == "disabled"


def test_optional_provider_initialization_failure_is_logged(tmp_path):
    factory = ProviderFactory()
    factory.register("mock", MockProvider)
    factory.register("openai", _BrokenProvider)

    logger = logging.getLogger("providers")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter(service="llm-cost-autopilot", environment="test"))
    logger.addHandler(handler)
    logger.setLevel(logging.ERROR)
    logger.propagate = False

    credential_store = _make_credential_store(tmp_path, openai_api_key="sk-test")
    ProviderManager(factory, credential_store)

    lines = [line for line in stream.getvalue().strip().splitlines() if line]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["message"] == "provider_initialization_failed"
    assert record["provider"] == "openai"
    assert record["level"] == "ERROR"

    logger.removeHandler(handler)


def test_mandatory_mock_provider_initialization_failure_is_not_swallowed(tmp_path):
    factory = ProviderFactory()
    factory.register("mock", _BrokenProvider)

    credential_store = _make_credential_store(tmp_path)

    with pytest.raises(RuntimeError):
        ProviderManager(factory, credential_store)


def test_reload_provider_activates_newly_saved_credential(tmp_path):
    credential_store = _make_credential_store(tmp_path)
    manager = ProviderManager(_make_factory(), credential_store)
    assert manager.is_provider_available("openai") is False

    credential_store.save("openai", api_key="sk-new", base_url=None)
    result = manager.reload_provider("openai")

    assert result is True
    assert manager.is_provider_available("openai") is True


def test_reload_provider_unregisters_when_credential_disabled(tmp_path):
    credential_store = _make_credential_store(tmp_path)
    credential_store.save("openai", api_key="sk-temp", base_url=None)
    manager = ProviderManager(_make_factory(), credential_store)
    assert manager.is_provider_available("openai") is True

    credential_store.set_enabled("openai", False)
    result = manager.reload_provider("openai")

    assert result is False
    assert manager.is_provider_available("openai") is False


def test_reload_provider_only_touches_the_named_provider(tmp_path):
    credential_store = _make_credential_store(tmp_path, openai_api_key="sk-env")
    manager = ProviderManager(_make_factory(), credential_store)
    mock_before = manager.get_provider("mock")

    credential_store.save("openai", api_key="sk-new", base_url=None)
    manager.reload_provider("openai")

    assert manager.get_provider("mock") is mock_before
