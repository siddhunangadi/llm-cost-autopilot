import io
import json
import logging

import pytest

from backend.config.settings import Settings
from backend.providers.base import BaseProvider
from backend.providers.factory import ProviderFactory
from backend.providers.manager import ProviderManager
from backend.providers.mock_provider import MockProvider
from backend.providers.openai_provider import OpenAIProvider
from backend.telemetry.logging import JsonFormatter


def _make_factory():
    factory = ProviderFactory()
    factory.register("mock", MockProvider)
    factory.register("openai", OpenAIProvider)
    return factory


class _BrokenProvider(BaseProvider):
    def __init__(self, settings):
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


def test_mock_provider_always_available():
    settings = Settings(_env_file=None)
    manager = ProviderManager(_make_factory(), settings)

    assert manager.is_provider_available("mock") is True
    assert isinstance(manager.get_provider("mock"), MockProvider)


def test_openai_disabled_without_key():
    settings = Settings(_env_file=None, openai_api_key=None)
    manager = ProviderManager(_make_factory(), settings)

    assert manager.is_provider_available("openai") is False
    with pytest.raises(KeyError):
        manager.get_provider("openai")


def test_openai_available_with_key():
    settings = Settings(_env_file=None, openai_api_key="sk-test")
    manager = ProviderManager(_make_factory(), settings)

    assert manager.is_provider_available("openai") is True
    assert isinstance(manager.get_provider("openai"), OpenAIProvider)


def test_list_providers_covers_known_providers():
    settings = Settings(_env_file=None, openai_api_key="sk-test")
    manager = ProviderManager(_make_factory(), settings)

    assert manager.list_providers() == {
        "openai": "available",
        "anthropic": "disabled",
        "ollama": "disabled",
    }


def test_optional_provider_initialization_failure_is_recorded_as_unavailable():
    factory = ProviderFactory()
    factory.register("mock", MockProvider)
    factory.register("openai", _BrokenProvider)

    settings = Settings(_env_file=None, openai_api_key="sk-test")
    manager = ProviderManager(factory, settings)

    assert manager.is_provider_available("mock") is True
    assert manager.is_provider_available("openai") is False
    assert manager.list_providers()["openai"] == "disabled"


def test_optional_provider_initialization_failure_is_logged():
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

    settings = Settings(_env_file=None, openai_api_key="sk-test")
    ProviderManager(factory, settings)

    lines = [line for line in stream.getvalue().strip().splitlines() if line]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["message"] == "provider_initialization_failed"
    assert record["provider"] == "openai"
    assert record["level"] == "ERROR"

    logger.removeHandler(handler)


def test_mandatory_mock_provider_initialization_failure_is_not_swallowed():
    factory = ProviderFactory()
    factory.register("mock", _BrokenProvider)

    settings = Settings(_env_file=None)

    with pytest.raises(RuntimeError):
        ProviderManager(factory, settings)
