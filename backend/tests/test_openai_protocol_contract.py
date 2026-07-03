from unittest.mock import AsyncMock

import pytest
from openai import OpenAIError

from backend.providers.base import ProviderError
from backend.providers.gemini_provider import GeminiProvider
from backend.providers.groq_provider import GroqProvider
from backend.providers.mistral_provider import MistralProvider
from backend.providers.nvidia_nim_provider import NvidiaNimProvider
from backend.providers.openai_provider import OpenAIProvider
from backend.providers.openrouter_provider import OpenRouterProvider
from backend.services.credential_store import ProviderCredential

ALL_PROTOCOL_PROVIDERS = [
    OpenAIProvider, GeminiProvider, NvidiaNimProvider,
    OpenRouterProvider, GroqProvider, MistralProvider,
]


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


def _make_provider(provider_cls):
    return provider_cls(ProviderCredential(provider_cls._NAME, "test-key", None))


@pytest.mark.parametrize("provider_cls", ALL_PROTOCOL_PROVIDERS)
async def test_generate_returns_completion_content(provider_cls, mocker):
    provider = _make_provider(provider_cls)
    mocker.patch.object(
        provider._client.chat.completions, "create",
        new_callable=AsyncMock, return_value=_FakeCompletion("hello world"),
    )

    assert await provider.generate("hi", model="some-model") == "hello world"


@pytest.mark.parametrize("provider_cls", ALL_PROTOCOL_PROVIDERS)
async def test_generate_translates_sdk_errors_into_provider_error(provider_cls, mocker):
    provider = _make_provider(provider_cls)
    mocker.patch.object(
        provider._client.chat.completions, "create",
        new_callable=AsyncMock, side_effect=OpenAIError("boom"),
    )

    with pytest.raises(ProviderError):
        await provider.generate("hi", model="some-model")


@pytest.mark.parametrize("provider_cls", ALL_PROTOCOL_PROVIDERS)
async def test_stream_translates_sdk_errors_into_provider_error(provider_cls, mocker):
    provider = _make_provider(provider_cls)
    mocker.patch.object(
        provider._client.chat.completions, "create",
        new_callable=AsyncMock, side_effect=OpenAIError("boom"),
    )

    with pytest.raises(ProviderError):
        async for _ in provider.stream("hi", model="some-model"):
            pass


@pytest.mark.parametrize("provider_cls", ALL_PROTOCOL_PROVIDERS)
async def test_health_check_true_when_models_list_succeeds(provider_cls, mocker):
    provider = _make_provider(provider_cls)
    mocker.patch.object(
        provider._client.models, "list", new_callable=AsyncMock, return_value=None,
    )

    assert await provider.health_check() is True


@pytest.mark.parametrize("provider_cls", ALL_PROTOCOL_PROVIDERS)
async def test_health_check_false_when_models_list_raises(provider_cls, mocker):
    provider = _make_provider(provider_cls)
    mocker.patch.object(
        provider._client.models, "list",
        new_callable=AsyncMock, side_effect=RuntimeError("down"),
    )

    assert await provider.health_check() is False


@pytest.mark.parametrize("provider_cls", ALL_PROTOCOL_PROVIDERS)
def test_count_tokens_is_positive(provider_cls):
    provider = _make_provider(provider_cls)
    assert provider.count_tokens("abcdefgh") == 2


@pytest.mark.parametrize("provider_cls", ALL_PROTOCOL_PROVIDERS)
def test_estimate_cost_matches_linear_formula(provider_cls):
    provider = _make_provider(provider_cls)
    cost = provider.estimate_cost(1_000_000, 1_000_000, 1.0, 2.0)
    assert cost == pytest.approx(3.0)


@pytest.mark.parametrize("provider_cls", ALL_PROTOCOL_PROVIDERS)
def test_base_url_reaches_the_constructed_client(provider_cls):
    provider = _make_provider(provider_cls)
    expected = provider_cls._BASE_URL or "https://api.openai.com/v1"
    assert str(provider._client.base_url).rstrip("/") == expected.rstrip("/")
