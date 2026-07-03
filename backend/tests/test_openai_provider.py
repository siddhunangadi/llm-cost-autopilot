from unittest.mock import AsyncMock

import pytest
from openai import OpenAIError

from backend.providers.base import ProviderError
from backend.providers.openai_provider import OpenAIProvider
from backend.services.credential_store import ProviderCredential


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


def _make_provider():
    credential = ProviderCredential("openai", "sk-test", None)
    return OpenAIProvider(credential)


def test_name_is_openai():
    provider = _make_provider()
    assert provider.name == "openai"


async def test_generate_returns_completion_content(mocker):
    provider = _make_provider()
    mocker.patch.object(
        provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=_FakeCompletion("hello world"),
    )

    result = await provider.generate("hi", model="gpt-4o-mini")
    assert result == "hello world"


async def test_generate_translates_openai_errors_into_provider_error(mocker):
    provider = _make_provider()
    mocker.patch.object(
        provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        side_effect=OpenAIError("boom"),
    )

    with pytest.raises(ProviderError):
        await provider.generate("hi", model="gpt-4o-mini")


async def test_generate_error_preserves_original_exception_as_cause(mocker):
    provider = _make_provider()
    original = OpenAIError("boom")
    mocker.patch.object(
        provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        side_effect=original,
    )

    with pytest.raises(ProviderError) as exc_info:
        await provider.generate("hi", model="gpt-4o-mini")

    assert exc_info.value.__cause__ is original


async def test_generate_does_not_swallow_non_openai_errors(mocker):
    provider = _make_provider()
    mocker.patch.object(
        provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        side_effect=ValueError("unexpected bug"),
    )

    with pytest.raises(ValueError):
        await provider.generate("hi", model="gpt-4o-mini")


async def test_stream_translates_openai_errors_into_provider_error(mocker):
    provider = _make_provider()
    mocker.patch.object(
        provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        side_effect=OpenAIError("boom"),
    )

    with pytest.raises(ProviderError):
        async for _ in provider.stream("hi", model="gpt-4o-mini"):
            pass


async def test_health_check_true_when_models_list_succeeds(mocker):
    provider = _make_provider()
    mocker.patch.object(
        provider._client.models, "list", new_callable=AsyncMock, return_value=None
    )

    assert await provider.health_check() is True


async def test_health_check_false_when_models_list_raises(mocker):
    provider = _make_provider()
    mocker.patch.object(
        provider._client.models,
        "list",
        new_callable=AsyncMock,
        side_effect=RuntimeError("down"),
    )

    assert await provider.health_check() is False


def test_count_tokens_is_positive():
    provider = _make_provider()
    assert provider.count_tokens("abcdefgh") == 2


def test_estimate_cost_matches_linear_formula():
    provider = _make_provider()
    cost = provider.estimate_cost(1_000_000, 1_000_000, 1.0, 2.0)
    assert cost == pytest.approx(3.0)
