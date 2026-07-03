from unittest.mock import AsyncMock

import pytest
from anthropic import AnthropicError

from backend.providers.anthropic_provider import AnthropicProvider
from backend.providers.base import ProviderError
from backend.services.credential_store import ProviderCredential


class _FakeTextBlock:
    def __init__(self, text):
        self.text = text


class _FakeMessage:
    def __init__(self, content):
        self.content = content


def _make_provider():
    credential = ProviderCredential("anthropic", "sk-ant-test", None)
    return AnthropicProvider(credential)


def test_name_is_anthropic():
    provider = _make_provider()
    assert provider.name == "anthropic"


async def test_generate_returns_joined_text_blocks(mocker):
    provider = _make_provider()
    mocker.patch.object(
        provider._client.messages, "create", new_callable=AsyncMock,
        return_value=_FakeMessage([_FakeTextBlock("hello "), _FakeTextBlock("world")]),
    )

    result = await provider.generate("hi", model="claude-3-5-sonnet-latest")
    assert result == "hello world"


async def test_generate_translates_anthropic_errors_into_provider_error(mocker):
    provider = _make_provider()
    mocker.patch.object(
        provider._client.messages, "create", new_callable=AsyncMock,
        side_effect=AnthropicError("boom"),
    )

    with pytest.raises(ProviderError):
        await provider.generate("hi", model="claude-3-5-sonnet-latest")


async def test_health_check_true_when_models_list_succeeds(mocker):
    provider = _make_provider()
    mocker.patch.object(
        provider._client.models, "list", new_callable=AsyncMock, return_value=None
    )

    assert await provider.health_check() is True


async def test_health_check_false_when_models_list_raises(mocker):
    provider = _make_provider()
    mocker.patch.object(
        provider._client.models, "list", new_callable=AsyncMock,
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
