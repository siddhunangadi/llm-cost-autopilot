from unittest.mock import AsyncMock

import httpx
import pytest

from backend.providers.base import ProviderError
from backend.providers.ollama_provider import OllamaProvider
from backend.services.credential_store import ProviderCredential


def _make_provider(base_url=None):
    credential = ProviderCredential("ollama", None, base_url)
    return OllamaProvider(credential)


def test_name_is_ollama():
    provider = _make_provider()
    assert provider.name == "ollama"


def test_defaults_to_localhost_when_no_base_url_given():
    provider = _make_provider(base_url=None)
    assert provider._base_url == "http://localhost:11434"


def test_uses_configured_base_url():
    provider = _make_provider(base_url="http://ollama.internal:11434")
    assert provider._base_url == "http://ollama.internal:11434"


async def test_generate_returns_response_field(mocker):
    provider = _make_provider()
    fake_response = mocker.Mock()
    fake_response.raise_for_status = mocker.Mock()
    fake_response.json = mocker.Mock(return_value={"response": "hello world"})
    mocker.patch.object(provider._client, "post", new_callable=AsyncMock, return_value=fake_response)

    result = await provider.generate("hi", model="llama3")
    assert result == "hello world"


async def test_generate_translates_http_errors_into_provider_error(mocker):
    provider = _make_provider()
    mocker.patch.object(
        provider._client, "post", new_callable=AsyncMock,
        side_effect=httpx.ConnectError("connection refused"),
    )

    with pytest.raises(ProviderError):
        await provider.generate("hi", model="llama3")


async def test_health_check_true_on_200(mocker):
    provider = _make_provider()
    fake_response = mocker.Mock(status_code=200)
    mocker.patch.object(provider._client, "get", new_callable=AsyncMock, return_value=fake_response)

    assert await provider.health_check() is True


async def test_health_check_false_on_connection_error(mocker):
    provider = _make_provider()
    mocker.patch.object(
        provider._client, "get", new_callable=AsyncMock,
        side_effect=httpx.ConnectError("connection refused"),
    )

    assert await provider.health_check() is False


def test_count_tokens_is_positive():
    provider = _make_provider()
    assert provider.count_tokens("abcdefgh") == 2


def test_estimate_cost_matches_linear_formula():
    provider = _make_provider()
    cost = provider.estimate_cost(1_000_000, 1_000_000, 1.0, 2.0)
    assert cost == pytest.approx(3.0)
