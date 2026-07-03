import pytest

from backend.providers.gemini_provider import GeminiProvider
from backend.providers.groq_provider import GroqProvider
from backend.providers.mistral_provider import MistralProvider
from backend.providers.nvidia_nim_provider import NvidiaNimProvider
from backend.providers.openai_protocol_provider import OpenAIProtocolProvider
from backend.providers.openrouter_provider import OpenRouterProvider
from backend.services.credential_store import ProviderCredential

_CASES = [
    (GeminiProvider, "gemini", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    (NvidiaNimProvider, "nvidia_nim", "https://integrate.api.nvidia.com/v1/"),
    (OpenRouterProvider, "openrouter", "https://openrouter.ai/api/v1/"),
    (GroqProvider, "groq", "https://api.groq.com/openai/v1/"),
    (MistralProvider, "mistral", "https://api.mistral.ai/v1/"),
]


@pytest.mark.parametrize("provider_cls,expected_name,expected_base_url", _CASES)
def test_provider_is_openai_protocol_subclass(provider_cls, expected_name, expected_base_url):
    assert issubclass(provider_cls, OpenAIProtocolProvider)


@pytest.mark.parametrize("provider_cls,expected_name,expected_base_url", _CASES)
def test_provider_name(provider_cls, expected_name, expected_base_url):
    provider = provider_cls(ProviderCredential(expected_name, "key", None))
    assert provider.name == expected_name


@pytest.mark.parametrize("provider_cls,expected_name,expected_base_url", _CASES)
def test_provider_base_url(provider_cls, expected_name, expected_base_url):
    provider = provider_cls(ProviderCredential(expected_name, "key", None))
    assert str(provider._client.base_url) == expected_base_url
