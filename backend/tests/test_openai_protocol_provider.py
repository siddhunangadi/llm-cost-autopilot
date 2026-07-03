from backend.providers.openai_protocol_provider import OpenAIProtocolProvider
from backend.services.credential_store import ProviderCredential


class _FakeProvider(OpenAIProtocolProvider):
    _NAME = "fake"
    _BASE_URL = "https://fake.example.com/v1"


def test_name_returns_class_constant():
    provider = _FakeProvider(ProviderCredential("fake", "key", None))
    assert provider.name == "fake"


def test_base_url_is_passed_to_client():
    provider = _FakeProvider(ProviderCredential("fake", "key", None))
    assert str(provider._client.base_url) == "https://fake.example.com/v1/"


def test_none_base_url_uses_sdk_default():
    class _DefaultProvider(OpenAIProtocolProvider):
        _NAME = "default"
        _BASE_URL = None

    provider = _DefaultProvider(ProviderCredential("default", "key", None))
    assert str(provider._client.base_url) == "https://api.openai.com/v1/"
