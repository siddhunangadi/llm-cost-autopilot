import pytest

from backend.providers.factory import ProviderFactory
from backend.providers.mock_provider import MockProvider
from backend.services.credential_store import ProviderCredential


def test_register_and_create_returns_instance():
    factory = ProviderFactory()
    factory.register("mock", MockProvider)

    provider = factory.create("mock", None)

    assert isinstance(provider, MockProvider)


def test_create_unregistered_provider_raises():
    factory = ProviderFactory()
    credential = ProviderCredential("unknown", None, None)

    with pytest.raises(KeyError):
        factory.create("unknown", credential)
