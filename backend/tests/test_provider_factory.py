import pytest

from backend.config.settings import Settings
from backend.providers.factory import ProviderFactory
from backend.providers.mock_provider import MockProvider


def test_register_and_create_returns_instance():
    factory = ProviderFactory()
    factory.register("mock", MockProvider)

    settings = Settings(_env_file=None)
    provider = factory.create("mock", settings)

    assert isinstance(provider, MockProvider)


def test_create_unregistered_provider_raises():
    factory = ProviderFactory()
    settings = Settings(_env_file=None)

    with pytest.raises(KeyError):
        factory.create("unknown", settings)
