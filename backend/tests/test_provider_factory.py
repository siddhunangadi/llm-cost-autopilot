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


def test_registered_names_excludes_non_user_configurable():
    factory = ProviderFactory()
    factory.register("mock", MockProvider, user_configurable=False)
    factory.register("openai_alias", MockProvider)

    assert factory.registered_names() == ("openai_alias",)


def test_registered_names_preserves_registration_order():
    factory = ProviderFactory()
    factory.register("b", MockProvider)
    factory.register("a", MockProvider)

    assert factory.registered_names() == ("b", "a")


def test_registered_names_defaults_to_user_configurable_true():
    factory = ProviderFactory()
    factory.register("mock", MockProvider)

    assert factory.registered_names() == ("mock",)
