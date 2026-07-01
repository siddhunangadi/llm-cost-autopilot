import pytest

from backend.providers.base import BaseProvider, ProviderError


def test_provider_error_is_an_exception():
    assert issubclass(ProviderError, Exception)


def test_base_provider_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseProvider()


class _CompleteProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "complete"

    async def generate(self, prompt, model, **kwargs):
        return "ok"

    async def stream(self, prompt, model, **kwargs):
        yield "ok"

    async def health_check(self):
        return True

    def count_tokens(self, text):
        return 1

    def estimate_cost(self, input_tokens, output_tokens, input_cost, output_cost):
        return 0.0


def test_complete_subclass_can_be_instantiated():
    provider = _CompleteProvider()
    assert isinstance(provider, BaseProvider)
    assert provider.name == "complete"


class _MissingNameProvider(BaseProvider):
    async def generate(self, prompt, model, **kwargs):
        return "ok"

    async def stream(self, prompt, model, **kwargs):
        yield "ok"

    async def health_check(self):
        return True

    def count_tokens(self, text):
        return 1

    def estimate_cost(self, input_tokens, output_tokens, input_cost, output_cost):
        return 0.0


def test_subclass_missing_name_property_cannot_be_instantiated():
    with pytest.raises(TypeError):
        _MissingNameProvider()
