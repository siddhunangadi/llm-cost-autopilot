from backend.config.settings import Settings
from backend.providers.base import BaseProvider


class ProviderFactory:
    def __init__(self) -> None:
        self._registry: dict[str, type[BaseProvider]] = {}

    def register(self, name: str, provider_cls: type[BaseProvider]) -> None:
        self._registry[name] = provider_cls

    def create(self, name: str, settings: Settings) -> BaseProvider:
        if name not in self._registry:
            raise KeyError(f"No provider registered under name '{name}'")
        return self._registry[name](settings)
