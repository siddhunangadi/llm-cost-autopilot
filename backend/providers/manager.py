from backend.config.settings import Settings
from backend.providers.base import BaseProvider
from backend.providers.factory import ProviderFactory
from backend.telemetry.logging import get_logger, request_context

KNOWN_PROVIDER_NAMES = ("openai", "anthropic", "ollama")


class ProviderManager:
    def __init__(self, factory: ProviderFactory, settings: Settings) -> None:
        self._logger = get_logger("providers")

        # "mock" is mandatory -- if it fails to construct there is no
        # sensible degraded mode, so the exception propagates and crashes
        # startup rather than being swallowed here.
        self._providers: dict[str, BaseProvider] = {"mock": factory.create("mock", settings)}

        if settings.openai_api_key:
            try:
                self._providers["openai"] = factory.create("openai", settings)
            except Exception:
                with request_context(provider="openai"):
                    self._logger.exception("provider_initialization_failed")

    def get_provider(self, name: str) -> BaseProvider:
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' is not available")
        return self._providers[name]

    def is_provider_available(self, name: str) -> bool:
        return name in self._providers

    def list_providers(self) -> dict[str, str]:
        return {
            name: ("available" if name in self._providers else "disabled")
            for name in KNOWN_PROVIDER_NAMES
        }
