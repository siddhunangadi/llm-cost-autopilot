from backend.providers.base import BaseProvider
from backend.providers.factory import ProviderFactory
from backend.services.credential_store import CredentialStore
from backend.telemetry.logging import get_logger, request_context

KNOWN_PROVIDER_NAMES = ("openai", "anthropic", "ollama")


class ProviderManager:
    def __init__(self, factory: ProviderFactory, credential_store: CredentialStore) -> None:
        self._logger = get_logger("providers")
        self._factory = factory
        self._credential_store = credential_store

        # "mock" is mandatory -- if it fails to construct there is no
        # sensible degraded mode, so the exception propagates and crashes
        # startup rather than being swallowed here.
        self._providers: dict[str, BaseProvider] = {"mock": factory.create("mock", None)}

        for name in KNOWN_PROVIDER_NAMES:
            credential = credential_store.get(name)
            if credential is not None:
                self._try_build(name, credential)

    def _try_build(self, name: str, credential) -> bool:
        try:
            self._providers[name] = self._factory.create(name, credential)
            return True
        except Exception:
            self._providers.pop(name, None)
            with request_context(provider=name):
                self._logger.exception("provider_initialization_failed")
            return False

    def reload_provider(self, name: str) -> bool:
        """Rebuilds exactly one provider from CredentialStore's current
        value for it, live -- no restart. Called after activation (a
        passing save/enable/disable/delete), never for validation."""
        credential = self._credential_store.get(name)
        if credential is None:
            self._providers.pop(name, None)
            return False
        return self._try_build(name, credential)

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
