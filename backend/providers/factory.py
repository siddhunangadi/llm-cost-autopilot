from backend.providers.base import BaseProvider
from backend.services.credential_store import ProviderCredential


class ProviderFactory:
    def __init__(self) -> None:
        self._registry: dict[str, type[BaseProvider]] = {}
        self._user_configurable: list[str] = []

    def register(
        self, name: str, provider_cls: type[BaseProvider], *, user_configurable: bool = True,
    ) -> None:
        self._registry[name] = provider_cls
        if user_configurable:
            self._user_configurable.append(name)

    def create(self, name: str, credential: ProviderCredential | None) -> BaseProvider:
        if name not in self._registry:
            raise KeyError(f"No provider registered under name '{name}'")
        return self._registry[name](credential)

    def registered_names(self) -> tuple[str, ...]:
        """Every provider name registered with user_configurable=True (the
        default), in registration order. The single source of truth for
        which providers are user-facing (configurable via Provider
        Configuration, listed on the dashboard, etc.) -- callers must
        never maintain their own copy of this set."""
        return tuple(self._user_configurable)
