from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import TypeVar

from backend.providers.base import ProviderError

T = TypeVar("T")


class BaseRetryPolicy(ABC):
    @abstractmethod
    async def execute(self, operation: Callable[[], Awaitable[T]]) -> T: ...


class ExponentialBackoffRetryPolicy(BaseRetryPolicy):
    """Retries any zero-argument async callable on ProviderError. Has no
    knowledge of providers, circuits, or events -- it only knows about
    its own attempt budget. Callers that need retry outcomes recorded
    anywhere (metrics, circuit breakers) must do so themselves; this
    class never emits anything."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 0.2,
        multiplier: float = 2.0,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        import asyncio

        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._multiplier = multiplier
        self._sleep = sleep if sleep is not None else asyncio.sleep

    async def execute(self, operation: Callable[[], Awaitable[T]]) -> T:
        last_exc: ProviderError | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return await operation()
            except ProviderError as exc:
                last_exc = exc
                if attempt < self._max_attempts:
                    await self._sleep(self._base_delay * (self._multiplier ** (attempt - 1)))
        assert last_exc is not None
        raise last_exc
