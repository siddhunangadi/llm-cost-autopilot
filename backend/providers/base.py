from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class BaseProvider(ABC):
    """Common interface every LLM provider implementation must satisfy."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier for this provider (e.g. "openai", "mock").
        Callers must never infer a provider's identity from its class name
        or type -- this property is the single source of truth."""
        ...

    @abstractmethod
    async def generate(self, prompt: str, model: str, **kwargs) -> str: ...

    @abstractmethod
    async def stream(self, prompt: str, model: str, **kwargs) -> AsyncIterator[str]: ...

    @abstractmethod
    async def health_check(self) -> bool: ...

    @abstractmethod
    def count_tokens(self, text: str) -> int: ...

    @abstractmethod
    def estimate_cost(
        self, input_tokens: int, output_tokens: int, input_cost: float, output_cost: float
    ) -> float: ...
