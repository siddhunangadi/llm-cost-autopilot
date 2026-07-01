import asyncio
import hashlib
from collections.abc import AsyncIterator

from backend.config.settings import Settings
from backend.providers.base import BaseProvider
from backend.services.cost_estimator import calculate_linear_cost


class MockProvider(BaseProvider):
    """Deterministic provider with no network calls. Used in tests and as
    a dev fallback when no real provider key is configured.

    Configurable for tests that need specific behavior:
    - response: fixed text returned by generate()/stream() instead of the
      default deterministic hash-based response.
    - latency_ms: simulated delay before generate()/stream() return.
    - input_tokens: fixed value returned by count_tokens() for any text,
      instead of the default length-based heuristic.
    - output_tokens: exposed as a read-only attribute for tests that need
      a stable simulated output token count without depending on
      generated response length. Not consumed by estimate_cost(), which
      always uses its own explicit arguments per the BaseProvider contract.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        response: str | None = None,
        latency_ms: float = 0,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        self._response = response
        self._latency_ms = latency_ms
        self._input_tokens = input_tokens
        self.output_tokens = output_tokens

    @property
    def name(self) -> str:
        return "mock"

    async def generate(self, prompt: str, model: str, **kwargs) -> str:
        if self._latency_ms:
            await asyncio.sleep(self._latency_ms / 1000)
        if self._response is not None:
            return self._response
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        return f"[mock:{model}] response-{digest}"

    async def stream(self, prompt: str, model: str, **kwargs) -> AsyncIterator[str]:
        response = await self.generate(prompt, model, **kwargs)
        words = response.split(" ")
        for index, word in enumerate(words):
            yield word if index == len(words) - 1 else word + " "

    async def health_check(self) -> bool:
        return True

    def count_tokens(self, text: str) -> int:
        if self._input_tokens is not None:
            return self._input_tokens
        return max(1, len(text) // 4)

    def estimate_cost(
        self, input_tokens: int, output_tokens: int, input_cost: float, output_cost: float
    ) -> float:
        return calculate_linear_cost(input_tokens, output_tokens, input_cost, output_cost)
