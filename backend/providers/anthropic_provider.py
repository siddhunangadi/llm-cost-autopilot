from collections.abc import AsyncIterator

from anthropic import AnthropicError, AsyncAnthropic

from backend.providers.base import BaseProvider, ProviderError
from backend.services.cost_estimator import calculate_linear_cost
from backend.services.credential_store import ProviderCredential


class AnthropicProvider(BaseProvider):
    """Thin adapter over the Anthropic SDK, mirroring OpenAIProvider's
    shape. No retries, caching, logging policy, budgeting, or failover --
    those belong above this layer."""

    def __init__(
        self, credential: ProviderCredential, client: AsyncAnthropic | None = None,
    ) -> None:
        self._client = client or AsyncAnthropic(api_key=credential.api_key if credential else None)

    @property
    def name(self) -> str:
        return "anthropic"

    async def generate(self, prompt: str, model: str, **kwargs) -> str:
        try:
            response = await self._client.messages.create(
                model=model, max_tokens=kwargs.pop("max_tokens", 1024),
                messages=[{"role": "user", "content": prompt}],
            )
        except AnthropicError as exc:
            raise ProviderError(f"Anthropic generate failed: {exc}") from exc
        return "".join(block.text for block in response.content if hasattr(block, "text"))

    async def stream(self, prompt: str, model: str, **kwargs) -> AsyncIterator[str]:
        try:
            async with self._client.messages.stream(
                model=model, max_tokens=kwargs.pop("max_tokens", 1024),
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except AnthropicError as exc:
            raise ProviderError(f"Anthropic stream failed: {exc}") from exc

    async def health_check(self) -> bool:
        # Cheap connectivity probe (list models) rather than a completion
        # request, mirroring OpenAIProvider.health_check's approach.
        try:
            await self._client.models.list(limit=1)
            return True
        except Exception:
            return False

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def estimate_cost(
        self, input_tokens: int, output_tokens: int, input_cost: float, output_cost: float
    ) -> float:
        return calculate_linear_cost(input_tokens, output_tokens, input_cost, output_cost)
