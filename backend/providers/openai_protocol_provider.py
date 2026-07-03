from collections.abc import AsyncIterator

from openai import AsyncOpenAI, OpenAIError

from backend.providers.base import BaseProvider, ProviderError
from backend.services.cost_estimator import calculate_linear_cost
from backend.services.credential_store import ProviderCredential


class OpenAIProtocolProvider(BaseProvider):
    """Shared adapter for any provider exposing an OpenAI-compatible
    chat-completions API. Subclasses declare `_NAME` and `_BASE_URL`;
    everything else is inherited. If a provider later needs capabilities
    beyond the OpenAI-compatible protocol, its subclass may override the
    relevant method(s) while still satisfying BaseProvider. No retries,
    caching, logging policy, budgeting, or failover -- those belong above
    this layer."""

    _NAME: str
    _BASE_URL: str | None = None

    def __init__(
        self, credential: ProviderCredential | None, client: AsyncOpenAI | None = None,
    ) -> None:
        self._client = client or AsyncOpenAI(
            api_key=credential.api_key if credential else None,
            base_url=self._BASE_URL,
        )

    @property
    def name(self) -> str:
        return self._NAME

    async def generate(self, prompt: str, model: str, **kwargs) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
        except OpenAIError as exc:
            raise ProviderError(f"{self._NAME} generate failed: {exc}") from exc
        return response.choices[0].message.content or ""

    async def stream(self, prompt: str, model: str, **kwargs) -> AsyncIterator[str]:
        try:
            stream = await self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except OpenAIError as exc:
            raise ProviderError(f"{self._NAME} stream failed: {exc}") from exc

    async def health_check(self) -> bool:
        # Cheap connectivity probe (list models) rather than a completion
        # request. A health probe reports status, it doesn't raise -- any
        # failure here just means "not available", not a bug to propagate.
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def estimate_cost(
        self, input_tokens: int, output_tokens: int, input_cost: float, output_cost: float
    ) -> float:
        return calculate_linear_cost(input_tokens, output_tokens, input_cost, output_cost)
