import json
from collections.abc import AsyncIterator

import httpx

from backend.providers.base import BaseProvider, ProviderError
from backend.services.cost_estimator import calculate_linear_cost
from backend.services.credential_store import ProviderCredential


class OllamaProvider(BaseProvider):
    """Thin adapter over a local Ollama server's HTTP API. No API key --
    identified by base_url only. No retries, caching, logging policy,
    budgeting, or failover -- those belong above this layer."""

    def __init__(
        self, credential: ProviderCredential | None, client: httpx.AsyncClient | None = None,
    ) -> None:
        base_url = credential.base_url if credential else None
        self._base_url = (base_url or "http://localhost:11434").rstrip("/")
        self._client = client or httpx.AsyncClient(base_url=self._base_url, timeout=30.0)

    @property
    def name(self) -> str:
        return "ollama"

    async def generate(self, prompt: str, model: str, **kwargs) -> str:
        try:
            response = await self._client.post(
                "/api/generate", json={"model": model, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama generate failed: {exc}") from exc
        return response.json().get("response", "")

    async def stream(self, prompt: str, model: str, **kwargs) -> AsyncIterator[str]:
        try:
            async with self._client.stream(
                "POST", "/api/generate", json={"model": model, "prompt": prompt, "stream": True},
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if chunk.get("response"):
                        yield chunk["response"]
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama stream failed: {exc}") from exc

    async def health_check(self) -> bool:
        try:
            response = await self._client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def estimate_cost(
        self, input_tokens: int, output_tokens: int, input_cost: float, output_cost: float
    ) -> float:
        return calculate_linear_cost(input_tokens, output_tokens, input_cost, output_cost)
