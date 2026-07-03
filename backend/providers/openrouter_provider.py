from backend.providers.openai_protocol_provider import OpenAIProtocolProvider


class OpenRouterProvider(OpenAIProtocolProvider):
    """OpenRouter, via its OpenAI-compatible endpoint."""

    _NAME = "openrouter"
    _BASE_URL = "https://openrouter.ai/api/v1"
