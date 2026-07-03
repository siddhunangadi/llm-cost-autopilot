from backend.providers.openai_protocol_provider import OpenAIProtocolProvider


class MistralProvider(OpenAIProtocolProvider):
    """Mistral AI, via its OpenAI-compatible endpoint."""

    _NAME = "mistral"
    _BASE_URL = "https://api.mistral.ai/v1"
