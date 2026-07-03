from backend.providers.openai_protocol_provider import OpenAIProtocolProvider


class GroqProvider(OpenAIProtocolProvider):
    """Groq, via its OpenAI-compatible endpoint."""

    _NAME = "groq"
    _BASE_URL = "https://api.groq.com/openai/v1"
