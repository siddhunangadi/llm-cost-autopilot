from backend.providers.openai_protocol_provider import OpenAIProtocolProvider


class GeminiProvider(OpenAIProtocolProvider):
    """Google Gemini, via its OpenAI-compatible endpoint."""

    _NAME = "gemini"
    _BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
