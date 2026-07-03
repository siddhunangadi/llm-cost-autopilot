from backend.providers.openai_protocol_provider import OpenAIProtocolProvider


class OpenAIProvider(OpenAIProtocolProvider):
    """OpenAI, using the SDK's default base_url (api.openai.com)."""

    _NAME = "openai"
    _BASE_URL = None
