from backend.providers.openai_protocol_provider import OpenAIProtocolProvider


class NvidiaNimProvider(OpenAIProtocolProvider):
    """NVIDIA NIM, via its OpenAI-compatible endpoint."""

    _NAME = "nvidia_nim"
    _BASE_URL = "https://integrate.api.nvidia.com/v1"
