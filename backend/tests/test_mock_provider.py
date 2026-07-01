import time

import pytest

from backend.providers.mock_provider import MockProvider


def test_name_is_mock():
    provider = MockProvider()
    assert provider.name == "mock"


async def test_generate_is_deterministic_for_same_prompt():
    provider = MockProvider()
    first = await provider.generate("hello", model="mock-1")
    second = await provider.generate("hello", model="mock-1")
    assert first == second
    assert "mock-1" in first


async def test_generate_differs_for_different_prompts():
    provider = MockProvider()
    a = await provider.generate("hello", model="mock-1")
    b = await provider.generate("goodbye", model="mock-1")
    assert a != b


async def test_stream_yields_words_matching_generate():
    provider = MockProvider()
    full = await provider.generate("hello world", model="mock-1")

    chunks = [chunk async for chunk in provider.stream("hello world", model="mock-1")]
    assert "".join(chunks).strip() == full


async def test_health_check_is_always_true():
    provider = MockProvider()
    assert await provider.health_check() is True


def test_count_tokens_is_positive():
    provider = MockProvider()
    assert provider.count_tokens("hello world") > 0


def test_default_count_tokens_uses_length_heuristic():
    provider = MockProvider()
    assert provider.count_tokens("hello world") == 2


def test_estimate_cost_matches_linear_formula():
    provider = MockProvider()
    cost = provider.estimate_cost(1_000_000, 1_000_000, 1.0, 2.0)
    assert cost == pytest.approx(3.0)


async def test_configured_response_overrides_default_generation():
    provider = MockProvider(response="Hello")
    result = await provider.generate("anything at all", model="mock-1")
    assert result == "Hello"


async def test_configured_response_also_overrides_stream():
    provider = MockProvider(response="Hello there")
    chunks = [chunk async for chunk in provider.stream("anything", model="mock-1")]
    assert "".join(chunks).strip() == "Hello there"


async def test_configured_latency_ms_delays_generate():
    provider = MockProvider(latency_ms=20)
    start = time.monotonic()
    await provider.generate("hi", model="mock-1")
    elapsed_ms = (time.monotonic() - start) * 1000
    assert elapsed_ms >= 10


def test_configured_input_tokens_overrides_count_tokens():
    provider = MockProvider(input_tokens=5)
    assert provider.count_tokens("literally any text, long or short") == 5


def test_configured_output_tokens_is_exposed_as_attribute():
    provider = MockProvider(output_tokens=7)
    assert provider.output_tokens == 7


def test_output_tokens_defaults_to_none_when_not_configured():
    provider = MockProvider()
    assert provider.output_tokens is None
