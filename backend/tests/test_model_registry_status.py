from types import MappingProxyType
from unittest.mock import AsyncMock

import pytest

from backend.tests.test_model_registry import _make_registry


async def test_refresh_marks_model_available_when_provider_healthy(tmp_path, mocker):
    registry = _make_registry(tmp_path, openai_key="sk-test")
    registry.reload()

    mocker.patch(
        "backend.providers.openai_provider.OpenAIProvider.health_check",
        new_callable=AsyncMock,
        return_value=True,
    )

    await registry.refresh_provider_status()

    assert registry.get_available_models()[0].available is True


async def test_refresh_marks_model_unavailable_when_provider_unhealthy(tmp_path, mocker):
    registry = _make_registry(tmp_path, openai_key="sk-test")
    registry.reload()

    mocker.patch(
        "backend.providers.openai_provider.OpenAIProvider.health_check",
        new_callable=AsyncMock,
        side_effect=RuntimeError("boom"),
    )

    await registry.refresh_provider_status()

    assert registry.get_models()[0].available is False


async def test_refresh_preserves_cache_immutability(tmp_path, mocker):
    registry = _make_registry(tmp_path, openai_key="sk-test")
    registry.reload()

    mocker.patch(
        "backend.providers.openai_provider.OpenAIProvider.health_check",
        new_callable=AsyncMock,
        return_value=True,
    )

    await registry.refresh_provider_status()

    assert isinstance(registry._cache, MappingProxyType)
    with pytest.raises(TypeError):
        registry._cache["gpt-4o-mini"] = None


def test_estimate_cost(tmp_path):
    registry = _make_registry(tmp_path, openai_key="sk-test")
    registry.reload()

    cost = registry.estimate_cost("gpt-4o-mini", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(0.15 + 0.60)
