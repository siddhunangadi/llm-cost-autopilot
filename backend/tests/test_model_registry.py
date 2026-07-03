import textwrap
from types import MappingProxyType

import pytest
import yaml as pyyaml
from pydantic import ValidationError

from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.events.bus import EventBus
from backend.providers.factory import ProviderFactory
from backend.providers.manager import ProviderManager
from backend.providers.mock_provider import MockProvider
from backend.providers.openai_provider import OpenAIProvider
from backend.services.cost_estimator import DefaultCostEstimator
from backend.services.credential_store import CredentialStore
from backend.services.model_registry import ModelRegistry

SAMPLE_YAML = textwrap.dedent("""
    models:
      - id: gpt-4o-mini
        provider: openai
        model: gpt-4o-mini
        pricing:
          input_cost: 0.15
          output_cost: 0.60
        limits:
          context_window: 128000
          max_output_tokens: 16384
        capabilities:
          supports_streaming: true
          supports_tools: true
          supports_json: true
          supports_vision: false
        metadata:
          benchmark_score: 0.82
          average_latency_ms: 450
""")

TWO_MODEL_YAML = textwrap.dedent("""
    models:
      - id: gpt-4o-mini
        provider: openai
        model: gpt-4o-mini
        pricing:
          input_cost: 0.15
          output_cost: 0.60
        limits:
          context_window: 128000
          max_output_tokens: 16384
        capabilities:
          supports_streaming: true
          supports_tools: true
          supports_json: true
          supports_vision: false
        metadata:
          benchmark_score: 0.82
          average_latency_ms: 450
      - id: gpt-4o
        provider: openai
        model: gpt-4o
        pricing:
          input_cost: 2.50
          output_cost: 10.00
        limits:
          context_window: 128000
          max_output_tokens: 16384
        capabilities:
          supports_streaming: true
          supports_tools: true
          supports_json: true
          supports_vision: true
        metadata:
          benchmark_score: 0.93
          average_latency_ms: 900
""")

DUPLICATE_ID_YAML = textwrap.dedent("""
    models:
      - id: gpt-4o-mini
        provider: openai
        model: gpt-4o-mini
        pricing: {input_cost: 0.15, output_cost: 0.60}
        limits: {context_window: 128000, max_output_tokens: 16384}
        capabilities: {supports_streaming: true, supports_tools: true, supports_json: true, supports_vision: false}
        metadata: {benchmark_score: 0.82, average_latency_ms: 450}
      - id: gpt-4o-mini
        provider: openai
        model: gpt-4o-mini
        pricing: {input_cost: 0.15, output_cost: 0.60}
        limits: {context_window: 128000, max_output_tokens: 16384}
        capabilities: {supports_streaming: true, supports_tools: true, supports_json: true, supports_vision: false}
        metadata: {benchmark_score: 0.82, average_latency_ms: 450}
""")

MISSING_PRICING_YAML = textwrap.dedent("""
    models:
      - id: broken-model
        provider: openai
        model: broken-model
        limits: {context_window: 128000, max_output_tokens: 16384}
        capabilities: {supports_streaming: true, supports_tools: true, supports_json: true, supports_vision: false}
        metadata: {benchmark_score: 0.82, average_latency_ms: 450}
""")

INVALID_YAML = "models:\n\t- id: broken\n"


def _make_registry(tmp_path, openai_key, yaml_text=SAMPLE_YAML):
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(yaml_text)

    settings = Settings(
        _env_file=None, database_url=f"sqlite:///{tmp_path}/test.db", openai_api_key=openai_key
    )
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    factory = ProviderFactory()
    factory.register("mock", MockProvider)
    factory.register("openai", OpenAIProvider)
    credential_store = CredentialStore(
        session_factory=session_factory, settings=settings,
        provider_names=("openai", "anthropic", "ollama"),
    )
    provider_manager = ProviderManager(factory, credential_store)

    return ModelRegistry(
        provider_manager=provider_manager,
        event_bus=EventBus(),
        cost_estimator=DefaultCostEstimator(),
        session_factory=session_factory,
        yaml_path=str(yaml_path),
    )


def test_reload_loads_models_into_cache(tmp_path):
    registry = _make_registry(tmp_path, openai_key="sk-test")
    registry.reload()

    models = registry.get_models()
    assert len(models) == 1
    assert models[0].id == "gpt-4o-mini"
    assert models[0].benchmark_score == 0.82


def test_get_available_models_respects_provider_key(tmp_path):
    registry = _make_registry(tmp_path, openai_key=None)
    registry.reload()

    assert registry.get_available_models() == []
    assert registry.get_models()[0].available is False


def test_get_model_unknown_raises(tmp_path):
    registry = _make_registry(tmp_path, openai_key="sk-test")
    registry.reload()

    with pytest.raises(KeyError):
        registry.get_model("nonexistent")


def test_get_provider_models(tmp_path):
    registry = _make_registry(tmp_path, openai_key="sk-test")
    registry.reload()

    assert len(registry.get_provider_models("openai")) == 1
    assert registry.get_provider_models("anthropic") == []


def test_cache_is_immutable_mapping_proxy_after_reload(tmp_path):
    registry = _make_registry(tmp_path, openai_key="sk-test")
    registry.reload()

    assert isinstance(registry._cache, MappingProxyType)
    with pytest.raises(TypeError):
        registry._cache["gpt-4o-mini"] = None


def test_reload_raises_on_duplicate_model_ids(tmp_path):
    registry = _make_registry(tmp_path, openai_key="sk-test", yaml_text=DUPLICATE_ID_YAML)

    with pytest.raises(ValueError):
        registry.reload()


def test_reload_raises_on_malformed_yaml(tmp_path):
    registry = _make_registry(tmp_path, openai_key="sk-test", yaml_text=INVALID_YAML)

    with pytest.raises(pyyaml.YAMLError):
        registry.reload()


def test_reload_raises_on_invalid_schema_missing_field(tmp_path):
    registry = _make_registry(tmp_path, openai_key="sk-test", yaml_text=MISSING_PRICING_YAML)

    with pytest.raises(ValidationError):
        registry.reload()


def test_reload_replaces_stale_cache_entries(tmp_path):
    registry = _make_registry(tmp_path, openai_key="sk-test", yaml_text=TWO_MODEL_YAML)
    registry.reload()
    assert {m.id for m in registry.get_models()} == {"gpt-4o-mini", "gpt-4o"}

    (tmp_path / "models.yaml").write_text(SAMPLE_YAML)
    registry.reload()

    assert {m.id for m in registry.get_models()} == {"gpt-4o-mini"}
    with pytest.raises(KeyError):
        registry.get_model("gpt-4o")


def test_failed_reload_does_not_corrupt_existing_cache(tmp_path):
    registry = _make_registry(tmp_path, openai_key="sk-test")
    registry.reload()
    assert len(registry.get_models()) == 1

    (tmp_path / "models.yaml").write_text(INVALID_YAML)
    with pytest.raises(pyyaml.YAMLError):
        registry.reload()

    assert len(registry.get_models()) == 1
    assert registry.get_model("gpt-4o-mini") is not None
