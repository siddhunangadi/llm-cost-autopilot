import textwrap

import pytest

from backend.analysis.prompt_analyzer import PromptAnalyzer
from backend.classifier.complexity_classifier import HeuristicComplexityClassifier
from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.events.bus import EventBus
from backend.providers.factory import ProviderFactory
from backend.providers.manager import ProviderManager
from backend.providers.mock_provider import MockProvider
from backend.providers.openai_provider import OpenAIProvider
from backend.routing.config import BalancedStrategyWeights, ClassifierPolicy, EligibilityPolicy
from backend.routing.engine import NoEligibleModelError, RoutingEngine
from backend.routing.explanation import ExplanationGenerator
from backend.routing.policy import RoutingPolicy
from backend.routing.strategies import BalancedStrategy, CostOptimizedStrategy
from backend.services.cost_estimator import DefaultCostEstimator
from backend.services.credential_store import CredentialStore
from backend.services.model_registry import ModelRegistry

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


def _routing_policy() -> RoutingPolicy:
    return RoutingPolicy({
        "simple": EligibilityPolicy(min_benchmark_score=0.0),
        "medium": EligibilityPolicy(min_benchmark_score=0.75),
        "complex": EligibilityPolicy(min_benchmark_score=0.90),
    })


def _make_engine(tmp_path, openai_key="sk-test", strategies=None):
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(TWO_MODEL_YAML)

    settings = Settings(
        _env_file=None, database_url=f"sqlite:///{tmp_path}/test.db", openai_api_key=openai_key
    )
    engine_db = create_engine_from_settings(settings)
    init_db(engine_db)
    session_factory = create_session_factory(engine_db)

    factory = ProviderFactory()
    factory.register("mock", MockProvider)
    factory.register("openai", OpenAIProvider)
    credential_store = CredentialStore(session_factory=session_factory, settings=settings)
    provider_manager = ProviderManager(factory, credential_store)

    model_registry = ModelRegistry(
        provider_manager=provider_manager,
        event_bus=EventBus(),
        cost_estimator=DefaultCostEstimator(),
        session_factory=session_factory,
        yaml_path=str(yaml_path),
    )
    model_registry.reload()

    return RoutingEngine(
        model_registry=model_registry,
        analyzer=PromptAnalyzer(),
        classifier=HeuristicComplexityClassifier(ClassifierPolicy(simple_max=1, medium_max=3)),
        routing_policy=_routing_policy(),
        strategies=strategies
        or {
            "balanced": BalancedStrategy(BalancedStrategyWeights()),
            "cost": CostOptimizedStrategy(),
        },
        explanation_generator=ExplanationGenerator(),
    )


def test_route_returns_decision_for_simple_prompt(tmp_path):
    engine = _make_engine(tmp_path)
    decision = engine.route("List three fruits.", strategy_name="cost")

    assert decision.complexity.value == "simple"
    assert decision.strategy == "cost"
    assert decision.selected_model in {"gpt-4o-mini", "gpt-4o"}
    assert decision.estimated_cost > 0
    assert decision.estimated_latency_ms > 0
    assert len(decision.reasoning) == 3


def test_route_complex_prompt_only_selects_high_benchmark_model(tmp_path):
    engine = _make_engine(tmp_path)
    complex_prompt = (
        "Analyze and compare these two algorithms, explain the reasoning step by step, "
        "calculate their time complexity, and format the answer as bullet points. "
        "You must include examples and should ensure correctness."
    )
    decision = engine.route(complex_prompt, strategy_name="cost")

    assert decision.complexity.value == "complex"
    assert decision.selected_model == "gpt-4o"


def test_route_raises_when_no_provider_available(tmp_path):
    engine = _make_engine(tmp_path, openai_key=None, strategies={"cost": CostOptimizedStrategy()})

    with pytest.raises(NoEligibleModelError):
        engine.route("Hello.", strategy_name="cost")


def test_route_raises_key_error_for_unknown_strategy(tmp_path):
    engine = _make_engine(tmp_path)
    with pytest.raises(KeyError):
        engine.route("Hello.", strategy_name="does-not-exist")


def test_route_excludes_specified_provider(tmp_path):
    engine = _make_engine(tmp_path, strategies={"cost": CostOptimizedStrategy()})

    with pytest.raises(NoEligibleModelError):
        engine.route("Hello.", strategy_name="cost", exclude_providers=frozenset({"openai"}))


def test_route_exclude_providers_has_no_effect_on_unrelated_provider(tmp_path):
    engine = _make_engine(tmp_path, strategies={"cost": CostOptimizedStrategy()})

    decision = engine.route("Hello.", strategy_name="cost", exclude_providers=frozenset({"mock"}))

    assert decision.selected_model in {"gpt-4o-mini", "gpt-4o"}


def test_route_default_exclude_providers_is_empty(tmp_path):
    engine = _make_engine(tmp_path, strategies={"cost": CostOptimizedStrategy()})

    decision = engine.route("Hello.", strategy_name="cost")

    assert decision.selected_model in {"gpt-4o-mini", "gpt-4o"}
