import pytest

from backend.analysis.prompt_analyzer import PromptAnalyzer
from backend.classifier.complexity_classifier import ComplexityTier
from backend.routing.config import BalancedStrategyWeights
from backend.routing.context import RoutingContext
from backend.routing.strategies import (
    BalancedStrategy,
    BaseRoutingStrategy,
    CostOptimizedStrategy,
    LatencyOptimizedStrategy,
    QualityOptimizedStrategy,
)
from backend.services.model_registry import ModelSpec


def _model(id, input_cost, output_cost, latency, benchmark) -> ModelSpec:
    return ModelSpec(
        id=id, provider="openai", model=id, input_cost=input_cost, output_cost=output_cost,
        context_window=128000, max_output_tokens=16384, supports_streaming=True,
        supports_tools=True, supports_json=True, supports_vision=False,
        benchmark_score=benchmark, average_latency_ms=latency, available=True,
    )


def _context(candidates) -> RoutingContext:
    features = PromptAnalyzer().analyze("test prompt")
    return RoutingContext(
        prompt="test prompt", features=features, complexity=ComplexityTier.SIMPLE,
        candidates=candidates,
    )


def test_base_routing_strategy_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseRoutingStrategy()


def test_cost_optimized_picks_cheapest():
    cheap = _model("cheap", 0.1, 0.1, 500, 0.8)
    expensive = _model("expensive", 5.0, 5.0, 100, 0.99)
    selected = CostOptimizedStrategy().select_model(_context([cheap, expensive]))
    assert selected.id == "cheap"


def test_latency_optimized_picks_fastest():
    fast = _model("fast", 1.0, 1.0, 100, 0.8)
    slow = _model("slow", 0.1, 0.1, 900, 0.99)
    selected = LatencyOptimizedStrategy().select_model(_context([fast, slow]))
    assert selected.id == "fast"


def test_quality_optimized_picks_highest_benchmark():
    low = _model("low", 0.1, 0.1, 100, 0.7)
    high = _model("high", 5.0, 5.0, 900, 0.99)
    selected = QualityOptimizedStrategy().select_model(_context([low, high]))
    assert selected.id == "high"


def test_balanced_strategy_with_single_candidate_returns_it():
    only = _model("only", 1.0, 1.0, 500, 0.85)
    weights = BalancedStrategyWeights(cost_weight=1 / 3, latency_weight=1 / 3, quality_weight=1 / 3)
    selected = BalancedStrategy(weights).select_model(_context([only]))
    assert selected.id == "only"


def test_balanced_strategy_picks_best_combined_score():
    balanced = _model("balanced", 0.15, 0.60, 450, 0.82)
    premium = _model("premium", 2.50, 10.00, 900, 0.93)
    weights = BalancedStrategyWeights(cost_weight=1 / 3, latency_weight=1 / 3, quality_weight=1 / 3)
    selected = BalancedStrategy(weights).select_model(_context([balanced, premium]))
    assert selected.id == "balanced"


def test_balanced_strategy_respects_quality_weight_override():
    balanced = _model("balanced", 0.15, 0.60, 450, 0.82)
    premium = _model("premium", 2.50, 10.00, 900, 0.93)
    weights = BalancedStrategyWeights(cost_weight=0.05, latency_weight=0.05, quality_weight=0.90)
    selected = BalancedStrategy(weights).select_model(_context([balanced, premium]))
    assert selected.id == "premium"


def test_balanced_strategy_handles_tied_metric_without_division_by_zero():
    tied_a = _model("a", 1.0, 1.0, 500, 0.80)
    tied_b = _model("b", 1.0, 1.0, 500, 0.90)
    weights = BalancedStrategyWeights(cost_weight=1 / 3, latency_weight=1 / 3, quality_weight=1 / 3)
    selected = BalancedStrategy(weights).select_model(_context([tied_a, tied_b]))
    assert selected.id == "b"
