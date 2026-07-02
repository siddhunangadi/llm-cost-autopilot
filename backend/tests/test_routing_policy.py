from backend.classifier.complexity_classifier import ComplexityTier
from backend.routing.config import EligibilityPolicy
from backend.routing.policy import RoutingPolicy
from backend.services.model_registry import ModelSpec


def _model(id: str, benchmark_score: float) -> ModelSpec:
    return ModelSpec(
        id=id, provider="openai", model=id, input_cost=0.15, output_cost=0.60,
        context_window=128000, max_output_tokens=16384, supports_streaming=True,
        supports_tools=True, supports_json=True, supports_vision=False,
        benchmark_score=benchmark_score, average_latency_ms=450, available=True,
    )


def _policy() -> RoutingPolicy:
    return RoutingPolicy({
        "simple": EligibilityPolicy(min_benchmark_score=0.0),
        "medium": EligibilityPolicy(min_benchmark_score=0.75),
        "complex": EligibilityPolicy(min_benchmark_score=0.90),
    })


def test_simple_allows_all_models():
    candidates = [_model("a", 0.5), _model("b", 0.95)]
    result = _policy().filter_candidates(ComplexityTier.SIMPLE, candidates)
    assert {m.id for m in result} == {"a", "b"}


def test_complex_excludes_low_benchmark_models():
    candidates = [_model("a", 0.82), _model("b", 0.93)]
    result = _policy().filter_candidates(ComplexityTier.COMPLEX, candidates)
    assert {m.id for m in result} == {"b"}


def test_medium_boundary_is_inclusive():
    candidates = [_model("a", 0.75)]
    result = _policy().filter_candidates(ComplexityTier.MEDIUM, candidates)
    assert {m.id for m in result} == {"a"}


def test_no_eligible_candidates_returns_empty_list():
    candidates = [_model("a", 0.5)]
    result = _policy().filter_candidates(ComplexityTier.COMPLEX, candidates)
    assert result == []
