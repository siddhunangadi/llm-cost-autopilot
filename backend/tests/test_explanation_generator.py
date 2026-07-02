from backend.analysis.prompt_analyzer import PromptAnalyzer
from backend.classifier.complexity_classifier import ClassificationResult, ComplexityTier
from backend.routing.context import RoutingContext
from backend.routing.explanation import ExplanationGenerator
from backend.services.model_registry import ModelSpec


def _model() -> ModelSpec:
    return ModelSpec(
        id="gpt-4o-mini", provider="openai", model="gpt-4o-mini", input_cost=0.15,
        output_cost=0.60, context_window=128000, max_output_tokens=16384,
        supports_streaming=True, supports_tools=True, supports_json=True,
        supports_vision=False, benchmark_score=0.82, average_latency_ms=450, available=True,
    )


def _context() -> RoutingContext:
    features = PromptAnalyzer().analyze("Explain why the sky is blue.")
    return RoutingContext(
        prompt="Explain why the sky is blue.", features=features,
        complexity=ComplexityTier.SIMPLE, candidates=[_model()],
    )


def test_generate_includes_signals_when_present():
    classification = ClassificationResult(
        tier=ComplexityTier.MEDIUM, score=2, confidence=0.66,
        signals=["reasoning keywords detected", "code content detected"],
    )
    reasoning = ExplanationGenerator().generate(_context(), _model(), "balanced", classification)

    assert "reasoning keywords detected" in reasoning[0]
    assert "code content detected" in reasoning[0]
    assert "medium" in reasoning[0]
    assert "0.66" in reasoning[0]
    assert "balanced" in reasoning[1]
    assert "1 eligible model" in reasoning[1]
    assert "gpt-4o-mini" in reasoning[2]


def test_generate_handles_no_signals():
    classification = ClassificationResult(
        tier=ComplexityTier.SIMPLE, score=0, confidence=0.66, signals=[]
    )
    reasoning = ExplanationGenerator().generate(_context(), _model(), "cost", classification)

    assert "no complexity signals detected" in reasoning[0]
