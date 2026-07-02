from backend.analysis.prompt_analyzer import PromptAnalyzer
from backend.classifier.complexity_classifier import ComplexityTier
from backend.routing.context import RoutingContext
from backend.services.model_registry import ModelSpec


def test_routing_context_holds_all_fields():
    features = PromptAnalyzer().analyze("Explain why the sky is blue.")
    model = ModelSpec(
        id="gpt-4o-mini", provider="openai", model="gpt-4o-mini", input_cost=0.15,
        output_cost=0.60, context_window=128000, max_output_tokens=16384,
        supports_streaming=True, supports_tools=True, supports_json=True,
        supports_vision=False, benchmark_score=0.82, average_latency_ms=450, available=True,
    )

    context = RoutingContext(
        prompt="Explain why the sky is blue.",
        features=features,
        complexity=ComplexityTier.SIMPLE,
        candidates=[model],
    )

    assert context.prompt == "Explain why the sky is blue."
    assert context.features == features
    assert context.complexity == ComplexityTier.SIMPLE
    assert context.candidates == [model]
