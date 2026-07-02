from pydantic import BaseModel

from backend.analysis.prompt_analyzer import PromptAnalyzer
from backend.classifier.complexity_classifier import BaseComplexityClassifier, ComplexityTier
from backend.routing.context import RoutingContext
from backend.routing.explanation import ExplanationGenerator
from backend.routing.policy import RoutingPolicy
from backend.routing.strategies import BaseRoutingStrategy
from backend.services.model_registry import ModelRegistry


class RoutingDecision(BaseModel):
    selected_model: str
    strategy: str
    complexity: ComplexityTier
    confidence: float
    estimated_cost: float
    estimated_latency_ms: float
    reasoning: list[str]


class NoEligibleModelError(Exception):
    pass


class RoutingEngine:
    def __init__(
        self,
        model_registry: ModelRegistry,
        analyzer: PromptAnalyzer,
        classifier: BaseComplexityClassifier,
        routing_policy: RoutingPolicy,
        strategies: dict[str, BaseRoutingStrategy],
        explanation_generator: ExplanationGenerator,
    ) -> None:
        self._model_registry = model_registry
        self._analyzer = analyzer
        self._classifier = classifier
        self._routing_policy = routing_policy
        self._strategies = strategies
        self._explanation_generator = explanation_generator

    def route(self, prompt: str, strategy_name: str = "balanced") -> RoutingDecision:
        features = self._analyzer.analyze(prompt)
        classification = self._classifier.classify(features)

        available = self._model_registry.get_available_models()
        candidates = self._routing_policy.filter_candidates(classification.tier, available)
        if not candidates:
            raise NoEligibleModelError(
                f"No available model meets the '{classification.tier.value}' complexity policy"
            )

        context = RoutingContext(
            prompt=prompt,
            features=features,
            complexity=classification.tier,
            candidates=candidates,
        )
        selected = self._strategies[strategy_name].select_model(context)

        estimated_cost = self._model_registry.estimate_cost(
            selected.id, features.estimated_tokens, features.estimated_output_tokens
        )
        reasoning = self._explanation_generator.generate(
            context, selected, strategy_name, classification
        )

        return RoutingDecision(
            selected_model=selected.id,
            strategy=strategy_name,
            complexity=classification.tier,
            confidence=classification.confidence,
            estimated_cost=estimated_cost,
            estimated_latency_ms=selected.average_latency_ms,
            reasoning=reasoning,
        )
