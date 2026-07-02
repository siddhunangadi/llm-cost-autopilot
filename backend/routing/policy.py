from backend.classifier.complexity_classifier import ComplexityTier
from backend.routing.config import EligibilityPolicy
from backend.services.model_registry import ModelSpec


class RoutingPolicy:
    def __init__(self, policies: dict[str, EligibilityPolicy]) -> None:
        self._policies = policies

    def filter_candidates(
        self, complexity: ComplexityTier, candidates: list[ModelSpec]
    ) -> list[ModelSpec]:
        policy = self._policies[complexity.value]
        return [c for c in candidates if c.benchmark_score >= policy.min_benchmark_score]
