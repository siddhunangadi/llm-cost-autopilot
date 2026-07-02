from abc import ABC, abstractmethod

from backend.routing.config import BalancedStrategyWeights
from backend.routing.context import RoutingContext
from backend.services.model_registry import ModelSpec


class BaseRoutingStrategy(ABC):
    @abstractmethod
    def select_model(self, context: RoutingContext) -> ModelSpec: ...


class CostOptimizedStrategy(BaseRoutingStrategy):
    def select_model(self, context: RoutingContext) -> ModelSpec:
        return min(context.candidates, key=lambda c: c.input_cost + c.output_cost)


class LatencyOptimizedStrategy(BaseRoutingStrategy):
    def select_model(self, context: RoutingContext) -> ModelSpec:
        return min(context.candidates, key=lambda c: c.average_latency_ms)


class QualityOptimizedStrategy(BaseRoutingStrategy):
    def select_model(self, context: RoutingContext) -> ModelSpec:
        return max(context.candidates, key=lambda c: c.benchmark_score)


def _normalize(values: list[float], invert: bool) -> list[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5] * len(values)
    if invert:
        return [(hi - v) / (hi - lo) for v in values]
    return [(v - lo) / (hi - lo) for v in values]


class BalancedStrategy(BaseRoutingStrategy):
    def __init__(self, weights: BalancedStrategyWeights) -> None:
        self._weights = weights

    def select_model(self, context: RoutingContext) -> ModelSpec:
        candidates = context.candidates
        if len(candidates) == 1:
            return candidates[0]

        costs = [c.input_cost + c.output_cost for c in candidates]
        latencies = [c.average_latency_ms for c in candidates]
        qualities = [c.benchmark_score for c in candidates]

        cost_scores = _normalize(costs, invert=True)
        latency_scores = _normalize(latencies, invert=True)
        quality_scores = _normalize(qualities, invert=False)

        combined = [
            cost * self._weights.cost_weight
            + latency * self._weights.latency_weight
            + quality * self._weights.quality_weight
            for cost, latency, quality in zip(cost_scores, latency_scores, quality_scores)
        ]
        best_index = combined.index(max(combined))
        return candidates[best_index]
