from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel

from backend.analysis.prompt_analyzer import PromptFeatures
from backend.routing.config import ClassifierPolicy


class ComplexityTier(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class ClassificationResult(BaseModel):
    tier: ComplexityTier
    score: int
    confidence: float
    signals: list[str]


class BaseComplexityClassifier(ABC):
    @abstractmethod
    def classify(self, features: PromptFeatures) -> ClassificationResult: ...


class HeuristicComplexityClassifier(BaseComplexityClassifier):
    def __init__(self, policy: ClassifierPolicy) -> None:
        self._policy = policy

    def classify(self, features: PromptFeatures) -> ClassificationResult:
        score = 0
        signals: list[str] = []

        if features.estimated_tokens > 200:
            score += 1
            signals.append("prompt exceeds 200 estimated tokens")
        if features.constraint_count >= 2:
            score += 1
            signals.append("multiple constraints detected")
        if features.has_code:
            score += 1
            signals.append("code content detected")
        if features.has_reasoning_keywords:
            score += 1
            signals.append("reasoning keywords detected")
        if features.has_comparison_keywords:
            score += 1
            signals.append("comparison keywords detected")
        if features.has_analysis_keywords:
            score += 1
            signals.append("analysis keywords detected")
        if features.has_math_indicators:
            score += 1
            signals.append("math indicators detected")
        if features.has_chain_of_thought_indicators:
            score += 1
            signals.append("chain-of-thought indicators detected")
        if features.requires_output_formatting:
            score += 1
            signals.append("output formatting requested")

        return ClassificationResult(
            tier=self._tier_for_score(score),
            score=score,
            confidence=self._confidence_for_score(score),
            signals=signals,
        )

    def _tier_for_score(self, score: int) -> ComplexityTier:
        if score <= self._policy.simple_max:
            return ComplexityTier.SIMPLE
        if score <= self._policy.medium_max:
            return ComplexityTier.MEDIUM
        return ComplexityTier.COMPLEX

    def _confidence_for_score(self, score: int) -> float:
        boundaries = (self._policy.simple_max, self._policy.medium_max)
        nearest_distance = min(abs(score - boundary) for boundary in boundaries)
        confidence = 0.5 + min(nearest_distance / 3, 1.0) * 0.49
        return round(confidence, 2)
