from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel

from backend.database.models import VerificationRow
from backend.verification.status import VerificationStatus


class RuleType(str, Enum):
    MODEL_COMPLEXITY = "model_complexity"
    COMPLEXITY_TIER = "complexity_tier"
    COST_OPTIMIZATION = "cost_optimization"


class Finding(BaseModel):
    rule_type: RuleType
    subject: str
    sample_size: int
    pass_rate: float
    threshold: float


@dataclass(frozen=True)
class DetectionRuleConfig:
    min_samples: int
    pass_rate_threshold: float


class BaseDetectionRule(ABC):
    @abstractmethod
    def evaluate(self, rows: list[VerificationRow]) -> list[Finding]: ...


def eligible_verification_rows(rows: list[VerificationRow]) -> list[VerificationRow]:
    return [r for r in rows if r.status == VerificationStatus.COMPLETED.value and r.passed is not None]


class ModelComplexityRule(BaseDetectionRule):
    def __init__(self, config: DetectionRuleConfig) -> None:
        self._config = config

    def evaluate(self, rows: list[VerificationRow]) -> list[Finding]:
        groups: dict[tuple[str, str], list[bool]] = defaultdict(list)
        for row in eligible_verification_rows(rows):
            groups[(row.routing_model, row.routing_complexity)].append(row.passed)

        findings = []
        for (model, complexity), outcomes in groups.items():
            sample_size = len(outcomes)
            if sample_size < self._config.min_samples:
                continue
            pass_rate = sum(outcomes) / sample_size
            if pass_rate < self._config.pass_rate_threshold:
                findings.append(Finding(
                    rule_type=RuleType.MODEL_COMPLEXITY,
                    subject=f"{model}:{complexity}",
                    sample_size=sample_size,
                    pass_rate=pass_rate,
                    threshold=self._config.pass_rate_threshold,
                ))
        return findings


class ComplexityTierRule(BaseDetectionRule):
    def __init__(self, config: DetectionRuleConfig) -> None:
        self._config = config

    def evaluate(self, rows: list[VerificationRow]) -> list[Finding]:
        groups: dict[str, list[bool]] = defaultdict(list)
        for row in eligible_verification_rows(rows):
            groups[row.routing_complexity].append(row.passed)

        findings = []
        for complexity, outcomes in groups.items():
            sample_size = len(outcomes)
            if sample_size < self._config.min_samples:
                continue
            pass_rate = sum(outcomes) / sample_size
            if pass_rate < self._config.pass_rate_threshold:
                findings.append(Finding(
                    rule_type=RuleType.COMPLEXITY_TIER,
                    subject=complexity,
                    sample_size=sample_size,
                    pass_rate=pass_rate,
                    threshold=self._config.pass_rate_threshold,
                ))
        return findings


class OverpoweredModelRule(BaseDetectionRule):
    """Flags a model that is reliably passing quality checks for a
    complexity tier -- a candidate for a cheaper replacement. This rule
    has no knowledge of pricing; it only measures observed quality.
    Selection of a cheaper alternative happens in RecommendationGenerator.
    """

    def __init__(self, config: DetectionRuleConfig) -> None:
        self._config = config

    def evaluate(self, rows: list[VerificationRow]) -> list[Finding]:
        groups: dict[tuple[str, str], list[bool]] = defaultdict(list)
        for row in eligible_verification_rows(rows):
            groups[(row.routing_model, row.routing_complexity)].append(row.passed)

        findings = []
        for model, complexity in sorted(groups.keys()):
            outcomes = groups[(model, complexity)]
            sample_size = len(outcomes)
            if sample_size < self._config.min_samples:
                continue
            pass_rate = sum(outcomes) / sample_size
            if pass_rate >= self._config.pass_rate_threshold:
                findings.append(Finding(
                    rule_type=RuleType.COST_OPTIMIZATION,
                    subject=f"{model}:{complexity}",
                    sample_size=sample_size,
                    pass_rate=pass_rate,
                    threshold=self._config.pass_rate_threshold,
                ))
        return findings
