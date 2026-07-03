from enum import Enum

from pydantic import BaseModel

from backend.learning.cost_metrics import ModelCostMetrics
from backend.learning.rules import Finding, RuleType

_LOW_SAVINGS_CEILING = 10.0
_MEDIUM_SAVINGS_CEILING = 100.0


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RecommendationSource(str, Enum):
    VERIFICATION = "verification"
    COST_OPTIMIZATION = "cost_optimization"


class ModelComparison(BaseModel):
    current_model: str
    suggested_model: str
    current_pass_rate: float
    suggested_pass_rate: float
    current_cost_per_request: float
    suggested_cost_per_request: float
    estimated_monthly_savings: float


class RecommendationEvidence(BaseModel):
    sample_size: int
    pass_rate: float
    threshold: float
    comparison: ModelComparison | None = None


class Recommendation(BaseModel):
    signature: str
    rule_type: RuleType
    subject: str
    text: str
    evidence_confidence: float
    severity: Severity
    evidence: RecommendationEvidence
    source: RecommendationSource = RecommendationSource.VERIFICATION


def _severity_from_savings(savings: float) -> Severity:
    if savings < _LOW_SAVINGS_CEILING:
        return Severity.LOW
    if savings <= _MEDIUM_SAVINGS_CEILING:
        return Severity.MEDIUM
    return Severity.HIGH


class RecommendationGenerator:
    def generate(
        self,
        findings: list[Finding],
        cost_metrics: dict[tuple[str, str], ModelCostMetrics] | None = None,
    ) -> list[Recommendation]:
        cost_metrics = cost_metrics or {}
        recommendations = []
        for finding in findings:
            rec = self._generate_one(finding, cost_metrics)
            if rec is not None:
                recommendations.append(rec)
        return recommendations

    def _generate_one(
        self, finding: Finding, cost_metrics: dict[tuple[str, str], ModelCostMetrics]
    ) -> Recommendation | None:
        if finding.rule_type == RuleType.COST_OPTIMIZATION:
            return self._generate_cost_optimization(finding, cost_metrics)

        signature = f"{finding.rule_type.value}:{finding.subject}"
        evidence_confidence = min(0.5 + (finding.sample_size / 200), 0.95)

        if finding.rule_type == RuleType.MODEL_COMPLEXITY:
            model, complexity = finding.subject.split(":", 1)
            text = (
                f"Model '{model}' has a {finding.pass_rate:.0%} pass rate for "
                f"'{complexity}' prompts ({finding.sample_size} samples) — "
                f"consider a higher-benchmark model for this tier."
            )
            severity = Severity.HIGH if finding.pass_rate < 0.4 else Severity.MEDIUM
        else:
            text = (
                f"'{finding.subject}' prompts have a {finding.pass_rate:.0%} overall "
                f"pass rate ({finding.sample_size} samples) — consider reviewing the "
                f"eligibility policy for this tier."
            )
            severity = Severity.HIGH if finding.pass_rate < 0.3 else Severity.MEDIUM

        return Recommendation(
            signature=signature,
            rule_type=finding.rule_type,
            subject=finding.subject,
            text=text,
            evidence_confidence=evidence_confidence,
            severity=severity,
            evidence=RecommendationEvidence(
                sample_size=finding.sample_size,
                pass_rate=finding.pass_rate,
                threshold=finding.threshold,
            ),
        )

    def _generate_cost_optimization(
        self, finding: Finding, cost_metrics: dict[tuple[str, str], ModelCostMetrics]
    ) -> Recommendation | None:
        model, complexity = finding.subject.split(":", 1)
        current = cost_metrics.get((model, complexity))
        if current is None:
            return None

        candidates = [
            m for (m_model, m_complexity), m in cost_metrics.items()
            if m_complexity == complexity and m.eligible_for_optimization
        ]
        if not candidates:
            return None

        best = min(candidates, key=lambda m: (m.avg_cost_per_request, m.model))
        if best.model == current.model:
            return None
        if current.avg_cost_per_request <= best.avg_cost_per_request:
            return None

        estimated_monthly_savings = round(
            (current.avg_cost_per_request - best.avg_cost_per_request) * current.requests_per_day * 30,
            2,
        )
        if estimated_monthly_savings <= 0:
            return None

        comparison = ModelComparison(
            current_model=current.model,
            suggested_model=best.model,
            current_pass_rate=current.pass_rate,
            suggested_pass_rate=best.pass_rate,
            current_cost_per_request=current.avg_cost_per_request,
            suggested_cost_per_request=best.avg_cost_per_request,
            estimated_monthly_savings=estimated_monthly_savings,
        )

        text = (
            f"Current model '{current.model}' consistently meets the quality threshold "
            f"for '{complexity}' prompts. A lower-cost model, '{best.model}', also meets "
            f"the threshold. Estimated monthly savings: ~${estimated_monthly_savings:.2f}."
        )

        return Recommendation(
            signature=f"{finding.rule_type.value}:{finding.subject}",
            rule_type=finding.rule_type,
            subject=finding.subject,
            text=text,
            evidence_confidence=min(0.5 + (finding.sample_size / 200), 0.95),
            severity=_severity_from_savings(estimated_monthly_savings),
            evidence=RecommendationEvidence(
                sample_size=finding.sample_size,
                pass_rate=finding.pass_rate,
                threshold=finding.threshold,
                comparison=comparison,
            ),
            source=RecommendationSource.COST_OPTIMIZATION,
        )
