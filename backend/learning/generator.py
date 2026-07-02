from enum import Enum

from pydantic import BaseModel

from backend.learning.rules import Finding, RuleType


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RecommendationSource(str, Enum):
    VERIFICATION = "verification"


class RecommendationEvidence(BaseModel):
    sample_size: int
    pass_rate: float
    threshold: float


class Recommendation(BaseModel):
    signature: str
    rule_type: RuleType
    subject: str
    text: str
    evidence_confidence: float
    severity: Severity
    evidence: RecommendationEvidence
    source: RecommendationSource = RecommendationSource.VERIFICATION


class RecommendationGenerator:
    def generate(self, findings: list[Finding]) -> list[Recommendation]:
        return [self._generate_one(f) for f in findings]

    def _generate_one(self, finding: Finding) -> Recommendation:
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
