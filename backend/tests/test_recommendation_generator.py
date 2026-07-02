import pytest

from backend.learning.generator import RecommendationGenerator, Severity
from backend.learning.rules import Finding, RuleType


def test_generate_model_complexity_signature_and_text():
    finding = Finding(
        rule_type=RuleType.MODEL_COMPLEXITY, subject="gpt-4o-mini:medium",
        sample_size=20, pass_rate=0.35, threshold=0.6,
    )
    [rec] = RecommendationGenerator().generate([finding])

    assert rec.signature == "model_complexity:gpt-4o-mini:medium"
    assert "gpt-4o-mini" in rec.text
    assert "medium" in rec.text
    assert "35%" in rec.text
    assert rec.severity == Severity.HIGH  # pass_rate 0.35 < 0.4
    assert rec.evidence.sample_size == 20
    assert rec.evidence.pass_rate == 0.35
    assert rec.evidence.threshold == 0.6


def test_generate_model_complexity_medium_severity_above_0_4():
    finding = Finding(
        rule_type=RuleType.MODEL_COMPLEXITY, subject="gpt-4o-mini:medium",
        sample_size=20, pass_rate=0.55, threshold=0.6,
    )
    [rec] = RecommendationGenerator().generate([finding])
    assert rec.severity == Severity.MEDIUM


def test_generate_complexity_tier_signature_and_text():
    finding = Finding(
        rule_type=RuleType.COMPLEXITY_TIER, subject="complex",
        sample_size=30, pass_rate=0.25, threshold=0.5,
    )
    [rec] = RecommendationGenerator().generate([finding])

    assert rec.signature == "complexity_tier:complex"
    assert "complex" in rec.text
    assert rec.severity == Severity.HIGH  # pass_rate 0.25 < 0.3


def test_generate_complexity_tier_medium_severity_above_0_3():
    finding = Finding(
        rule_type=RuleType.COMPLEXITY_TIER, subject="complex",
        sample_size=30, pass_rate=0.45, threshold=0.5,
    )
    [rec] = RecommendationGenerator().generate([finding])
    assert rec.severity == Severity.MEDIUM


def test_evidence_confidence_scales_with_sample_size():
    small = Finding(rule_type=RuleType.COMPLEXITY_TIER, subject="complex", sample_size=30, pass_rate=0.4, threshold=0.5)
    large = Finding(rule_type=RuleType.COMPLEXITY_TIER, subject="complex", sample_size=300, pass_rate=0.4, threshold=0.5)

    [rec_small] = RecommendationGenerator().generate([small])
    [rec_large] = RecommendationGenerator().generate([large])

    assert rec_small.evidence_confidence == pytest.approx(0.5 + 30 / 200)
    assert rec_large.evidence_confidence == pytest.approx(0.95)  # capped


def test_generate_default_source_is_verification():
    finding = Finding(rule_type=RuleType.COMPLEXITY_TIER, subject="complex", sample_size=30, pass_rate=0.4, threshold=0.5)
    [rec] = RecommendationGenerator().generate([finding])
    assert rec.source.value == "verification"
