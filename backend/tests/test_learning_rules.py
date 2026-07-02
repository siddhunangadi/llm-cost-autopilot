import pytest

from backend.learning.rules import BaseDetectionRule, DetectionRuleConfig, Finding, RuleType


def test_base_detection_rule_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseDetectionRule()


def test_detection_rule_config_is_frozen():
    config = DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.6)
    with pytest.raises(Exception):
        config.min_samples = 30  # dataclasses.FrozenInstanceError


def test_finding_holds_all_fields():
    finding = Finding(
        rule_type=RuleType.MODEL_COMPLEXITY, subject="gpt-4o-mini:medium",
        sample_size=25, pass_rate=0.4, threshold=0.6,
    )
    assert finding.rule_type == RuleType.MODEL_COMPLEXITY
    assert finding.subject == "gpt-4o-mini:medium"
    assert finding.sample_size == 25
    assert finding.pass_rate == 0.4
    assert finding.threshold == 0.6


from backend.database.models import VerificationRow
from backend.learning.rules import ComplexityTierRule, ModelComplexityRule
from backend.verification.status import VerificationStatus


def _row(model, strategy, complexity, passed, status=VerificationStatus.COMPLETED.value):
    return VerificationRow(
        request_id="req", status=status, routing_model=model,
        routing_strategy=strategy, routing_complexity=complexity, passed=passed,
    )


def test_model_complexity_rule_emits_finding_below_threshold():
    rows = (
        [_row("gpt-4o-mini", "balanced", "medium", passed=False) for _ in range(13)]
        + [_row("gpt-4o-mini", "balanced", "medium", passed=True) for _ in range(7)]
    )  # 20 samples, pass_rate = 0.35 < 0.6
    rule = ModelComplexityRule(DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.6))

    findings = rule.evaluate(rows)

    assert len(findings) == 1
    assert findings[0].rule_type == RuleType.MODEL_COMPLEXITY
    assert findings[0].subject == "gpt-4o-mini:medium"
    assert findings[0].sample_size == 20
    assert findings[0].pass_rate == pytest.approx(0.35)
    assert findings[0].threshold == 0.6


def test_model_complexity_rule_skips_below_min_samples():
    rows = [_row("gpt-4o-mini", "balanced", "medium", passed=False) for _ in range(5)]
    rule = ModelComplexityRule(DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.6))

    assert rule.evaluate(rows) == []


def test_model_complexity_rule_skips_above_threshold():
    rows = (
        [_row("gpt-4o-mini", "balanced", "medium", passed=True) for _ in range(18)]
        + [_row("gpt-4o-mini", "balanced", "medium", passed=False) for _ in range(2)]
    )  # pass_rate = 0.9
    rule = ModelComplexityRule(DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.6))

    assert rule.evaluate(rows) == []


def test_model_complexity_rule_excludes_non_completed_rows():
    rows = (
        [_row("gpt-4o-mini", "balanced", "medium", passed=False) for _ in range(20)]
        + [
            _row("gpt-4o-mini", "balanced", "medium", passed=None, status=VerificationStatus.FAILED.value)
            for _ in range(50)
        ]
    )
    rule = ModelComplexityRule(DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.6))

    findings = rule.evaluate(rows)

    assert len(findings) == 1
    assert findings[0].sample_size == 20  # the 50 FAILED-status rows are excluded


def test_complexity_tier_rule_emits_finding_below_threshold():
    rows = (
        [_row("gpt-4o-mini", "balanced", "complex", passed=False) for _ in range(20)]
        + [_row("gpt-4o", "quality", "complex", passed=True) for _ in range(10)]
    )  # 30 samples, pass_rate = 10/30 = 0.333 < 0.5
    rule = ComplexityTierRule(DetectionRuleConfig(min_samples=30, pass_rate_threshold=0.5))

    findings = rule.evaluate(rows)

    assert len(findings) == 1
    assert findings[0].rule_type == RuleType.COMPLEXITY_TIER
    assert findings[0].subject == "complex"
    assert findings[0].sample_size == 30
    assert findings[0].pass_rate == pytest.approx(1 / 3)


def test_complexity_tier_rule_skips_below_min_samples():
    rows = [_row("gpt-4o-mini", "balanced", "complex", passed=False) for _ in range(10)]
    rule = ComplexityTierRule(DetectionRuleConfig(min_samples=30, pass_rate_threshold=0.5))

    assert rule.evaluate(rows) == []
