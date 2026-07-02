from backend.database.models import VerificationRow
from backend.learning.detector import FailurePatternDetector
from backend.learning.rules import ComplexityTierRule, DetectionRuleConfig, Finding, ModelComplexityRule, RuleType
from backend.verification.status import VerificationStatus


def _row(model, complexity, passed):
    return VerificationRow(
        request_id="req", status=VerificationStatus.COMPLETED.value,
        routing_model=model, routing_strategy="balanced",
        routing_complexity=complexity, passed=passed,
    )


def _rows():
    return (
        [_row("gpt-4o-mini", "medium", False) for _ in range(13)]
        + [_row("gpt-4o-mini", "medium", True) for _ in range(7)]
    )


def _detector():
    return FailurePatternDetector(rules=[
        ModelComplexityRule(DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.6)),
        ComplexityTierRule(DetectionRuleConfig(min_samples=30, pass_rate_threshold=0.5)),
    ])


def test_detect_runs_all_registered_rules():
    findings = _detector().detect(_rows())
    assert any(f.rule_type == RuleType.MODEL_COMPLEXITY for f in findings)


def test_detect_is_deterministic():
    rows = _rows()
    first = _detector().detect(rows)
    second = _detector().detect(rows)
    assert first == second


def test_detect_rules_are_independent():
    calls: list[list[Finding]] = []

    class _RecordingRule:
        def evaluate(self, rows):
            calls.append(list(rows))
            return []

    detector = FailurePatternDetector(rules=[_RecordingRule(), _RecordingRule()])
    rows = _rows()
    detector.detect(rows)

    # each rule received the full, unfiltered row list -- never another rule's findings
    assert len(calls) == 2
    assert calls[0] == rows
    assert calls[1] == rows
