from backend.database.models import VerificationRow
from backend.learning.rules import BaseDetectionRule, Finding


class FailurePatternDetector:
    def __init__(self, rules: list[BaseDetectionRule]) -> None:
        self._rules = rules

    def detect(self, rows: list[VerificationRow]) -> list[Finding]:
        return [finding for rule in self._rules for finding in rule.evaluate(rows)]
