# Phase 4 Implementation Plan: Self-Improvement & Optimization

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Phase 3's verification data into advisory recommendations — deterministic detection rules find failure patterns, a pure generator turns them into recommendations, and `LearningService` persists them (upsert by signature, never touching human-set `status`) behind three read endpoints.

**Architecture:** One new package (`backend/learning/`) plus one new router. `BaseDetectionRule` implementations are pure and independent (`list[VerificationRow] -> list[Finding]`); `FailurePatternDetector` just runs all registered rules; `RecommendationGenerator` is pure (`list[Finding] -> list[Recommendation]`); `LearningService` is the only component that touches the database, computed entirely on-demand — no scheduler, no background worker.

**Tech Stack:** Same as Phases 1-3 — Python 3.11+, `uv`, FastAPI, Pydantic v2, SQLAlchemy 2.0. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-02-phase4-learning-design.md` (frozen — implement exactly).

## Global Constraints

- Same `uv`-managed Python 3.11+ project as Phases 1-3; no new dependencies.
- One batch (5-8 tightly related tasks), one full regression run, one manual end-to-end verification, one commit, then tag `v0.4.0`.
- `BaseDetectionRule` implementations are pure: no DB, no clock, no randomness, no dependence on dict/set iteration order.
- **Rule independence invariant:** no rule ever reads another rule's `Finding`s; `FailurePatternDetector` shares no state between rule invocations.
- **Determinism invariant:** `FailurePatternDetector.detect(rows)` called twice with the same `rows` (same order) returns identical `Finding`s in identical order.
- `RecommendationGenerator` is pure and is the *only* place a recommendation `signature` is constructed (`f"{rule_type.value}:{subject}"`).
- `LearningService` is the only component that persists `RecommendationRow`s. It upserts by `signature`; on update it never writes `status` — status is owned exclusively by humans.
- `evidence_confidence` is computed as `min(0.5 + (sample_size / 200), 0.95)` and is explicitly distinct from `ClassificationResult.confidence` (Phase 2) and `JudgeVerdict.confidence` (Phase 3) — do not rename or conflate.
- `ModelComplexityRule` uses `DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.6)`; `ComplexityTierRule` uses `DetectionRuleConfig(min_samples=30, pass_rate_threshold=0.5)`. These are hardcoded at construction in `main.py`, not YAML-configured.
- No automatic routing/classifier/policy changes anywhere in this phase. No scheduler, cron, or worker process. No learning-dataset-export subsystem. No `status`-mutating endpoint (the column exists for a future phase, this phase never writes to it after creation).
- No placeholder code, no TODOs, no speculative abstractions.

---

## Batch 1: Full Learning Subsystem (Tasks 35-41)

### Task 35: Finding, RuleType & BaseDetectionRule

**Files:**
- Create: `backend/learning/__init__.py` (empty)
- Create: `backend/learning/rules.py`
- Test: `backend/tests/test_learning_rules.py`

**Interfaces:**
- Produces: `RuleType(str, Enum)` (`MODEL_COMPLEXITY`, `COMPLEXITY_TIER`), `Finding(rule_type, subject, sample_size, pass_rate, threshold)`, `DetectionRuleConfig(min_samples: int, pass_rate_threshold: float)` (frozen dataclass), `BaseDetectionRule` ABC (`evaluate(rows: list[VerificationRow]) -> list[Finding]`). Consumed by `ModelComplexityRule`/`ComplexityTierRule` (Task 36), `FailurePatternDetector` (Task 37).

- [ ] **Step 1: Create the package directory**

Run:
```bash
mkdir -p backend/learning
touch backend/learning/__init__.py
```

- [ ] **Step 2: Write the failing test**

```python
# backend/tests/test_learning_rules.py
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_learning_rules.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.learning.rules'`

- [ ] **Step 4: Write the implementation**

```python
# backend/learning/rules.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel

from backend.database.models import VerificationRow


class RuleType(str, Enum):
    MODEL_COMPLEXITY = "model_complexity"
    COMPLEXITY_TIER = "complexity_tier"


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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_learning_rules.py -v`
Expected: PASS (3 tests)

### Task 36: ModelComplexityRule & ComplexityTierRule

**Files:**
- Modify: `backend/learning/rules.py`
- Test: `backend/tests/test_learning_rules.py` (append)

**Interfaces:**
- Consumes: `BaseDetectionRule`, `Finding`, `RuleType`, `DetectionRuleConfig` (Task 35), `VerificationRow` (Phase 3, `backend/database/models.py`).
- Produces: `ModelComplexityRule(config: DetectionRuleConfig)`, `ComplexityTierRule(config: DetectionRuleConfig)`, both with `evaluate(rows) -> list[Finding]`. Consumed by `FailurePatternDetector` (Task 37), `main.py` (Task 41).

Only `COMPLETED` `VerificationRow`s with a non-`None` `passed` are considered eligible input for both rules — a `FAILED`-status verification (judge itself errored, per Phase 3) contributes no pass/fail signal and must be excluded, otherwise a burst of judge outages would masquerade as a routing quality problem.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_learning_rules.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_learning_rules.py -v`
Expected: FAIL — `ImportError: cannot import name 'ModelComplexityRule'`

- [ ] **Step 3: Write the implementation**

Append to `backend/learning/rules.py`:

```python
from collections import defaultdict

from backend.verification.status import VerificationStatus


def _eligible_rows(rows: list[VerificationRow]) -> list[VerificationRow]:
    return [r for r in rows if r.status == VerificationStatus.COMPLETED.value and r.passed is not None]


class ModelComplexityRule(BaseDetectionRule):
    def __init__(self, config: DetectionRuleConfig) -> None:
        self._config = config

    def evaluate(self, rows: list[VerificationRow]) -> list[Finding]:
        groups: dict[tuple[str, str], list[bool]] = defaultdict(list)
        for row in _eligible_rows(rows):
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
        for row in _eligible_rows(rows):
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_learning_rules.py -v`
Expected: PASS (9 tests)

### Task 37: FailurePatternDetector

**Files:**
- Create: `backend/learning/detector.py`
- Test: `backend/tests/test_failure_pattern_detector.py`

**Interfaces:**
- Consumes: `BaseDetectionRule`, `Finding` (Task 35/36).
- Produces: `FailurePatternDetector(rules: list[BaseDetectionRule])`, `FailurePatternDetector.detect(rows) -> list[Finding]`. Consumed by `LearningService` (Task 40), `main.py` (Task 41).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_failure_pattern_detector.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_failure_pattern_detector.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.learning.detector'`

- [ ] **Step 3: Write the implementation**

```python
# backend/learning/detector.py
from backend.database.models import VerificationRow
from backend.learning.rules import BaseDetectionRule, Finding


class FailurePatternDetector:
    def __init__(self, rules: list[BaseDetectionRule]) -> None:
        self._rules = rules

    def detect(self, rows: list[VerificationRow]) -> list[Finding]:
        return [finding for rule in self._rules for finding in rule.evaluate(rows)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_failure_pattern_detector.py -v`
Expected: PASS (3 tests)

### Task 38: RecommendationGenerator

**Files:**
- Create: `backend/learning/generator.py`
- Test: `backend/tests/test_recommendation_generator.py`

**Interfaces:**
- Consumes: `Finding`, `RuleType` (Task 35).
- Produces: `Severity(str, Enum)`, `RecommendationSource(str, Enum)`, `RecommendationEvidence(sample_size, pass_rate, threshold)`, `Recommendation(signature, rule_type, subject, text, evidence_confidence, severity, evidence, source)`, `RecommendationGenerator.generate(findings) -> list[Recommendation]`. Consumed by `LearningService` (Task 40), `main.py` (Task 41).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_recommendation_generator.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_recommendation_generator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.learning.generator'`

- [ ] **Step 3: Write the implementation**

```python
# backend/learning/generator.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_recommendation_generator.py -v`
Expected: PASS (6 tests)

### Task 39: RecommendationRow

**Files:**
- Modify: `backend/database/models.py`
- Test: `backend/tests/test_recommendation_row.py`

**Interfaces:**
- Produces: `RecommendationRow` (SQLAlchemy model). Consumed by `LearningService` (Task 40), `backend/api/routers/learning.py` (Task 41 provides the endpoint that reads it).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_recommendation_row.py
from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.database.models import RecommendationRow


def test_recommendation_row_round_trip(tmp_path):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    with session_factory() as session:
        session.add(RecommendationRow(
            signature="model_complexity:gpt-4o-mini:medium",
            rule_type="model_complexity",
            subject="gpt-4o-mini:medium",
            recommendation_text="Consider a higher-benchmark model.",
            evidence_confidence=0.6,
            severity="high",
            evidence={"sample_size": 20, "pass_rate": 0.35, "threshold": 0.6},
            source="verification",
        ))
        session.commit()

    with session_factory() as session:
        row = session.query(RecommendationRow).filter_by(
            signature="model_complexity:gpt-4o-mini:medium"
        ).one()
        assert row.status == "new"  # default
        assert row.severity == "high"
        assert row.evidence["pass_rate"] == 0.35
        assert row.updated_at is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_recommendation_row.py -v`
Expected: FAIL — `ImportError: cannot import name 'RecommendationRow'`

- [ ] **Step 3: Write the implementation**

Append to `backend/database/models.py`:

```python
class RecommendationRow(Base):
    __tablename__ = "learning_recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signature: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    rule_type: Mapped[str] = mapped_column(String, nullable=False)
    subject: Mapped[str] = mapped_column(String, nullable=False)

    recommendation_text: Mapped[str] = mapped_column(String, nullable=False)
    evidence_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    severity: Mapped[str] = mapped_column(String, nullable=False)
    evidence: Mapped[dict] = mapped_column(JSON, nullable=False)

    status: Mapped[str] = mapped_column(String, nullable=False, default="new")
    source: Mapped[str] = mapped_column(String, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_recommendation_row.py -v`
Expected: PASS (1 test)

### Task 40: LearningService

**Files:**
- Create: `backend/learning/service.py`
- Test: `backend/tests/test_learning_service.py`

**Interfaces:**
- Consumes: `FailurePatternDetector` (Task 37), `RecommendationGenerator` (Task 38), `RecommendationRow` (Task 39), `VerificationRow` (Phase 3).
- Produces: `LearningService(detector, generator, session_factory)`, `LearningService.refresh_recommendations() -> list[RecommendationRow]`. Consumed by `backend/api/routers/learning.py` and `main.py` (Task 41).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_learning_service.py
from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.database.models import RecommendationRow, RequestRow, VerificationRow
from backend.learning.detector import FailurePatternDetector
from backend.learning.generator import RecommendationGenerator
from backend.learning.rules import ComplexityTierRule, DetectionRuleConfig, ModelComplexityRule
from backend.learning.service import LearningService
from backend.verification.status import VerificationStatus


def _make_service(tmp_path):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    detector = FailurePatternDetector(rules=[
        ModelComplexityRule(DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.6)),
        ComplexityTierRule(DetectionRuleConfig(min_samples=30, pass_rate_threshold=0.5)),
    ])
    service = LearningService(
        detector=detector, generator=RecommendationGenerator(), session_factory=session_factory
    )
    return service, session_factory


def _seed_failing_model(session_factory, count=20, passed_count=7):
    with session_factory() as session:
        for i in range(count):
            request_id = f"req-{i}"
            session.add(RequestRow(request_id=request_id, prompt="hi", strategy="balanced"))
            session.add(VerificationRow(
                request_id=request_id, status=VerificationStatus.COMPLETED.value,
                routing_model="gpt-4o-mini", routing_strategy="balanced",
                routing_complexity="medium", passed=(i < passed_count),
            ))
        session.commit()


def test_refresh_inserts_new_recommendation(tmp_path):
    service, session_factory = _make_service(tmp_path)
    _seed_failing_model(session_factory)

    results = service.refresh_recommendations()

    assert len(results) == 1
    assert results[0].signature == "model_complexity:gpt-4o-mini:medium"
    assert results[0].status == "new"


def test_refresh_is_idempotent_no_duplicates(tmp_path):
    service, session_factory = _make_service(tmp_path)
    _seed_failing_model(session_factory)

    service.refresh_recommendations()
    results = service.refresh_recommendations()

    with session_factory() as session:
        count = session.query(RecommendationRow).count()
    assert count == 1
    assert len(results) == 1


def test_refresh_updates_evidence_but_preserves_human_set_status(tmp_path):
    service, session_factory = _make_service(tmp_path)
    _seed_failing_model(session_factory, count=20, passed_count=7)
    service.refresh_recommendations()

    with session_factory() as session:
        row = session.query(RecommendationRow).filter_by(
            signature="model_complexity:gpt-4o-mini:medium"
        ).one()
        row.status = "acknowledged"
        session.commit()

    _seed_failing_model(session_factory, count=20, passed_count=5)  # shifts pass_rate lower
    results = service.refresh_recommendations()

    assert len(results) == 1
    assert results[0].status == "acknowledged"  # untouched by refresh
    assert results[0].evidence["sample_size"] == 40  # evidence did update


def test_refresh_returns_empty_list_when_no_findings(tmp_path):
    service, session_factory = _make_service(tmp_path)
    results = service.refresh_recommendations()
    assert results == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_learning_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.learning.service'`

- [ ] **Step 3: Write the implementation**

```python
# backend/learning/service.py
from datetime import datetime, timezone

from sqlalchemy.orm import sessionmaker

from backend.database.models import RecommendationRow, VerificationRow
from backend.learning.detector import FailurePatternDetector
from backend.learning.generator import RecommendationGenerator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LearningService:
    def __init__(
        self,
        detector: FailurePatternDetector,
        generator: RecommendationGenerator,
        session_factory: sessionmaker,
    ) -> None:
        self._detector = detector
        self._generator = generator
        self._session_factory = session_factory

    def refresh_recommendations(self) -> list[RecommendationRow]:
        with self._session_factory() as session:
            rows = session.query(VerificationRow).order_by(VerificationRow.id).all()

        findings = self._detector.detect(rows)
        recommendations = self._generator.generate(findings)

        with self._session_factory() as session:
            for rec in recommendations:
                existing = (
                    session.query(RecommendationRow)
                    .filter_by(signature=rec.signature)
                    .first()
                )
                if existing is None:
                    session.add(RecommendationRow(
                        signature=rec.signature,
                        rule_type=rec.rule_type.value,
                        subject=rec.subject,
                        recommendation_text=rec.text,
                        evidence_confidence=rec.evidence_confidence,
                        severity=rec.severity.value,
                        evidence=rec.evidence.model_dump(),
                        status="new",
                        source=rec.source.value,
                    ))
                else:
                    existing.recommendation_text = rec.text
                    existing.evidence_confidence = rec.evidence_confidence
                    existing.severity = rec.severity.value
                    existing.evidence = rec.evidence.model_dump()
                    existing.updated_at = _utcnow()
                    # existing.status is intentionally never modified here --
                    # status is owned exclusively by humans.
            session.commit()

        with self._session_factory() as session:
            return (
                session.query(RecommendationRow)
                .order_by(
                    RecommendationRow.severity.desc(),
                    RecommendationRow.evidence_confidence.desc(),
                    RecommendationRow.updated_at.desc(),
                )
                .all()
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_learning_service.py -v`
Expected: PASS (4 tests)

### Task 41: Learning API Endpoints & Wiring, Tag v0.4.0

**Files:**
- Create: `backend/api/routers/learning.py`
- Modify: `backend/api/dependencies.py`
- Modify: `backend/api/main.py`
- Test: `backend/tests/test_learning_router.py`

**Interfaces:**
- Consumes: `LearningService` (Task 40), `RecommendationRow` (Task 39), `VerificationRow` (Phase 3), `SessionFactoryDep` (Phase 1).
- Produces: `LearningServiceDep`, `LearningSummary`, `FailureRecord`, `RecommendationResponse`, `GET /v1/learning/summary`, `GET /v1/learning/failures`, `GET /v1/learning/recommendations`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_learning_router.py
import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.database.models import RequestRow, VerificationRow
from backend.verification.status import VerificationStatus


def _seed(session_factory):
    with session_factory() as session:
        for i in range(20):
            request_id = f"req-{i}"
            session.add(RequestRow(request_id=request_id, prompt="hi", strategy="balanced"))
            session.add(VerificationRow(
                request_id=request_id, status=VerificationStatus.COMPLETED.value,
                routing_model="gpt-4o-mini", routing_strategy="balanced",
                routing_complexity="medium", passed=(i < 7), score=0.4 if i >= 7 else 0.9,
                rationale="Incomplete answer." if i >= 7 else "Good answer.",
            ))
        session.commit()


def test_learning_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()
    with TestClient(app) as client:
        _seed(app.state.session_factory)

        response = client.get("/v1/learning/summary")

        assert response.status_code == 200
        body = response.json()
        assert body["total_verified"] == 20
        assert body["overall_pass_rate"] == pytest.approx(7 / 20)
        assert body["by_model"]["gpt-4o-mini"] == pytest.approx(7 / 20)


def test_learning_failures(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()
    with TestClient(app) as client:
        _seed(app.state.session_factory)

        response = client.get("/v1/learning/failures")

        assert response.status_code == 200
        body = response.json()
        assert len(body) == 13  # 20 - 7 passed
        assert all(r["routing_model"] == "gpt-4o-mini" for r in body)


def test_learning_recommendations_triggers_refresh(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()
    with TestClient(app) as client:
        _seed(app.state.session_factory)

        response = client.get("/v1/learning/recommendations")

        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["signature"] == "model_complexity:gpt-4o-mini:medium"
        assert body[0]["status"] == "new"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_learning_router.py -v`
Expected: FAIL — `404 Not Found` for all three routes

- [ ] **Step 3: Write `backend/api/routers/learning.py`**

```python
# backend/api/routers/learning.py
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel

from backend.api.dependencies import LearningServiceDep, SessionFactoryDep
from backend.database.models import VerificationRow
from backend.learning.generator import RecommendationEvidence, RecommendationSource, Severity
from backend.learning.rules import RuleType
from backend.verification.status import VerificationStatus

router = APIRouter()


class LearningSummary(BaseModel):
    total_verified: int
    overall_pass_rate: float
    by_model: dict[str, float]
    by_strategy: dict[str, float]
    by_complexity: dict[str, float]


class FailureRecord(BaseModel):
    request_id: str
    routing_model: str
    routing_strategy: str
    routing_complexity: str
    score: float | None
    rationale: str | None
    created_at: datetime


class RecommendationResponse(BaseModel):
    signature: str
    rule_type: RuleType
    subject: str
    text: str
    evidence_confidence: float
    severity: Severity
    evidence: RecommendationEvidence
    status: str
    source: RecommendationSource
    created_at: datetime
    updated_at: datetime


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _group_pass_rate(rows: list[VerificationRow], key: str) -> dict[str, float]:
    grouped: dict[str, list[bool]] = {}
    for row in rows:
        grouped.setdefault(getattr(row, key), []).append(1.0 if row.passed else 0.0)
    return {name: _avg(outcomes) for name, outcomes in grouped.items()}


@router.get("/learning/summary", response_model=LearningSummary)
async def get_learning_summary(session_factory: SessionFactoryDep) -> LearningSummary:
    with session_factory() as session:
        rows = (
            session.query(VerificationRow)
            .filter_by(status=VerificationStatus.COMPLETED.value)
            .all()
        )

    return LearningSummary(
        total_verified=len(rows),
        overall_pass_rate=_avg([1.0 if r.passed else 0.0 for r in rows]),
        by_model=_group_pass_rate(rows, "routing_model"),
        by_strategy=_group_pass_rate(rows, "routing_strategy"),
        by_complexity=_group_pass_rate(rows, "routing_complexity"),
    )


@router.get("/learning/failures", response_model=list[FailureRecord])
async def get_learning_failures(session_factory: SessionFactoryDep) -> list[FailureRecord]:
    with session_factory() as session:
        rows = (
            session.query(VerificationRow)
            .filter_by(status=VerificationStatus.COMPLETED.value, passed=False)
            .order_by(VerificationRow.created_at.desc())
            .all()
        )
        return [
            FailureRecord(
                request_id=r.request_id, routing_model=r.routing_model,
                routing_strategy=r.routing_strategy, routing_complexity=r.routing_complexity,
                score=r.score, rationale=r.rationale, created_at=r.created_at,
            )
            for r in rows
        ]


@router.get("/learning/recommendations", response_model=list[RecommendationResponse])
async def get_learning_recommendations(
    learning_service: LearningServiceDep,
) -> list[RecommendationResponse]:
    rows = learning_service.refresh_recommendations()
    return [
        RecommendationResponse(
            signature=r.signature, rule_type=RuleType(r.rule_type), subject=r.subject,
            text=r.recommendation_text, evidence_confidence=r.evidence_confidence,
            severity=Severity(r.severity), evidence=RecommendationEvidence(**r.evidence),
            status=r.status, source=RecommendationSource(r.source),
            created_at=r.created_at, updated_at=r.updated_at,
        )
        for r in rows
    ]
```

- [ ] **Step 4: Modify `backend/api/dependencies.py`**

Change:
```python
from typing import Annotated

from fastapi import Depends, Request

from backend.chat.service import ChatService
from backend.config.settings import Settings
from backend.events.bus import EventBus
from backend.providers.manager import ProviderManager
from backend.services.model_registry import ModelRegistry
```

To:
```python
from typing import Annotated

from fastapi import Depends, Request

from backend.chat.service import ChatService
from backend.config.settings import Settings
from backend.events.bus import EventBus
from backend.learning.service import LearningService
from backend.providers.manager import ProviderManager
from backend.services.model_registry import ModelRegistry
```

Add, immediately after `get_chat_service`:
```python
def get_learning_service(request: Request) -> LearningService:
    return request.app.state.learning_service
```

Add, immediately after `ChatServiceDep`:
```python
LearningServiceDep = Annotated[LearningService, Depends(get_learning_service)]
```

- [ ] **Step 5: Modify `backend/api/main.py`**

Add to the import block (alongside the other `backend.api.routers.*` imports):
```python
from backend.api.routers.learning import router as learning_router
```

Add, alongside the other `backend.learning`/`backend.verification` imports:
```python
from backend.learning.detector import FailurePatternDetector
from backend.learning.generator import RecommendationGenerator
from backend.learning.rules import ComplexityTierRule, DetectionRuleConfig, ModelComplexityRule
from backend.learning.service import LearningService
```

Change `APP_VERSION = "0.3.0"` to `APP_VERSION = "0.4.0"`.

In `lifespan`, immediately after the `chat_service = ChatService(...)` block, add:
```python
    learning_service = LearningService(
        detector=FailurePatternDetector(rules=[
            ModelComplexityRule(DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.6)),
            ComplexityTierRule(DetectionRuleConfig(min_samples=30, pass_rate_threshold=0.5)),
        ]),
        generator=RecommendationGenerator(),
        session_factory=session_factory,
    )
```

In the `app.state.*` assignment block, add:
```python
    app.state.learning_service = learning_service
```

In `create_app()`, add:
```python
    app.include_router(learning_router, prefix="/v1")
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_learning_router.py -v`
Expected: PASS (3 tests)

- [ ] **Batch verification & commit**

Run the full suite:
```bash
uv run pytest -v
```
Expected: all tests pass (190 existing + new tests from Tasks 35-41; verify against actual collected count rather than assuming an exact number).

Manual end-to-end verification: seeding 20+ verification rows isn't practical via curl alone, so run the same boot-and-route smoke test as Phases 2-3 — confirm the app boots and the three new routes exist and return valid (empty/zero) responses against a fresh database:
```bash
uv run uvicorn backend.api.main:app --reload
```
```bash
curl -s http://localhost:8000/v1/health | python3 -m json.tool   # confirm version "0.4.0"
curl -s http://localhost:8000/v1/learning/summary | python3 -m json.tool        # expect total_verified: 0
curl -s http://localhost:8000/v1/learning/failures | python3 -m json.tool       # expect []
curl -s http://localhost:8000/v1/learning/recommendations | python3 -m json.tool # expect []
```

Commit and tag:
```bash
git add backend/learning backend/database/models.py backend/api/routers/learning.py backend/api/dependencies.py backend/api/main.py backend/tests/test_learning_rules.py backend/tests/test_failure_pattern_detector.py backend/tests/test_recommendation_generator.py backend/tests/test_recommendation_row.py backend/tests/test_learning_service.py backend/tests/test_learning_router.py
git commit -m "feat: add learning subsystem (detection rules, recommendations, learning API)"
git tag v0.4.0
```
