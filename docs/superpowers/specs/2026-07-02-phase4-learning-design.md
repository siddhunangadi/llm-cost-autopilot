# LLM Cost Autopilot — Phase 4 Design: Self-Improvement & Optimization

Status: **Approved — frozen as implementation contract**
Date: 2026-07-02

## 1. Purpose & Scope

Phase 4 answers: **given what verification has learned, what should change?**
It turns Phase 3's verification data into advisory recommendations —
never automatic routing changes. The system observes and recommends; a
human decides.

```
VerificationRows (ordered by id)
        │
        ▼
Detection Rules (BaseDetectionRule implementations, run independently)
        │
        ▼
Findings
        │
        ▼
RecommendationGenerator
        │
        ▼
Recommendations
        │
        ▼
LearningService (upsert by signature)
        │
        ▼
learning_recommendations table

GET /v1/learning/summary          -> aggregate pass rates (no persistence)
GET /v1/learning/failures         -> individual failed VerificationRow records
GET /v1/learning/recommendations  -> refresh + return persisted recommendations
```

**In scope:**
- `BaseDetectionRule` + two implementations (`ModelComplexityRule`,
  `ComplexityTierRule`), each pure (`list[VerificationRow] -> list[Finding]`)
- `FailurePatternDetector` (runs all registered rules, deterministic)
- `RecommendationGenerator` (pure: `list[Finding] -> list[Recommendation]`,
  owns signature generation)
- `LearningService` (the only component that persists — upserts by
  signature, never touches `status`)
- `RecommendationRow` table
- `GET /v1/learning/summary`, `GET /v1/learning/failures`,
  `GET /v1/learning/recommendations`

**Explicitly out of scope for Phase 4** (deferred to a later phase):
- Automatic classifier retraining or online learning
- Automatic routing changes — recommendations are never auto-applied
- Reinforcement learning
- Autonomous prompt optimization
- A dedicated "learning dataset builder" / training-data export format —
  `GET /v1/learning/failures` already exposes the underlying records;
  building export/formatting logic for a training consumer that doesn't
  exist yet would be speculative
- A scheduler, cron, or background worker for recommendation refresh —
  computation stays on-demand, matching Phase 1-3's no-worker-process
  constraint
- YAML-configurable rule thresholds — see §4

## 2. Directory Structure

```
backend/
  learning/
    __init__.py
    rules.py              # BaseDetectionRule, DetectionRuleConfig, ModelComplexityRule, ComplexityTierRule
    detector.py             # Finding, FailurePatternDetector
    generator.py              # RecommendationEvidence, Recommendation, RuleType, Severity,
                                # RecommendationSource, RecommendationGenerator
    service.py                 # LearningService
  api/
    routers/
      learning.py                # LearningSummary, FailureRecord, RecommendationResponse,
                                   # GET /v1/learning/summary|failures|recommendations
  database/
    models.py                     # + RecommendationRow (modify)
```

## 3. Findings & Detection Rules

```python
class Finding(BaseModel):
    rule_type: RuleType
    subject: str              # e.g. "gpt-4o-mini:medium" or "complex"
    sample_size: int
    pass_rate: float
    threshold: float


class RuleType(str, Enum):
    MODEL_COMPLEXITY = "model_complexity"
    COMPLEXITY_TIER = "complexity_tier"
```

`Finding` answers "what statistically happened?" — no recommendation
text, no severity, no confidence. Those belong to the generator layer
(§4).

```python
class BaseDetectionRule(ABC):
    @abstractmethod
    def evaluate(self, rows: list[VerificationRow]) -> list[Finding]: ...
```

**Rule independence (invariant):** each rule receives the complete,
unfiltered `rows` list and returns its own `list[Finding]`. No rule ever
reads another rule's output, and `FailurePatternDetector` never shares
state between rule invocations. This is what keeps the rule set a true
plugin list — adding a third rule later means adding a class, never
touching the other two.

```python
class FailurePatternDetector:
    def __init__(self, rules: list[BaseDetectionRule]) -> None:
        self._rules = rules

    def detect(self, rows: list[VerificationRow]) -> list[Finding]:
        return [finding for rule in self._rules for finding in rule.evaluate(rows)]
```

**Determinism invariant:** `FailurePatternDetector.detect()` run twice
over the same `rows` (in the same order) must produce identical
`Finding`s, in the same order, every time. This holds because: `rules`
iterates in fixed construction order; each rule iterates `rows` in the
order given; no rule reads a clock, a random source, or depends on
dict/set iteration order. Callers are responsible for supplying `rows`
in a stable order — `LearningService` queries `VerificationRow` with
`ORDER BY id` (§6).

## 4. `DetectionRuleConfig` & Rule Implementations

```python
@dataclass(frozen=True)
class DetectionRuleConfig:
    min_samples: int
    pass_rate_threshold: float
```

Encapsulating the threshold pair in a small immutable object (rather than
bare module-level constants) means a future YAML-driven config can be
introduced later without changing `BaseDetectionRule`'s interface — but
no config file is built in Phase 4 itself: these are internal detection
heuristics, not user-facing policy, and adding YAML for two numbers now
would be premature relative to Phase 2/3's genuinely tunable
(`routing.yaml`, `verification.yaml`) config.

```python
class ModelComplexityRule(BaseDetectionRule):
    def __init__(self, config: DetectionRuleConfig) -> None:
        self._config = config

    def evaluate(self, rows: list[VerificationRow]) -> list[Finding]:
        # group rows by (routing_model, routing_complexity), compute pass_rate per group
        # emit a Finding for each group where sample_size >= config.min_samples
        # and pass_rate < config.pass_rate_threshold
        ...


class ComplexityTierRule(BaseDetectionRule):
    def __init__(self, config: DetectionRuleConfig) -> None:
        self._config = config

    def evaluate(self, rows: list[VerificationRow]) -> list[Finding]:
        # group rows by routing_complexity only (across all models), compute pass_rate per group
        # emit a Finding for each group where sample_size >= config.min_samples
        # and pass_rate < config.pass_rate_threshold
        ...
```

Constructed in `main.py` (Phase 4's wiring task) as:
```python
model_complexity_rule = ModelComplexityRule(
    DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.6)
)
complexity_tier_rule = ComplexityTierRule(
    DetectionRuleConfig(min_samples=30, pass_rate_threshold=0.5)
)
detector = FailurePatternDetector(rules=[model_complexity_rule, complexity_tier_rule])
```

## 5. `RecommendationGenerator`

```python
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
```

`evidence_confidence` is **statistical confidence in the recommendation
given the available sample size** — explicitly distinct from
`ClassificationResult.confidence` (Phase 2, boundary-distance in the
complexity classifier) and `JudgeVerdict.confidence` (Phase 3, the
judge's self-reported confidence in its own scoring). The name makes
this unambiguous without needing a comment at every call site. Computed
as `min(0.5 + (sample_size / 200), 0.95)` — more samples raise
confidence, capped below `1.0` since these are heuristic thresholds, not
causal proof.

```python
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
        else:  # RuleType.COMPLEXITY_TIER
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

`RecommendationGenerator` owns signature generation exclusively — no
caller constructs a signature — guaranteeing one canonical format
(`f"{rule_type.value}:{subject}"`).

## 6. `RecommendationRow` & `LearningService`

```
id, signature (unique, not null), rule_type, subject,
recommendation_text, evidence_confidence, severity,
evidence (JSON: sample_size, pass_rate, threshold),
status (default "new"), source,
created_at, updated_at
```

```python
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
                        signature=rec.signature, rule_type=rec.rule_type.value,
                        subject=rec.subject, recommendation_text=rec.text,
                        evidence_confidence=rec.evidence_confidence,
                        severity=rec.severity.value, evidence=rec.evidence.model_dump(),
                        status="new", source=rec.source.value,
                    ))
                else:
                    existing.recommendation_text = rec.text
                    existing.evidence_confidence = rec.evidence_confidence
                    existing.severity = rec.severity.value
                    existing.evidence = rec.evidence.model_dump()
                    existing.updated_at = _utcnow()
                    # existing.status is never touched here
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

**Status is owned exclusively by humans.** `refresh_recommendations()`
updates evidence fields (`recommendation_text`, `evidence_confidence`,
`severity`, `evidence`, `updated_at`) on an existing row but never writes
`status` — an operator's `acknowledged`/`dismissed` marking survives
every subsequent refresh even as the underlying pass rates shift. The
detector owns evidence; humans own workflow. (Phase 4 does not add an
endpoint to *change* `status` — that's a natural, separately-scoped
follow-up once the read-side API below is in place — but the column and
this invariant exist now so a future `PATCH` doesn't require a schema
migration.)

Ordering (`severity desc, evidence_confidence desc, updated_at desc`) is
fully deterministic even when two recommendations tie on both severity
and confidence.

## 7. API Endpoints (`backend/api/routers/learning.py`)

```python
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


@router.get("/learning/summary", response_model=LearningSummary)
async def get_learning_summary(session_factory: SessionFactoryDep) -> LearningSummary: ...

@router.get("/learning/failures", response_model=list[FailureRecord])
async def get_learning_failures(session_factory: SessionFactoryDep) -> list[FailureRecord]: ...

@router.get("/learning/recommendations", response_model=list[RecommendationResponse])
async def get_learning_recommendations(learning_service: LearningServiceDep) -> list[RecommendationResponse]: ...
```

- `GET /v1/learning/summary` — same on-demand aggregation style as
  Phase 3's `GET /v1/metrics/quality`, no persistence, computed directly
  over `COMPLETED` `VerificationRow`s.
- `GET /v1/learning/failures` — `VerificationRow` where `passed == False`,
  ordered by `created_at desc`.
- `GET /v1/learning/recommendations` — calls
  `LearningService.refresh_recommendations()` and returns the result
  directly (already ordered per §6).

`LearningServiceDep` (`backend/api/dependencies.py`, modified) follows
the exact same `Depends()`-reading-`app.state` pattern as `ChatServiceDep`
(Phase 2) and `SessionFactoryDep` (Phase 1); `LearningService` is
constructed once in `main.py`'s `lifespan`, alongside everything else.

## 8. Testing

Same discipline as Phases 1-3: `BaseDetectionRule` cannot be instantiated
without `evaluate()`; each rule is tested with fixed `VerificationRow`
fixtures crossing the `min_samples`/`pass_rate_threshold` boundaries
(below both, above one, above both); `FailurePatternDetector.detect()` is
tested for determinism (same input twice -> identical output) and for
rule independence (a rule's `evaluate()` is never called with another
rule's `Finding`s); `RecommendationGenerator` is tested for exact
signature format, exact `evidence_confidence` values at known sample
sizes, and severity thresholds; `LearningService` is tested end-to-end
against a real SQLite test database for insert, update-preserves-status,
and ordering; endpoint tests assert response shapes and that
`GET /v1/learning/recommendations` triggers a refresh (new
`VerificationRow` data between two calls changes the second response).

## 9. Tooling

No new dependencies — `pydantic`, SQLAlchemy, and the standard library
`dataclasses`/`enum` (already used throughout Phases 1-3) cover Phase 4.
