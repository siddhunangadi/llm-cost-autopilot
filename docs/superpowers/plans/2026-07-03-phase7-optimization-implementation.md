# Phase 7 Optimization Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a cost-optimization recommendation to the existing Phase 4 learning pipeline that flags models delivering reliably passing quality at a complexity tier when a cheaper model, with its own proven passing pass-rate, is available.

**Architecture:** Extends `backend/learning/` in place — a new detection rule (`OverpoweredModelRule`) that stays ignorant of pricing, a new cost-metrics builder that joins `VerificationRow`/`ResponseRow`/`ModelRegistry`, and a generator branch that performs alternative selection and savings math. No new top-level package, no schema migration, no new API endpoints, no dashboard template changes.

**Tech Stack:** Python 3.13, SQLAlchemy 2.0 (ORM), Pydantic v2, pytest.

## Global Constraints

- Frozen design doc: `docs/superpowers/specs/2026-07-03-phase7-optimization-design.md` — every task below implements one section of it; do not deviate without checking that doc first.
- Optimization recommendations are advisory only — never write to routing policy, model registry, or provider config.
- `OverpoweredModelRule` must never access `ModelRegistry`, pricing, or `ResponseRow` — detector stays pricing-ignorant (spec §2).
- `Finding` model is reused verbatim — no new fields added to it (spec §2).
- `RecommendationRow` schema is unchanged — no migration. New data goes into the existing `evidence` JSON column (spec §5).
- Savings must never be negative; a skip (no recommendation) is always preferred over a wrong number (spec §4).
- `estimated_monthly_savings` is rounded to 2 decimal places before it reaches evidence or text (spec §4).
- Severity bands: `< $10/month` → LOW, `$10.00–$100.00/month` → MEDIUM, `> $100.00/month` → HIGH (spec §4).

---

## Task 1: `OverpoweredModelRule` detector

**Files:**
- Modify: `backend/learning/rules.py`
- Test: `backend/tests/test_learning_rules.py`

**Interfaces:**
- Consumes: existing `BaseDetectionRule`, `Finding`, `DetectionRuleConfig`, `VerificationRow`, `VerificationStatus`.
- Produces: `RuleType.COST_OPTIMIZATION` (new enum member), `OverpoweredModelRule` class, and a renamed public helper `eligible_verification_rows(rows: list[VerificationRow]) -> list[VerificationRow]` (was private `_eligible_rows`) — Task 2 imports this helper.

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/test_learning_rules.py` (below the existing `ComplexityTierRule` tests, reusing the existing `_row` helper already defined in that file):

```python
from backend.learning.rules import OverpoweredModelRule


def test_overpowered_model_rule_emits_finding_at_or_above_threshold():
    rows = (
        [_row("gpt-4o", "balanced", "complex", passed=True) for _ in range(18)]
        + [_row("gpt-4o", "balanced", "complex", passed=False) for _ in range(2)]
    )  # 20 samples, pass_rate = 0.9 >= 0.7
    rule = OverpoweredModelRule(DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.7))

    findings = rule.evaluate(rows)

    assert len(findings) == 1
    assert findings[0].rule_type == RuleType.COST_OPTIMIZATION
    assert findings[0].subject == "gpt-4o:complex"
    assert findings[0].sample_size == 20
    assert findings[0].pass_rate == pytest.approx(0.9)
    assert findings[0].threshold == 0.7


def test_overpowered_model_rule_skips_below_min_samples():
    rows = [_row("gpt-4o", "balanced", "complex", passed=True) for _ in range(5)]
    rule = OverpoweredModelRule(DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.7))

    assert rule.evaluate(rows) == []


def test_overpowered_model_rule_skips_below_pass_rate():
    rows = (
        [_row("gpt-4o", "balanced", "complex", passed=True) for _ in range(13)]
        + [_row("gpt-4o", "balanced", "complex", passed=False) for _ in range(7)]
    )  # pass_rate = 0.65 < 0.7
    rule = OverpoweredModelRule(DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.7))

    assert rule.evaluate(rows) == []


def test_overpowered_model_rule_excludes_non_completed_rows():
    rows = (
        [_row("gpt-4o", "balanced", "complex", passed=True) for _ in range(20)]
        + [
            _row("gpt-4o", "balanced", "complex", passed=None, status=VerificationStatus.FAILED.value)
            for _ in range(50)
        ]
    )
    rule = OverpoweredModelRule(DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.7))

    findings = rule.evaluate(rows)

    assert len(findings) == 1
    assert findings[0].sample_size == 20  # the 50 non-completed rows are excluded


def test_overpowered_model_rule_at_most_one_finding_per_pair():
    rows = [_row("gpt-4o", "balanced", "complex", passed=True) for _ in range(500)]
    rule = OverpoweredModelRule(DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.7))

    findings = rule.evaluate(rows)

    assert len(findings) == 1


def test_overpowered_model_rule_deterministic_ordering():
    rows = (
        [_row("gpt-4o-mini", "balanced", "simple", passed=True) for _ in range(20)]
        + [_row("gpt-4o", "balanced", "complex", passed=True) for _ in range(20)]
        + [_row("claude-3-haiku", "balanced", "medium", passed=True) for _ in range(20)]
    )
    rule = OverpoweredModelRule(DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.7))

    findings = rule.evaluate(rows)

    subjects = [f.subject for f in findings]
    assert subjects == sorted(subjects)  # (model, complexity) ascending
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest backend/tests/test_learning_rules.py -v -k overpowered`
Expected: FAIL with `ImportError: cannot import name 'OverpoweredModelRule'`

- [ ] **Step 3: Implement `OverpoweredModelRule` and rename `_eligible_rows`**

In `backend/learning/rules.py`, make these exact changes:

Replace:
```python
class RuleType(str, Enum):
    MODEL_COMPLEXITY = "model_complexity"
    COMPLEXITY_TIER = "complexity_tier"
```
with:
```python
class RuleType(str, Enum):
    MODEL_COMPLEXITY = "model_complexity"
    COMPLEXITY_TIER = "complexity_tier"
    COST_OPTIMIZATION = "cost_optimization"
```

Replace:
```python
def _eligible_rows(rows: list[VerificationRow]) -> list[VerificationRow]:
    return [r for r in rows if r.status == VerificationStatus.COMPLETED.value and r.passed is not None]
```
with:
```python
def eligible_verification_rows(rows: list[VerificationRow]) -> list[VerificationRow]:
    return [r for r in rows if r.status == VerificationStatus.COMPLETED.value and r.passed is not None]
```

Update both call sites (`ModelComplexityRule.evaluate` and `ComplexityTierRule.evaluate`) from `_eligible_rows(rows)` to `eligible_verification_rows(rows)`.

Append a new class after `ComplexityTierRule`:

```python
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
        for (model, complexity) in sorted(groups.keys()):
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest backend/tests/test_learning_rules.py -v`
Expected: PASS (all tests, including the pre-existing ones — the rename must not break `ModelComplexityRule`/`ComplexityTierRule` tests)

- [ ] **Step 5: Run the full regression suite**

Run: `source .venv/bin/activate && pytest -q`
Expected: all tests pass (286 pre-existing + new ones)

- [ ] **Step 6: Commit**

```bash
git add backend/learning/rules.py backend/tests/test_learning_rules.py
git commit -m "feat: add OverpoweredModelRule detection rule for Phase 7"
```

---

## Task 2: `ModelCostMetrics` builder

**Files:**
- Create: `backend/learning/cost_metrics.py`
- Test: `backend/tests/test_cost_metrics.py`

**Interfaces:**
- Consumes: `VerificationRow`, `eligible_verification_rows` (from Task 1), `DetectionRuleConfig` (from `backend/learning/rules.py`). Consumes a `model_registry` object satisfying `get_model(model_id: str) -> <object with .input_cost, .output_cost>` — production callers pass a real `ModelRegistry`; tests pass a minimal fake.
- Produces: `ModelCostMetrics` (Pydantic model) and `build_model_cost_metrics(verification_rows, cost_by_request_id, model_registry, config) -> dict[tuple[str, str], ModelCostMetrics]` — Task 3 (generator) and Task 4 (`LearningService`) both use this.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_cost_metrics.py`:

```python
from types import SimpleNamespace

import pytest

from backend.database.models import VerificationRow
from backend.learning.cost_metrics import build_model_cost_metrics
from backend.learning.rules import DetectionRuleConfig
from backend.verification.status import VerificationStatus


class _FakeModelRegistry:
    def __init__(self, pricing: dict[str, tuple[float, float]]) -> None:
        self._pricing = pricing

    def get_model(self, model_id: str):
        input_cost, output_cost = self._pricing[model_id]
        return SimpleNamespace(input_cost=input_cost, output_cost=output_cost)


def _row(model, complexity, passed, request_id, created_at, status=VerificationStatus.COMPLETED.value):
    return VerificationRow(
        request_id=request_id, status=status, routing_model=model,
        routing_strategy="balanced", routing_complexity=complexity, passed=passed,
        created_at=created_at,
    )


def _dt(day):
    from datetime import datetime, timezone
    return datetime(2026, 7, day, tzinfo=timezone.utc)


def test_builds_metrics_for_pair_with_cost_data():
    rows = [
        _row("gpt-4o", "complex", True, f"req-{i}", _dt(1 + i % 5)) for i in range(20)
    ]
    costs = {f"req-{i}": 0.05 for i in range(20)}
    registry = _FakeModelRegistry({"gpt-4o": (2.50, 10.00)})
    config = DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.7)

    metrics = build_model_cost_metrics(rows, costs, registry, config)

    key = ("gpt-4o", "complex")
    assert key in metrics
    assert metrics[key].model == "gpt-4o"
    assert metrics[key].complexity == "complex"
    assert metrics[key].input_cost == 2.50
    assert metrics[key].output_cost == 10.00
    assert metrics[key].avg_cost_per_request == pytest.approx(0.05)
    assert metrics[key].pass_rate == pytest.approx(1.0)
    assert metrics[key].eligible_for_optimization is True


def test_skips_pair_with_no_cost_data():
    rows = [_row("gpt-4o", "complex", True, f"req-{i}", _dt(1)) for i in range(20)]
    registry = _FakeModelRegistry({"gpt-4o": (2.50, 10.00)})
    config = DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.7)

    metrics = build_model_cost_metrics(rows, {}, registry, config)

    assert metrics == {}


def test_eligible_for_optimization_false_below_min_samples():
    rows = [_row("gpt-4o", "complex", True, f"req-{i}", _dt(1)) for i in range(5)]
    costs = {f"req-{i}": 0.05 for i in range(5)}
    registry = _FakeModelRegistry({"gpt-4o": (2.50, 10.00)})
    config = DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.7)

    metrics = build_model_cost_metrics(rows, costs, registry, config)

    key = ("gpt-4o", "complex")
    assert metrics[key].eligible_for_optimization is False


def test_eligible_for_optimization_false_below_pass_rate():
    rows = (
        [_row("gpt-4o", "complex", True, f"req-{i}", _dt(1)) for i in range(10)]
        + [_row("gpt-4o", "complex", False, f"req-fail-{i}", _dt(1)) for i in range(10)]
    )  # pass_rate = 0.5 < 0.7
    costs = {**{f"req-{i}": 0.05 for i in range(10)}, **{f"req-fail-{i}": 0.05 for i in range(10)}}
    registry = _FakeModelRegistry({"gpt-4o": (2.50, 10.00)})
    config = DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.7)

    metrics = build_model_cost_metrics(rows, costs, registry, config)

    key = ("gpt-4o", "complex")
    assert metrics[key].eligible_for_optimization is False


def test_requests_per_day_uses_observed_date_span():
    rows = [_row("gpt-4o", "complex", True, f"req-{i}", _dt(1 + (i % 5))) for i in range(20)]
    costs = {f"req-{i}": 0.05 for i in range(20)}
    registry = _FakeModelRegistry({"gpt-4o": (2.50, 10.00)})
    config = DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.7)

    metrics = build_model_cost_metrics(rows, costs, registry, config)

    # created_at spans day 1 to day 5 -> window_days = 4, sample_size = 20
    assert metrics[("gpt-4o", "complex")].requests_per_day == pytest.approx(20 / 4)


def test_skips_pair_when_model_unknown_to_registry():
    rows = [_row("retired-model", "complex", True, f"req-{i}", _dt(1)) for i in range(20)]
    costs = {f"req-{i}": 0.05 for i in range(20)}
    registry = _FakeModelRegistry({})  # empty -- get_model raises KeyError
    config = DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.7)

    metrics = build_model_cost_metrics(rows, costs, registry, config)

    assert metrics == {}


def test_excludes_non_completed_rows():
    rows = (
        [_row("gpt-4o", "complex", True, f"req-{i}", _dt(1)) for i in range(20)]
        + [
            _row("gpt-4o", "complex", None, f"req-fail-{i}", _dt(1), status=VerificationStatus.FAILED.value)
            for i in range(50)
        ]
    )
    costs = {f"req-{i}": 0.05 for i in range(20)}
    registry = _FakeModelRegistry({"gpt-4o": (2.50, 10.00)})
    config = DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.7)

    metrics = build_model_cost_metrics(rows, costs, registry, config)

    assert metrics[("gpt-4o", "complex")].pass_rate == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest backend/tests/test_cost_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.learning.cost_metrics'`

- [ ] **Step 3: Implement `backend/learning/cost_metrics.py`**

```python
from collections import defaultdict
from typing import Protocol

from pydantic import BaseModel

from backend.database.models import VerificationRow
from backend.learning.rules import DetectionRuleConfig, eligible_verification_rows


class _ModelPricing(Protocol):
    input_cost: float
    output_cost: float


class _ModelRegistryLike(Protocol):
    def get_model(self, model_id: str) -> _ModelPricing: ...


class ModelCostMetrics(BaseModel):
    model: str
    complexity: str
    input_cost: float
    output_cost: float
    avg_cost_per_request: float
    requests_per_day: float
    pass_rate: float
    eligible_for_optimization: bool


def build_model_cost_metrics(
    verification_rows: list[VerificationRow],
    cost_by_request_id: dict[str, float],
    model_registry: _ModelRegistryLike,
    config: DetectionRuleConfig,
) -> dict[tuple[str, str], ModelCostMetrics]:
    groups: dict[tuple[str, str], list[VerificationRow]] = defaultdict(list)
    for row in eligible_verification_rows(verification_rows):
        groups[(row.routing_model, row.routing_complexity)].append(row)

    metrics: dict[tuple[str, str], ModelCostMetrics] = {}
    for (model, complexity), rows in groups.items():
        sample_size = len(rows)
        pass_rate = sum(1 for r in rows if r.passed) / sample_size

        costed_rows = [r for r in rows if r.request_id in cost_by_request_id]
        if not costed_rows:
            continue

        try:
            spec = model_registry.get_model(model)
        except KeyError:
            continue

        avg_cost_per_request = (
            sum(cost_by_request_id[r.request_id] for r in costed_rows) / len(costed_rows)
        )
        window_days = max(
            (max(r.created_at for r in costed_rows) - min(r.created_at for r in costed_rows)).days,
            1,
        )
        requests_per_day = sample_size / window_days

        metrics[(model, complexity)] = ModelCostMetrics(
            model=model,
            complexity=complexity,
            input_cost=spec.input_cost,
            output_cost=spec.output_cost,
            avg_cost_per_request=avg_cost_per_request,
            requests_per_day=requests_per_day,
            pass_rate=pass_rate,
            eligible_for_optimization=(
                sample_size >= config.min_samples and pass_rate >= config.pass_rate_threshold
            ),
        )
    return metrics
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest backend/tests/test_cost_metrics.py -v`
Expected: PASS (all 7 tests)

- [ ] **Step 5: Run the full regression suite**

Run: `source .venv/bin/activate && pytest -q`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add backend/learning/cost_metrics.py backend/tests/test_cost_metrics.py
git commit -m "feat: add ModelCostMetrics builder joining verification and cost data"
```

---

## Task 3: `RecommendationGenerator` cost-optimization branch

**Files:**
- Modify: `backend/learning/generator.py`
- Test: `backend/tests/test_recommendation_generator.py`

**Interfaces:**
- Consumes: `Finding`, `RuleType.COST_OPTIMIZATION` (Task 1), `ModelCostMetrics` (Task 2).
- Produces: `RecommendationSource.COST_OPTIMIZATION`, `ModelComparison`, `RecommendationEvidence.comparison` field, and the new `RecommendationGenerator.generate(findings, cost_metrics=None)` signature — Task 4 (`LearningService`) calls this with real `cost_metrics`.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_recommendation_generator.py`:

```python
from backend.learning.cost_metrics import ModelCostMetrics
from backend.learning.generator import ModelComparison, RecommendationSource


def _metrics(model, complexity, avg_cost, requests_per_day=10.0, pass_rate=0.9, eligible=True):
    return ModelCostMetrics(
        model=model, complexity=complexity, input_cost=1.0, output_cost=1.0,
        avg_cost_per_request=avg_cost, requests_per_day=requests_per_day,
        pass_rate=pass_rate, eligible_for_optimization=eligible,
    )


def test_generate_cost_optimization_picks_cheapest_eligible_alternative():
    finding = Finding(
        rule_type=RuleType.COST_OPTIMIZATION, subject="gpt-4o:complex",
        sample_size=20, pass_rate=0.9, threshold=0.7,
    )
    cost_metrics = {
        ("gpt-4o", "complex"): _metrics("gpt-4o", "complex", avg_cost=0.10),
        ("gpt-4o-mini", "complex"): _metrics("gpt-4o-mini", "complex", avg_cost=0.02, pass_rate=0.75),
        ("claude-3-haiku", "complex"): _metrics("claude-3-haiku", "complex", avg_cost=0.05, pass_rate=0.8),
    }

    [rec] = RecommendationGenerator().generate([finding], cost_metrics)

    assert rec.signature == "cost_optimization:gpt-4o:complex"
    assert rec.source == RecommendationSource.COST_OPTIMIZATION
    assert rec.evidence.comparison.suggested_model == "gpt-4o-mini"  # cheapest eligible, not claude


def test_generate_cost_optimization_computes_rounded_monthly_savings():
    finding = Finding(
        rule_type=RuleType.COST_OPTIMIZATION, subject="gpt-4o:complex",
        sample_size=20, pass_rate=0.9, threshold=0.7,
    )
    cost_metrics = {
        ("gpt-4o", "complex"): _metrics("gpt-4o", "complex", avg_cost=0.10, requests_per_day=10.0),
        ("gpt-4o-mini", "complex"): _metrics("gpt-4o-mini", "complex", avg_cost=0.03),
    }

    [rec] = RecommendationGenerator().generate([finding], cost_metrics)

    # (0.10 - 0.03) * 10 requests/day * 30 = 21.00
    assert rec.evidence.comparison.estimated_monthly_savings == pytest.approx(21.00)


def test_generate_cost_optimization_skips_when_current_model_is_cheapest():
    finding = Finding(
        rule_type=RuleType.COST_OPTIMIZATION, subject="gpt-4o-mini:complex",
        sample_size=20, pass_rate=0.9, threshold=0.7,
    )
    cost_metrics = {
        ("gpt-4o-mini", "complex"): _metrics("gpt-4o-mini", "complex", avg_cost=0.02),
        ("gpt-4o", "complex"): _metrics("gpt-4o", "complex", avg_cost=0.10),
    }

    recs = RecommendationGenerator().generate([finding], cost_metrics)

    assert recs == []


def test_generate_cost_optimization_skips_when_current_not_in_cost_metrics():
    finding = Finding(
        rule_type=RuleType.COST_OPTIMIZATION, subject="gpt-4o:complex",
        sample_size=20, pass_rate=0.9, threshold=0.7,
    )
    cost_metrics = {("gpt-4o-mini", "complex"): _metrics("gpt-4o-mini", "complex", avg_cost=0.02)}

    recs = RecommendationGenerator().generate([finding], cost_metrics)

    assert recs == []


def test_generate_cost_optimization_ignores_ineligible_candidates():
    finding = Finding(
        rule_type=RuleType.COST_OPTIMIZATION, subject="gpt-4o:complex",
        sample_size=20, pass_rate=0.9, threshold=0.7,
    )
    cost_metrics = {
        ("gpt-4o", "complex"): _metrics("gpt-4o", "complex", avg_cost=0.10),
        # cheaper but not eligible (didn't clear the pass-rate/sample bar itself)
        ("gpt-4o-mini", "complex"): _metrics("gpt-4o-mini", "complex", avg_cost=0.01, eligible=False),
    }

    recs = RecommendationGenerator().generate([finding], cost_metrics)

    assert recs == []


def test_generate_cost_optimization_severity_bands():
    def _rec_for_savings(daily_delta):
        finding = Finding(
            rule_type=RuleType.COST_OPTIMIZATION, subject="gpt-4o:complex",
            sample_size=20, pass_rate=0.9, threshold=0.7,
        )
        cost_metrics = {
            ("gpt-4o", "complex"): _metrics("gpt-4o", "complex", avg_cost=daily_delta, requests_per_day=1.0 / 30),
            ("gpt-4o-mini", "complex"): _metrics("gpt-4o-mini", "complex", avg_cost=0.0),
        }
        [rec] = RecommendationGenerator().generate([finding], cost_metrics)
        return rec

    assert _rec_for_savings(9.99).severity == Severity.LOW
    assert _rec_for_savings(10.00).severity == Severity.MEDIUM
    assert _rec_for_savings(100.00).severity == Severity.MEDIUM
    assert _rec_for_savings(100.01).severity == Severity.HIGH


def test_generate_cost_optimization_text_mentions_both_models_and_savings():
    finding = Finding(
        rule_type=RuleType.COST_OPTIMIZATION, subject="gpt-4o:complex",
        sample_size=20, pass_rate=0.9, threshold=0.7,
    )
    cost_metrics = {
        ("gpt-4o", "complex"): _metrics("gpt-4o", "complex", avg_cost=0.10, requests_per_day=10.0),
        ("gpt-4o-mini", "complex"): _metrics("gpt-4o-mini", "complex", avg_cost=0.03),
    }

    [rec] = RecommendationGenerator().generate([finding], cost_metrics)

    assert "gpt-4o" in rec.text
    assert "gpt-4o-mini" in rec.text
    assert "21.00" in rec.text


def test_generate_mixed_findings_quality_and_cost():
    quality_finding = Finding(
        rule_type=RuleType.MODEL_COMPLEXITY, subject="gpt-4o-mini:medium",
        sample_size=20, pass_rate=0.35, threshold=0.6,
    )
    cost_finding = Finding(
        rule_type=RuleType.COST_OPTIMIZATION, subject="gpt-4o:complex",
        sample_size=20, pass_rate=0.9, threshold=0.7,
    )
    cost_metrics = {
        ("gpt-4o", "complex"): _metrics("gpt-4o", "complex", avg_cost=0.10, requests_per_day=10.0),
        ("gpt-4o-mini", "complex"): _metrics("gpt-4o-mini", "complex", avg_cost=0.03),
    }

    recs = RecommendationGenerator().generate([quality_finding, cost_finding], cost_metrics)

    assert len(recs) == 2
    assert {r.source for r in recs} == {RecommendationSource.VERIFICATION, RecommendationSource.COST_OPTIMIZATION}


def test_generate_without_cost_metrics_argument_still_works_for_quality_findings():
    finding = Finding(
        rule_type=RuleType.COMPLEXITY_TIER, subject="complex", sample_size=30, pass_rate=0.4, threshold=0.5,
    )
    [rec] = RecommendationGenerator().generate([finding])  # no cost_metrics arg -- backward compatible
    assert rec.source == RecommendationSource.VERIFICATION
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest backend/tests/test_recommendation_generator.py -v -k cost_optimization`
Expected: FAIL with `ImportError: cannot import name 'ModelComparison'`

- [ ] **Step 3: Implement the generator changes**

Replace the full contents of `backend/learning/generator.py`:

```python
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

        best = min(candidates, key=lambda m: m.avg_cost_per_request)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest backend/tests/test_recommendation_generator.py -v`
Expected: PASS (all tests, including pre-existing ones)

- [ ] **Step 5: Run the full regression suite**

Run: `source .venv/bin/activate && pytest -q`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add backend/learning/generator.py backend/tests/test_recommendation_generator.py
git commit -m "feat: add cost-optimization recommendation generation with savings math"
```

---

## Task 4: Wire `LearningService`, `main.py`, and dashboard end-to-end

**Files:**
- Modify: `backend/learning/service.py`
- Modify: `backend/api/main.py`
- Modify: `backend/tests/test_learning_service.py`
- Modify: `backend/tests/test_dashboard_ui.py`

**Interfaces:**
- Consumes: `build_model_cost_metrics` (Task 2), `RecommendationGenerator.generate(findings, cost_metrics)` (Task 3), `OverpoweredModelRule` (Task 1).
- Produces: `LearningService.__init__(detector, generator, session_factory, model_registry, cost_optimization_config)` — the final wiring; nothing downstream depends on this beyond `main.py`.

- [ ] **Step 1: Write the failing tests**

Replace the `_make_service` helper at the top of `backend/tests/test_learning_service.py` and add new tests. First, update imports and the helper:

```python
from datetime import datetime, timezone
from types import SimpleNamespace

from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.database.models import RecommendationRow, RequestRow, ResponseRow, VerificationRow
from backend.learning.detector import FailurePatternDetector
from backend.learning.generator import RecommendationGenerator
from backend.learning.rules import (
    ComplexityTierRule, DetectionRuleConfig, ModelComplexityRule, OverpoweredModelRule,
)
from backend.learning.service import LearningService
from backend.verification.status import VerificationStatus


class _FakeModelRegistry:
    def __init__(self, pricing: dict[str, tuple[float, float]]) -> None:
        self._pricing = pricing

    def get_model(self, model_id: str):
        input_cost, output_cost = self._pricing[model_id]
        return SimpleNamespace(input_cost=input_cost, output_cost=output_cost)


def _make_service(tmp_path, pricing=None):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    detector = FailurePatternDetector(rules=[
        ModelComplexityRule(DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.6)),
        ComplexityTierRule(DetectionRuleConfig(min_samples=30, pass_rate_threshold=0.5)),
        OverpoweredModelRule(DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.7)),
    ])
    service = LearningService(
        detector=detector,
        generator=RecommendationGenerator(),
        session_factory=session_factory,
        model_registry=_FakeModelRegistry(pricing or {}),
        cost_optimization_config=DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.7),
    )
    return service, session_factory
```

Every existing call site of `_make_service(tmp_path)` in this file stays as-is (the new `pricing` parameter defaults to `None`). Now append new tests at the end of the file:

```python
def _seed_passing_model_with_cost(session_factory, model, cost, count=20, prefix="req"):
    with session_factory() as session:
        base_day = 1
        for i in range(count):
            request_id = f"{prefix}-{model}-{i}"
            created = datetime(2026, 7, base_day + (i % 5), tzinfo=timezone.utc)
            session.add(RequestRow(request_id=request_id, prompt="hi", strategy="balanced"))
            session.add(VerificationRow(
                request_id=request_id, status=VerificationStatus.COMPLETED.value,
                routing_model=model, routing_strategy="balanced",
                routing_complexity="complex", passed=True, created_at=created,
            ))
            session.add(ResponseRow(request_id=request_id, response_text="ok", actual_cost=cost))
        session.commit()


def test_refresh_inserts_cost_optimization_recommendation(tmp_path):
    pricing = {"gpt-4o": (2.50, 10.00), "gpt-4o-mini": (0.15, 0.60)}
    service, session_factory = _make_service(tmp_path, pricing=pricing)
    _seed_passing_model_with_cost(session_factory, "gpt-4o", cost=0.10, prefix="expensive")
    _seed_passing_model_with_cost(session_factory, "gpt-4o-mini", cost=0.02, prefix="cheap")

    results = service.refresh_recommendations()

    cost_recs = [r for r in results if r.source == "cost_optimization"]
    assert len(cost_recs) == 1
    assert cost_recs[0].signature == "cost_optimization:gpt-4o:complex"
    assert cost_recs[0].evidence["comparison"]["suggested_model"] == "gpt-4o-mini"


def test_refresh_omits_cost_recommendation_when_no_cheaper_alternative(tmp_path):
    pricing = {"gpt-4o-mini": (0.15, 0.60)}
    service, session_factory = _make_service(tmp_path, pricing=pricing)
    _seed_passing_model_with_cost(session_factory, "gpt-4o-mini", cost=0.02, prefix="only")

    results = service.refresh_recommendations()

    assert [r for r in results if r.source == "cost_optimization"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest backend/tests/test_learning_service.py -v`
Expected: FAIL with `TypeError: LearningService.__init__() missing 2 required keyword-only arguments`

- [ ] **Step 3: Implement `LearningService` changes**

Replace the full contents of `backend/learning/service.py`:

```python
from datetime import datetime, timezone

from sqlalchemy.orm import sessionmaker

from backend.database.models import RecommendationRow, ResponseRow, VerificationRow
from backend.learning.cost_metrics import build_model_cost_metrics
from backend.learning.detector import FailurePatternDetector
from backend.learning.generator import RecommendationGenerator
from backend.learning.rules import DetectionRuleConfig


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LearningService:
    def __init__(
        self,
        detector: FailurePatternDetector,
        generator: RecommendationGenerator,
        session_factory: sessionmaker,
        model_registry,
        cost_optimization_config: DetectionRuleConfig,
    ) -> None:
        self._detector = detector
        self._generator = generator
        self._session_factory = session_factory
        self._model_registry = model_registry
        self._cost_optimization_config = cost_optimization_config

    def refresh_recommendations(self) -> list[RecommendationRow]:
        with self._session_factory() as session:
            rows = session.query(VerificationRow).order_by(VerificationRow.id).all()
            cost_by_request_id = {
                r.request_id: r.actual_cost
                for r in session.query(ResponseRow).all()
                if r.actual_cost is not None
            }

        findings = self._detector.detect(rows)
        cost_metrics = build_model_cost_metrics(
            rows, cost_by_request_id, self._model_registry, self._cost_optimization_config
        )
        recommendations = self._generator.generate(findings, cost_metrics)

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

    def get_recommendations(self) -> list[RecommendationRow]:
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

(Only `__init__` and the top of `refresh_recommendations` changed from the existing file — the persistence loop and `get_recommendations` are unchanged; shown in full above to avoid ambiguity.)

- [ ] **Step 4: Wire `main.py`**

In `backend/api/main.py`, change the import line:

Replace:
```python
from backend.learning.rules import ComplexityTierRule, DetectionRuleConfig, ModelComplexityRule
```
with:
```python
from backend.learning.rules import (
    ComplexityTierRule, DetectionRuleConfig, ModelComplexityRule, OverpoweredModelRule,
)
```

Replace:
```python
    detector = FailurePatternDetector(rules=[
        ModelComplexityRule(DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.6)),
        ComplexityTierRule(DetectionRuleConfig(min_samples=30, pass_rate_threshold=0.5)),
    ])
    learning_service = LearningService(
        detector=detector,
        generator=RecommendationGenerator(),
        session_factory=session_factory,
    )
```
with:
```python
    cost_optimization_config = DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.75)
    detector = FailurePatternDetector(rules=[
        ModelComplexityRule(DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.6)),
        ComplexityTierRule(DetectionRuleConfig(min_samples=30, pass_rate_threshold=0.5)),
        OverpoweredModelRule(cost_optimization_config),
    ])
    learning_service = LearningService(
        detector=detector,
        generator=RecommendationGenerator(),
        session_factory=session_factory,
        model_registry=model_registry,
        cost_optimization_config=cost_optimization_config,
    )
```

Also bump the version constant. Replace:
```python
APP_VERSION = "0.6.1"
```
with:
```python
APP_VERSION = "0.7.0"
```

- [ ] **Step 5: Extend the dashboard end-to-end test**

In `backend/tests/test_dashboard_ui.py`, find the test that seeds a `RecommendationRow` and asserts it renders (or the fixture that populates the dashboard). Add a new test at the end of the file:

```python
def test_dashboard_renders_cost_optimization_recommendation(client, session_factory):
    with session_factory() as session:
        session.add(RecommendationRow(
            signature="cost_optimization:gpt-4o:complex",
            rule_type="cost_optimization",
            subject="gpt-4o:complex",
            recommendation_text=(
                "Current model 'gpt-4o' consistently meets the quality threshold for "
                "'complex' prompts. A lower-cost model, 'gpt-4o-mini', also meets the "
                "threshold. Estimated monthly savings: ~$21.00."
            ),
            evidence_confidence=0.6,
            severity="high",
            evidence={"sample_size": 20, "pass_rate": 0.9, "threshold": 0.7, "comparison": {
                "current_model": "gpt-4o", "suggested_model": "gpt-4o-mini",
                "current_pass_rate": 0.9, "suggested_pass_rate": 0.85,
                "current_cost_per_request": 0.10, "suggested_cost_per_request": 0.03,
                "estimated_monthly_savings": 21.00,
            }},
            status="new",
            source="cost_optimization",
        ))
        session.commit()

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert "gpt-4o-mini" in response.text
    assert "$21.00" in response.text
```

Check the existing test file's fixtures (`client`, `session_factory`) match this signature — if the existing tests in that file use a different fixture name or a different import for `RecommendationRow`, match those exactly rather than the names shown here (read the file's existing imports and fixtures first, they are the source of truth).

- [ ] **Step 6: Run all affected tests**

Run: `source .venv/bin/activate && pytest backend/tests/test_learning_service.py backend/tests/test_dashboard_ui.py -v`
Expected: PASS (all tests)

- [ ] **Step 7: Run the full regression suite**

Run: `source .venv/bin/activate && pytest -q`
Expected: all tests pass, 0 failures

- [ ] **Step 8: Manual end-to-end verification**

```bash
source .venv/bin/activate
cp .env.example .env  # if not already present
uvicorn backend.api.main:app --port 8000 &
sleep 2
curl -s http://127.0.0.1:8000/v1/health | head -c 200
curl -s http://127.0.0.1:8000/dashboard | grep -c "section-recommendations"
kill %1
```
Expected: health check returns 200; dashboard contains the recommendations section (count >= 1). Since no real traffic has been seeded, the recommendations list will legitimately be empty on a fresh DB — confirm the *plumbing* works (no 500 error, `cost_optimization_config` wired without a `TypeError` on startup), not that a fabricated recommendation appears.

- [ ] **Step 9: Commit**

```bash
git add backend/learning/service.py backend/api/main.py backend/tests/test_learning_service.py backend/tests/test_dashboard_ui.py
git commit -m "feat: wire cost-optimization recommendations through LearningService and bump to v0.7.0"
```

- [ ] **Step 10: Tag the release**

```bash
git tag v0.7.0
```

---

## Post-implementation checklist

- [ ] Full regression suite green (`pytest -q`)
- [ ] Manual verification: dashboard `/dashboard` loads without error, `section-recommendations` present
- [ ] Whole-branch review requested (per project convention: a reviewer checks plan alignment, code quality, architecture, testing, before merge/tag is considered final)
- [ ] `git tag --list` shows `v0.7.0`
- [ ] Update project memory (`~/.claude/projects/-Users-siddhunangadi/memory/project_llm_cost_autopilot_v040.md`) with the new version and Phase 8 as the next unscoped phase
