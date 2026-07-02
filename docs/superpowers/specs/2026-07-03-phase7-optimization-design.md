# LLM Cost Autopilot — Phase 7 Design: Optimization Engine

Status: **Approved — frozen as implementation contract**
Date: 2026-07-03

## 1. Purpose & Scope

Phase 7 answers: **is the platform spending more than it needs to for the
quality it's actually getting?** It extends the Phase 4 learning pipeline
(detector → generator → recommendation) with a second kind of finding —
cost-optimization, alongside the existing quality-failure findings — rather
than introducing a parallel `backend/optimization/` subsystem.

**Invariant carried over from Phase 4, restated explicitly for Phase 7:**
Optimization recommendations are **advisory only**. They never modify
routing policies, eligibility rules, model registry configuration, or
provider settings automatically. A human reads a recommendation and decides
whether to act on it, exactly as with quality recommendations today.

```
VerificationRows ──► FailurePatternDetector
                        ├─ ComplexityTierRule     (existing, unchanged)
                        ├─ ModelComplexityRule    (existing, unchanged)
                        └─ OverpoweredModelRule   (NEW)
                                │
                                ▼
                          list[Finding]
                                │
        ResponseRow + ModelRegistry ──► ModelCostMetrics   (NEW, built by LearningService)
                                │                    │
                                ▼                    ▼
                        RecommendationGenerator.generate(findings, cost_metrics)
                                ├─ quality findings         → existing text/severity logic (unchanged)
                                └─ COST_OPTIMIZATION findings → new selection + savings logic
                                │
                                ▼
                          list[Recommendation]
                                │
                        LearningService.refresh_recommendations()  (existing, unchanged persistence logic)
                                │
                        RecommendationRow  (evidence JSON gains optional `comparison`)
                                │
                        Dashboard "Learning Recommendations" card grid (existing, unchanged template)
```

**In scope:**
- `RuleType.COST_OPTIMIZATION` (new enum value in `backend/learning/rules.py`)
- `OverpoweredModelRule` (new `BaseDetectionRule`) — detects models
  passing reliably at a complexity tier; carries no pricing knowledge
- `ModelCostMetrics` (new value object) — per-`(model, complexity)`
  pricing + observed cost/volume data, including an
  `eligible_for_optimization` flag, built by `LearningService`
- `RecommendationSource.COST_OPTIMIZATION` (new enum value in
  `backend/learning/generator.py`)
- `RecommendationGenerator` extended: `generate(findings, cost_metrics)`
  — cost-optimization branch does alternative selection, savings math,
  severity computation, and text generation
- `ModelComparison` (new nested value object in evidence)
- Dashboard: no template changes — cost-optimization recommendations
  render in the existing "Learning Recommendations" card grid,
  distinguished by `source`/`rule_type`, same as quality recommendations
  today

**Explicitly out of scope for Phase 7:**
- Underutilized-provider detection, cheaper-provider (cross-provider,
  same-model) swaps — deferred; this phase only covers same-tier
  same-complexity model-to-model swaps within the existing routing
  eligibility model
- A new `backend/optimization/` package, new API endpoints, new
  database tables/columns, or a new dashboard section — everything
  routes through the existing Phase 4 recommendation pipeline
- Auto-applying any recommendation (see invariant above)
- Forecasting beyond a simple 30-day linear extrapolation of currently
  observed daily volume (Phase 8's concern)
- Provider API key management / multi-operator configuration
  (explicitly deferred, unrelated concern — see project memory)

## 2. Detector: `OverpoweredModelRule`

Same dependency shape as the existing two rules — only `VerificationRow`s,
**no** access to `ModelRegistry`, pricing, or `ResponseRow`. The detector
answers exactly one question:

> "Is this model consistently delivering acceptable quality for this
> complexity tier?"

Nothing about cheaper models, providers, costs, routing, savings, or
recommendations — those all belong to the generator.

**Logic:** for each `(model, complexity)` group (using
`row.routing_model`, `row.routing_complexity` from eligible
`VerificationRow`s, same grouping as `ModelComplexityRule`):
- if `sample_size < min_samples` → **no Finding emitted** for that pair.
  Not a low-confidence finding, not an informational one — silently
  ignored.
- if `sample_size >= min_samples` and `pass_rate >= minimum_pass_rate`
  (the field is named `minimum_pass_rate` for this rule, not `threshold`
  — the value means "good enough to optimize," the inverse sense of the
  existing rules' `threshold`, so the name must say so) → emit exactly
  **one** `Finding`:
  - `rule_type=RuleType.COST_OPTIMIZATION`
  - `subject="{model}:{complexity}"` — canonical model ID, not a display
    name, so signatures stay stable if a friendly name changes later
  - `sample_size`, `pass_rate` as observed
  - `threshold=minimum_pass_rate`

**Invariants:**
- Deterministic ordering: findings are emitted ordered by
  `(model ASC, complexity ASC)`, matching the stable iteration order
  the existing rules already rely on. Recommendation ordering must never
  change between runs on the same data.
- At most one `Finding` per `(model, complexity)` pair, regardless of
  input row volume — this must hold even if the underlying aggregation
  query changes later.
- Reuses the existing `Finding` model verbatim — no new fields added to
  `Finding` itself.

`DetectionRuleConfig` is reused as-is (`min_samples`,
`pass_rate_threshold` — for this rule instantiated with the
"minimum-to-optimize" value, which may differ from the quality rules'
threshold; both are just config values passed at construction).

## 3. `ModelCostMetrics`: cost data, owned by `LearningService`

The generator does not query the database or `ModelRegistry` directly —
`LearningService` builds this data once per `refresh_recommendations()`
call and passes it in, keeping the generator's dependencies explicit and
testable with plain objects.

```python
class ModelCostMetrics(BaseModel):
    model: str
    complexity: str
    input_cost: float             # per-million-token pricing, from ModelRegistry
    output_cost: float
    avg_cost_per_request: float   # from ResponseRow.actual_cost, same (model, complexity) window
    requests_per_day: float       # observed volume over the analysis window / window days
    pass_rate: float              # same value the detector computed, carried alongside for evidence
    eligible_for_optimization: bool
```

`eligible_for_optimization` is `True` when the same `(model, complexity)`
pair satisfies the detector's own criteria (`sample_size >= min_samples`
and `pass_rate >= minimum_pass_rate`) — computed independently by
`LearningService` from the same underlying data, not by reading the
`Finding` list. This is a deliberate decoupling: the generator's
candidate search filters `cost_metrics` directly
(`same complexity AND eligible_for_optimization`), not another list of
findings. A `Finding` only ever says "this model passed" for the
*current* model being evaluated; it says nothing about other models at
the same tier.

`LearningService` builds `dict[(model, complexity), ModelCostMetrics]`
by joining `VerificationRow` (for `pass_rate`/`sample_size`, mirroring the
detector's own grouping) with `ResponseRow.actual_cost` (for
`avg_cost_per_request` and `requests_per_day`) and `ModelRegistry` (for
per-model pricing).

## 4. Generator: selection, savings, and recommendation text

`RecommendationGenerator.generate()` signature changes from
`generate(findings)` to `generate(findings, cost_metrics)`. The
`cost_metrics` argument is only consulted for `COST_OPTIMIZATION`-type
findings — the two existing quality-recommendation code paths
(`MODEL_COMPLEXITY`, `COMPLEXITY_TIER`) are untouched.

**For each `COST_OPTIMIZATION` finding on `(current_model, complexity)`:**

1. Build the candidate set: all entries in `cost_metrics` where
   `complexity` matches and `eligible_for_optimization` is `True`
   (this naturally includes `current_model` itself).
2. Pick the candidate with the lowest `avg_cost_per_request` — the
   **best eligible cheaper model**, not merely "the cheapest model
   overall." If that candidate *is* `current_model` (no cheaper eligible
   alternative exists), **no recommendation is emitted** for this
   finding — skip silently, same treatment as the sample-size case in
   the detector.
3. Invariant: **savings must never be negative.** If
   `current_cost_per_request <= suggested_cost_per_request` for any
   reason, skip and emit nothing. A recommendation must never read
   "save -$8/month."
4. Compute `estimated_monthly_savings =`
   `round((current_cost_per_request - suggested_cost_per_request) * requests_per_day * 30, 2)`
   — rounded to 2 decimal places before persistence, so floating-point
   noise never leaks into the stored JSON or the rendered text.

**`ModelComparison`** (new, nested inside `RecommendationEvidence`):

```python
class ModelComparison(BaseModel):
    current_model: str
    suggested_model: str
    current_pass_rate: float
    suggested_pass_rate: float
    current_cost_per_request: float
    suggested_cost_per_request: float
    estimated_monthly_savings: float
```

Carrying both pass rates makes the comparison self-contained: a reader
can see *why* the cheaper model is an acceptable substitute without
cross-referencing anything else.

`RecommendationEvidence` gains an optional field:
`comparison: ModelComparison | None = None` — `None` for existing
quality recommendations, populated only for `COST_OPTIMIZATION` ones.
No migration needed; `evidence` is already a JSON column.

**Severity**, computed from `estimated_monthly_savings` using named
constants (matching the Phase 4 style of centralized-but-hardcoded
config, not YAML):

```python
_LOW_SAVINGS_CEILING = 10.0     # < $10/month -> LOW
_MEDIUM_SAVINGS_CEILING = 100.0 # $10-100/month -> MEDIUM
                                 # > $100/month -> HIGH
```

Reuses the existing `RecommendationRow.severity` column and its existing
sort order (`severity.desc()`) — no schema change. `severity` becomes a
generic "how much does this matter" signal, computed differently per
`source` but stored and sorted identically.

**Recommendation text** — templated from the `Finding`/`ModelComparison`
data, deliberately *not* hardcoding percentages into prose (so wording
stays stable if thresholds change later):

> "Current model '{current_model}' consistently meets the quality
> threshold for '{complexity}' prompts. A lower-cost model,
> '{suggested_model}', also meets the threshold. Estimated monthly
> savings: ~${estimated_monthly_savings}."

`RecommendationSource.COST_OPTIMIZATION` is added alongside the existing
`RecommendationSource.VERIFICATION`.

## 5. Persistence & dashboard — unchanged

`LearningService.refresh_recommendations()` keeps its existing logic
verbatim: upsert by `signature`, never overwrite human-owned `status`.
`RecommendationRow` schema is unchanged (`evidence` JSON column already
supports the new nested `comparison` shape). The dashboard's "Learning
Recommendations" card grid (`backend/templates/dashboard.html`,
`section-recommendations`) requires no template changes — it already
renders `rec.subject`, `rec.text`, `rec.evidence_confidence` generically
regardless of `source`.

## 6. Testing

Matches Phase 4's existing test structure and file layout
(`backend/tests/test_learning_*.py`) — no new test infrastructure.

- **`OverpoweredModelRule`**: mirrors `ModelComplexityRule`'s existing
  tests — passing/failing threshold boundary, sample-size boundary,
  deterministic `(model, complexity)` ordering, at-most-one-Finding-per-pair.
- **`RecommendationGenerator` cost path**: candidate filtering
  (`eligible_for_optimization` + same complexity), cheapest-selection,
  no-cheaper-alternative skip, negative-savings skip, rounding, severity
  boundaries (`$9.99`→LOW, `$10.00`→MEDIUM, `$100.00`→MEDIUM,
  `$100.01`→HIGH).
- **`LearningService`**: integration test verifying `ModelCostMetrics`
  construction from seeded `ResponseRow`/`VerificationRow` data produces
  correct `avg_cost_per_request`/`requests_per_day`/
  `eligible_for_optimization`, and that a full `refresh_recommendations()`
  run persists a `COST_OPTIMIZATION` recommendation with the expected
  `comparison` evidence.
- **End-to-end**: extend the existing dashboard fixture test to assert a
  cost-optimization recommendation renders in the "Learning
  Recommendations" card grid alongside a quality one.

## 7. Implementation workflow

Same pattern as Phases 1-6:

1. Write implementation plan (next step after this spec is reviewed).
2. Implement in 2-3 batched iterations (batched subagent dispatch, task
   reviewer after each batch).
3. Full regression suite after each batch.
4. One manual end-to-end verification: confirm a cost-optimization
   recommendation appears in the running dashboard.
5. Whole-branch review.
6. Tag **v0.7.0**.
