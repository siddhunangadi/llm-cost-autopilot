# Changelog

All notable changes to this project are documented here. Versions
correspond to the `vX.Y.0` git tags marking the end of each phase.

## v0.8.0 — Advanced Analytics & Reporting (2026-07-03)

Adds historical, trend-oriented analytics on top of the Phase 6/6b
operations dashboard and Phase 7 optimization engine. Where the
operations dashboard answers "what is happening right now," this phase
answers "how has the platform evolved over the last N days." This is the
last phase on the project's roadmap.

**Added**
- `AnalyticsService` — read-only, composes five daily trends into one
  `AnalyticsReport`; never writes, never refreshes recommendations, never
  touches routing or learning state (same invariant as `DashboardService`)
- `DashboardRepository.get_failover_trend` / `get_routing_distribution` /
  `get_recommendation_trend` — three new query methods, following the
  existing fetch-then-group-in-Python pattern already used by
  `get_cost_trend`/`get_quality_trend`
- `GET /v1/analytics/report?days=30` — JSON API, reuses the existing
  `TimeWindow` abstraction, `days` query param
- `GET /dashboard/analytics?days=30` — single-render HTML page (cost,
  quality, routing distribution, failover, and recommendation-generation
  trend charts via Chart.js); explicitly **not** HTMX-polled, unlike
  `/dashboard` — analytics is historical, not live operational state
- Nav link between `/dashboard` and `/dashboard/analytics`

**Explicitly out of scope** (deferred, not planned for a future phase):
spend forecasting, provider-performance-over-time trends, recommendation
*impact* tracking (dollars actually saved after a recommendation was
acted on), any new database tables or migrations.

## v0.3.0 — Quality Verification & Evaluation (2026-07-02)

Adds an in-process, background LLM-as-judge pipeline that scores every
chat response for quality after `ChatService` returns it, without adding
latency to `/v1/chat` or risking its availability.

**Added**
- `BaseJudge` / `LLMJudge` — pure `(prompt, response) -> JudgeVerdict`
  evaluator; parses judge output through a Pydantic schema
  (`_JudgeResponseSchema.model_validate_json`), never manual `json.loads`
- `VerificationDimensions` (correctness, completeness,
  instruction_following, format_adherence) and `JudgeVerdict`, all score
  fields bounded to `[0.0, 1.0]`
- `JudgeEngine` — times a judge call, keeps `BaseJudge` implementations pure
- `VerificationService` — owns the full DB lifecycle
  (`PENDING -> RUNNING -> COMPLETED | FAILED`), persists before emitting
  events, snapshots the routing decision (`selected_model`, `strategy`,
  `complexity`) being verified
- `VerificationRow` table, `VerificationStatus` enum
- Typed verification events (`VerificationStarted`, `VerificationCompleted`,
  `VerificationFailed`) over the existing `EventBus`
- `GET /v1/chat/{request_id}/verification` — single verification result
- `GET /v1/metrics/quality` — aggregate quality metrics (average score,
  confidence, pass rate, queue/evaluation/total timing, breakdowns by
  model/strategy/complexity)
- `backend/config/verification.yaml` — judge model, pass threshold,
  prompt/schema version, loaded via `VerificationConfigLoader`
- `ChatService` schedules verification as a best-effort `BackgroundTasks`
  side effect — a broken or unavailable judge provider degrades to a
  `FAILED` verification row, never to a failed or delayed chat response

**Explicitly out of scope** (deferred): auto-escalation on low scores,
classifier retraining, feedback loop into routing, prompt optimization,
retry policies for failed verifications, distributed workers/queues.

**Stats:** 190 tests passing (29 new), 0 regressions.

## v0.2.0 — Intelligent Routing Engine (2026-07-02)

Adds a heuristic routing pipeline that analyzes a prompt, classifies its
complexity, selects a model via a pluggable strategy, and exposes it
through `POST /v1/chat` with full request/response/routing persistence.

**Added**
- `PromptAnalyzer` / `PromptFeatures` — deterministic feature extraction
  (constraint count, code/JSON/reasoning/comparison/analysis/creative/math/
  chain-of-thought signals, output-formatting detection, an
  `estimated_output_tokens` heuristic distinct from input length)
- `HeuristicComplexityClassifier` behind `BaseComplexityClassifier`,
  producing a `ClassificationResult` with human-readable `signals` and a
  boundary-distance `confidence` score
- `RoutingPolicy` — config-driven eligibility filtering by complexity tier
- Four routing strategies (`cost`, `latency`, `quality`, `balanced`) behind
  `BaseRoutingStrategy`, operating on a single immutable `RoutingContext`
- `ExplanationGenerator` — builds human-readable routing reasoning from the
  classifier's own signals, kept fully decoupled from `RoutingEngine`
- `RoutingEngine` — pure orchestration (`RoutingDecision`,
  `NoEligibleModelError`); never calls a provider or touches the database
- `ChatService` — the only component that calls both `RoutingEngine` and a
  provider; persists `requests`, `responses`, `routing_events`
- `POST /v1/chat` with a `strategy` parameter
  (`cost` / `latency` / `quality` / `balanced`)
- `backend/config/routing.yaml` — classifier thresholds, eligibility
  policy per tier, balanced-strategy weights, loaded via a single
  `RoutingConfigLoader` (no per-component YAML parsing)

**Stats:** 161 tests passing (64 new), 0 regressions.

## v0.1.0 — Project Skeleton & Provider Foundation

Establishes the provider abstraction, model registry, event bus, and
configuration/logging foundation the rest of the platform builds on.

**Added**
- `GET /v1/health` — service, database, and provider status
- `GET /v1/models` — full model registry (pricing, limits, capabilities,
  benchmark info)
- `ModelRegistry` — memory-first, backed by `backend/config/models.yaml`
  and persisted to SQLite; immutable read-only cache with atomic
  `reload()` / `refresh_provider_status()`; fails fast on
  malformed/invalid/duplicate config
- `OpenAIProvider` and `MockProvider` behind a shared `BaseProvider`
  interface; SDK exceptions translated into `ProviderError`, never leaked
- `ProviderManager` — the mandatory `mock` provider crashes startup if it
  fails to construct; the optional `openai` provider degrades gracefully
  to "disabled" if its construction fails
- In-process `EventBus` (`PROVIDER_AVAILABLE`, `PROVIDER_DISABLED`,
  `PROVIDER_FAILED`, `MODEL_REGISTERED`) with per-subscriber exception
  isolation
- Structured JSON logging (console + rotating file) with
  `contextvars`-based request context

**Stats:** 97 tests passing.
