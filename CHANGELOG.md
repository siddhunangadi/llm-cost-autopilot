# Changelog

All notable changes to this project are documented here. Versions
correspond to the `vX.Y.0` git tags marking the end of each phase.

## Unreleased

Fixes `GET /` returning `{"detail":"Not Found"}` on the live demo -- no
route existed at the bare root, only `/dashboard` and the `/v1/*` API. Adds
a redirect from `/` to `/dashboard` so the root URL works.

Adds `benchmarks/run_benchmarks.py` (Phase A: Production Validation, per
CLAUDE.md's Portfolio Completion criteria -- prove documented performance
claims with evidence instead of assertion). Reuses the real
`RoutingEngine`/`ModelRegistry`/`HeuristicComplexityClassifier`/
`ProviderExecutor`/`CircuitBreaker` against production `models.yaml` and
`routing.yaml`, with `MockProvider` standing in for every provider (no
network calls, no real API keys). Four sections, written to
`benchmarks/report.md`:

- **Routing latency**: 300 iterations, avg 0.06ms / P95 0.12ms vs. the
  <50ms target -- PASS.
- **Classifier latency**: 300 iterations, avg 0.002ms / P95 0.003ms vs.
  the <10ms target -- PASS.
- **Load test**: 500 requests, 80.1% cost savings vs. the highest-cost
  model, 93.2% quality parity, full routing distribution. "Quality
  parity" here is `mean(selected model benchmark_score) /
  mean(baseline benchmark_score)` -- the static per-model quality rating
  from `models.yaml` metadata, not a live LLM-judge score or
  `VerificationService` pass rate (that real measured-output-quality data
  already exists separately on the dashboard's Quality metrics, just
  isn't wired into this static load test, which never calls a provider).
- **Provider failover**: drives a real `CircuitBreaker` through
  closed -> open -> half-open -> closed against a provider scripted to
  fail 3 times then recover, with assertions on each transition and the
  full transition log in the report -- not just "failover exists."

Covered by `backend/tests/test_benchmarks.py`, a fast smoke test at low
iteration counts so import/signature drift is caught in the normal test
run without needing a full benchmark pass every time.

Adds the Product Vision, User Journey, and Decision Hierarchy sections to
`CLAUDE.md` -- shifts the project's framing from "LLM router" to "AI Cost
Optimization platform" and establishes that features should be prioritized
by user outcome (savings, trust, insight) over engineering interest, with
an explicit instruction to inspect existing subsystems before adding new
ones.

Upgrades the waste-detection/cost-optimization recommendation card (Loop
3, Phase C: the last item in the "prove it saves money -> prove trust ->
prove it finds more savings" journey). Per the new Decision Hierarchy, this
was inspect-first: `OverpoweredModelRule` + `RecommendationGenerator`
already fully implement waste detection (current vs. suggested model,
cost-per-request delta, quality parity, estimated monthly savings,
confidence) -- the only gap was presentation. `dashboard.html`'s
recommendation card now renders `evidence.comparison` (when present) as a
structured layout matching the target UX (headline daily-savings figure,
model move, request count, cost/quality deltas, monthly total) instead of
a single prose sentence. Purely template-level -- no backend/API changes.
Existing `test_dashboard_renders_cost_optimization_recommendation` was
tightened to assert on the new structured fields (daily figure, request
count) rather than just substring-matching the old sentence.

Adds the routing decision card (Loop 3, Phase B: explainability, after
Phase A's savings KPI established the product's headline value). `RoutingEngine.route`
now computes `AlternativeModel` entries for every other eligible candidate
the policy considered -- estimated cost (via the same
`ModelRegistry.estimate_cost` call and token counts already used for the
selected model, no duplicated pricing math), cost delta vs. the selected
model, and quality delta (`benchmark_score` difference). Persisted on
`RoutingEventRow.alternatives` (new nullable JSON column) alongside the
existing `reasoning` field, and surfaced end-to-end:
`DashboardRepository.get_recent_requests` now returns `reasoning` and
`alternatives` on `RecentRequestRow`, rendered as an expandable native
`<details>` "Why {model}?" card under each row in the Recent Requests
table -- reasoning bullets plus an alternatives table showing cost/quality
deltas for models the router considered but didn't pick. Covered by a new
routing-engine test asserting alternatives are computed correctly for a
2-model eligible set.

Adds the "Savings vs Baseline" headline KPI (Loop 3, Phase A: the product's
core value proposition -- "how much is this actually saving me?" -- surfaced
before routing-decision explainability or waste insights, per the reasoning
that trust in the product's value has to land before trust in its mechanics).
`DashboardService._compute_savings` picks a baseline model (configurable via
`COST_BASELINE_MODEL_ID`; defaults to the highest combined
input+output-cost model in the registry when unset) and computes what this
window's actual traffic would have cost entirely on that one model, using
each response's real token counts (`DashboardRepository.get_token_totals`,
reusing `ModelRegistry.estimate_cost` -- no duplicated pricing math).
Exposed as `DashboardOverview.savings` and new
`total_cost`/`savings_amount`/`savings_percent`/`baseline_model_id` keys on
`get_overview_fragment()`, and rendered as the first thing on the dashboard
(`overview.html` fragment) when a baseline model exists. Covered by 3 new
tests verifying the default-baseline selection, an explicitly configured
baseline, and the zero-models edge case.

Adds automatic escalation: `VerificationService.verify` now marks
`VerificationRow.escalated` and emits `EventType.ESCALATION_TRIGGERED` whenever
a verdict falls below `pass_threshold`. This closes a gap where the CLAUDE.md
"Automatic escalation" objective and Verification Rules ("trigger escalation")
had zero implementation anywhere in `backend/` (confirmed via a full-repo
search for "escalat"). Escalated verifications feed the existing learning
pipeline unchanged — no changes to `backend/learning` were needed since it
already keys off `VerificationRow.passed`.

Extends escalation to a full audit-and-learning workflow, not just an event:
on a failed verdict, `VerificationService` automatically re-runs the same
prompt against a configured higher-tier model (`escalation_model_id` in
`verification.yaml`, reusing `ProviderExecutor.generate` and
`ModelRegistry.estimate_cost` -- no duplicated retry/circuit-breaker/pricing
logic) and records `escalated_model`, `escalation_cost_delta`,
`escalation_latency_ms`, and `quality_gap` (`pass_threshold - score`) on
`VerificationRow`, plus the same fields on `ESCALATION_TRIGGERED`. The
original user-facing response is never replaced -- verification (and
therefore escalation) runs after the response has already been returned,
per the existing "user latency must not increase because of verification"
rule -- so this is an audit/learning-pipeline signal, not response
replacement. Regeneration failures (model missing, provider down) are
caught and never propagate; `quality_gap` and the event are still recorded.

Fixes a structured-logging defect found during a field-by-field audit
against CLAUDE.md's Logging requirements: `logging.LoggerAdapter`'s default
`process()` silently replaced any caller-supplied `extra={...}` with the
adapter's own `{"component": ...}` dict, so `subscribers.py`'s event
payloads (verification score, escalation cost delta, provider failover
reason, etc.) never reached the logs -- confirmed live via a Docker
container's `event_emitted` lines showing `provider: null` during real
request handling. `get_logger()` now returns a `_MergingLoggerAdapter` that
merges instead of overwrites, and `JsonFormatter` now surfaces any extra
fields a caller passes rather than silently dropping them.

Also closes the remaining Logging-section gap: `complexity`, `final_model`,
`routing_reason`, token counts, `estimated_cost`, and request latency were
persisted to the database but never logged. `ChatService.chat` now emits one
`chat_request_completed` log line per request carrying all of them, plus a
SHA-256 `prompt_hash` -- never the raw prompt, per the "never log raw
prompts" rule.

Fixes a dashboard correctness bug found during a data-source audit:
`DashboardRepository.get_quality_aggregation` was the only aggregation
method with no `TimeWindow` filter -- every sibling method
(`get_cost_trend`, `get_quality_trend`, `get_failover_summary`,
`get_failover_events`, `get_cost_by_model`) already had one, confirmed by
each having an "excludes data outside window" test that this method
lacked. Since `DashboardService.get_overview`/`get_overview_fragment` call
it alongside the windowed cost/failover queries, the dashboard's "This
Week" selector silently showed all-time quality numbers next to windowed
cost numbers on the same page. `get_quality_aggregation` now accepts an
optional `window` and filters by `VerificationRow.created_at`;
`/v1/metrics/quality` (which has no date param) keeps its existing
all-time behavior by passing no window.

Adds Docker support: a `Dockerfile` (multi-stage `uv sync --frozen --no-dev`,
runs via `uv run --no-sync` so the container never re-resolves dev
dependencies at startup) and a matching `.dockerignore`. Verified end-to-end
with `docker build` + `docker run` against `/v1/health`. Also fixes
`test_new_provider_models_are_registered`, which was missing the `tmp_path`
database isolation used by every other test in the file and so could hit
the real dev-mode `llm_cost_autopilot.db` on disk.

Performance: `LearningService.refresh_recommendations` fetched the entire
`ResponseRow` table on every call to build a `request_id -> cost` lookup,
even though only rows matching the `VerificationRow`s already fetched were
ever used. Now filters `ResponseRow` by `request_id.in_(...)`, so cost is
bounded by verification volume, not total response history.

Performance: `DashboardRepository.get_quality_aggregation` fetched every
completed `VerificationRow` as full ORM objects (including `dimensions`,
`rationale`, `raw_judge_response` -- unbounded text/JSON blobs never used
in this aggregation) to average `score`/`confidence`/`passed` and group by
`routing_model`/`routing_strategy`/`routing_complexity` in Python.
`total_verified`, `average_score`, `average_confidence`, `pass_rate`,
`average_evaluation_duration_ms`, and the three `by_*` breakdowns are now
computed with SQL `COUNT`/`AVG`/`GROUP BY`. Only `average_queue_delay_ms`
and `average_total_verification_ms` still run in Python (they're
timestamp subtraction, not portable across sqlite/postgres dialects), but
now fetch only the 3 needed timestamp columns instead of full rows.
Behavior-preserving: covered by the existing
`test_get_quality_aggregation_matches_verification_data` plus a new
`test_get_quality_aggregation_computes_confidence_pass_rate_and_duration_via_sql`
regression test.

Performance: adds DB indexes on columns filtered or joined on in every
dashboard/learning query but never indexed before: `request_id` on
`responses`, `routing_events`, and `verifications` (all joined via
`.in_(request_ids)` in `DashboardRepository`), `status` on `verifications`
(filtered on every `get_quality_aggregation` call), and `created_at` on
`requests`, `responses`, `routing_events`, `verifications`, and
`learning_recommendations` (every dashboard query filters by
`TimeWindow.cutoff` on this column). No migration tool exists yet
(`init_db` is `Base.metadata.create_all`), so these take effect on any
freshly created database; an existing on-disk SQLite file would need
`CREATE INDEX` run manually or the file recreated.

Security: `backend/api/routers/providers_config.py` write routes (save,
delete, enable, disable, test) accepted requests with no authentication --
anyone able to reach the API could overwrite or delete a provider's stored
API key. Adds an opt-in shared-secret gate: when `ADMIN_API_KEY` is set,
these 5 routes require a matching `X-Admin-Key` header (constant-time
compared, `require_admin_key` dependency in `backend/api/dependencies.py`)
or return 401. Follows the same opt-in convention as
`PROVIDER_CREDENTIAL_ENCRYPTION_KEY`: unset by default so local dev, tests,
and the public demo are unaffected; read-only routes
(`GET /v1/providers/config`, the `/dashboard/providers` page) stay open
since they only ever return masked keys. Not a full auth system (no
sessions, users, or JWTs) -- scoped to the one exposed write surface.

## v0.9.1 — Provider Expansion (2026-07-04)

Adds five new LLM providers -- Google Gemini, NVIDIA NIM, OpenRouter, Groq,
and Mistral AI -- all via a shared `OpenAIProtocolProvider` base class,
since each exposes an OpenAI-compatible chat-completions API differing
only in base_url and API key. `OpenAIProvider` is refactored to inherit
from it with no behavior change. `ProviderFactory` becomes the single
source of truth for which provider names exist: the two previously
duplicated `KNOWN_PROVIDER_NAMES` tuples (in `manager.py` and
`credential_store.py`) are removed, and every consumer now reads names
from `ProviderFactory.registered_names()` (directly, or via
`ProviderManager.registered_names()` / an injected `provider_names` tuple).

**Added**
- `OpenAIProtocolProvider` -- shared adapter for any OpenAI-compatible
  provider; subclasses declare only `_NAME` and `_BASE_URL`
- `GeminiProvider`, `NvidiaNimProvider`, `OpenRouterProvider`,
  `GroqProvider`, `MistralProvider` -- registered in `ProviderFactory`
  alongside the existing 4; configurable via Provider Configuration
  (API key only -- base_url is fixed per provider, these are hosted
  cloud APIs with one canonical endpoint) with `.env` fallback
  (`GEMINI_API_KEY`, `NVIDIA_NIM_API_KEY`, `OPENROUTER_API_KEY`,
  `GROQ_API_KEY`, `MISTRAL_API_KEY`)
- `ProviderFactory.registered_names()` / `register(..., user_configurable=)`
  -- the single source of truth for the user-facing provider set; `mock`
  is registered `user_configurable=False` and stays internal-only
- Six curated models across the new providers in `models.yaml`
  (`gemini-2.5-pro`, `gemini-2.5-flash`, `llama-3.3-70b-versatile`,
  `mistral-large-latest`, `meta/llama-3.3-70b-instruct`,
  `openai/gpt-4.1-mini`) -- model ids stored exactly as each vendor's API
  requires, no normalization layer

**Changed**
- `OpenAIProvider` now inherits from `OpenAIProtocolProvider`
  (`_NAME="openai"`, `_BASE_URL=None`) -- behavior-preserving refactor,
  verified by the full pre-existing `OpenAIProvider` test suite passing
  unchanged
- `CredentialStore.__init__` takes an explicit `provider_names` argument
  instead of reading a module-level constant
- `main.py`'s provider circuit-breaker map and `ProviderManager`
  construction now derive their provider set from
  `ProviderFactory.registered_names()`

**Stats:** 455 tests passing (75 new), 0 regressions.

## v0.9.0 — Provider Configuration (2026-07-03)

Replaces `.env`-only provider API keys with dashboard-managed,
encrypted, live-reloadable credentials for `openai`, `anthropic`, and
`ollama`. Moves the platform from a single-operator, restart-required
configuration model toward a runtime-managed one, without expanding the
model registry or routing surface. Phases 1–8 delivered the project's
planned roadmap; this release rounds it out with a practical
operational feature and is the final release on top of that roadmap.

**Added**
- `AnthropicProvider` / `OllamaProvider` — implement `BaseProvider`
  identically to `OpenAIProvider`'s pattern, registered in
  `ProviderFactory` alongside `openai`/`mock`
- `CredentialStore` — Fernet-encrypted `provider_credentials` CRUD
  (`get`, `get_stored`, `save`, `record_health_check_failure`,
  `set_enabled`, `delete`, `list_status`); the only layer that knows
  encryption exists, with existing `.env` values as a fallback so a
  deployment with only environment variables configured behaves exactly
  as it did before this release
- `ProviderManager.reload_provider(name)` — live, zero-downtime
  credential swap; `ProviderFactory`/`ProviderManager` now take a
  `ProviderCredential` value object instead of reading `Settings`
  directly
- Validate-before-persist save flow: a candidate credential is health-
  checked before anything is written, so an invalid key never takes
  down a working provider
- `POST /v1/providers/{name}/config`, `DELETE /v1/providers/{name}/config`,
  `POST /v1/providers/{name}/test`, `POST /v1/providers/{name}/enable`,
  `POST /v1/providers/{name}/disable`, `GET /v1/providers/config`
- `/dashboard/providers` page — one form per provider, masked keys,
  Test/Save/Enable/Disable/Delete; nav link added to `/dashboard` and
  `/dashboard/analytics`
- `is_enabled` flag (disable without deleting); disabling or deleting a
  provider is authoritative and is never silently overridden by an
  environment-variable credential
- `ProviderUnavailableError` — a provider removed/disabled/reloaded
  concurrently with an in-flight request now fails over gracefully
  through the existing resilience path instead of raising an unhandled
  error

**Explicitly out of scope** (deferred, not planned for a future phase):
Gemini/Groq/OpenRouter provider classes; populating `models.yaml` so
chat requests actually route to Anthropic/Ollama (this release makes
providers connectable and health-checkable, not routable — routing
remains OpenAI-only until model registry work happens separately);
multi-organization/multi-tenant credential scoping; credential rotation
history/audit log; any UI/API auth (matches the existing dashboard's
current posture); capability/metadata discovery per provider;
encryption key rotation.

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
