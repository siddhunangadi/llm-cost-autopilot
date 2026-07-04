# Changelog

All notable changes to this project are documented here. Versions
correspond to the `vX.Y.0` git tags marking the end of each phase.

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
