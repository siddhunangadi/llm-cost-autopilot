# Architecture

## Configuration

`Settings` (pydantic-settings) reads `.env`/environment variables into a
strongly-typed, fail-fast model — invalid `environment`/`log_level`
values or blank `database_url`/`models_yaml_path` raise
`pydantic.ValidationError` at construction. `Settings` only carries
`models_yaml_path` as a string; it never reads or parses the YAML file
itself — that's `ModelRegistry`'s job.

## Provider Layer

`BaseProvider` defines `name`, `generate`, `stream`, `health_check`,
`count_tokens`, and `estimate_cost`. Every provider must self-identify via
`name` — callers never infer identity from class/type. Concrete providers
translate SDK-specific exceptions into `ProviderError`; nothing outside
`providers/openai_provider.py` imports the OpenAI SDK.

`ProviderFactory` registers provider classes by name (`register`/`create`
only — no caching, no discovery, `create()` raises `KeyError` for an
unregistered name). `ProviderManager` builds one instance per provider
that has the configuration it needs: `MockProvider` always (mandatory —
a construction failure crashes startup, since there's no sensible
degraded mode without it), `OpenAIProvider` when `OPENAI_API_KEY` is set
(optional — a construction failure is logged and the provider is simply
left unavailable, never crashes the app). `ProviderManager` is the only
source of truth for provider availability.

## Registry

`ModelRegistry` loads `backend/config/models.yaml`, validates it into
`ModelSpec` objects, and keeps an in-memory cache (`types.MappingProxyType`,
read-only) that all reads go through — the database is a write target for
persistence/analytics, never read on the lookup path.

`reload()` re-reads YAML and upserts the `models` table. It fails fast on
malformed YAML, invalid schema, or duplicate model IDs — *before*
touching the database or the cache — and swaps the entire cache
atomically at the end, so a failed reload never partially applies and
stale entries removed from YAML are dropped rather than left lingering.

`refresh_provider_status()` pings each configured provider's
`health_check()` and upserts the `providers` table — kept separate from
`reload()` because one is a config concern and the other is a runtime
concern. It follows the same snapshot-then-swap pattern as `reload()`:
build a whole new cache with updated availability, then swap the
reference once, so readers never see a partially-updated snapshot.

`estimate_cost()` purely coordinates — it looks up the spec and delegates
to `BaseCostEstimator`; no pricing formula lives in the registry itself.

## Database

Three single-purpose functions, not one: `create_engine_from_settings`
(engine only, no DDL), `init_db` (issues `CREATE TABLE`, and is therefore
the fail-fast point — a bad `DATABASE_URL` crashes startup here, not on
first request), and `create_session_factory` (session factory only,
takes an already-built engine). Two tables: `providers`, `models`.

## Events

An in-process, synchronous `EventBus` (`subscribe`/`emit`, no external
broker). A subscriber that raises is logged and skipped — it never
prevents other subscribers for the same event from running. Phase 1
emits `PROVIDER_AVAILABLE`, `PROVIDER_DISABLED`, `PROVIDER_FAILED`, and
`MODEL_REGISTERED`; a logging subscriber writes every event to the
structured logger.

## Logging

Structured JSON (console + rotating file). `get_logger()` is the only
sanctioned way to obtain a logger — nothing else calls
`logging.getLogger()` directly. Request-scoped fields (`request_id`,
`trace_id`, `provider`, `model`, `latency_ms`, `cost_estimate`) are
carried via `contextvars`, isolated per asyncio task, through the
`request_context()` context manager.

## API

FastAPI. All routes mounted under `/v1`. Dependencies (`Settings`,
`EventBus`, `ProviderManager`, `ModelRegistry`, DB session factory, app
version/start time) are constructed once in the `lifespan` startup
handler and exposed to routers via `Depends()` — routers never
instantiate services directly, and nothing runs at import time.
`GET /v1/health` and `GET /v1/models` only report current state; neither
triggers a provider health check or a registry reload.

## Classification

`PromptAnalyzer.analyze()` extracts a deterministic `PromptFeatures` from
a prompt (regex/keyword-based, no ML) — including `estimated_output_tokens`,
which uses explicit brevity/word-count/long-form signals rather than
assuming output length mirrors input length. `HeuristicComplexityClassifier`
(behind `BaseComplexityClassifier`) turns those features into a
`ClassificationResult` (tier, score, confidence, human-readable `signals`)
via an additive, YAML-configurable threshold score — designed so a future
ML classifier is a drop-in replacement with zero call-site changes.

## Routing

`RoutingPolicy` filters `ModelRegistry`'s available models by a
per-complexity-tier minimum `benchmark_score` (YAML-configurable) — the
engine never hardcodes eligibility. Four strategies
(`CostOptimizedStrategy`, `LatencyOptimizedStrategy`,
`QualityOptimizedStrategy`, `BalancedStrategy`) implement
`BaseRoutingStrategy.select_model(context: RoutingContext) -> ModelSpec`;
`BalancedStrategy`'s weights are also YAML-configurable, defaulting to
equal thirds. `ExplanationGenerator` builds the human-readable reasoning
from the classifier's own `signals` rather than rediscovering them, kept
entirely separate from `RoutingEngine` so the orchestrator never
accumulates conditional string-building. `RoutingEngine` is pure
orchestration — it never calls a provider or touches the database.
`RoutingConfigLoader` is the single owner of `routing.yaml` file I/O;
`ClassifierPolicy`, `RoutingPolicy`, and `BalancedStrategy` all receive
already-parsed, already-validated configuration.

`ChatService` is the one component that calls both `RoutingEngine` and a
provider: it routes, persists the request + routing event immediately,
calls `provider.generate()`, and persists the response (or the error, on
`ProviderError`, before re-raising). `POST /v1/chat` is a thin HTTP layer
mapping `NoEligibleModelError` → 503 and `ProviderError` → 502.

## Verification

After `ChatService` persists a response, it schedules
`VerificationService.verify(request_id, prompt, response)` as a FastAPI
`BackgroundTasks` side effect — always via `try/except` around
`add_task()` so a scheduling failure is logged and never affects the
`/v1/chat` response. `VerificationService` owns the full DB lifecycle
(`PENDING -> RUNNING -> COMPLETED | FAILED`), opening a fresh,
independent `session_factory()` transaction for every state
transition — no ORM instance is ever held across transaction
boundaries. Each write commits before its corresponding
`VerificationStarted`/`VerificationCompleted`/`VerificationFailed` event
is published on the existing `EventBus`. A verification snapshot always
copies `routing_model`/`routing_strategy`/`routing_complexity` from the
`RoutingEventRow` that produced the response, so historical scores stay
tied to the routing decision even if `routing.yaml` changes later.

`BaseJudge`/`LLMJudge` are pure — `(prompt, response) -> JudgeVerdict`,
no persistence, no retries, no knowledge of `request_id`. `LLMJudge`
parses the judge model's raw text through `_JudgeResponseSchema
.model_validate_json()`, so malformed JSON, missing fields, and
out-of-range dimension scores (outside `[0.0, 1.0]`) all surface as a
`pydantic.ValidationError`, which `VerificationService.verify()` catches
and records as a `FAILED` verification rather than propagating.
`JudgeEngine` is the sole place that times a judge call
(`run() -> (JudgeVerdict, duration_ms)`); `VerificationService` calls
`JudgeEngine`, never `BaseJudge` directly. `verify()` itself never
raises — every exception path is swallowed internally, matching the
best-effort contract with `ChatService`.

`GET /v1/chat/{request_id}/verification` returns a single
`VerificationResult` (404 if none exists); `raw_judge_response` is
persisted for audit but deliberately excluded from this and every other
API response. `GET /v1/metrics/quality` aggregates `COMPLETED` rows —
average score/confidence, pass rate, timing breakdowns, and per-model /
per-strategy / per-complexity averages — plus a `FAILED`-row count. Both
routers, the judge, engine, and service are wired up in `main.py`'s
`lifespan`, which loads `verification.yaml` once at startup via
`VerificationConfigLoader`. If the configured judge model's provider is
unavailable at startup (e.g. no API key), the app still boots — a
fallback judge is substituted that fails cleanly (`FAILED` status) on
every verification attempt instead of crashing startup, since
verification is never a prerequisite for `/v1/chat`.

No auto-escalation, no classifier retraining, no feedback loop into
routing, no prompt optimization, no retry policies for failed
verifications, and no separate worker process/queue — all explicitly
deferred to a later phase.

## Learning

_Not built yet._

## Dashboard

_Not built yet._
