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

## Routing

_Not built yet — Phase 2._

## Classification

_Not built yet — Phase 3._

## Verification

_Not built yet — Phase 4._

## Learning

_Not built yet._

## Dashboard

_Not built yet._
