# LLM Cost Autopilot — Phase 1 Design: Skeleton, Provider Foundation & Event Bus

Status: **Approved — frozen as implementation contract**
Date: 2026-07-02

## 1. Purpose & Scope

Phase 1 delivers a running, testable FastAPI service with a real provider
integration (OpenAI), a memory-first model registry backed by YAML config
and SQLite persistence, an in-process event bus, and structured logging —
the foundation the routing engine, classifier, verifier, and dashboard
(Phase 2+) will build on.

**In scope:**
- Project scaffold (`uv`-managed Python 3.11+ project)
- `Settings` via `pydantic-settings` (`.env` + YAML)
- `ModelRegistry` service (memory-first, YAML-driven, DB-backed persistence)
- `BaseProvider` interface + `OpenAIProvider` + `MockProvider`
- `ProviderFactory` + `ProviderManager`
- In-process `EventBus`
- SQLAlchemy models for `providers` and `models` tables, SQLite
- Structured JSON logging with rotating file handler
- `GET /v1/health`, `GET /v1/models`
- Dependency injection via FastAPI `Depends()`
- pytest suite, no network calls required to pass
- `README.md` (current-state only) + `docs/ARCHITECTURE.md` (section
  headers, Phase 1 sections filled in)

**Explicitly out of scope for Phase 1** (deferred to later phases):
- Routing engine, complexity classifier, quality verifier, background
  worker, learning loop
- Anthropic and Ollama provider implementations (interfaces support them;
  not built yet)
- Capability discovery subsystem (providers self-reporting capabilities)
- Streamlit dashboard, Docker/docker-compose
- Full DB schema (`requests`, `responses`, `routing_events`,
  `verification_results`, `classifier_feedback`, `daily_metrics`) — only
  `providers` and `models` exist in Phase 1
- Any `system_health` table (rejected in design review; `providers` +
  `models` cover Phase 1's needs)

No placeholder code, stub classes, or TODO implementations for out-of-scope
items. Extension points (factory registration, event types, provider
interface) are designed to make later additions additive, not a rewrite —
but nothing is pre-built for them.

## 2. Directory Structure

```
llm-cost-autopilot/
  backend/
    api/
      main.py                # FastAPI app factory, lifespan startup, mounts /v1
      dependencies.py         # Depends() providers for Settings/EventBus/ProviderManager/ModelRegistry/DB session
      routers/
        health.py              # GET /v1/health
        models.py               # GET /v1/models
    services/
      model_registry.py         # ModelRegistry
      cost_estimator.py          # BaseCostEstimator + default implementation
    providers/
      base.py                     # BaseProvider ABC
      openai_provider.py
      mock_provider.py
      factory.py                   # ProviderFactory
      manager.py                    # ProviderManager
    events/
      bus.py                         # EventBus (in-process, synchronous)
      types.py                        # EventType enum + payload dataclasses/TypedDicts
      subscribers.py                   # logging subscriber
    config/
      settings.py                       # pydantic-settings BaseSettings
      models.yaml                        # model registry source config
    database/
      base.py                             # engine, session factory, declarative Base
      models.py                            # Provider, ModelEntry ORM tables
    telemetry/
      logging.py                            # JSON formatter, RotatingFileHandler, LogContext
    tests/
      test_model_registry.py
      test_provider_manager.py
      test_provider_factory.py
      test_event_bus.py
      test_health_endpoint.py
      test_models_endpoint.py
      test_cost_estimator.py
  docs/
    superpowers/specs/            # design docs (this file)
    ARCHITECTURE.md                # section headers; Phase 1 sections filled
  pyproject.toml                   # uv-managed
  .env.example
  README.md                         # documents only what exists today
```

## 3. Configuration

### 3.1 Settings (`config/settings.py`)

`pydantic-settings` `BaseSettings` subclass. Reads `.env` for secrets
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `ENVIRONMENT`, `LOG_LEVEL`,
`DATABASE_URL`) and exposes a `models_yaml_path` pointing at
`config/models.yaml`. Missing optional provider keys are `None`, not
errors — `ProviderFactory`/`ProviderManager` decide what that means, not
`Settings`.

### 3.2 `models.yaml` schema

Nested, capability/limits/pricing/metadata grouping so new fields (vision,
function calling, embeddings) slot in without a schema rewrite:

```yaml
models:
  - id: gpt-4o-mini
    provider: openai
    model: gpt-4o-mini
    pricing:
      input_cost: 0.15
      output_cost: 0.60
    limits:
      context_window: 128000
      max_output_tokens: 16384
    capabilities:
      supports_streaming: true
      supports_tools: true
      supports_json: true
      supports_vision: false
    metadata:
      benchmark_score: 0.82
      average_latency_ms: 450
```

`benchmark_score` (not `quality_score`) — this is the static,
config-supplied capability rating. The Phase 4 verifier will later produce
a distinct runtime `quality_score` per response; the two must not be
conflated.

## 4. Provider Layer

### 4.1 `BaseProvider` (`providers/base.py`)

Abstract async methods: `generate()`, `stream()`, `health_check()`,
`count_tokens()`, `estimate_cost()`. `OpenAIProvider` implements this
against the real OpenAI SDK. `MockProvider` implements it with
deterministic canned responses — used in tests and as a dev fallback when
no provider key is configured, so the app always boots and the test suite
never needs network access or a real key.

### 4.2 `ProviderFactory` (`providers/factory.py`)

```python
class ProviderFactory:
    def register(self, name: str, provider_cls: type[BaseProvider]) -> None: ...
    def create(self, name: str, settings: Settings) -> BaseProvider: ...
```

Phase 1 registers `openai` → `OpenAIProvider`, `mock` → `MockProvider`.
Adding a provider later is a `register()` call at startup — no change to
`ProviderManager` or anything downstream.

### 4.3 `ProviderManager` (`providers/manager.py`)

```python
class ProviderManager:
    def __init__(self, factory: ProviderFactory, settings: Settings): ...
    def get_provider(self, name: str) -> BaseProvider: ...
    def is_provider_available(self, name: str) -> bool: ...
    def list_providers(self) -> dict[str, ProviderStatus]: ...
```

Builds one instance per provider that has the config it needs (`openai`
only instantiated if `OPENAI_API_KEY` is set; `mock` always instantiated).
Never constructs providers itself — always goes through the factory.
`ModelRegistry` asks this for availability; it never reads `Settings` or
env vars directly.

## 5. ModelRegistry (`services/model_registry.py`)

Lives in `services/`, not a standalone `registry/` package — it does more
than register models (loads, validates, persists, emits events), so it's
a service like anything else in `services/`.

```python
class ModelRegistry(BaseRegistry):
    def __init__(self, provider_manager: ProviderManager, event_bus: EventBus, cost_estimator: BaseCostEstimator): ...
    def get_model(self, model_id: str) -> ModelSpec: ...
    def get_models(self) -> list[ModelSpec]: ...
    def get_available_models(self) -> list[ModelSpec]: ...
    def get_provider_models(self, provider: str) -> list[ModelSpec]: ...
    def reload(self) -> None: ...
    def refresh_provider_status(self) -> None: ...
    def estimate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float: ...
```

**Memory-first, always.** All read methods (`get_model`, `get_models`,
`get_available_models`, `get_provider_models`, `estimate_cost`) hit an
in-memory cache built by `reload()`. Zero DB queries on the read path —
the database exists for the dashboard/analytics/history to query directly
in later phases, not for routing-time lookups.

**`reload()`** — re-reads `models.yaml`, validates into `ModelSpec`
objects, rebuilds the in-memory cache, upserts rows into the `models`
table. Emits `MODEL_REGISTERED` per model. Static-config concern only —
does not contact providers.

**`refresh_provider_status()`** — separate from `reload()`. Calls
`ProviderManager` → each provider's `health_check()`, updates the
`providers` table (`status`, `last_checked_at`, `last_error`), emits
`PROVIDER_AVAILABLE` / `PROVIDER_DISABLED` / `PROVIDER_FAILED`. Runtime
concern, kept out of `reload()`.

## 6. Cost Estimation

`BaseCostEstimator` (thin ABC, `estimate(model_spec, input_tokens,
output_tokens) -> float`) with a default linear implementation in
`services/cost_estimator.py`. `ModelRegistry.estimate_cost()` delegates to
this rather than computing inline, so tiered/volume pricing can replace
the default implementation later without touching the registry.

## 7. Event Bus (`events/bus.py`)

**In-process only.** No Redis, no Kafka, no NATS, no Celery in Phase 1.

```python
class EventBus:
    def subscribe(self, event_type: EventType, handler: Callable[[dict], None]) -> None: ...
    def emit(self, event_type: EventType, payload: dict) -> None:
        for handler in self._subscribers.get(event_type, []):
            handler(payload)
```

Synchronous, in-memory dict of subscriber lists. The module docstring
states explicitly that this is a Phase 1 in-process implementation and
that `emit`/`subscribe` are the stable interface a future broker-backed
implementation would preserve.

`EventType` (Phase 1 members only): `PROVIDER_AVAILABLE`,
`PROVIDER_DISABLED`, `PROVIDER_FAILED`, `MODEL_REGISTERED`. Events like
`MODEL_SELECTED`, `REQUEST_STARTED`, `REQUEST_FINISHED` are not defined
yet — there is no router or request pipeline in Phase 1 to emit them.

One subscriber wired at startup: a logging subscriber
(`events/subscribers.py`) that writes every emitted event through the
structured logger.

## 8. Database

SQLite via SQLAlchemy, engine/session/declarative base in
`database/base.py`. Two tables only:

- **`providers`**: `id, name, status, last_checked_at, last_error, updated_at`
- **`models`**: `id, model_id, provider, model_name, input_cost,
  output_cost, context_window, benchmark_score, supports_streaming,
  supports_tools, supports_json, average_latency_ms, available, updated_at`

(Table is named `models`, not `model_registry` — matches the naming
convention the rest of the schema will use in later phases:
`requests`, `responses`, `routing_events`, etc.)

## 9. API

All routers mounted under `/v1` in `main.py` — no unversioned routes,
ever, even in Phase 1.

**`GET /v1/health`:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "environment": "development",
  "database": "healthy",
  "providers": {"openai": "available", "anthropic": "disabled", "ollama": "disabled"},
  "loaded_models": 7,
  "uptime_seconds": 42.1
}
```

**`GET /v1/models`:** full `ModelSpec` dump per model — `provider`,
`model`, `available`, pricing, limits, capabilities, `benchmark_score`,
`average_latency_ms`.

Dependencies (`Settings`, `EventBus`, `ProviderManager`, `ModelRegistry`,
DB session) are constructed once in the FastAPI lifespan handler and
exposed to routers via `Depends()` — routers never instantiate services
directly.

## 10. Logging

Structured JSON via stdlib `logging` + `RotatingFileHandler`, console
output in dev. Every log line carries: `request_id, trace_id, provider,
model, latency_ms, cost_estimate, timestamp, component, environment,
hostname, service`. `service="llm-cost-autopilot"`, `hostname` from
`socket.gethostname()`, `environment` from `Settings`. Fields with no
value yet (`request_id`, `trace_id`, `provider`, `model`, `latency_ms`,
`cost_estimate` — no request pipeline exists yet) are `null`, not omitted,
so the log schema doesn't change shape when Phase 2 adds the pipeline.

## 11. Interfaces

`BaseProvider`, `BaseRegistry`, `BaseCostEstimator` — each a small ABC (2-6
methods), no logic in the interface itself. Added because concrete
Phase-2+ consumers exist (routing strategies need a stable `BaseRegistry`
surface; alternative providers need `BaseProvider`), not speculatively.

## 12. Testing

pytest. `MockProvider` and a mocked OpenAI client (`pytest-mock`/`respx`)
mean the full suite runs with zero API keys and zero network calls.
Coverage target for Phase 1 code: the project-wide >80% bar applies from
the start, not deferred.

## 13. Documentation

`README.md` documents only what's buildable today: setup with `uv`,
running the test suite, hitting `/v1/health` and `/v1/models`.
`docs/ARCHITECTURE.md` has section headers for every planned subsystem
(Provider Layer, Registry, Events, Routing, Verification, Learning,
Dashboard, ...) — only the Phase 1 sections (Provider Layer, Registry,
Events) have real content; the rest stay as headers until their phase
lands. No documentation is written for unbuilt features.

## 14. Tooling

`uv`-managed `pyproject.toml`. Git repository initialized at project root.
