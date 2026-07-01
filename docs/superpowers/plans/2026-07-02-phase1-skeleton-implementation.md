# LLM Cost Autopilot — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a running, testable FastAPI service (`GET /v1/health`, `GET /v1/models`) backed by a memory-first `ModelRegistry`, an `OpenAIProvider`/`MockProvider` pair behind `BaseProvider`, SQLite persistence, an in-process event bus, and structured JSON logging, per the frozen Phase 1 design spec.

**Architecture:** Clean layering — `database` → `providers` (+ `services.cost_estimator`) → `services.model_registry` → `api`, with `events` and `telemetry` as cross-cutting concerns. Dependency injection via FastAPI `Depends()`, all services constructed once at startup and exposed through `app.state`.

**Tech Stack:** Python 3.11+, `uv`, FastAPI, Pydantic v2 / pydantic-settings, SQLAlchemy 2.0 + SQLite, PyYAML, OpenAI SDK, pytest + pytest-asyncio + pytest-mock + httpx.

**Spec:** `docs/superpowers/specs/2026-07-02-phase1-skeleton-design.md` (frozen — implement exactly, no new abstractions beyond what's specified below).

## Global Constraints

- Python 3.11+, `uv`-managed project, `[tool.uv] package = false` (this is an app, not an installable library).
- All API routes mounted under `/v1` — no unversioned routes.
- `ModelRegistry` reads are memory-only; the `models`/`providers` DB tables are write targets for persistence, never read on the routing/lookup path.
- `EventBus` is in-process and synchronous only — no Redis/Kafka/NATS/Celery.
- `benchmark_score` (not `quality_score`) is the YAML/DB field name.
- Every log record includes: `request_id, trace_id, provider, model, latency_ms, cost_estimate, timestamp, component, environment, hostname, service` (null where not yet applicable).
- No provider stubs for Anthropic/Ollama — only `openai` and `mock` are registered in Phase 1.
- No placeholder code, no TODOs, no unused abstractions.
- Commit after every task.

---

### Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `backend/__init__.py`, `backend/api/__init__.py`, `backend/api/routers/__init__.py`, `backend/services/__init__.py`, `backend/providers/__init__.py`, `backend/events/__init__.py`, `backend/config/__init__.py`, `backend/database/__init__.py`, `backend/telemetry/__init__.py`, `backend/tests/__init__.py` (all empty)

**Interfaces:**
- Produces: a `uv sync`-able project with `backend` importable as `backend.*` from the project root (via `pythonpath = ["."]` in pytest config).

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "llm-cost-autopilot"
version = "0.1.0"
description = "Intelligent cost-aware routing layer for multiple LLM providers"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "pydantic>=2.9.0",
    "pydantic-settings>=2.6.0",
    "sqlalchemy>=2.0.0",
    "pyyaml>=6.0.2",
    "openai>=1.54.0",
]

[dependency-groups]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "pytest-mock>=3.14.0",
    "httpx>=0.27.0",
]

[tool.uv]
package = false

[tool.pytest.ini_options]
pythonpath = ["."]
asyncio_mode = "auto"
testpaths = ["backend/tests"]
```

- [ ] **Step 2: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
*.db
logs/
.env
.coverage
htmlcov/
```

- [ ] **Step 3: Write `.env.example`**

```
ENVIRONMENT=development
LOG_LEVEL=INFO
DATABASE_URL=sqlite:///./llm_cost_autopilot.db
MODELS_YAML_PATH=backend/config/models.yaml
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

- [ ] **Step 4: Create package directories and empty `__init__.py` files**

Run:
```bash
mkdir -p backend/api/routers backend/services backend/providers backend/events backend/config backend/database backend/telemetry backend/tests
touch backend/__init__.py backend/api/__init__.py backend/api/routers/__init__.py \
      backend/services/__init__.py backend/providers/__init__.py backend/events/__init__.py \
      backend/config/__init__.py backend/database/__init__.py backend/telemetry/__init__.py \
      backend/tests/__init__.py
```

- [ ] **Step 5: Sync dependencies**

Run: `uv sync`
Expected: creates `.venv/` and `uv.lock`, no errors.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore .env.example backend uv.lock
git commit -m "chore: project scaffold"
```

---

### Task 2: Settings

**Files:**
- Create: `backend/config/settings.py`
- Test: `backend/tests/test_settings.py`

**Interfaces:**
- Produces: `Settings` (pydantic-settings `BaseSettings`) with fields `environment: Literal["development", "test", "staging", "production"]`, `log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]`, `database_url: str` (min_length=1), `models_yaml_path: str` (min_length=1), `openai_api_key: str | None`, `anthropic_api_key: str | None`. Invalid/blank values raise `pydantic.ValidationError` at construction (fail-fast). Settings only carries `models_yaml_path` as a string — it does not read or validate YAML content; that responsibility belongs to `ModelRegistry` (Task 13). All other tasks construct `Settings(...)` directly (keyword overrides for tests) or `Settings()` (reads `.env`/env) in `main.py`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_settings.py
import pytest
from pydantic import ValidationError

from backend.config.settings import Settings


def test_settings_successful_load_with_defaults():
    settings = Settings(_env_file=None)
    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.database_url == "sqlite:///./llm_cost_autopilot.db"
    assert settings.models_yaml_path == "backend/config/models.yaml"
    assert settings.openai_api_key is None
    assert settings.anthropic_api_key is None


def test_settings_successful_load_with_explicit_values():
    settings = Settings(
        _env_file=None,
        environment="production",
        log_level="ERROR",
        database_url="sqlite:///./prod.db",
        models_yaml_path="config/models.yaml",
        openai_api_key="sk-live",
    )
    assert settings.environment == "production"
    assert settings.log_level == "ERROR"
    assert settings.database_url == "sqlite:///./prod.db"
    assert settings.models_yaml_path == "config/models.yaml"
    assert settings.openai_api_key == "sk-live"


def test_settings_rejects_invalid_environment():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="production-ish")


def test_settings_rejects_invalid_log_level():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, log_level="VERBOSE")


def test_settings_rejects_blank_database_url():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, database_url="")


def test_settings_rejects_blank_models_yaml_path():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, models_yaml_path="")


def test_settings_optional_provider_keys_default_to_none_when_missing():
    # Missing provider keys are intentionally not an error -- Phase 1 must
    # work with zero provider keys configured (ProviderManager decides what
    # "no key" means later, not Settings).
    settings = Settings(_env_file=None)
    assert settings.openai_api_key is None
    assert settings.anthropic_api_key is None


def test_settings_reads_env_var_overrides(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

    settings = Settings(_env_file=None)

    assert settings.environment == "test"
    assert settings.log_level == "DEBUG"
    assert settings.openai_api_key == "sk-from-env"


def test_settings_models_yaml_path_is_a_plain_path_string_not_parsed():
    # Settings only carries the path -- ModelRegistry (Task 13) owns reading
    # and validating the YAML content itself, per the frozen design's split
    # between config-loading and registry concerns.
    settings = Settings(_env_file=None, models_yaml_path="some/nonexistent/models.yaml")
    assert settings.models_yaml_path == "some/nonexistent/models.yaml"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_settings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.config.settings'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/config/settings.py
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: Literal["development", "test", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    database_url: str = Field(default="sqlite:///./llm_cost_autopilot.db", min_length=1)
    models_yaml_path: str = Field(default="backend/config/models.yaml", min_length=1)

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_settings.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/config/settings.py backend/tests/test_settings.py
git commit -m "feat: add Settings via pydantic-settings"
```

---

### Task 3: Cost Estimator

**Files:**
- Create: `backend/services/cost_estimator.py`
- Test: `backend/tests/test_cost_estimator.py`

**Interfaces:**
- Produces: `calculate_linear_cost(input_tokens: int, output_tokens: int, input_cost: float, output_cost: float) -> float` (pure function, per-million-token pricing, raises `ValueError` on negative token counts or negative pricing), `BaseCostEstimator` (ABC with `estimate(...)`), `DefaultCostEstimator(BaseCostEstimator)`. Consumed by `providers/mock_provider.py`, `providers/openai_provider.py` (Tasks 9-10) and `services/model_registry.py` (Task 14, constructed as `DefaultCostEstimator()`).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_cost_estimator.py
import pytest

from backend.services.cost_estimator import (
    BaseCostEstimator,
    DefaultCostEstimator,
    calculate_linear_cost,
)


def test_input_token_cost_only():
    cost = calculate_linear_cost(input_tokens=1_000_000, output_tokens=0, input_cost=3.0, output_cost=15.0)
    assert cost == pytest.approx(3.0)


def test_output_token_cost_only():
    cost = calculate_linear_cost(input_tokens=0, output_tokens=1_000_000, input_cost=3.0, output_cost=15.0)
    assert cost == pytest.approx(15.0)


def test_zero_tokens_costs_nothing():
    assert calculate_linear_cost(0, 0, 1.0, 2.0) == 0.0


def test_combined_input_and_output_cost():
    cost = calculate_linear_cost(1_000_000, 1_000_000, 1.0, 2.0)
    assert cost == pytest.approx(3.0)


def test_large_token_counts():
    cost = calculate_linear_cost(
        input_tokens=500_000_000, output_tokens=250_000_000, input_cost=0.15, output_cost=0.60
    )
    assert cost == pytest.approx(500 * 0.15 + 250 * 0.60)


def test_decimal_precision_is_preserved():
    cost = calculate_linear_cost(
        input_tokens=333_333, output_tokens=666_667, input_cost=0.15, output_cost=0.60
    )
    expected = (333_333 / 1_000_000) * 0.15 + (666_667 / 1_000_000) * 0.60
    assert cost == pytest.approx(expected, rel=1e-9)


def test_negative_token_counts_raise_value_error():
    with pytest.raises(ValueError):
        calculate_linear_cost(-1, 0, 1.0, 2.0)


def test_negative_pricing_raises_value_error():
    with pytest.raises(ValueError):
        calculate_linear_cost(1000, 1000, -1.0, 2.0)


def test_default_cost_estimator_delegates_to_linear_formula():
    estimator: BaseCostEstimator = DefaultCostEstimator()
    cost = estimator.estimate(500_000, 500_000, 2.0, 4.0)
    assert cost == pytest.approx(1.0 + 2.0)


def test_default_cost_estimator_is_a_base_cost_estimator():
    assert isinstance(DefaultCostEstimator(), BaseCostEstimator)


# Note: "missing model" isn't a concept this module knows about --
# calculate_linear_cost/DefaultCostEstimator operate on raw token counts
# and pricing, not model lookups. Unknown-model-id handling belongs to
# ModelRegistry.get_model (Task 13's test_get_model_unknown_raises).
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_cost_estimator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.services.cost_estimator'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/services/cost_estimator.py
from abc import ABC, abstractmethod


def calculate_linear_cost(
    input_tokens: int, output_tokens: int, input_cost: float, output_cost: float
) -> float:
    """Cost in dollars given per-million-token pricing."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must not be negative")
    if input_cost < 0 or output_cost < 0:
        raise ValueError("pricing must not be negative")
    return (input_tokens / 1_000_000) * input_cost + (output_tokens / 1_000_000) * output_cost


class BaseCostEstimator(ABC):
    @abstractmethod
    def estimate(
        self, input_tokens: int, output_tokens: int, input_cost: float, output_cost: float
    ) -> float: ...


class DefaultCostEstimator(BaseCostEstimator):
    def estimate(
        self, input_tokens: int, output_tokens: int, input_cost: float, output_cost: float
    ) -> float:
        return calculate_linear_cost(input_tokens, output_tokens, input_cost, output_cost)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_cost_estimator.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/services/cost_estimator.py backend/tests/test_cost_estimator.py
git commit -m "feat: add BaseCostEstimator/DefaultCostEstimator with fail-fast validation"
```

---

### Task 4: Structured Logging

**Files:**
- Create: `backend/telemetry/logging.py`
- Test: `backend/tests/test_logging.py`

**Interfaces:**
- Produces: `JsonFormatter(service, environment)` (logging.Formatter), `configure_logging(settings: Settings, log_dir: str = "logs") -> None`, `get_logger(component: str) -> logging.LoggerAdapter` (the **only** sanctioned way for application code to obtain a logger — never call `logging.getLogger()` directly outside this module), `request_context(**fields)` (context manager, `contextvars`-backed, isolated per asyncio task), `clear_request_context() -> None`, `get_request_context() -> dict`. Consumed by `events/subscribers.py` (Task 6, `get_logger("events")` — no `settings` arg) and `api/main.py` (Task 16, `configure_logging(settings)`).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_logging.py
import asyncio
import io
import json
import logging

import pytest

from backend.config.settings import Settings
from backend.telemetry.logging import (
    JsonFormatter,
    clear_request_context,
    configure_logging,
    get_logger,
    get_request_context,
    request_context,
)


def _capture_logger(component: str) -> tuple[logging.LoggerAdapter, io.StringIO]:
    adapter = get_logger(component)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter(service="llm-cost-autopilot", environment="test"))
    adapter.logger.addHandler(handler)
    adapter.logger.setLevel(logging.INFO)
    adapter.logger.propagate = False
    return adapter, stream


def _read_last_record(stream: io.StringIO) -> dict:
    lines = [line for line in stream.getvalue().strip().splitlines() if line]
    return json.loads(lines[-1])


def test_log_record_contains_all_required_fields():
    logger, stream = _capture_logger("test_component")
    logger.info("hello")

    record = _read_last_record(stream)

    for field in (
        "timestamp", "level", "message", "service", "environment", "hostname",
        "component", "request_id", "trace_id", "provider", "model",
        "latency_ms", "cost_estimate",
    ):
        assert field in record

    assert record["message"] == "hello"
    assert record["level"] == "INFO"
    assert record["service"] == "llm-cost-autopilot"
    assert record["environment"] == "test"
    assert record["component"] == "test_component"
    assert record["hostname"]


def test_json_output_is_valid_json():
    logger, stream = _capture_logger("test_component")
    logger.info("hello")

    record = _read_last_record(stream)
    assert isinstance(record, dict)


def test_optional_request_fields_serialize_as_null_when_unset():
    clear_request_context()
    logger, stream = _capture_logger("test_component")
    logger.info("hello")

    record = _read_last_record(stream)
    assert record["request_id"] is None
    assert record["trace_id"] is None
    assert record["provider"] is None
    assert record["model"] is None
    assert record["latency_ms"] is None
    assert record["cost_estimate"] is None


def test_request_context_sets_fields_on_log_records():
    clear_request_context()
    logger, stream = _capture_logger("test_component")

    with request_context(request_id="req-1", trace_id="trace-1", provider="openai", model="gpt-4o-mini"):
        logger.info("inside context")

    record = _read_last_record(stream)
    assert record["request_id"] == "req-1"
    assert record["trace_id"] == "trace-1"
    assert record["provider"] == "openai"
    assert record["model"] == "gpt-4o-mini"


def test_request_context_resets_after_block_exits():
    clear_request_context()
    logger, stream = _capture_logger("test_component")

    with request_context(request_id="req-1"):
        pass
    logger.info("outside context")

    record = _read_last_record(stream)
    assert record["request_id"] is None


def test_request_context_rejects_unknown_fields():
    with pytest.raises(ValueError):
        with request_context(not_a_real_field="oops"):
            pass


def test_clear_request_context_resets_to_defaults():
    with request_context(request_id="req-1"):
        clear_request_context()
        assert get_request_context()["request_id"] is None


async def test_context_isolation_between_concurrent_tasks():
    results = {}

    async def run(name, request_id):
        with request_context(request_id=request_id):
            await asyncio.sleep(0.01)
            results[name] = get_request_context()["request_id"]

    await asyncio.gather(run("a", "req-a"), run("b", "req-b"))

    assert results == {"a": "req-a", "b": "req-b"}


def test_configure_logging_creates_log_dir_and_writes_json_lines(tmp_path):
    settings = Settings(_env_file=None, environment="test")
    log_dir = str(tmp_path / "logs")

    configure_logging(settings, log_dir=log_dir)
    logging.getLogger("smoke").info("boot")

    log_file = tmp_path / "logs" / "app.log"
    assert log_file.exists()
    last_line = [line for line in log_file.read_text().strip().splitlines() if line][-1]
    json.loads(last_line)


def test_get_logger_returns_logger_adapter_with_component_bound():
    logger = get_logger("my_component")
    assert isinstance(logger, logging.LoggerAdapter)
    assert logger.extra["component"] == "my_component"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_logging.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.telemetry.logging'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/telemetry/logging.py
import contextvars
import json
import logging
import logging.handlers
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from backend.config.settings import Settings

_REQUEST_CONTEXT_FIELDS = (
    "request_id",
    "trace_id",
    "provider",
    "model",
    "latency_ms",
    "cost_estimate",
)

_DEFAULT_REQUEST_CONTEXT: dict = {field: None for field in _REQUEST_CONTEXT_FIELDS}

_request_context: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "request_context", default=_DEFAULT_REQUEST_CONTEXT
)


@contextmanager
def request_context(**fields) -> Iterator[None]:
    """Temporarily bind request-scoped fields onto every log record emitted
    within this block. Backed by contextvars, so concurrent asyncio tasks
    each see their own isolated copy -- one request's fields never leak
    into another's log lines."""
    unknown = set(fields) - set(_REQUEST_CONTEXT_FIELDS)
    if unknown:
        raise ValueError(f"Unknown request context field(s): {sorted(unknown)}")

    merged = {**_request_context.get(), **fields}
    token = _request_context.set(merged)
    try:
        yield
    finally:
        _request_context.reset(token)


def clear_request_context() -> None:
    _request_context.set(_DEFAULT_REQUEST_CONTEXT)


def get_request_context() -> dict:
    return dict(_request_context.get())


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str, environment: str) -> None:
        super().__init__()
        self._service = service
        self._environment = environment
        self._hostname = socket.gethostname()

    def format(self, record: logging.LogRecord) -> str:
        context = _request_context.get()
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "message": record.getMessage(),
            "service": self._service,
            "environment": self._environment,
            "hostname": self._hostname,
            "component": getattr(record, "component", None),
        }
        for field in _REQUEST_CONTEXT_FIELDS:
            payload[field] = context.get(field)
        return json.dumps(payload)


def configure_logging(settings: Settings, log_dir: str = "logs") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(settings.log_level)
    root.handlers.clear()

    formatter = JsonFormatter(service="llm-cost-autopilot", environment=settings.environment)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        f"{log_dir}/app.log", maxBytes=10_000_000, backupCount=5
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def get_logger(component: str) -> logging.LoggerAdapter:
    """Centralized logger accessor. Application code must always go
    through this instead of calling logging.getLogger() directly, so
    every log line consistently carries `component` and merges the
    request-scoped context fields via JsonFormatter."""
    return logging.LoggerAdapter(logging.getLogger(component), {"component": component})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_logging.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/telemetry/logging.py backend/tests/test_logging.py
git commit -m "feat: add structured JSON logging with contextvars request context"
```

---

### Task 5: Event Bus

**Files:**
- Create: `backend/events/types.py`
- Create: `backend/events/bus.py`
- Test: `backend/tests/test_event_bus.py`

**Interfaces:**
- Produces: `EventType(str, Enum)` with members `PROVIDER_AVAILABLE, PROVIDER_DISABLED, PROVIDER_FAILED, MODEL_REGISTERED`; `EventBus` with `subscribe(event_type: EventType, handler: Callable[[dict], None]) -> None` and `emit(event_type: EventType, payload: dict) -> None`. Consumed by `events/subscribers.py` (Task 6), `services/model_registry.py` (Task 14), `api/main.py` (Task 16).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_event_bus.py
from backend.events.bus import EventBus
from backend.events.types import EventType


def test_emit_calls_subscribed_handler_with_payload():
    bus = EventBus()
    received = []
    bus.subscribe(EventType.MODEL_REGISTERED, lambda payload: received.append(payload))

    bus.emit(EventType.MODEL_REGISTERED, {"model_id": "gpt-4o-mini"})

    assert received == [{"model_id": "gpt-4o-mini"}]


def test_emit_with_no_subscribers_does_not_raise():
    bus = EventBus()
    bus.emit(EventType.PROVIDER_FAILED, {"provider": "openai"})


def test_multiple_subscribers_all_called():
    bus = EventBus()
    calls = []
    bus.subscribe(EventType.PROVIDER_AVAILABLE, lambda p: calls.append("a"))
    bus.subscribe(EventType.PROVIDER_AVAILABLE, lambda p: calls.append("b"))

    bus.emit(EventType.PROVIDER_AVAILABLE, {})

    assert calls == ["a", "b"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_event_bus.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.events.bus'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/events/types.py
from enum import Enum


class EventType(str, Enum):
    PROVIDER_AVAILABLE = "provider_available"
    PROVIDER_DISABLED = "provider_disabled"
    PROVIDER_FAILED = "provider_failed"
    MODEL_REGISTERED = "model_registered"
```

```python
# backend/events/bus.py
from collections import defaultdict
from typing import Callable

from backend.events.types import EventType

EventHandler = Callable[[dict], None]


class EventBus:
    """In-process, synchronous event bus for Phase 1.

    No external broker (Redis/NATS/Kafka) is used. `emit` calls each
    subscribed handler synchronously in registration order. A future
    broker-backed implementation would preserve this subscribe/emit
    interface.
    """

    def __init__(self) -> None:
        self._subscribers: dict[EventType, list[EventHandler]] = defaultdict(list)

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        self._subscribers[event_type].append(handler)

    def emit(self, event_type: EventType, payload: dict) -> None:
        for handler in self._subscribers.get(event_type, []):
            handler(payload)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_event_bus.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/events/types.py backend/events/bus.py backend/tests/test_event_bus.py
git commit -m "feat: add in-process EventBus and EventType"
```

- [ ] **Step 6 (refinement, bundled with Task 6): Add subscriber exception isolation**

A subscriber that raises must not prevent other subscribers for the same
event from running, and the failure must be logged via the centralized
`get_logger()` (Task 4) rather than swallowed silently. This doesn't
change `EventBus`'s public API (`subscribe`/`emit` signatures are
unchanged).

Add to `backend/tests/test_event_bus.py`:

```python
import io
import json
import logging

from backend.telemetry.logging import JsonFormatter


def test_subscriber_exception_does_not_prevent_other_subscribers():
    bus = EventBus()
    calls = []

    def failing_handler(payload):
        raise RuntimeError("boom")

    def working_handler(payload):
        calls.append("worked")

    bus.subscribe(EventType.MODEL_REGISTERED, failing_handler)
    bus.subscribe(EventType.MODEL_REGISTERED, working_handler)

    bus.emit(EventType.MODEL_REGISTERED, {})

    assert calls == ["worked"]


def test_subscriber_exception_is_logged():
    bus = EventBus()

    logger = logging.getLogger("events")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter(service="llm-cost-autopilot", environment="test"))
    logger.addHandler(handler)
    logger.setLevel(logging.ERROR)
    logger.propagate = False

    def failing_handler(payload):
        raise RuntimeError("boom")

    bus.subscribe(EventType.MODEL_REGISTERED, failing_handler)
    bus.emit(EventType.MODEL_REGISTERED, {})

    lines = [line for line in stream.getvalue().strip().splitlines() if line]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["message"] == "event_subscriber_failed"
    assert record["level"] == "ERROR"

    logger.removeHandler(handler)
```

Update `backend/events/bus.py`:

```python
# backend/events/bus.py
from collections import defaultdict
from typing import Callable

from backend.events.types import EventType
from backend.telemetry.logging import get_logger

EventHandler = Callable[[dict], None]


class EventBus:
    """In-process, synchronous event bus for Phase 1.

    No external broker (Redis/NATS/Kafka) is used. `emit` calls each
    subscribed handler synchronously in registration order. A future
    broker-backed implementation would preserve this subscribe/emit
    interface. A subscriber that raises is logged and skipped -- it never
    prevents other subscribers for the same event from running.
    """

    def __init__(self) -> None:
        self._subscribers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._logger = get_logger("events")

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        self._subscribers[event_type].append(handler)

    def emit(self, event_type: EventType, payload: dict) -> None:
        for handler in self._subscribers.get(event_type, []):
            try:
                handler(payload)
            except Exception:
                self._logger.exception(
                    "event_subscriber_failed", extra={"event_type": event_type.value}
                )
```

Run: `uv run pytest backend/tests/test_event_bus.py -v`
Expected: PASS (5 tests)

Commit separately from Task 6 proper:

```bash
git add backend/events/bus.py backend/tests/test_event_bus.py
git commit -m "fix: isolate EventBus subscriber exceptions and log failures"
```

---

### Task 6: Event Logging Subscriber

**Files:**
- Create: `backend/events/subscribers.py`
- Test: `backend/tests/test_subscribers.py`

**Interfaces:**
- Produces: `register_logging_subscriber(event_bus: EventBus) -> None`. Consumed by `api/main.py` (Task 16). No longer takes `settings` — `get_logger()` (Task 4) doesn't need it either, since environment/service/hostname are baked into the `JsonFormatter` once by `configure_logging()`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_subscribers.py
import io
import json
import logging

from backend.events.bus import EventBus
from backend.events.subscribers import register_logging_subscriber
from backend.events.types import EventType
from backend.telemetry.logging import JsonFormatter


def test_registered_subscriber_logs_every_event_type():
    bus = EventBus()
    register_logging_subscriber(bus)

    logger = logging.getLogger("events")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter(service="llm-cost-autopilot", environment="test"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    bus.emit(EventType.MODEL_REGISTERED, {"model_id": "gpt-4o-mini"})

    lines = [line for line in stream.getvalue().strip().splitlines() if line]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["message"] == "event_emitted"
    assert record["component"] == "events"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_subscribers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.events.subscribers'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/events/subscribers.py
from backend.events.bus import EventBus
from backend.events.types import EventType
from backend.telemetry.logging import get_logger


def register_logging_subscriber(event_bus: EventBus) -> None:
    logger = get_logger("events")

    def handler(payload: dict) -> None:
        logger.info("event_emitted", extra={"payload": payload})

    for event_type in EventType:
        event_bus.subscribe(event_type, handler)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_subscribers.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add backend/events/subscribers.py backend/tests/test_subscribers.py
git commit -m "feat: log every emitted event via a logging subscriber"
```

---

### Task 7: Database Layer

**Files:**
- Create: `backend/database/base.py`
- Create: `backend/database/models.py`
- Test: `backend/tests/test_database.py`

**Interfaces:**
- Produces: `Base` (SQLAlchemy `DeclarativeBase`), `create_engine_from_settings(settings: Settings) -> Engine` (engine factory — no DDL, no tables touched), `init_db(engine: Engine) -> None` (issues `CREATE TABLE`, raises immediately on a bad connection — this is Phase 1's fail-fast-at-startup point), `create_session_factory(engine: Engine) -> sessionmaker` (session factory only, takes an engine, not settings), `ProviderRow`, `ModelRow` ORM classes (tables `providers`, `models`). Three separate single-purpose functions instead of one function that did engine+DDL+session-factory together, per explicit engine/session/model separation. Consumed by `services/model_registry.py` (Tasks 13-14, calls all three), `api/dependencies.py` (Task 15), `api/main.py` (Task 16, calls `create_engine_from_settings` then `init_db` then `create_session_factory` in that order at startup, with no try/except around any of them — a bad `DATABASE_URL` must crash startup, not degrade silently).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_database.py
import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import OperationalError

from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.database.models import ModelRow, ProviderRow


def _sample_provider_row() -> ProviderRow:
    return ProviderRow(name="openai", status="available")


def _sample_model_row() -> ModelRow:
    return ModelRow(
        model_id="gpt-4o-mini",
        provider="openai",
        model_name="gpt-4o-mini",
        input_cost=0.15,
        output_cost=0.60,
        context_window=128000,
        benchmark_score=0.82,
        supports_streaming=True,
        supports_tools=True,
        supports_json=True,
        average_latency_ms=450,
        available=True,
    )


def test_create_engine_from_settings_returns_a_bound_engine(tmp_path):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)

    assert str(engine.url) == settings.database_url


def test_init_db_creates_providers_and_models_tables(tmp_path):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)

    init_db(engine)

    table_names = set(inspect(engine).get_table_names())
    assert {"providers", "models"}.issubset(table_names)


def test_session_factory_returns_a_new_session_each_call(tmp_path):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    session_a = session_factory()
    session_b = session_factory()

    assert session_a is not session_b
    session_a.close()
    session_b.close()


def test_crud_insert_and_query_provider_and_model_rows(tmp_path):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    with session_factory() as session:
        session.add(_sample_provider_row())
        session.add(_sample_model_row())
        session.commit()

    with session_factory() as session:
        provider_row = session.query(ProviderRow).filter_by(name="openai").one()
        model_row = session.query(ModelRow).filter_by(model_id="gpt-4o-mini").one()

    assert provider_row.status == "available"
    assert model_row.benchmark_score == 0.82


def test_crud_update_and_delete_provider_row(tmp_path):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    with session_factory() as session:
        session.add(_sample_provider_row())
        session.commit()

    with session_factory() as session:
        row = session.query(ProviderRow).filter_by(name="openai").one()
        row.status = "disabled"
        session.commit()

    with session_factory() as session:
        row = session.query(ProviderRow).filter_by(name="openai").one()
        assert row.status == "disabled"
        session.delete(row)
        session.commit()

    with session_factory() as session:
        assert session.query(ProviderRow).filter_by(name="openai").one_or_none() is None


def test_rollback_discards_uncommitted_changes(tmp_path):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    with session_factory() as session:
        session.add(_sample_provider_row())
        session.flush()
        session.rollback()

    with session_factory() as session:
        assert session.query(ProviderRow).filter_by(name="openai").one_or_none() is None


def test_init_db_fails_fast_on_unwritable_database_path():
    settings = Settings(
        _env_file=None, database_url="sqlite:////nonexistent-directory-xyz/test.db"
    )
    engine = create_engine_from_settings(settings)

    with pytest.raises(OperationalError):
        init_db(engine)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_database.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.database.base'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/database/base.py
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.config.settings import Settings


class Base(DeclarativeBase):
    pass


def create_engine_from_settings(settings: Settings) -> Engine:
    connect_args = (
        {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
    )
    return create_engine(settings.database_url, connect_args=connect_args)


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)
```

```python
# backend/database/models.py
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.database.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ProviderRow(Base):
    __tablename__ = "providers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="unknown")
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(String, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class ModelRow(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    provider: Mapped[str] = mapped_column(String, nullable=False)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    input_cost: Mapped[float] = mapped_column(Float, nullable=False)
    output_cost: Mapped[float] = mapped_column(Float, nullable=False)
    context_window: Mapped[int] = mapped_column(Integer, nullable=False)
    benchmark_score: Mapped[float] = mapped_column(Float, nullable=False)
    supports_streaming: Mapped[bool] = mapped_column(Boolean, default=False)
    supports_tools: Mapped[bool] = mapped_column(Boolean, default=False)
    supports_json: Mapped[bool] = mapped_column(Boolean, default=False)
    average_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    available: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_database.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/database/base.py backend/database/models.py backend/tests/test_database.py
git commit -m "feat: add SQLAlchemy providers and models tables"
```

---

### Task 8: BaseProvider Interface

**Files:**
- Create: `backend/providers/base.py`
- Test: `backend/tests/test_base_provider.py`

**Interfaces:**
- Produces: `BaseProvider` ABC with abstract async `generate(prompt, model, **kwargs) -> str`, async `stream(prompt, model, **kwargs) -> AsyncIterator[str]`, async `health_check() -> bool`, sync `count_tokens(text) -> int`, sync `estimate_cost(input_tokens, output_tokens, input_cost, output_cost) -> float`. Consumed by `providers/mock_provider.py` (Task 9), `providers/openai_provider.py` (Task 10).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_base_provider.py
import pytest

from backend.providers.base import BaseProvider


def test_base_provider_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseProvider()


class _CompleteProvider(BaseProvider):
    async def generate(self, prompt, model, **kwargs):
        return "ok"

    async def stream(self, prompt, model, **kwargs):
        yield "ok"

    async def health_check(self):
        return True

    def count_tokens(self, text):
        return 1

    def estimate_cost(self, input_tokens, output_tokens, input_cost, output_cost):
        return 0.0


def test_complete_subclass_can_be_instantiated():
    provider = _CompleteProvider()
    assert isinstance(provider, BaseProvider)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_base_provider.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.providers.base'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/providers/base.py
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class BaseProvider(ABC):
    """Common interface every LLM provider implementation must satisfy."""

    @abstractmethod
    async def generate(self, prompt: str, model: str, **kwargs) -> str: ...

    @abstractmethod
    async def stream(self, prompt: str, model: str, **kwargs) -> AsyncIterator[str]: ...

    @abstractmethod
    async def health_check(self) -> bool: ...

    @abstractmethod
    def count_tokens(self, text: str) -> int: ...

    @abstractmethod
    def estimate_cost(
        self, input_tokens: int, output_tokens: int, input_cost: float, output_cost: float
    ) -> float: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_base_provider.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/providers/base.py backend/tests/test_base_provider.py
git commit -m "feat: add BaseProvider interface"
```

---

### Task 9: MockProvider

**Files:**
- Create: `backend/providers/mock_provider.py`
- Test: `backend/tests/test_mock_provider.py`

**Interfaces:**
- Produces: `MockProvider(BaseProvider)`, constructor `MockProvider(settings: Settings | None = None)`. Consumed by `providers/factory.py` (Task 11).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_mock_provider.py
import pytest

from backend.providers.mock_provider import MockProvider


async def test_generate_is_deterministic_for_same_prompt():
    provider = MockProvider()
    first = await provider.generate("hello", model="mock-1")
    second = await provider.generate("hello", model="mock-1")
    assert first == second
    assert "mock-1" in first


async def test_generate_differs_for_different_prompts():
    provider = MockProvider()
    a = await provider.generate("hello", model="mock-1")
    b = await provider.generate("goodbye", model="mock-1")
    assert a != b


async def test_stream_yields_words_matching_generate():
    provider = MockProvider()
    full = await provider.generate("hello world", model="mock-1")

    chunks = [chunk async for chunk in provider.stream("hello world", model="mock-1")]
    assert "".join(chunks).strip() == full


async def test_health_check_is_always_true():
    provider = MockProvider()
    assert await provider.health_check() is True


def test_count_tokens_is_positive():
    provider = MockProvider()
    assert provider.count_tokens("hello world") > 0


def test_estimate_cost_matches_linear_formula():
    provider = MockProvider()
    cost = provider.estimate_cost(1_000_000, 1_000_000, 1.0, 2.0)
    assert cost == pytest.approx(3.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_mock_provider.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.providers.mock_provider'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/providers/mock_provider.py
import hashlib
from collections.abc import AsyncIterator

from backend.config.settings import Settings
from backend.providers.base import BaseProvider
from backend.services.cost_estimator import calculate_linear_cost


class MockProvider(BaseProvider):
    """Deterministic provider with no network calls. Used in tests and as
    a dev fallback when no real provider key is configured."""

    def __init__(self, settings: Settings | None = None) -> None:
        pass

    async def generate(self, prompt: str, model: str, **kwargs) -> str:
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        return f"[mock:{model}] response-{digest}"

    async def stream(self, prompt: str, model: str, **kwargs) -> AsyncIterator[str]:
        response = await self.generate(prompt, model, **kwargs)
        words = response.split(" ")
        for index, word in enumerate(words):
            yield word if index == len(words) - 1 else word + " "

    async def health_check(self) -> bool:
        return True

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def estimate_cost(
        self, input_tokens: int, output_tokens: int, input_cost: float, output_cost: float
    ) -> float:
        return calculate_linear_cost(input_tokens, output_tokens, input_cost, output_cost)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_mock_provider.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/providers/mock_provider.py backend/tests/test_mock_provider.py
git commit -m "feat: add MockProvider"
```

---

### Task 10: OpenAIProvider

**Files:**
- Create: `backend/providers/openai_provider.py`
- Test: `backend/tests/test_openai_provider.py`

**Interfaces:**
- Produces: `OpenAIProvider(BaseProvider)`, constructor `OpenAIProvider(settings: Settings)`, internal `self._client: AsyncOpenAI`. Consumed by `providers/factory.py` (Task 11).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_openai_provider.py
from unittest.mock import AsyncMock

import pytest

from backend.config.settings import Settings
from backend.providers.openai_provider import OpenAIProvider


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


def _make_provider():
    settings = Settings(_env_file=None, openai_api_key="sk-test")
    return OpenAIProvider(settings)


async def test_generate_returns_completion_content(mocker):
    provider = _make_provider()
    mocker.patch.object(
        provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=_FakeCompletion("hello world"),
    )

    result = await provider.generate("hi", model="gpt-4o-mini")
    assert result == "hello world"


async def test_health_check_true_when_models_list_succeeds(mocker):
    provider = _make_provider()
    mocker.patch.object(
        provider._client.models, "list", new_callable=AsyncMock, return_value=None
    )

    assert await provider.health_check() is True


async def test_health_check_false_when_models_list_raises(mocker):
    provider = _make_provider()
    mocker.patch.object(
        provider._client.models,
        "list",
        new_callable=AsyncMock,
        side_effect=RuntimeError("down"),
    )

    assert await provider.health_check() is False


def test_count_tokens_is_positive():
    provider = _make_provider()
    assert provider.count_tokens("abcdefgh") == 2


def test_estimate_cost_matches_linear_formula():
    provider = _make_provider()
    cost = provider.estimate_cost(1_000_000, 1_000_000, 1.0, 2.0)
    assert cost == pytest.approx(3.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_openai_provider.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.providers.openai_provider'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/providers/openai_provider.py
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from backend.config.settings import Settings
from backend.providers.base import BaseProvider
from backend.services.cost_estimator import calculate_linear_cost


class OpenAIProvider(BaseProvider):
    def __init__(self, settings: Settings, client: AsyncOpenAI | None = None) -> None:
        self._client = client or AsyncOpenAI(api_key=settings.openai_api_key)

    async def generate(self, prompt: str, model: str, **kwargs) -> str:
        response = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    async def stream(self, prompt: str, model: str, **kwargs) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def health_check(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def estimate_cost(
        self, input_tokens: int, output_tokens: int, input_cost: float, output_cost: float
    ) -> float:
        return calculate_linear_cost(input_tokens, output_tokens, input_cost, output_cost)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_openai_provider.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/providers/openai_provider.py backend/tests/test_openai_provider.py
git commit -m "feat: add OpenAIProvider"
```

---

### Task 11: ProviderFactory

**Files:**
- Create: `backend/providers/factory.py`
- Test: `backend/tests/test_provider_factory.py`

**Interfaces:**
- Produces: `ProviderFactory` with `register(name: str, provider_cls: type[BaseProvider]) -> None` and `create(name: str, settings: Settings) -> BaseProvider`. Consumed by `providers/manager.py` (Task 12), `api/main.py` (Task 16).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_provider_factory.py
import pytest

from backend.config.settings import Settings
from backend.providers.factory import ProviderFactory
from backend.providers.mock_provider import MockProvider


def test_register_and_create_returns_instance():
    factory = ProviderFactory()
    factory.register("mock", MockProvider)

    settings = Settings(_env_file=None)
    provider = factory.create("mock", settings)

    assert isinstance(provider, MockProvider)


def test_create_unregistered_provider_raises():
    factory = ProviderFactory()
    settings = Settings(_env_file=None)

    with pytest.raises(KeyError):
        factory.create("unknown", settings)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_provider_factory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.providers.factory'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/providers/factory.py
from backend.config.settings import Settings
from backend.providers.base import BaseProvider


class ProviderFactory:
    def __init__(self) -> None:
        self._registry: dict[str, type[BaseProvider]] = {}

    def register(self, name: str, provider_cls: type[BaseProvider]) -> None:
        self._registry[name] = provider_cls

    def create(self, name: str, settings: Settings) -> BaseProvider:
        if name not in self._registry:
            raise KeyError(f"No provider registered under name '{name}'")
        return self._registry[name](settings)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_provider_factory.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/providers/factory.py backend/tests/test_provider_factory.py
git commit -m "feat: add ProviderFactory"
```

---

### Task 12: ProviderManager

**Files:**
- Create: `backend/providers/manager.py`
- Test: `backend/tests/test_provider_manager.py`

**Interfaces:**
- Produces: `KNOWN_PROVIDER_NAMES = ("openai", "anthropic", "ollama")`, `ProviderManager` with constructor `ProviderManager(factory: ProviderFactory, settings: Settings)`, methods `get_provider(name) -> BaseProvider`, `is_provider_available(name) -> bool`, `list_providers() -> dict[str, str]`. Consumed by `services/model_registry.py` (Tasks 13-14), `api/main.py` (Task 16), `api/routers/health.py` (Task 16).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_provider_manager.py
import pytest

from backend.config.settings import Settings
from backend.providers.factory import ProviderFactory
from backend.providers.manager import ProviderManager
from backend.providers.mock_provider import MockProvider
from backend.providers.openai_provider import OpenAIProvider


def _make_factory():
    factory = ProviderFactory()
    factory.register("mock", MockProvider)
    factory.register("openai", OpenAIProvider)
    return factory


def test_mock_provider_always_available():
    settings = Settings(_env_file=None)
    manager = ProviderManager(_make_factory(), settings)

    assert manager.is_provider_available("mock") is True
    assert isinstance(manager.get_provider("mock"), MockProvider)


def test_openai_disabled_without_key():
    settings = Settings(_env_file=None, openai_api_key=None)
    manager = ProviderManager(_make_factory(), settings)

    assert manager.is_provider_available("openai") is False
    with pytest.raises(KeyError):
        manager.get_provider("openai")


def test_openai_available_with_key():
    settings = Settings(_env_file=None, openai_api_key="sk-test")
    manager = ProviderManager(_make_factory(), settings)

    assert manager.is_provider_available("openai") is True
    assert isinstance(manager.get_provider("openai"), OpenAIProvider)


def test_list_providers_covers_known_providers():
    settings = Settings(_env_file=None, openai_api_key="sk-test")
    manager = ProviderManager(_make_factory(), settings)

    assert manager.list_providers() == {
        "openai": "available",
        "anthropic": "disabled",
        "ollama": "disabled",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_provider_manager.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.providers.manager'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/providers/manager.py
from backend.config.settings import Settings
from backend.providers.base import BaseProvider
from backend.providers.factory import ProviderFactory

KNOWN_PROVIDER_NAMES = ("openai", "anthropic", "ollama")


class ProviderManager:
    def __init__(self, factory: ProviderFactory, settings: Settings) -> None:
        self._providers: dict[str, BaseProvider] = {"mock": factory.create("mock", settings)}

        if settings.openai_api_key:
            self._providers["openai"] = factory.create("openai", settings)

    def get_provider(self, name: str) -> BaseProvider:
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' is not available")
        return self._providers[name]

    def is_provider_available(self, name: str) -> bool:
        return name in self._providers

    def list_providers(self) -> dict[str, str]:
        return {
            name: ("available" if name in self._providers else "disabled")
            for name in KNOWN_PROVIDER_NAMES
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_provider_manager.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/providers/manager.py backend/tests/test_provider_manager.py
git commit -m "feat: add ProviderManager"
```

---

### Task 13: ModelRegistry — Config Loading & Reads

**Files:**
- Create: `backend/config/models.yaml`
- Create: `backend/services/model_registry.py`
- Test: `backend/tests/test_model_registry.py`

**Interfaces:**
- Produces: `ModelSpec` (Pydantic model with fields `id, provider, model, input_cost, output_cost, context_window, max_output_tokens, supports_streaming, supports_tools, supports_json, supports_vision, benchmark_score, average_latency_ms, available`), `BaseRegistry` ABC (`get_model, get_models, get_available_models, get_provider_models, reload`), `ModelRegistry(BaseRegistry)` constructor `ModelRegistry(provider_manager, event_bus, cost_estimator, session_factory, yaml_path)`, methods `reload() -> None`, `get_model(model_id) -> ModelSpec`, `get_models() -> list[ModelSpec]`, `get_available_models() -> list[ModelSpec]`, `get_provider_models(provider) -> list[ModelSpec]`. Consumed by `api/routers/models.py` (Task 17), `api/main.py` (Task 16). Task 14 extends this same file with `refresh_provider_status()` and `estimate_cost()`.

- [ ] **Step 1: Write `backend/config/models.yaml`**

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
  - id: gpt-4o
    provider: openai
    model: gpt-4o
    pricing:
      input_cost: 2.50
      output_cost: 10.00
    limits:
      context_window: 128000
      max_output_tokens: 16384
    capabilities:
      supports_streaming: true
      supports_tools: true
      supports_json: true
      supports_vision: true
    metadata:
      benchmark_score: 0.93
      average_latency_ms: 900
```

- [ ] **Step 2: Write the failing test**

```python
# backend/tests/test_model_registry.py
import textwrap

import pytest

from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.events.bus import EventBus
from backend.providers.factory import ProviderFactory
from backend.providers.manager import ProviderManager
from backend.providers.mock_provider import MockProvider
from backend.providers.openai_provider import OpenAIProvider
from backend.services.cost_estimator import DefaultCostEstimator
from backend.services.model_registry import ModelRegistry

SAMPLE_YAML = textwrap.dedent("""
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
""")


def _make_registry(tmp_path, openai_key):
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(SAMPLE_YAML)

    settings = Settings(
        _env_file=None, database_url=f"sqlite:///{tmp_path}/test.db", openai_api_key=openai_key
    )
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    factory = ProviderFactory()
    factory.register("mock", MockProvider)
    factory.register("openai", OpenAIProvider)
    provider_manager = ProviderManager(factory, settings)

    return ModelRegistry(
        provider_manager=provider_manager,
        event_bus=EventBus(),
        cost_estimator=DefaultCostEstimator(),
        session_factory=session_factory,
        yaml_path=str(yaml_path),
    )


def test_reload_loads_models_into_cache(tmp_path):
    registry = _make_registry(tmp_path, openai_key="sk-test")
    registry.reload()

    models = registry.get_models()
    assert len(models) == 1
    assert models[0].id == "gpt-4o-mini"
    assert models[0].benchmark_score == 0.82


def test_get_available_models_respects_provider_key(tmp_path):
    registry = _make_registry(tmp_path, openai_key=None)
    registry.reload()

    assert registry.get_available_models() == []
    assert registry.get_models()[0].available is False


def test_get_model_unknown_raises(tmp_path):
    registry = _make_registry(tmp_path, openai_key="sk-test")
    registry.reload()

    with pytest.raises(KeyError):
        registry.get_model("nonexistent")


def test_get_provider_models(tmp_path):
    registry = _make_registry(tmp_path, openai_key="sk-test")
    registry.reload()

    assert len(registry.get_provider_models("openai")) == 1
    assert registry.get_provider_models("anthropic") == []
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_model_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.services.model_registry'`

- [ ] **Step 4: Write minimal implementation**

```python
# backend/services/model_registry.py
from abc import ABC, abstractmethod

import yaml
from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker

from backend.database.models import ModelRow
from backend.events.bus import EventBus
from backend.events.types import EventType
from backend.providers.manager import ProviderManager
from backend.services.cost_estimator import BaseCostEstimator


class _Pricing(BaseModel):
    input_cost: float
    output_cost: float


class _Limits(BaseModel):
    context_window: int
    max_output_tokens: int


class _Capabilities(BaseModel):
    supports_streaming: bool
    supports_tools: bool
    supports_json: bool
    supports_vision: bool


class _Metadata(BaseModel):
    benchmark_score: float
    average_latency_ms: float


class _ModelYamlEntry(BaseModel):
    id: str
    provider: str
    model: str
    pricing: _Pricing
    limits: _Limits
    capabilities: _Capabilities
    metadata: _Metadata


class ModelSpec(BaseModel):
    id: str
    provider: str
    model: str
    input_cost: float
    output_cost: float
    context_window: int
    max_output_tokens: int
    supports_streaming: bool
    supports_tools: bool
    supports_json: bool
    supports_vision: bool
    benchmark_score: float
    average_latency_ms: float
    available: bool = False

    @classmethod
    def from_yaml_entry(cls, entry: _ModelYamlEntry, available: bool) -> "ModelSpec":
        return cls(
            id=entry.id,
            provider=entry.provider,
            model=entry.model,
            input_cost=entry.pricing.input_cost,
            output_cost=entry.pricing.output_cost,
            context_window=entry.limits.context_window,
            max_output_tokens=entry.limits.max_output_tokens,
            supports_streaming=entry.capabilities.supports_streaming,
            supports_tools=entry.capabilities.supports_tools,
            supports_json=entry.capabilities.supports_json,
            supports_vision=entry.capabilities.supports_vision,
            benchmark_score=entry.metadata.benchmark_score,
            average_latency_ms=entry.metadata.average_latency_ms,
            available=available,
        )


class BaseRegistry(ABC):
    @abstractmethod
    def get_model(self, model_id: str) -> ModelSpec: ...

    @abstractmethod
    def get_models(self) -> list[ModelSpec]: ...

    @abstractmethod
    def get_available_models(self) -> list[ModelSpec]: ...

    @abstractmethod
    def get_provider_models(self, provider: str) -> list[ModelSpec]: ...

    @abstractmethod
    def reload(self) -> None: ...


class ModelRegistry(BaseRegistry):
    def __init__(
        self,
        provider_manager: ProviderManager,
        event_bus: EventBus,
        cost_estimator: BaseCostEstimator,
        session_factory: sessionmaker,
        yaml_path: str,
    ) -> None:
        self._provider_manager = provider_manager
        self._event_bus = event_bus
        self._cost_estimator = cost_estimator
        self._session_factory = session_factory
        self._yaml_path = yaml_path
        self._cache: dict[str, ModelSpec] = {}
        self._provider_health: dict[str, bool] = {}

    def _is_available(self, provider: str) -> bool:
        return self._provider_manager.is_provider_available(
            provider
        ) and self._provider_health.get(provider, True)

    def reload(self) -> None:
        with open(self._yaml_path) as f:
            raw = yaml.safe_load(f)

        entries = [_ModelYamlEntry.model_validate(item) for item in raw["models"]]
        cache: dict[str, ModelSpec] = {}

        with self._session_factory() as session:
            for entry in entries:
                spec = ModelSpec.from_yaml_entry(entry, available=self._is_available(entry.provider))
                cache[spec.id] = spec

                row = session.query(ModelRow).filter_by(model_id=spec.id).one_or_none()
                if row is None:
                    row = ModelRow(model_id=spec.id)
                    session.add(row)
                row.provider = spec.provider
                row.model_name = spec.model
                row.input_cost = spec.input_cost
                row.output_cost = spec.output_cost
                row.context_window = spec.context_window
                row.benchmark_score = spec.benchmark_score
                row.supports_streaming = spec.supports_streaming
                row.supports_tools = spec.supports_tools
                row.supports_json = spec.supports_json
                row.average_latency_ms = spec.average_latency_ms
                row.available = spec.available

                self._event_bus.emit(
                    EventType.MODEL_REGISTERED, {"model_id": spec.id, "provider": spec.provider}
                )

            session.commit()

        self._cache = cache

    def get_model(self, model_id: str) -> ModelSpec:
        if model_id not in self._cache:
            raise KeyError(f"Unknown model_id '{model_id}'")
        return self._cache[model_id]

    def get_models(self) -> list[ModelSpec]:
        return list(self._cache.values())

    def get_available_models(self) -> list[ModelSpec]:
        return [spec for spec in self._cache.values() if spec.available]

    def get_provider_models(self, provider: str) -> list[ModelSpec]:
        return [spec for spec in self._cache.values() if spec.provider == provider]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_model_registry.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add backend/config/models.yaml backend/services/model_registry.py backend/tests/test_model_registry.py
git commit -m "feat: add ModelRegistry config loading and read API"
```

---

### Task 14: ModelRegistry — Provider Status Refresh & Cost Estimation

**Files:**
- Modify: `backend/services/model_registry.py` (add `refresh_provider_status`, `estimate_cost`; add `ProviderRow`, `datetime`/`timezone` imports)
- Test: `backend/tests/test_model_registry_status.py`

**Interfaces:**
- Consumes: `ModelRegistry` from Task 13, `KNOWN_PROVIDER_NAMES` from `providers/manager.py` (Task 12).
- Produces: `ModelRegistry.refresh_provider_status() -> None` (async), `ModelRegistry.estimate_cost(model_id, input_tokens, output_tokens) -> float`. Consumed by `api/main.py` (Task 16).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_model_registry_status.py
from unittest.mock import AsyncMock

import pytest

from backend.tests.test_model_registry import _make_registry


async def test_refresh_marks_model_available_when_provider_healthy(tmp_path, mocker):
    registry = _make_registry(tmp_path, openai_key="sk-test")
    registry.reload()

    mocker.patch(
        "backend.providers.openai_provider.OpenAIProvider.health_check",
        new_callable=AsyncMock,
        return_value=True,
    )

    await registry.refresh_provider_status()

    assert registry.get_available_models()[0].available is True


async def test_refresh_marks_model_unavailable_when_provider_unhealthy(tmp_path, mocker):
    registry = _make_registry(tmp_path, openai_key="sk-test")
    registry.reload()

    mocker.patch(
        "backend.providers.openai_provider.OpenAIProvider.health_check",
        new_callable=AsyncMock,
        side_effect=RuntimeError("boom"),
    )

    await registry.refresh_provider_status()

    assert registry.get_models()[0].available is False


def test_estimate_cost(tmp_path):
    registry = _make_registry(tmp_path, openai_key="sk-test")
    registry.reload()

    cost = registry.estimate_cost("gpt-4o-mini", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(0.15 + 0.60)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_model_registry_status.py -v`
Expected: FAIL — `AttributeError: 'ModelRegistry' object has no attribute 'refresh_provider_status'`

- [ ] **Step 3: Add to `backend/services/model_registry.py`**

Add these imports at the top (alongside existing ones):

```python
from datetime import datetime, timezone

from backend.database.models import ProviderRow
from backend.providers.manager import KNOWN_PROVIDER_NAMES
```

Add these two methods to the `ModelRegistry` class (after `get_provider_models`):

```python
    async def refresh_provider_status(self) -> None:
        with self._session_factory() as session:
            for provider_name in KNOWN_PROVIDER_NAMES:
                row = session.query(ProviderRow).filter_by(name=provider_name).one_or_none()
                if row is None:
                    row = ProviderRow(name=provider_name)
                    session.add(row)

                if not self._provider_manager.is_provider_available(provider_name):
                    row.status = "disabled"
                    row.last_error = None
                    row.last_checked_at = datetime.now(timezone.utc)
                    self._event_bus.emit(EventType.PROVIDER_DISABLED, {"provider": provider_name})
                    continue

                provider = self._provider_manager.get_provider(provider_name)
                try:
                    healthy = await provider.health_check()
                    row.last_error = None
                except Exception as exc:
                    healthy = False
                    row.last_error = str(exc)

                self._provider_health[provider_name] = healthy
                row.status = "available" if healthy else "error"
                row.last_checked_at = datetime.now(timezone.utc)

                event_type = EventType.PROVIDER_AVAILABLE if healthy else EventType.PROVIDER_FAILED
                self._event_bus.emit(event_type, {"provider": provider_name, "status": row.status})

            session.commit()

        for spec_id, spec in list(self._cache.items()):
            self._cache[spec_id] = spec.model_copy(
                update={"available": self._is_available(spec.provider)}
            )

    def estimate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        spec = self.get_model(model_id)
        return self._cost_estimator.estimate(
            input_tokens, output_tokens, spec.input_cost, spec.output_cost
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_model_registry_status.py backend/tests/test_model_registry.py -v`
Expected: PASS (7 tests total)

- [ ] **Step 5: Commit**

```bash
git add backend/services/model_registry.py backend/tests/test_model_registry_status.py
git commit -m "feat: add ModelRegistry.refresh_provider_status and estimate_cost"
```

---

### Task 15: API Dependencies (DI Wiring)

**Files:**
- Create: `backend/api/dependencies.py`
- Test: `backend/tests/test_dependencies.py`

**Interfaces:**
- Produces: `get_settings, get_event_bus, get_provider_manager, get_model_registry, get_session_factory, get_app_version, get_app_start_time` (each `Callable[[Request], T]`), and their `Annotated[...]` aliases `SettingsDep, EventBusDep, ProviderManagerDep, ModelRegistryDep, SessionFactoryDep, AppVersionDep, AppStartTimeDep`. Consumed by `api/routers/health.py` and `api/routers/models.py` (Tasks 16-17), `api/main.py` (Task 16, sets the corresponding `app.state` attributes these read from).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_dependencies.py
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.dependencies import (
    AppStartTimeDep,
    AppVersionDep,
    EventBusDep,
    ModelRegistryDep,
    ProviderManagerDep,
    SessionFactoryDep,
    SettingsDep,
)
from backend.config.settings import Settings
from backend.events.bus import EventBus


def test_dependencies_read_from_app_state():
    app = FastAPI()
    app.state.settings = Settings(_env_file=None, environment="test")
    app.state.event_bus = EventBus()
    app.state.provider_manager = "fake-provider-manager"
    app.state.model_registry = "fake-model-registry"
    app.state.session_factory = "fake-session-factory"
    app.state.version = "0.1.0"
    app.state.start_time = 123.0

    @app.get("/probe")
    def probe(
        settings: SettingsDep,
        event_bus: EventBusDep,
        provider_manager: ProviderManagerDep,
        model_registry: ModelRegistryDep,
        session_factory: SessionFactoryDep,
        version: AppVersionDep,
        start_time: AppStartTimeDep,
    ):
        return {
            "environment": settings.environment,
            "event_bus_type": type(event_bus).__name__,
            "provider_manager": provider_manager,
            "model_registry": model_registry,
            "session_factory": session_factory,
            "version": version,
            "start_time": start_time,
        }

    client = TestClient(app)
    response = client.get("/probe")

    assert response.status_code == 200
    body = response.json()
    assert body["environment"] == "test"
    assert body["event_bus_type"] == "EventBus"
    assert body["provider_manager"] == "fake-provider-manager"
    assert body["model_registry"] == "fake-model-registry"
    assert body["session_factory"] == "fake-session-factory"
    assert body["version"] == "0.1.0"
    assert body["start_time"] == 123.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_dependencies.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.api.dependencies'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/api/dependencies.py
from typing import Annotated

from fastapi import Depends, Request

from backend.config.settings import Settings
from backend.events.bus import EventBus
from backend.providers.manager import ProviderManager
from backend.services.model_registry import ModelRegistry


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_event_bus(request: Request) -> EventBus:
    return request.app.state.event_bus


def get_provider_manager(request: Request) -> ProviderManager:
    return request.app.state.provider_manager


def get_model_registry(request: Request) -> ModelRegistry:
    return request.app.state.model_registry


def get_session_factory(request: Request):
    return request.app.state.session_factory


def get_app_version(request: Request) -> str:
    return request.app.state.version


def get_app_start_time(request: Request) -> float:
    return request.app.state.start_time


SettingsDep = Annotated[Settings, Depends(get_settings)]
EventBusDep = Annotated[EventBus, Depends(get_event_bus)]
ProviderManagerDep = Annotated[ProviderManager, Depends(get_provider_manager)]
ModelRegistryDep = Annotated[ModelRegistry, Depends(get_model_registry)]
SessionFactoryDep = Annotated[object, Depends(get_session_factory)]
AppVersionDep = Annotated[str, Depends(get_app_version)]
AppStartTimeDep = Annotated[float, Depends(get_app_start_time)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_dependencies.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add backend/api/dependencies.py backend/tests/test_dependencies.py
git commit -m "feat: add FastAPI dependency injection wiring"
```

---

### Task 16: FastAPI App + GET /v1/health

**Files:**
- Create: `backend/api/main.py`
- Create: `backend/api/routers/health.py`
- Test: `backend/tests/test_main.py`
- Test: `backend/tests/test_health_endpoint.py`

**Interfaces:**
- Consumes: `Settings` (Task 2), `configure_logging` (Task 4), `EventBus`/`register_logging_subscriber` (Tasks 5-6), `create_session_factory` (Task 7), `BaseProvider` impls + `ProviderFactory`/`ProviderManager` (Tasks 9-12), `ModelRegistry` (Tasks 13-14), all `*Dep` aliases (Task 15).
- Produces: `create_app() -> FastAPI`, module-level `app`, `APP_VERSION = "0.1.0"`. `GET /v1/health` route.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_main.py
from backend.api.main import create_app


def test_create_app_registers_health_route():
    app = create_app()
    paths = [route.path for route in app.routes]
    assert "/v1/health" in paths
```

```python
# backend/tests/test_health_endpoint.py
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.dependencies import (
    get_app_start_time,
    get_app_version,
    get_model_registry,
    get_provider_manager,
    get_session_factory,
    get_settings,
)
from backend.api.routers.health import router as health_router
from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db


class _FakeProviderManager:
    def list_providers(self):
        return {"openai": "available", "anthropic": "disabled", "ollama": "disabled"}


class _FakeModelRegistry:
    def get_models(self):
        return [object(), object()]


def test_health_endpoint_returns_expected_shape(tmp_path):
    app = FastAPI()
    app.include_router(health_router, prefix="/v1")

    settings = Settings(_env_file=None, environment="test", database_url=f"sqlite:///{tmp_path}/t.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_app_version] = lambda: "0.1.0"
    app.dependency_overrides[get_app_start_time] = lambda: time.time() - 10
    app.dependency_overrides[get_provider_manager] = lambda: _FakeProviderManager()
    app.dependency_overrides[get_model_registry] = lambda: _FakeModelRegistry()
    app.dependency_overrides[get_session_factory] = lambda: session_factory

    client = TestClient(app)
    response = client.get("/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["version"] == "0.1.0"
    assert body["environment"] == "test"
    assert body["database"] == "healthy"
    assert body["providers"] == {
        "openai": "available",
        "anthropic": "disabled",
        "ollama": "disabled",
    }
    assert body["loaded_models"] == 2
    assert body["uptime_seconds"] >= 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest backend/tests/test_main.py backend/tests/test_health_endpoint.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.api.main'`

- [ ] **Step 3: Write `backend/api/routers/health.py`**

```python
# backend/api/routers/health.py
import time

from fastapi import APIRouter
from sqlalchemy import text

from backend.api.dependencies import (
    AppStartTimeDep,
    AppVersionDep,
    ModelRegistryDep,
    ProviderManagerDep,
    SessionFactoryDep,
    SettingsDep,
)

router = APIRouter()


@router.get("/health")
def get_health(
    settings: SettingsDep,
    version: AppVersionDep,
    start_time: AppStartTimeDep,
    provider_manager: ProviderManagerDep,
    model_registry: ModelRegistryDep,
    session_factory: SessionFactoryDep,
):
    try:
        with session_factory() as session:
            session.execute(text("SELECT 1"))
        database_status = "healthy"
    except Exception:
        database_status = "unhealthy"

    return {
        "status": "healthy",
        "version": version,
        "environment": settings.environment,
        "database": database_status,
        "providers": provider_manager.list_providers(),
        "loaded_models": len(model_registry.get_models()),
        "uptime_seconds": round(time.time() - start_time, 1),
    }
```

- [ ] **Step 4: Write `backend/api/main.py`**

```python
# backend/api/main.py
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.api.routers.health import router as health_router
from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.events.bus import EventBus
from backend.events.subscribers import register_logging_subscriber
from backend.providers.factory import ProviderFactory
from backend.providers.manager import ProviderManager
from backend.providers.mock_provider import MockProvider
from backend.providers.openai_provider import OpenAIProvider
from backend.services.cost_estimator import DefaultCostEstimator
from backend.services.model_registry import ModelRegistry
from backend.telemetry.logging import configure_logging

APP_VERSION = "0.1.0"


def _build_provider_factory() -> ProviderFactory:
    factory = ProviderFactory()
    factory.register("mock", MockProvider)
    factory.register("openai", OpenAIProvider)
    return factory


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    configure_logging(settings)

    event_bus = EventBus()
    register_logging_subscriber(event_bus)

    # No try/except around DB init: a bad DATABASE_URL must crash startup.
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    provider_manager = ProviderManager(_build_provider_factory(), settings)

    model_registry = ModelRegistry(
        provider_manager=provider_manager,
        event_bus=event_bus,
        cost_estimator=DefaultCostEstimator(),
        session_factory=session_factory,
        yaml_path=settings.models_yaml_path,
    )
    model_registry.reload()
    await model_registry.refresh_provider_status()

    app.state.settings = settings
    app.state.event_bus = event_bus
    app.state.provider_manager = provider_manager
    app.state.model_registry = model_registry
    app.state.session_factory = session_factory
    app.state.version = APP_VERSION
    app.state.start_time = time.time()

    yield


def create_app() -> FastAPI:
    app = FastAPI(title="LLM Cost Autopilot", version=APP_VERSION, lifespan=lifespan)
    app.include_router(health_router, prefix="/v1")
    return app


app = create_app()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest backend/tests/test_main.py backend/tests/test_health_endpoint.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add backend/api/main.py backend/api/routers/health.py backend/tests/test_main.py backend/tests/test_health_endpoint.py
git commit -m "feat: add FastAPI app factory and GET /v1/health"
```

---

### Task 17: GET /v1/models

**Files:**
- Create: `backend/api/routers/models.py`
- Modify: `backend/api/main.py` (mount the new router)
- Test: `backend/tests/test_models_endpoint.py`

**Interfaces:**
- Consumes: `ModelRegistryDep` (Task 15), `ModelSpec` (Task 13).
- Produces: `GET /v1/models` route returning a list of full `ModelSpec` dicts.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_models_endpoint.py
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.dependencies import get_model_registry
from backend.api.routers.models import router as models_router
from backend.services.model_registry import ModelSpec


class _FakeModelRegistry:
    def get_models(self):
        return [
            ModelSpec(
                id="gpt-4o-mini",
                provider="openai",
                model="gpt-4o-mini",
                input_cost=0.15,
                output_cost=0.60,
                context_window=128000,
                max_output_tokens=16384,
                supports_streaming=True,
                supports_tools=True,
                supports_json=True,
                supports_vision=False,
                benchmark_score=0.82,
                average_latency_ms=450,
                available=True,
            )
        ]


def test_list_models_returns_full_spec():
    app = FastAPI()
    app.include_router(models_router, prefix="/v1")
    app.dependency_overrides[get_model_registry] = lambda: _FakeModelRegistry()

    client = TestClient(app)
    response = client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["provider"] == "openai"
    assert body[0]["benchmark_score"] == 0.82
    assert body[0]["available"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_models_endpoint.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.api.routers.models'`

- [ ] **Step 3: Write `backend/api/routers/models.py`**

```python
# backend/api/routers/models.py
from fastapi import APIRouter

from backend.api.dependencies import ModelRegistryDep

router = APIRouter()


@router.get("/models")
def list_models(model_registry: ModelRegistryDep):
    return [
        {
            "provider": spec.provider,
            "model": spec.model,
            "available": spec.available,
            "input_cost": spec.input_cost,
            "output_cost": spec.output_cost,
            "context_window": spec.context_window,
            "max_output_tokens": spec.max_output_tokens,
            "supports_streaming": spec.supports_streaming,
            "supports_tools": spec.supports_tools,
            "supports_json": spec.supports_json,
            "supports_vision": spec.supports_vision,
            "benchmark_score": spec.benchmark_score,
            "average_latency_ms": spec.average_latency_ms,
        }
        for spec in model_registry.get_models()
    ]
```

- [ ] **Step 4: Modify `backend/api/main.py`**

Add the import near the other router import:

```python
from backend.api.routers.models import router as models_router
```

Add this line in `create_app()`, right after the existing `app.include_router(health_router, prefix="/v1")`:

```python
    app.include_router(models_router, prefix="/v1")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_models_endpoint.py backend/tests/test_main.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add backend/api/routers/models.py backend/api/main.py backend/tests/test_models_endpoint.py
git commit -m "feat: add GET /v1/models"
```

---

### Task 18: Docs, Full Suite, Manual Verification

**Files:**
- Create: `README.md`
- Create: `docs/ARCHITECTURE.md`

**Interfaces:**
- None — documentation and verification only.

- [ ] **Step 1: Write `README.md`**

```markdown
# LLM Cost Autopilot

Phase 1: project skeleton, provider foundation, and event bus for an
intelligent cost-aware LLM routing layer.

## What exists today

- `GET /v1/health` — service, database, and provider status
- `GET /v1/models` — full model registry (pricing, limits, capabilities)
- `ModelRegistry`, backed by `backend/config/models.yaml` and persisted to SQLite
- `OpenAIProvider` and `MockProvider` behind a shared `BaseProvider` interface
- In-process event bus (`PROVIDER_AVAILABLE`, `PROVIDER_DISABLED`, `PROVIDER_FAILED`, `MODEL_REGISTERED`)
- Structured JSON logging to console and rotating file

Routing, classification, verification, and the dashboard are not built yet
— see `docs/superpowers/specs/` for the full roadmap.

## Setup

```bash
uv sync
cp .env.example .env   # add OPENAI_API_KEY to enable the OpenAI provider
```

## Run tests

```bash
uv run pytest
```

## Run the API

```bash
uv run uvicorn backend.api.main:app --reload
```

Then:

```bash
curl http://127.0.0.1:8000/v1/health
curl http://127.0.0.1:8000/v1/models
```
```

- [ ] **Step 2: Write `docs/ARCHITECTURE.md`**

```markdown
# Architecture

## Provider Layer

`BaseProvider` defines `generate`, `stream`, `health_check`, `count_tokens`,
and `estimate_cost`. `ProviderFactory` registers provider classes by name;
`ProviderManager` builds one instance per provider that has the
configuration it needs (`MockProvider` always, `OpenAIProvider` when
`OPENAI_API_KEY` is set) and is the only source of truth for provider
availability.

## Registry

`ModelRegistry` loads `backend/config/models.yaml`, validates it into
`ModelSpec` objects, and keeps an in-memory cache that all reads go
through. `reload()` re-reads YAML and upserts the `models` table.
`refresh_provider_status()` pings each configured provider's
`health_check()` and upserts the `providers` table — kept separate from
`reload()` because one is a config concern and the other is a runtime
concern.

## Events

An in-process, synchronous `EventBus` (`subscribe`/`emit`, no external
broker). Phase 1 emits `PROVIDER_AVAILABLE`, `PROVIDER_DISABLED`,
`PROVIDER_FAILED`, and `MODEL_REGISTERED`; a logging subscriber writes
every event to the structured logger.

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
```

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest -v`
Expected: PASS (all tests across all tasks)

- [ ] **Step 4: Manual verification — run the server and hit both endpoints**

```bash
uv run uvicorn backend.api.main:app --port 8000 &
sleep 2
curl -s http://127.0.0.1:8000/v1/health | python3 -m json.tool
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
kill %1
```

Expected: `/v1/health` returns `status: healthy`; `/v1/models` returns the 2 models from `models.yaml` (both `available: false` unless `OPENAI_API_KEY` is set in `.env`).

- [ ] **Step 5: Commit**

```bash
git add README.md docs/ARCHITECTURE.md
git commit -m "docs: add README and Phase 1 architecture notes"
```
