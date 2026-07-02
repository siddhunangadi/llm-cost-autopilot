# Phase 5 Implementation Plan: Provider Fault Tolerance

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ChatService` resilient to provider failures — retry transient errors, trip a per-provider circuit breaker on sustained failure, and fail over exactly once to a different provider/model via `RoutingEngine`, all without `RoutingEngine` or `ProviderManager` knowing resilience exists.

**Architecture:** Two new pure/testable primitives (`BaseRetryPolicy`/`ExponentialBackoffRetryPolicy`, `CircuitBreaker`) composed by a new orchestrator (`ProviderExecutor`) that wraps `ProviderManager`. `RoutingEngine.route()` gains an `exclude_providers` filter param. `ChatService` coordinates: primary attempt with retry, on failure re-route excluding the failed provider, one retry-free failover attempt, persisting an ordered two-row routing audit trail and a standardized `error_type` on failure.

**Tech Stack:** Same as Phases 1-4 — Python 3.11+, `uv`, FastAPI, Pydantic v2, SQLAlchemy 2.0. No new dependencies; `asyncio.sleep`/`time.monotonic` (stdlib) provide the DI seams for deterministic tests.

**Spec:** `docs/superpowers/specs/2026-07-02-phase5-resilience-design.md` (frozen — implement exactly).

## Global Constraints

- Same `uv`-managed Python 3.11+ project as Phases 1-4; no new dependencies.
- One batch (Tasks 42-48), one full regression run, one manual end-to-end verification, one commit, then tag `v0.5.0`.
- `BaseRetryPolicy`/`ExponentialBackoffRetryPolicy` never emit events, never touch a `CircuitBreaker`, never know about providers by name.
- `CircuitBreaker` only sees `allow_request()` / `record_success()` / `record_failure()` — no I/O, no event emission, no knowledge of retries.
- **Sole-emitter invariant:** `ProviderExecutor` is the only component that emits resilience events (`CIRCUIT_OPENED`, `CIRCUIT_HALF_OPEN`, `CIRCUIT_CLOSED`, `PROVIDER_FAILOVER_TRIGGERED`).
- **Final-outcome-only invariant:** a `generate()` call records exactly one `record_success()`/`record_failure()` to the breaker regardless of how many internal retry attempts occurred.
- **Per-provider circuit breakers:** one `CircuitBreaker` instance per entry in `KNOWN_PROVIDER_NAMES` (`openai`, `anthropic`, `ollama`) — never global, never per-model.
- **No recursive failover:** exactly one re-route, exactly one retry-free attempt against the replacement provider, ever.
- `CircuitBreaker` defaults: `failure_threshold=5`, `open_timeout=30.0` seconds. `ExponentialBackoffRetryPolicy` defaults: `max_attempts=3`, `base_delay=0.2`, `multiplier=2.0`. Hardcoded at construction in `main.py`, not YAML-configured.
- No placeholder code, no TODOs, no speculative abstractions.

---

## Batch 1: Full Provider Fault Tolerance (Tasks 42-48)

### Task 42: `BaseRetryPolicy` & `ExponentialBackoffRetryPolicy`

**Files:**
- Create: `backend/providers/retry.py`
- Test: `backend/tests/test_retry_policy.py`

**Interfaces:**
- Produces: `BaseRetryPolicy` ABC (`async execute(operation: Callable[[], Awaitable[T]]) -> T`), `ExponentialBackoffRetryPolicy(max_attempts=3, base_delay=0.2, multiplier=2.0, sleep=asyncio.sleep)`. Consumed by `ProviderExecutor` (Task 44).

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_retry_policy.py
import pytest

from backend.providers.base import ProviderError
from backend.providers.retry import BaseRetryPolicy, ExponentialBackoffRetryPolicy


def test_base_retry_policy_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseRetryPolicy()


async def test_execute_returns_result_on_first_success():
    calls = []

    async def operation():
        calls.append(1)
        return "ok"

    policy = ExponentialBackoffRetryPolicy(sleep=_no_op_sleep)
    result = await policy.execute(operation)

    assert result == "ok"
    assert len(calls) == 1


async def test_execute_retries_until_success():
    attempts = {"count": 0}

    async def operation():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ProviderError("transient")
        return "ok"

    policy = ExponentialBackoffRetryPolicy(max_attempts=3, sleep=_no_op_sleep)
    result = await policy.execute(operation)

    assert result == "ok"
    assert attempts["count"] == 3


async def test_execute_raises_after_max_attempts_exhausted():
    attempts = {"count": 0}

    async def operation():
        attempts["count"] += 1
        raise ProviderError(f"failure-{attempts['count']}")

    policy = ExponentialBackoffRetryPolicy(max_attempts=3, sleep=_no_op_sleep)

    with pytest.raises(ProviderError, match="failure-3"):
        await policy.execute(operation)

    assert attempts["count"] == 3


async def test_execute_uses_exponential_backoff_delays():
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    async def operation():
        raise ProviderError("always fails")

    policy = ExponentialBackoffRetryPolicy(
        max_attempts=3, base_delay=0.2, multiplier=2.0, sleep=fake_sleep
    )

    with pytest.raises(ProviderError):
        await policy.execute(operation)

    assert sleeps == [0.2, 0.4]  # 2 sleeps between 3 attempts, exponential


async def _no_op_sleep(delay: float) -> None:
    pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest backend/tests/test_retry_policy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.providers.retry'`

- [ ] **Step 3: Write the implementation**

```python
# backend/providers/retry.py
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import TypeVar

from backend.providers.base import ProviderError

T = TypeVar("T")


class BaseRetryPolicy(ABC):
    @abstractmethod
    async def execute(self, operation: Callable[[], Awaitable[T]]) -> T: ...


class ExponentialBackoffRetryPolicy(BaseRetryPolicy):
    """Retries any zero-argument async callable on ProviderError. Has no
    knowledge of providers, circuits, or events -- it only knows about
    its own attempt budget. Callers that need retry outcomes recorded
    anywhere (metrics, circuit breakers) must do so themselves; this
    class never emits anything."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 0.2,
        multiplier: float = 2.0,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        import asyncio

        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._multiplier = multiplier
        self._sleep = sleep if sleep is not None else asyncio.sleep

    async def execute(self, operation: Callable[[], Awaitable[T]]) -> T:
        last_exc: ProviderError | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return await operation()
            except ProviderError as exc:
                last_exc = exc
                if attempt < self._max_attempts:
                    await self._sleep(self._base_delay * (self._multiplier ** (attempt - 1)))
        assert last_exc is not None
        raise last_exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest backend/tests/test_retry_policy.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/providers/retry.py backend/tests/test_retry_policy.py
git commit -m "feat: add BaseRetryPolicy and ExponentialBackoffRetryPolicy"
```

### Task 43: `CircuitBreaker`

**Files:**
- Create: `backend/providers/circuit_breaker.py`
- Test: `backend/tests/test_circuit_breaker.py`

**Interfaces:**
- Produces: `CircuitState(str, Enum)` (`CLOSED`, `OPEN`, `HALF_OPEN`), `CircuitBreaker(failure_threshold=5, open_timeout=30.0, clock=time.monotonic)` with `allow_request() -> bool`, `retry_after_seconds() -> float`, `record_success() -> None`, `record_failure() -> None`, and read-only properties `state`, `consecutive_failures`, `successes`, `failures`, `last_failure_at`. Consumed by `ProviderExecutor` (Task 44).

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_circuit_breaker.py
from backend.providers.circuit_breaker import CircuitBreaker, CircuitState


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_starts_closed():
    breaker = CircuitBreaker()
    assert breaker.state == CircuitState.CLOSED
    assert breaker.allow_request() is True


def test_opens_after_failure_threshold_consecutive_failures():
    breaker = CircuitBreaker(failure_threshold=5)
    for _ in range(4):
        breaker.record_failure()
    assert breaker.state == CircuitState.CLOSED

    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN
    assert breaker.consecutive_failures == 5


def test_success_resets_consecutive_failures_and_stays_closed():
    breaker = CircuitBreaker(failure_threshold=5)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()

    assert breaker.state == CircuitState.CLOSED
    assert breaker.consecutive_failures == 0


def test_open_blocks_requests_until_open_timeout_elapses():
    clock = _FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, open_timeout=30.0, clock=clock)

    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN
    assert breaker.allow_request() is False

    clock.advance(29.9)
    assert breaker.allow_request() is False

    clock.advance(0.2)
    assert breaker.allow_request() is True
    assert breaker.state == CircuitState.HALF_OPEN


def test_half_open_success_closes_circuit():
    clock = _FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, open_timeout=30.0, clock=clock)
    breaker.record_failure()
    clock.advance(30.0)
    breaker.allow_request()  # transitions to HALF_OPEN
    assert breaker.state == CircuitState.HALF_OPEN

    breaker.record_success()
    assert breaker.state == CircuitState.CLOSED


def test_half_open_failure_reopens_circuit():
    clock = _FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, open_timeout=30.0, clock=clock)
    breaker.record_failure()
    clock.advance(30.0)
    breaker.allow_request()  # transitions to HALF_OPEN
    assert breaker.state == CircuitState.HALF_OPEN

    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN
    assert breaker.allow_request() is False  # freshly opened, timer restarted


def test_retry_after_seconds_zero_when_not_open():
    breaker = CircuitBreaker()
    assert breaker.retry_after_seconds() == 0.0


def test_retry_after_seconds_counts_down_while_open():
    clock = _FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, open_timeout=30.0, clock=clock)
    breaker.record_failure()

    assert breaker.retry_after_seconds() == 30.0
    clock.advance(10.0)
    assert breaker.retry_after_seconds() == 20.0


def test_tracks_successes_and_failures_counts():
    breaker = CircuitBreaker(failure_threshold=5)
    breaker.record_success()
    breaker.record_success()
    breaker.record_failure()

    assert breaker.successes == 2
    assert breaker.failures == 1


def test_last_failure_at_records_clock_value():
    clock = _FakeClock(start=100.0)
    breaker = CircuitBreaker(clock=clock)

    breaker.record_failure()

    assert breaker.last_failure_at == 100.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest backend/tests/test_circuit_breaker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.providers.circuit_breaker'`

- [ ] **Step 3: Write the implementation**

```python
# backend/providers/circuit_breaker.py
import time
from collections.abc import Callable
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Tracks attempt outcomes for a single provider and exposes the
    resulting state. Has no knowledge of retries, event buses, or which
    provider it belongs to -- the owner (ProviderExecutor) supplies that
    context and does all I/O/emission itself."""

    def __init__(
        self,
        failure_threshold: int = 5,
        open_timeout: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._open_timeout = open_timeout
        self._clock = clock
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._successes = 0
        self._failures = 0
        self._last_failure_at: float | None = None
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def successes(self) -> int:
        return self._successes

    @property
    def failures(self) -> int:
        return self._failures

    @property
    def last_failure_at(self) -> float | None:
        return self._last_failure_at

    def allow_request(self) -> bool:
        if self._state == CircuitState.OPEN:
            assert self._opened_at is not None
            if self._clock() - self._opened_at >= self._open_timeout:
                self._state = CircuitState.HALF_OPEN
                return True
            return False
        return True

    def retry_after_seconds(self) -> float:
        if self._state != CircuitState.OPEN:
            return 0.0
        assert self._opened_at is not None
        return max(0.0, self._open_timeout - (self._clock() - self._opened_at))

    def record_success(self) -> None:
        self._successes += 1
        self._consecutive_failures = 0
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self._failures += 1
        self._consecutive_failures += 1
        self._last_failure_at = self._clock()
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._opened_at = self._clock()
        elif self._consecutive_failures >= self._failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = self._clock()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest backend/tests/test_circuit_breaker.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/providers/circuit_breaker.py backend/tests/test_circuit_breaker.py
git commit -m "feat: add CircuitBreaker with explicit state machine"
```

### Task 44: `ProviderExecutor` & `CircuitOpenError`

**Files:**
- Create: `backend/providers/executor.py`
- Modify: `backend/events/types.py`
- Test: `backend/tests/test_provider_executor.py`

**Interfaces:**
- Consumes: `BaseRetryPolicy` (Task 42), `CircuitBreaker`/`CircuitState` (Task 43), `ProviderManager`/`ProviderError` (Phase 1), `EventBus`/`EventType` (Phase 1).
- Produces: `CircuitOpenError(provider, state, consecutive_failures, retry_after_seconds)`, `ProviderExecutor(provider_manager, retry_policy, circuit_breakers, event_bus)` with `async generate(provider_name, prompt, model, *, retry: bool) -> str`, `circuit_states() -> dict[str, dict]`, `emit_failover_triggered(*, failed_provider, replacement_provider, original_model, replacement_model, reason, attempt_number) -> None`. Consumed by `ChatService` (Task 46), `main.py` (Task 48).

- [ ] **Step 1: Add the four new `EventType` values**

Modify `backend/events/types.py`:

```python
from enum import Enum


class EventType(str, Enum):
    PROVIDER_AVAILABLE = "provider_available"
    PROVIDER_DISABLED = "provider_disabled"
    PROVIDER_FAILED = "provider_failed"
    MODEL_REGISTERED = "model_registered"
    VERIFICATION_STARTED = "verification_started"
    VERIFICATION_COMPLETED = "verification_completed"
    VERIFICATION_FAILED = "verification_failed"
    CIRCUIT_OPENED = "circuit_opened"
    CIRCUIT_HALF_OPEN = "circuit_half_open"
    CIRCUIT_CLOSED = "circuit_closed"
    PROVIDER_FAILOVER_TRIGGERED = "provider_failover_triggered"
```

- [ ] **Step 2: Write the failing tests**

```python
# backend/tests/test_provider_executor.py
import pytest

from backend.events.bus import EventBus
from backend.events.types import EventType
from backend.providers.base import BaseProvider, ProviderError
from backend.providers.circuit_breaker import CircuitBreaker
from backend.providers.executor import CircuitOpenError, ProviderExecutor
from backend.providers.retry import ExponentialBackoffRetryPolicy


class _ScriptedProvider(BaseProvider):
    """Returns pre-scripted outcomes in order; raises IndexError if
    called more times than scripted (a bug in the test, not the code
    under test)."""

    def __init__(self, outcomes: list[Exception | str]) -> None:
        self._outcomes = list(outcomes)
        self.call_count = 0

    @property
    def name(self) -> str:
        return "scripted"

    async def generate(self, prompt: str, model: str, **kwargs) -> str:
        self.call_count += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def stream(self, prompt: str, model: str, **kwargs):
        yield await self.generate(prompt, model, **kwargs)

    async def health_check(self) -> bool:
        return True

    def count_tokens(self, text: str) -> int:
        return len(text)

    def estimate_cost(self, input_tokens, output_tokens, input_cost, output_cost) -> float:
        return 0.0


class _StubProviderManager:
    def __init__(self, providers: dict[str, BaseProvider]) -> None:
        self._providers = providers

    def get_provider(self, name: str) -> BaseProvider:
        return self._providers[name]


async def _no_op_sleep(delay: float) -> None:
    pass


def _executor(provider: BaseProvider, provider_name: str = "primary", failure_threshold=5, bus=None):
    manager = _StubProviderManager({provider_name: provider})
    breakers = {provider_name: CircuitBreaker(failure_threshold=failure_threshold)}
    retry_policy = ExponentialBackoffRetryPolicy(max_attempts=3, sleep=_no_op_sleep)
    return ProviderExecutor(
        provider_manager=manager, retry_policy=retry_policy,
        circuit_breakers=breakers, event_bus=bus or EventBus(),
    ), breakers[provider_name]


async def test_generate_returns_result_on_success():
    provider = _ScriptedProvider(["hello"])
    executor, breaker = _executor(provider)

    result = await executor.generate("primary", "hi", "model-x", retry=True)

    assert result == "hello"
    assert breaker.successes == 1
    assert breaker.failures == 0


async def test_generate_retries_and_records_one_success():
    provider = _ScriptedProvider([ProviderError("blip"), ProviderError("blip"), "hello"])
    executor, breaker = _executor(provider)

    result = await executor.generate("primary", "hi", "model-x", retry=True)

    assert result == "hello"
    assert provider.call_count == 3
    assert breaker.successes == 1
    assert breaker.failures == 0  # intermediate retry failures never recorded


async def test_generate_records_one_failure_after_retries_exhausted():
    provider = _ScriptedProvider([ProviderError("a"), ProviderError("b"), ProviderError("c")])
    executor, breaker = _executor(provider)

    with pytest.raises(ProviderError):
        await executor.generate("primary", "hi", "model-x", retry=True)

    assert provider.call_count == 3
    assert breaker.failures == 1
    assert breaker.successes == 0


async def test_generate_retry_false_makes_exactly_one_attempt():
    provider = _ScriptedProvider([ProviderError("fails once, no retry")])
    executor, breaker = _executor(provider)

    with pytest.raises(ProviderError):
        await executor.generate("primary", "hi", "model-x", retry=False)

    assert provider.call_count == 1
    assert breaker.failures == 1


async def test_generate_raises_circuit_open_without_calling_provider():
    provider = _ScriptedProvider([ProviderError("a")] * 5)
    executor, breaker = _executor(provider, failure_threshold=1)

    with pytest.raises(ProviderError):
        await executor.generate("primary", "hi", "model-x", retry=False)
    assert breaker.state.value == "open"

    with pytest.raises(CircuitOpenError) as exc_info:
        await executor.generate("primary", "hi", "model-x", retry=True)

    assert provider.call_count == 1  # no second call was made
    assert exc_info.value.provider == "primary"
    assert exc_info.value.consecutive_failures == 1
    assert exc_info.value.retry_after_seconds > 0


async def test_generate_emits_circuit_opened_on_transition():
    events: list[tuple[EventType, dict]] = []
    bus = EventBus()
    bus.subscribe(EventType.CIRCUIT_OPENED, lambda p: events.append((EventType.CIRCUIT_OPENED, p)))

    provider = _ScriptedProvider([ProviderError("a")])
    executor, _ = _executor(provider, failure_threshold=1, bus=bus)

    with pytest.raises(ProviderError):
        await executor.generate("primary", "hi", "model-x", retry=False)

    assert len(events) == 1
    assert events[0][1] == {"provider": "primary", "from": "closed", "to": "open"}


async def test_generate_does_not_emit_when_state_unchanged():
    events: list[tuple[EventType, dict]] = []
    bus = EventBus()
    bus.subscribe(EventType.CIRCUIT_CLOSED, lambda p: events.append((EventType.CIRCUIT_CLOSED, p)))

    provider = _ScriptedProvider(["ok", "ok"])
    executor, _ = _executor(provider, bus=bus)

    await executor.generate("primary", "hi", "model-x", retry=True)
    await executor.generate("primary", "hi", "model-x", retry=True)

    assert events == []  # already closed both times -- no transition, no event


async def test_circuit_states_reports_all_breakers():
    provider = _ScriptedProvider(["ok"])
    executor, _ = _executor(provider)

    states = executor.circuit_states()

    assert states == {
        "primary": {"state": "closed", "consecutive_failures": 0, "successes": 0, "failures": 0}
    }


async def test_emit_failover_triggered_publishes_full_payload():
    events: list[dict] = []
    bus = EventBus()
    bus.subscribe(EventType.PROVIDER_FAILOVER_TRIGGERED, events.append)

    provider = _ScriptedProvider(["ok"])
    executor, _ = _executor(provider, bus=bus)

    executor.emit_failover_triggered(
        failed_provider="openai", replacement_provider="mock",
        original_model="gpt-4o-mini", replacement_model="mock-model",
        reason="provider_error", attempt_number=2,
    )

    assert events == [{
        "failed_provider": "openai", "replacement_provider": "mock",
        "original_model": "gpt-4o-mini", "replacement_model": "mock-model",
        "reason": "provider_error", "attempt_number": 2,
    }]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest backend/tests/test_provider_executor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.providers.executor'`

- [ ] **Step 4: Write the implementation**

```python
# backend/providers/executor.py
from backend.events.bus import EventBus
from backend.events.types import EventType
from backend.providers.base import ProviderError
from backend.providers.circuit_breaker import CircuitBreaker, CircuitState
from backend.providers.manager import ProviderManager
from backend.providers.retry import BaseRetryPolicy


class CircuitOpenError(Exception):
    def __init__(
        self, provider: str, state: CircuitState,
        consecutive_failures: int, retry_after_seconds: float,
    ) -> None:
        self.provider = provider
        self.state = state
        self.consecutive_failures = consecutive_failures
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            f"Circuit for '{provider}' is {state.value} "
            f"({consecutive_failures} consecutive failures, "
            f"retry after {retry_after_seconds:.1f}s)"
        )


class ProviderExecutor:
    """Retry attempts are considered part of a single logical provider
    execution. Metrics, breaker state, and events are emitted once per
    logical execution rather than once per individual retry attempt.

    The sole owner of the resilience lifecycle: BaseRetryPolicy and
    CircuitBreaker never emit events or touch each other -- only this
    class does, and only here."""

    def __init__(
        self,
        provider_manager: ProviderManager,
        retry_policy: BaseRetryPolicy,
        circuit_breakers: dict[str, CircuitBreaker],
        event_bus: EventBus,
    ) -> None:
        self._provider_manager = provider_manager
        self._retry_policy = retry_policy
        self._breakers = circuit_breakers
        self._event_bus = event_bus

    async def generate(self, provider_name: str, prompt: str, model: str, *, retry: bool) -> str:
        breaker = self._breakers[provider_name]

        if not breaker.allow_request():
            raise CircuitOpenError(
                provider=provider_name,
                state=breaker.state,
                consecutive_failures=breaker.consecutive_failures,
                retry_after_seconds=breaker.retry_after_seconds(),
            )

        provider = self._provider_manager.get_provider(provider_name)

        async def operation() -> str:
            return await provider.generate(prompt, model=model)

        prior_state = breaker.state
        try:
            if retry:
                result = await self._retry_policy.execute(operation)
            else:
                result = await operation()
        except ProviderError:
            breaker.record_failure()
            self._emit_circuit_transition(provider_name, prior_state, breaker.state)
            raise
        else:
            breaker.record_success()
            self._emit_circuit_transition(provider_name, prior_state, breaker.state)
            return result

    def circuit_states(self) -> dict[str, dict]:
        return {
            name: {
                "state": breaker.state.value,
                "consecutive_failures": breaker.consecutive_failures,
                "successes": breaker.successes,
                "failures": breaker.failures,
            }
            for name, breaker in self._breakers.items()
        }

    def emit_failover_triggered(
        self, *, failed_provider: str, replacement_provider: str,
        original_model: str, replacement_model: str, reason: str, attempt_number: int,
    ) -> None:
        self._event_bus.emit(EventType.PROVIDER_FAILOVER_TRIGGERED, {
            "failed_provider": failed_provider,
            "replacement_provider": replacement_provider,
            "original_model": original_model,
            "replacement_model": replacement_model,
            "reason": reason,
            "attempt_number": attempt_number,
        })

    def _emit_circuit_transition(self, provider: str, before: CircuitState, after: CircuitState) -> None:
        if before == after:
            return
        event_type = {
            CircuitState.OPEN: EventType.CIRCUIT_OPENED,
            CircuitState.HALF_OPEN: EventType.CIRCUIT_HALF_OPEN,
            CircuitState.CLOSED: EventType.CIRCUIT_CLOSED,
        }[after]
        self._event_bus.emit(event_type, {
            "provider": provider, "from": before.value, "to": after.value,
        })
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest backend/tests/test_provider_executor.py -v`
Expected: PASS (9 tests)

- [ ] **Step 6: Commit**

```bash
git add backend/providers/executor.py backend/events/types.py backend/tests/test_provider_executor.py
git commit -m "feat: add ProviderExecutor with retry/circuit-breaker orchestration"
```

### Task 45: `RoutingEngine.route()` — `exclude_providers`

**Files:**
- Modify: `backend/routing/engine.py:43-52`
- Test: `backend/tests/test_routing_engine.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces: `RoutingEngine.route(prompt, strategy_name="balanced", exclude_providers: frozenset[str] = frozenset()) -> RoutingDecision`. Consumed by `ChatService` (Task 46).

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_routing_engine.py`:

```python
def test_route_excludes_specified_provider(tmp_path):
    engine = _make_engine(tmp_path, strategies={"cost": CostOptimizedStrategy()})

    with pytest.raises(NoEligibleModelError):
        engine.route("Hello.", strategy_name="cost", exclude_providers=frozenset({"openai"}))


def test_route_exclude_providers_has_no_effect_on_unrelated_provider(tmp_path):
    engine = _make_engine(tmp_path, strategies={"cost": CostOptimizedStrategy()})

    decision = engine.route("Hello.", strategy_name="cost", exclude_providers=frozenset({"mock"}))

    assert decision.selected_model in {"gpt-4o-mini", "gpt-4o"}


def test_route_default_exclude_providers_is_empty(tmp_path):
    engine = _make_engine(tmp_path, strategies={"cost": CostOptimizedStrategy()})

    decision = engine.route("Hello.", strategy_name="cost")

    assert decision.selected_model in {"gpt-4o-mini", "gpt-4o"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest backend/tests/test_routing_engine.py -v -k exclude_providers`
Expected: FAIL — `TypeError: route() got an unexpected keyword argument 'exclude_providers'`

- [ ] **Step 3: Modify `backend/routing/engine.py`**

Change:
```python
    def route(self, prompt: str, strategy_name: str = "balanced") -> RoutingDecision:
        features = self._analyzer.analyze(prompt)
        classification = self._classifier.classify(features)

        available = self._model_registry.get_available_models()
        candidates = self._routing_policy.filter_candidates(classification.tier, available)
```

To:
```python
    def route(
        self,
        prompt: str,
        strategy_name: str = "balanced",
        exclude_providers: frozenset[str] = frozenset(),
    ) -> RoutingDecision:
        features = self._analyzer.analyze(prompt)
        classification = self._classifier.classify(features)

        available = self._model_registry.get_available_models()
        available = [m for m in available if m.provider not in exclude_providers]
        candidates = self._routing_policy.filter_candidates(classification.tier, available)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest backend/tests/test_routing_engine.py -v`
Expected: PASS (all tests, 3 new)

- [ ] **Step 5: Commit**

```bash
git add backend/routing/engine.py backend/tests/test_routing_engine.py
git commit -m "feat: add exclude_providers filter to RoutingEngine.route()"
```

### Task 46: `ResponseRow.error_type` & `ChatService` Failover Orchestration

**Files:**
- Modify: `backend/database/models.py:57-67` (`ResponseRow`)
- Modify: `backend/chat/service.py`
- Modify: `backend/tests/test_chat_service.py` (its existing `_make_chat_service` helper, plus append new tests)

**Interfaces:**
- Consumes: `ProviderExecutor` (Task 44), `CircuitOpenError` (Task 44), `exclude_providers` (Task 45).
- Produces: `ChatService.__init__` now takes `provider_executor: ProviderExecutor` instead of relying on raw `provider_manager` for generation (it still keeps `provider_manager`/`model_registry` for lookups). `ResponseRow.error_type: str | None`. Consumed by `main.py` (Task 48).

Note: `grep -rn "ChatService(" backend/tests backend/api` shows exactly two real construction sites in the whole repo — `test_chat_service.py`'s `_make_chat_service` helper (fixed in Step 2 below) and `backend/api/main.py` (Task 48). `test_chat_endpoint.py` only ever constructs a `_FakeChatService` test double via `app.dependency_overrides`, and `test_integration_chat_flow.py` goes through `create_app()`'s real lifespan — neither needs any change from this plan; both are automatically covered once Task 48 wires `ProviderExecutor` into `main.py`. `test_chat_database.py` tests the ORM models directly and never constructs `ChatService` at all.

- [ ] **Step 1: Add `error_type` to `ResponseRow`**

Modify `backend/database/models.py`:

```python
class ResponseRow(Base):
    __tablename__ = "responses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String, ForeignKey("requests.request_id"), nullable=False)
    response_text: Mapped[str | None] = mapped_column(String, nullable=True)
    actual_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    actual_output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    actual_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    error_type: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
```

- [ ] **Step 2: Update the existing `_make_chat_service` helper, then write the failing tests**

`_make_chat_service` (the helper every existing test in this file already
uses) constructs `ChatService(...)` without `provider_executor` — once
Task 46 Step 4 lands, every existing test in this file will fail with
`TypeError: missing 1 required positional argument: 'provider_executor'`
unless this helper is fixed first. Add these imports to the top of
`backend/tests/test_chat_service.py` (`ProviderError`, `AsyncMock`,
`mocker` are already imported):

```python
from backend.providers.circuit_breaker import CircuitBreaker
from backend.providers.executor import ProviderExecutor
from backend.providers.retry import ExponentialBackoffRetryPolicy


async def _no_op_sleep(delay: float) -> None:
    pass
```

Change the existing `_make_chat_service` function:

```python
def _make_chat_service(tmp_path, verification_service=None):
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(ONE_MODEL_YAML)

    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine_db = create_engine_from_settings(settings)
    init_db(engine_db)
    session_factory = create_session_factory(engine_db)

    factory = ProviderFactory()
    factory.register("mock", MockProvider)
    provider_manager = ProviderManager(factory, settings)
```

To (add the `provider_executor` construction and pass it through the
final `ChatService(...)` call):

```python
def _make_chat_service(tmp_path, verification_service=None):
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(ONE_MODEL_YAML)

    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine_db = create_engine_from_settings(settings)
    init_db(engine_db)
    session_factory = create_session_factory(engine_db)

    factory = ProviderFactory()
    factory.register("mock", MockProvider)
    provider_manager = ProviderManager(factory, settings)
    provider_executor = ProviderExecutor(
        provider_manager=provider_manager,
        retry_policy=ExponentialBackoffRetryPolicy(max_attempts=3, sleep=_no_op_sleep),
        circuit_breakers={"mock": CircuitBreaker(failure_threshold=5)},
        event_bus=EventBus(),
    )
```

and change the trailing `ChatService(...)` construction:

```python
    chat_service = ChatService(
        routing_engine=routing_engine,
        provider_manager=provider_manager,
        provider_executor=provider_executor,
        model_registry=model_registry,
        session_factory=session_factory,
        verification_service=verification_service,
    )
    return chat_service, session_factory
```

Now append the new failover-specific fixture and tests to the same file:

```python
TWO_PROVIDER_YAML = textwrap.dedent("""
    models:
      - id: primary-model
        provider: primary
        model: primary-model
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
      - id: backup-model
        provider: backup
        model: backup-model
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


class _TwoProviderManager:
    """Duck-typed ProviderManager double exposing two independently
    named providers, each backed by its own MockProvider instance --
    ProviderManager itself only supports 'mock' and 'openai', so a test
    double is required to exercise real cross-provider failover."""

    def __init__(self, primary: MockProvider, backup: MockProvider) -> None:
        self._providers = {"primary": primary, "backup": backup}

    def get_provider(self, name: str):
        return self._providers[name]

    def is_provider_available(self, name: str) -> bool:
        return name in self._providers

    def list_providers(self):
        return {name: "available" for name in self._providers}


async def _no_op_sleep(delay: float) -> None:
    pass


def _make_failover_chat_service(tmp_path):
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(TWO_PROVIDER_YAML)

    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine_db = create_engine_from_settings(settings)
    init_db(engine_db)
    session_factory = create_session_factory(engine_db)

    primary = MockProvider(response="primary-response")
    backup = MockProvider(response="backup-response")
    provider_manager = _TwoProviderManager(primary, backup)

    model_registry = ModelRegistry(
        provider_manager=provider_manager,
        event_bus=EventBus(),
        cost_estimator=DefaultCostEstimator(),
        session_factory=session_factory,
        yaml_path=str(yaml_path),
    )
    model_registry.reload()

    routing_engine = RoutingEngine(
        model_registry=model_registry,
        analyzer=PromptAnalyzer(),
        classifier=HeuristicComplexityClassifier(ClassifierPolicy(simple_max=1, medium_max=3)),
        routing_policy=RoutingPolicy({
            "simple": EligibilityPolicy(min_benchmark_score=0.0),
            "medium": EligibilityPolicy(min_benchmark_score=0.75),
            "complex": EligibilityPolicy(min_benchmark_score=0.90),
        }),
        strategies={"balanced": BalancedStrategy(BalancedStrategyWeights())},
        explanation_generator=ExplanationGenerator(),
    )

    provider_executor = ProviderExecutor(
        provider_manager=provider_manager,
        retry_policy=ExponentialBackoffRetryPolicy(max_attempts=3, sleep=_no_op_sleep),
        circuit_breakers={
            "primary": CircuitBreaker(failure_threshold=5),
            "backup": CircuitBreaker(failure_threshold=5),
        },
        event_bus=EventBus(),
    )

    judge = LLMJudge(provider=MockProvider(response="{}"), model="mock", pass_threshold=0.7)
    verification_service = VerificationService(
        judge_engine=JudgeEngine(judge=judge, judge_model_id="mock"),
        session_factory=session_factory, event_bus=EventBus(), judge_prompt_version="v1",
    )

    chat_service = ChatService(
        routing_engine=routing_engine,
        provider_manager=provider_manager,
        provider_executor=provider_executor,
        model_registry=model_registry,
        session_factory=session_factory,
        verification_service=verification_service,
    )
    return chat_service, session_factory, primary, backup


async def test_chat_fails_over_to_backup_provider_after_primary_exhausts_retries(tmp_path, mocker):
    chat_service, session_factory, primary, backup = _make_failover_chat_service(tmp_path)

    mocker.patch.object(
        primary, "generate", new_callable=AsyncMock, side_effect=ProviderError("primary down"),
    )

    result = await chat_service.chat("Hello.", strategy="balanced", background_tasks=BackgroundTasks())

    assert result.response == "backup-response"
    assert result.routing.selected_model == "backup-model"

    with session_factory() as session:
        routing_events = (
            session.query(RoutingEventRow)
            .filter_by(request_id=result.request_id)
            .order_by(RoutingEventRow.id)
            .all()
        )
        response_row = session.query(ResponseRow).filter_by(request_id=result.request_id).one()

    assert len(routing_events) == 2
    assert routing_events[0].selected_model == "primary-model"
    assert routing_events[1].selected_model == "backup-model"
    assert response_row.response_text == "backup-response"
    assert response_row.error is None


async def test_chat_persists_error_type_when_both_primary_and_failover_fail(tmp_path, mocker):
    chat_service, session_factory, primary, backup = _make_failover_chat_service(tmp_path)

    mocker.patch.object(
        primary, "generate", new_callable=AsyncMock, side_effect=ProviderError("primary down"),
    )
    mocker.patch.object(
        backup, "generate", new_callable=AsyncMock, side_effect=ProviderError("backup down"),
    )

    with pytest.raises(ProviderError):
        await chat_service.chat("Hello.", strategy="balanced", background_tasks=BackgroundTasks())

    with session_factory() as session:
        response_row = session.query(ResponseRow).one()

    assert response_row.response_text is None
    assert response_row.error == "backup down"
    assert response_row.error_type == "provider_error"


async def test_chat_failover_attempt_makes_exactly_one_call_no_retry(tmp_path, mocker):
    chat_service, session_factory, primary, backup = _make_failover_chat_service(tmp_path)

    mocker.patch.object(
        primary, "generate", new_callable=AsyncMock, side_effect=ProviderError("primary down"),
    )
    backup_spy = mocker.patch.object(
        backup, "generate", new_callable=AsyncMock, side_effect=ProviderError("backup down"),
    )

    with pytest.raises(ProviderError):
        await chat_service.chat("Hello.", strategy="balanced", background_tasks=BackgroundTasks())

    assert backup_spy.await_count == 1  # no retry on the failover attempt
```

`test_chat_returns_result_and_persists_rows` (already in the file)
already asserts `response_row.error is None` on the success path; no
separate test is needed to confirm `error_type` stays unset on success
since `ResponseRow(...)` is only ever constructed with `error_type` set
on the two explicit error-persistence call sites in `chat/service.py`
(Task 46 Step 4) — a success-path `ResponseRow` never passes it, so it
defaults to `None` via the column's `nullable=True` with no explicit
default needed.

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest backend/tests/test_chat_service.py -v`
Expected: FAIL — `TypeError: ChatService.__init__() got an unexpected keyword argument 'provider_executor'` (and `AttributeError`/`sqlalchemy` errors for `error_type` until Step 1 lands — Step 1 already applied above, so failures here are purely about `ChatService`'s missing constructor param and orchestration logic)

- [ ] **Step 4: Modify `backend/chat/service.py`**

```python
# backend/chat/service.py
import json
import uuid

from fastapi import BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker

from backend.database.models import RequestRow, ResponseRow, RoutingEventRow
from backend.providers.base import ProviderError
from backend.providers.executor import CircuitOpenError, ProviderExecutor
from backend.providers.manager import ProviderManager
from backend.routing.engine import NoEligibleModelError, RoutingDecision, RoutingEngine
from backend.services.model_registry import ModelRegistry
from backend.telemetry.logging import get_logger
from backend.verification.service import VerificationService


class ChatResult(BaseModel):
    request_id: str
    response: str
    routing: RoutingDecision


class ChatService:
    def __init__(
        self,
        routing_engine: RoutingEngine,
        provider_manager: ProviderManager,
        provider_executor: ProviderExecutor,
        model_registry: ModelRegistry,
        session_factory: sessionmaker,
        verification_service: VerificationService,
    ) -> None:
        self._routing_engine = routing_engine
        self._provider_manager = provider_manager
        self._provider_executor = provider_executor
        self._model_registry = model_registry
        self._session_factory = session_factory
        self._verification_service = verification_service
        self._logger = get_logger("chat")

    async def chat(
        self, prompt: str, strategy: str, background_tasks: BackgroundTasks
    ) -> ChatResult:
        request_id = str(uuid.uuid4())
        decision = self._routing_engine.route(prompt, strategy_name=strategy)

        with self._session_factory() as session:
            session.add(RequestRow(request_id=request_id, prompt=prompt, strategy=strategy))
            session.commit()
        self._persist_routing_event(request_id, decision)

        spec = self._model_registry.get_model(decision.selected_model)

        try:
            response_text = await self._provider_executor.generate(
                spec.provider, prompt, spec.model, retry=True
            )
        except (ProviderError, CircuitOpenError) as exc:
            reason = "circuit_open" if isinstance(exc, CircuitOpenError) else "provider_error"
            try:
                failover_decision = self._routing_engine.route(
                    prompt, strategy_name=strategy,
                    exclude_providers=frozenset({spec.provider}),
                )
            except NoEligibleModelError as no_eligible:
                self._persist_error_response(
                    request_id, error_type="no_eligible_model", error=str(no_eligible)
                )
                raise

            self._persist_routing_event(request_id, failover_decision)
            new_spec = self._model_registry.get_model(failover_decision.selected_model)
            self._provider_executor.emit_failover_triggered(
                failed_provider=spec.provider, replacement_provider=new_spec.provider,
                original_model=spec.id, replacement_model=new_spec.id,
                reason=reason, attempt_number=2,
            )

            try:
                response_text = await self._provider_executor.generate(
                    new_spec.provider, prompt, new_spec.model, retry=False
                )
            except (ProviderError, CircuitOpenError) as exc2:
                error_type = "circuit_open" if isinstance(exc2, CircuitOpenError) else "provider_error"
                self._persist_error_response(request_id, error_type=error_type, error=str(exc2))
                raise

            decision, spec = failover_decision, new_spec

        provider = self._provider_manager.get_provider(spec.provider)
        input_tokens = provider.count_tokens(prompt)
        output_tokens = provider.count_tokens(response_text)
        actual_cost = self._model_registry.estimate_cost(spec.id, input_tokens, output_tokens)

        with self._session_factory() as session:
            session.add(ResponseRow(
                request_id=request_id,
                response_text=response_text,
                actual_input_tokens=input_tokens,
                actual_output_tokens=output_tokens,
                actual_cost=actual_cost,
            ))
            session.commit()

        try:
            background_tasks.add_task(
                self._verification_service.verify, request_id, prompt, response_text
            )
        except Exception:
            self._logger.exception(
                "verification_scheduling_failed", extra={"request_id": request_id}
            )

        return ChatResult(request_id=request_id, response=response_text, routing=decision)

    def _persist_routing_event(self, request_id: str, decision: RoutingDecision) -> None:
        with self._session_factory() as session:
            session.add(RoutingEventRow(
                request_id=request_id,
                complexity=decision.complexity.value,
                confidence=decision.confidence,
                selected_model=decision.selected_model,
                selected_strategy=decision.strategy,
                estimated_cost=decision.estimated_cost,
                estimated_latency_ms=decision.estimated_latency_ms,
                reasoning=json.dumps(decision.reasoning),
            ))
            session.commit()

    def _persist_error_response(self, request_id: str, *, error_type: str, error: str) -> None:
        with self._session_factory() as session:
            session.add(ResponseRow(request_id=request_id, error_type=error_type, error=error))
            session.commit()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest backend/tests/test_chat_service.py -v`
Expected: PASS (all tests — the 4 pre-existing tests now passing again via the fixed `_make_chat_service` helper, plus 3 new failover tests)

- [ ] **Step 6: Run the full existing suite to verify no other fixture was missed**

Run: `.venv/bin/python -m pytest backend/tests -v 2>&1 | grep -E "FAILED|ERROR"`
Expected: no output. (Per the Task 46 header note, `test_chat_endpoint.py` only builds a `_FakeChatService` double and `test_integration_chat_flow.py` goes through `create_app()`'s real lifespan — both are unaffected until Task 48 wires `ProviderExecutor` into `main.py`, at which point they pass through the real construction path automatically. `test_chat_database.py` never constructs `ChatService`. If this grep surfaces any other `TypeError: ... missing ... 'provider_executor'`, a `ChatService(` construction site was missed by the repo-wide grep above — find it with the same grep and apply the same fix as Step 2.)

- [ ] **Step 7: Commit**

```bash
git add backend/database/models.py backend/chat/service.py backend/tests/test_chat_service.py
git commit -m "feat: add ChatService failover orchestration and ResponseRow.error_type"
```

### Task 47: `/v1/health` Circuit State Extension

**Files:**
- Modify: `backend/api/dependencies.py`
- Modify: `backend/api/routers/health.py`
- Test: `backend/tests/test_health_endpoint.py` (append)

**Interfaces:**
- Consumes: `ProviderExecutor.circuit_states()` (Task 44).
- Produces: `ProviderExecutorDep`, `GET /v1/health` response gains a `"circuits"` key. Consumed by `main.py` (Task 48, wiring only).

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_health_endpoint.py`:

```python
from backend.api.dependencies import get_provider_executor


class _FakeProviderExecutor:
    def circuit_states(self):
        return {
            "openai": {"state": "closed", "consecutive_failures": 0, "successes": 3, "failures": 0},
            "anthropic": {"state": "closed", "consecutive_failures": 0, "successes": 0, "failures": 0},
            "ollama": {"state": "closed", "consecutive_failures": 0, "successes": 0, "failures": 0},
        }


def test_health_endpoint_includes_circuit_states(tmp_path):
    app = FastAPI()
    app.include_router(health_router, prefix="/v1")

    settings = Settings(_env_file=None, environment="test", database_url=f"sqlite:///{tmp_path}/t2.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_app_version] = lambda: "0.5.0"
    app.dependency_overrides[get_app_start_time] = lambda: time.time() - 5
    app.dependency_overrides[get_provider_manager] = lambda: _FakeProviderManager()
    app.dependency_overrides[get_model_registry] = lambda: _FakeModelRegistry()
    app.dependency_overrides[get_session_factory] = lambda: session_factory
    app.dependency_overrides[get_provider_executor] = lambda: _FakeProviderExecutor()

    client = TestClient(app)
    response = client.get("/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["circuits"] == {
        "openai": {"state": "closed", "consecutive_failures": 0, "successes": 3, "failures": 0},
        "anthropic": {"state": "closed", "consecutive_failures": 0, "successes": 0, "failures": 0},
        "ollama": {"state": "closed", "consecutive_failures": 0, "successes": 0, "failures": 0},
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest backend/tests/test_health_endpoint.py -v -k circuit_states`
Expected: FAIL — `ImportError: cannot import name 'get_provider_executor'`

- [ ] **Step 3: Modify `backend/api/dependencies.py`**

Add import:
```python
from backend.providers.executor import ProviderExecutor
```

Add, immediately after `get_learning_service`:
```python
def get_provider_executor(request: Request) -> ProviderExecutor:
    return request.app.state.provider_executor
```

Add, immediately after `LearningServiceDep`:
```python
ProviderExecutorDep = Annotated[ProviderExecutor, Depends(get_provider_executor)]
```

- [ ] **Step 4: Modify `backend/api/routers/health.py`**

Change:
```python
from backend.api.dependencies import (
    AppStartTimeDep,
    AppVersionDep,
    ModelRegistryDep,
    ProviderManagerDep,
    SessionFactoryDep,
    SettingsDep,
)
```

To:
```python
from backend.api.dependencies import (
    AppStartTimeDep,
    AppVersionDep,
    ModelRegistryDep,
    ProviderExecutorDep,
    ProviderManagerDep,
    SessionFactoryDep,
    SettingsDep,
)
```

Change:
```python
@router.get("/health")
def get_health(
    settings: SettingsDep,
    version: AppVersionDep,
    start_time: AppStartTimeDep,
    provider_manager: ProviderManagerDep,
    model_registry: ModelRegistryDep,
    session_factory: SessionFactoryDep,
):
```

To:
```python
@router.get("/health")
def get_health(
    settings: SettingsDep,
    version: AppVersionDep,
    start_time: AppStartTimeDep,
    provider_manager: ProviderManagerDep,
    provider_executor: ProviderExecutorDep,
    model_registry: ModelRegistryDep,
    session_factory: SessionFactoryDep,
):
```

Change the return statement:
```python
    return {
        "status": "healthy",
        "version": version,
        "environment": settings.environment,
        "database": database_status,
        "providers": provider_manager.list_providers(),
        "circuits": provider_executor.circuit_states(),
        "loaded_models": len(model_registry.get_models()),
        "uptime_seconds": round(time.time() - start_time, 1),
    }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest backend/tests/test_health_endpoint.py -v`
Expected: PASS (all tests, 1 new)

- [ ] **Step 6: Commit**

```bash
git add backend/api/dependencies.py backend/api/routers/health.py backend/tests/test_health_endpoint.py
git commit -m "feat: expose circuit breaker state on GET /v1/health"
```

### Task 48: Wiring in `main.py`, Tag `v0.5.0`

**Files:**
- Modify: `backend/api/main.py`

**Interfaces:**
- Consumes: `ProviderExecutor`, `CircuitBreaker`, `ExponentialBackoffRetryPolicy` (Tasks 42-44), `ChatService(provider_executor=...)` (Task 46), `ProviderExecutorDep` (Task 47, wiring only — no router change needed since `/v1/health` already reads `app.state.provider_executor` via the dependency added in Task 47).

- [ ] **Step 1: Add imports**

Add to the import block in `backend/api/main.py`, alongside the other `backend.providers.*` imports:
```python
from backend.providers.circuit_breaker import CircuitBreaker
from backend.providers.executor import ProviderExecutor
from backend.providers.manager import KNOWN_PROVIDER_NAMES
from backend.providers.retry import ExponentialBackoffRetryPolicy
```

- [ ] **Step 2: Change `APP_VERSION`**

Change `APP_VERSION = "0.4.0"` to `APP_VERSION = "0.5.0"`.

- [ ] **Step 3: Construct `ProviderExecutor` in `lifespan`**

In `lifespan`, immediately after the `chat_service = ChatService(...)` block is removed and replaced (since `ChatService` now needs `provider_executor` at construction time), restructure to:

```python
    provider_executor = ProviderExecutor(
        provider_manager=provider_manager,
        retry_policy=ExponentialBackoffRetryPolicy(),
        circuit_breakers={name: CircuitBreaker() for name in KNOWN_PROVIDER_NAMES},
        event_bus=event_bus,
    )

    chat_service = ChatService(
        routing_engine=routing_engine,
        provider_manager=provider_manager,
        provider_executor=provider_executor,
        model_registry=model_registry,
        session_factory=session_factory,
        verification_service=verification_service,
    )
```

(`ProviderExecutor` uses its class defaults — `failure_threshold=5`, `open_timeout=30.0` for each `CircuitBreaker`; `max_attempts=3`, `base_delay=0.2`, `multiplier=2.0` for `ExponentialBackoffRetryPolicy` — matching §Global Constraints.)

- [ ] **Step 4: Add to `app.state`**

In the `app.state.*` assignment block, add:
```python
    app.state.provider_executor = provider_executor
```

- [ ] **Step 5: Run the full regression suite**

Run: `.venv/bin/python -m pytest -v`
Expected: all tests pass (216 existing + new tests from Tasks 42-47; verify against actual collected count rather than assuming an exact number).

- [ ] **Step 6: Manual end-to-end verification**

```bash
.venv/bin/uvicorn backend.api.main:app --reload
```
```bash
curl -s http://localhost:8000/v1/health | python3 -m json.tool
```
Expected: `"version": "0.5.0"`, and a `"circuits"` object with `openai`, `anthropic`, `ollama` all `{"state": "closed", "consecutive_failures": 0, "successes": 0, "failures": 0}` on a fresh boot.

```bash
curl -s -X POST http://localhost:8000/v1/chat -H 'Content-Type: application/json' \
  -d '{"prompt": "List three fruits.", "strategy": "balanced"}' | python3 -m json.tool
```
Expected: a normal successful response (no provider is actually down in this manual check — this confirms the new `ProviderExecutor` path doesn't break the happy path). Then re-check `curl -s http://localhost:8000/v1/health` and confirm `circuits.<provider used>.successes` incremented by 1.

- [ ] **Step 7: Commit and tag**

```bash
git add backend/api/main.py
git commit -m "feat: wire ProviderExecutor into app lifespan, bump to v0.5.0"
git tag v0.5.0
```
