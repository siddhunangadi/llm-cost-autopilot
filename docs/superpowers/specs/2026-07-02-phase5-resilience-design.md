# LLM Cost Autopilot — Phase 5 Design: Provider Fault Tolerance

Status: **Approved — frozen as implementation contract**
Date: 2026-07-02

## 1. Purpose & Scope

Phase 5 answers: **what happens when a provider fails mid-request?**
Phases 1-4 assumed providers respond. Phase 5 makes `ChatService`
resilient to transient failures (retry) and sustained outages (circuit
breaker + failover to another model/provider), without touching routing
intelligence itself — `RoutingEngine` stays unaware that resilience
exists; it only ever answers "given these constraints, what's best?"

```
ChatService.chat(prompt, strategy)
        │
        ▼
RoutingEngine.route(prompt, strategy)  ──────────────► RoutingEventRow #1 (primary)
        │
        ▼
ProviderExecutor.generate(provider, prompt, model, retry=True)
        │
   ┌────┴─────┐
   ▼          ▼
CircuitBreaker  ExponentialBackoffRetryPolicy
(per provider)  (up to 3 attempts, exponential backoff)
        │
        ▼
   success ──────────────────────────────────────────► ResponseRow (success)
        │
   exhausted / circuit open
        │
        ▼
RoutingEngine.route(prompt, strategy, exclude_providers={failed})
        │                                             ──► RoutingEventRow #2 (failover)
        ▼
ProviderExecutor.generate(new_provider, prompt, new_model, retry=False)
        │
   ┌────┴─────┐
   ▼          ▼
 success    failure ─────────────────────────────────► ResponseRow (error)
   │
   ▼
ResponseRow (success) ──► ChatResult(routing=failover_decision)

GET /v1/health  -> extended with per-provider circuit_state, consecutive_failures
```

**In scope:**
- `BaseRetryPolicy` + `ExponentialBackoffRetryPolicy` (pure, generic —
  retries any async callable, no provider- or circuit-specific knowledge)
- `CircuitBreaker` (per-provider outcome tracking + state machine, no
  I/O, no event emission)
- `ProviderExecutor` (the sole orchestrator of retry + circuit breaker +
  resilience events; wraps `ProviderManager`, never replaces it)
- `RoutingEngine.route()` gains `exclude_providers: frozenset[str]`
- `ChatService` failover orchestration (re-route once, single attempt,
  no recursive failover)
- `ResponseRow.error_type` column (standardizes error persistence across
  every failure path)
- Two-`RoutingEventRow`-per-request audit trail on failover
- `/v1/health` circuit state exposure
- New `EventType`s: `CIRCUIT_OPENED`, `CIRCUIT_HALF_OPEN`,
  `CIRCUIT_CLOSED`, `PROVIDER_FAILOVER_TRIGGERED`

**Explicitly out of scope for Phase 5** (deferred to a later phase):
- Streaming resilience (`BaseProvider.stream` is untouched — `ChatService`
  only calls `generate` today, and streaming resilience — mid-stream
  retry/failover — is a materially different problem)
- Cost/rate/budget safeguards (a separate future phase)
- Recursive/multi-hop failover (A → B → C) — exactly one failover
  attempt, ever, to keep worst-case latency bounded
- YAML-configurable retry/circuit thresholds — same reasoning as Phase
  4's `DetectionRuleConfig`: these are internal resilience heuristics,
  hardcoded at construction in `main.py`
- A `PATCH`-style manual circuit reset endpoint
- Per-model circuit granularity (per-provider only — see §4)

## 2. Directory Structure

```
backend/
  providers/
    retry.py            # BaseRetryPolicy, ExponentialBackoffRetryPolicy
    circuit_breaker.py    # CircuitBreaker, CircuitState
    executor.py              # ProviderExecutor, CircuitOpenError
    manager.py                  # unchanged — lookup/lifecycle only
  routing/
    engine.py                     # route() gains exclude_providers (modify)
  chat/
    service.py                      # failover orchestration (modify)
  database/
    models.py                         # + ResponseRow.error_type (modify)
  events/
    types.py                            # + 4 EventTypes (modify)
  api/
    routers/
      health.py                           # + circuit state (modify)
```

## 3. `BaseRetryPolicy` & `ExponentialBackoffRetryPolicy`

```python
class BaseRetryPolicy(ABC):
    @abstractmethod
    async def execute(self, operation: Callable[[], Awaitable[T]]) -> T: ...
```

Generic over any zero-argument async callable — it has no knowledge of
providers, prompts, or models. This mirrors `BaseRoutingStrategy` and
`BaseJudge`: one abstraction, swappable implementations, even though
only one implementation exists today.

```python
class ExponentialBackoffRetryPolicy(BaseRetryPolicy):
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 0.2,
        multiplier: float = 2.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._multiplier = multiplier
        self._sleep = sleep

    async def execute(self, operation: Callable[[], Awaitable[T]]) -> T:
        last_exc: ProviderError | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return await operation()
            except ProviderError as exc:
                last_exc = exc
                if attempt < self._max_attempts:
                    await self._sleep(self._base_delay * (self._multiplier ** (attempt - 1)))
        raise last_exc
```

**Retry policy never emits events, never touches a circuit breaker, and
never knows how many attempts preceded it in a larger flow.** It only
knows about its own `max_attempts`. The injected `sleep` callable is
what makes this testable without real wall-clock waits — tests pass a
no-op async function and assert on call count/args instead.

## 4. `CircuitBreaker`

```python
class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
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
    def state(self) -> CircuitState: ...
    @property
    def consecutive_failures(self) -> int: ...
    @property
    def successes(self) -> int: ...
    @property
    def failures(self) -> int: ...
    @property
    def last_failure_at(self) -> float | None: ...

    def allow_request(self) -> bool:
        if self._state == CircuitState.OPEN:
            if self._clock() - self._opened_at >= self._open_timeout:
                self._state = CircuitState.HALF_OPEN
                return True
            return False
        return True  # CLOSED or HALF_OPEN

    def retry_after_seconds(self) -> float:
        if self._state != CircuitState.OPEN:
            return 0.0
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

**State machine (explicit, the implementation contract):**

```
CLOSED
   │  consecutive_failures reaches failure_threshold (5)
   ▼
OPEN
   │  clock() - opened_at >= open_timeout (30s), next allow_request() call
   ▼
HALF_OPEN
   │  record_success()            │  record_failure()
   ▼                              ▼
CLOSED                           OPEN (opened_at reset to now)
```

**`CircuitBreaker` only sees attempt outcomes — `record_success()` /
`record_failure()` / `allow_request()`.** It has no knowledge of retries,
providers by name (each instance is already scoped to one provider by
the caller), or events. This is what makes it reusable and trivially
unit-testable with an injected `clock`.

**Per-provider invariant:** `ProviderExecutor` holds exactly one
`CircuitBreaker` instance per entry in `KNOWN_PROVIDER_NAMES`
(`openai`, `anthropic`, `ollama`) — never one shared/global breaker,
never one per model. A provider-level outage (auth, network, rate
limit) affects all of that provider's models together; tracking at
model granularity would fragment the same signal for no benefit.

## 5. `ProviderExecutor`

```python
class CircuitOpenError(Exception):
    def __init__(self, provider: str, state: CircuitState, consecutive_failures: int, retry_after_seconds: float) -> None:
        self.provider = provider
        self.state = state
        self.consecutive_failures = consecutive_failures
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            f"Circuit for '{provider}' is {state.value} "
            f"({consecutive_failures} consecutive failures, retry after {retry_after_seconds:.1f}s)"
        )


class ProviderExecutor:
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
                provider=provider_name, state=breaker.state,
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
                result = await operation()  # failover attempt: exactly one try, no retry
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
                "state": b.state.value,
                "consecutive_failures": b.consecutive_failures,
                "successes": b.successes,
                "failures": b.failures,
            }
            for name, b in self._breakers.items()
        }

    def _emit_circuit_transition(self, provider: str, before: CircuitState, after: CircuitState) -> None:
        if before == after:
            return
        event_type = {
            CircuitState.OPEN: EventType.CIRCUIT_OPENED,
            CircuitState.HALF_OPEN: EventType.CIRCUIT_HALF_OPEN,
            CircuitState.CLOSED: EventType.CIRCUIT_CLOSED,
        }[after]
        self._event_bus.emit(event_type, {"provider": provider, "from": before.value, "to": after.value})

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
```

**`ProviderExecutor` is the sole owner of the resilience lifecycle:**
retry counts (implicitly, via `retry_policy` calls it makes), failover
counts (via `emit_failover_triggered`, called once per failover by
`ChatService`), and breaker transitions (`_emit_circuit_transition`).
Neither `BaseRetryPolicy` nor `CircuitBreaker` emits anything — this is
an explicit invariant, not an implementation detail: **`RetryPolicy`
never emits events. `CircuitBreaker` never emits events. `ProviderExecutor`
is the only component that emits resilience events.**

**Breaker records only the final outcome, never intermediate retry
attempts.** When `retry=True`, `retry_policy.execute(operation)` may
internally call `operation()` up to 3 times, but `ProviderExecutor` calls
`breaker.record_failure()` / `record_success()` exactly once per
`generate()` call — after `retry_policy.execute` has fully resolved
(either returned or raised). Two failed attempts followed by a third
success is one `record_success()`, not two failures then a success —
otherwise transient blips within a single request's retry budget could
trip a circuit that a human would consider healthy.

## 6. `RoutingEngine.route()` — `exclude_providers`

```python
def route(
    self, prompt: str, strategy_name: str = "balanced",
    exclude_providers: frozenset[str] = frozenset(),
) -> RoutingDecision:
    ...
    available = self._model_registry.get_available_models()
    available = [m for m in available if m.provider not in exclude_providers]
    candidates = self._routing_policy.filter_candidates(classification.tier, available)
    if not candidates:
        raise NoEligibleModelError(...)
    ...
```

`RoutingEngine` gains one filtering step and nothing else — it remains
completely unaware that resilience, retries, or circuit breakers exist.
`exclude_providers` reads the same as any other candidate filter
(complexity policy, availability); the caller (`ChatService`) is the
only place that knows *why* a provider is being excluded.

## 7. `ChatService` Failover Orchestration

```python
async def chat(self, prompt: str, strategy: str, background_tasks: BackgroundTasks) -> ChatResult:
    request_id = str(uuid.uuid4())
    decision = self._routing_engine.route(prompt, strategy_name=strategy)
    self._persist_request_and_routing_event(request_id, prompt, strategy, decision)

    spec = self._model_registry.get_model(decision.selected_model)

    try:
        response_text = await self._provider_executor.generate(
            spec.provider, prompt, spec.model, retry=True
        )
    except (ProviderError, CircuitOpenError) as exc:
        reason = "circuit_open" if isinstance(exc, CircuitOpenError) else "provider_error"
        try:
            failover_decision = self._routing_engine.route(
                prompt, strategy_name=strategy, exclude_providers=frozenset({spec.provider})
            )
        except NoEligibleModelError as noe:
            self._persist_error_response(request_id, error_type="no_eligible_model", error=str(noe))
            raise

        self._persist_routing_event(request_id, failover_decision)  # RoutingEventRow #2
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

        decision, spec = failover_decision, new_spec  # downstream uses whichever actually served

    input_tokens = ...  # unchanged, uses `spec`/`decision` as before
    ...
    return ChatResult(request_id=request_id, response=response_text, routing=decision)
```

**No recursive failover.** Exactly one re-route, exactly one
retry-free attempt against the replacement. If that also fails, the
error propagates — `ChatService` does not attempt a third provider.
This bounds worst-case latency to (primary: up to 3 attempts with
backoff) + (failover: 1 attempt), never an unbounded chain.

**Failover decision becomes the returned routing decision.** `ChatResult.routing`
reflects whichever `RoutingDecision` actually produced the response the
client received — the audit trail (both `RoutingEventRow`s, ordered by
`created_at`, sharing `request_id`) preserves the fact that a different
model was originally selected and why the change happened.

## 8. `ResponseRow.error_type` — Standardized Error Persistence

```python
class ResponseRow(Base):
    __tablename__ = "responses"
    ...
    error: Mapped[str | None] = mapped_column(String, nullable=True)         # existing
    error_type: Mapped[str | None] = mapped_column(String, nullable=True)    # new
    ...
```

Every failure path — primary failure with no failover candidate
(`NoEligibleModelError`), failover attempt failure (`ProviderError` or
`CircuitOpenError`) — writes both `error_type` (one of
`"provider_error"`, `"circuit_open"`, `"no_eligible_model"`) and `error`
(the exception's string message) through the same
`_persist_error_response(request_id, error_type, error)` helper. No
failure path populates a different subset of fields than another — this
is what makes `error_type` usable for later analytics/dashboards without
special-casing per exception type.

## 9. Two-`RoutingEventRow` Audit Trail

Both `RoutingEventRow`s for a request share the same `request_id` and
represent an ordered routing history — `RoutingEventRow #1` is always
the primary decision (persisted before any provider call is attempted,
exactly as in Phases 1-4), and `RoutingEventRow #2`, when present, is
always the failover re-route (persisted only after a primary failure,
before the replacement provider call is attempted). A request has either
one row (no failover needed) or two (failover occurred) — never more,
per the no-recursive-failover invariant (§7). Consumers reading a
request's routing history should order by `created_at ascending` and
treat a second row as evidence of a mid-request provider failure, not as
an unrelated independent event.

## 10. `EventType` Additions

```python
class EventType(str, Enum):
    ...  # existing Phase 1-4 values unchanged
    CIRCUIT_OPENED = "circuit_opened"
    CIRCUIT_HALF_OPEN = "circuit_half_open"
    CIRCUIT_CLOSED = "circuit_closed"
    PROVIDER_FAILOVER_TRIGGERED = "provider_failover_triggered"
```

## 11. `/v1/health` Extension

`provider_manager.list_providers()` (unchanged) continues to report
`available`/`disabled`. The health endpoint additionally merges in
`provider_executor.circuit_states()` per provider:

```json
{
  "providers": {
    "openai": "available",
    "anthropic": "disabled",
    "ollama": "disabled"
  },
  "circuits": {
    "openai": {"state": "closed", "consecutive_failures": 0, "successes": 12, "failures": 1},
    "anthropic": {"state": "closed", "consecutive_failures": 0, "successes": 0, "failures": 0},
    "ollama": {"state": "closed", "consecutive_failures": 0, "successes": 0, "failures": 0}
  }
}
```

`ProviderExecutor` is constructed with one `CircuitBreaker` per entry in
`KNOWN_PROVIDER_NAMES` regardless of whether that provider is currently
available — a disabled provider's breaker simply never receives any
`record_*` calls and stays `closed` with zero counts, which is accurate
(it has never been attempted, not that it is healthy).

## 12. Testing

Same discipline as Phases 1-4, with the DI-for-determinism pattern
extended to timing:

- `ExponentialBackoffRetryPolicy`: injected `sleep` (no-op async fake)
  — tests assert attempt count against a fake `operation` that fails N
  times then succeeds/never succeeds, and assert the fake `sleep` was
  called with the expected exponential delays, with zero real wall-clock
  time spent.
- `CircuitBreaker`: injected `clock` (fake monotonic counter) — tests
  drive `record_failure()` to the threshold and assert `CLOSED → OPEN`;
  advance the fake clock past `open_timeout` and assert `allow_request()`
  flips to `HALF_OPEN`; assert `HALF_OPEN` + `record_success()` →
  `CLOSED` and `HALF_OPEN` + `record_failure()` → `OPEN` (reusing the
  existing `opened_at`/timeout mechanics, not a fresh threshold count).
- `ProviderExecutor`: unit tests with a stub `ProviderManager` (returns
  a fake `BaseProvider` whose `generate` is scripted to fail/succeed) and
  real `CircuitBreaker`/`ExponentialBackoffRetryPolicy` instances with
  injected fakes — assert circuit-open short-circuits with zero calls to
  the underlying provider; assert `retry=True` retries and records one
  outcome; assert `retry=False` makes exactly one call; assert breaker
  transition events are emitted on the `EventBus` exactly on state
  changes (not on every call).
- `RoutingEngine.route()`: existing tests extended with
  `exclude_providers` cases — excluding the only available provider
  raises `NoEligibleModelError`; excluding an unrelated provider has no
  effect on the selected model.
- `ChatService`: integration tests (real SQLite, stub providers) covering
  the full failover path — primary provider fails 3 times, failover
  succeeds, assert two `RoutingEventRow`s persisted in order, assert
  `ChatResult.routing` reflects the failover decision, assert
  `PROVIDER_FAILOVER_TRIGGERED` emitted with the full payload; a second
  test where both primary and failover fail, asserting a single
  `ResponseRow` with `error_type="provider_error"` and no unbounded
  retry chain (exact call count assertions on the stub providers).
- `/v1/health`: extended test asserting the `circuits` key is present
  with all three `KNOWN_PROVIDER_NAMES` and default `closed` state on a
  fresh app.

## 13. Tooling

No new dependencies — `asyncio.sleep`/`time.monotonic` (standard
library) cover the DI seams; `pydantic`, SQLAlchemy, and FastAPI remain
the only external dependencies, consistent with Phases 1-4.
