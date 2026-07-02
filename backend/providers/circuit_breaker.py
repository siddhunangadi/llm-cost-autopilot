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
