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
