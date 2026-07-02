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
