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

        async def operation() -> str:
            # Resolved inside the retry-guarded closure (not once up front)
            # so a provider that's disabled/deleted/reloaded mid-request --
            # including between individual retry attempts -- raises
            # ProviderUnavailableError (a ProviderError) here, where it's
            # already handled by the except clause below, instead of a bare
            # KeyError escaping as an unhandled 500.
            provider = self._provider_manager.get_provider(provider_name)
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
