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
