"""Smoke test for benchmarks/run_benchmarks.py -- keeps the benchmark
harness itself from silently breaking (e.g. an import that moved, a
constructor signature that changed) between full manual runs."""

import tempfile
from pathlib import Path

import pytest

from benchmarks.run_benchmarks import (
    _build_routing_stack, benchmark_classifier_latency, benchmark_routing_latency,
    run_failover_demo, run_load_test,
)


def test_build_routing_stack_and_run_all_sections_at_low_volume():
    with tempfile.TemporaryDirectory() as tmp:
        engine, model_registry = _build_routing_stack(Path(tmp))

        routing = benchmark_routing_latency(engine, iterations=5)
        assert routing["iterations"] == 5
        assert routing["avg_ms"] >= 0

        classifier = benchmark_classifier_latency(iterations=5)
        assert classifier["iterations"] == 5

        load_test = run_load_test(engine, model_registry, request_count=5)
        assert load_test["request_count"] == 5
        assert sum(load_test["distribution"].values()) == 5


@pytest.mark.asyncio
async def test_failover_demo_opens_rejects_and_recovers():
    result = await run_failover_demo(open_timeout_seconds=0.05)

    assert result["rejected_while_open"] is True
    assert result["recovered"] is True
    assert len(result["transitions"]) == 4
