from types import SimpleNamespace

import pytest

from backend.database.models import VerificationRow
from backend.learning.cost_metrics import build_model_cost_metrics
from backend.learning.rules import DetectionRuleConfig
from backend.verification.status import VerificationStatus


class _FakeModelRegistry:
    def __init__(self, pricing: dict[str, tuple[float, float]]) -> None:
        self._pricing = pricing

    def get_model(self, model_id: str):
        input_cost, output_cost = self._pricing[model_id]
        return SimpleNamespace(input_cost=input_cost, output_cost=output_cost)


def _row(model, complexity, passed, request_id, created_at, status=VerificationStatus.COMPLETED.value):
    return VerificationRow(
        request_id=request_id, status=status, routing_model=model,
        routing_strategy="balanced", routing_complexity=complexity, passed=passed,
        created_at=created_at,
    )


def _dt(day):
    from datetime import datetime, timezone
    return datetime(2026, 7, day, tzinfo=timezone.utc)


def test_builds_metrics_for_pair_with_cost_data():
    rows = [
        _row("gpt-4o", "complex", True, f"req-{i}", _dt(1 + i % 5)) for i in range(20)
    ]
    costs = {f"req-{i}": 0.05 for i in range(20)}
    registry = _FakeModelRegistry({"gpt-4o": (2.50, 10.00)})
    config = DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.7)

    metrics = build_model_cost_metrics(rows, costs, registry, config)

    key = ("gpt-4o", "complex")
    assert key in metrics
    assert metrics[key].model == "gpt-4o"
    assert metrics[key].complexity == "complex"
    assert metrics[key].input_cost == 2.50
    assert metrics[key].output_cost == 10.00
    assert metrics[key].avg_cost_per_request == pytest.approx(0.05)
    assert metrics[key].pass_rate == pytest.approx(1.0)
    assert metrics[key].eligible_for_optimization is True


def test_skips_pair_with_no_cost_data():
    rows = [_row("gpt-4o", "complex", True, f"req-{i}", _dt(1)) for i in range(20)]
    registry = _FakeModelRegistry({"gpt-4o": (2.50, 10.00)})
    config = DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.7)

    metrics = build_model_cost_metrics(rows, {}, registry, config)

    assert metrics == {}


def test_eligible_for_optimization_false_below_min_samples():
    rows = [_row("gpt-4o", "complex", True, f"req-{i}", _dt(1)) for i in range(5)]
    costs = {f"req-{i}": 0.05 for i in range(5)}
    registry = _FakeModelRegistry({"gpt-4o": (2.50, 10.00)})
    config = DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.7)

    metrics = build_model_cost_metrics(rows, costs, registry, config)

    key = ("gpt-4o", "complex")
    assert metrics[key].eligible_for_optimization is False


def test_eligible_for_optimization_false_below_pass_rate():
    rows = (
        [_row("gpt-4o", "complex", True, f"req-{i}", _dt(1)) for i in range(10)]
        + [_row("gpt-4o", "complex", False, f"req-fail-{i}", _dt(1)) for i in range(10)]
    )  # pass_rate = 0.5 < 0.7
    costs = {**{f"req-{i}": 0.05 for i in range(10)}, **{f"req-fail-{i}": 0.05 for i in range(10)}}
    registry = _FakeModelRegistry({"gpt-4o": (2.50, 10.00)})
    config = DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.7)

    metrics = build_model_cost_metrics(rows, costs, registry, config)

    key = ("gpt-4o", "complex")
    assert metrics[key].eligible_for_optimization is False


def test_requests_per_day_uses_observed_date_span():
    rows = [_row("gpt-4o", "complex", True, f"req-{i}", _dt(1 + (i % 5))) for i in range(20)]
    costs = {f"req-{i}": 0.05 for i in range(20)}
    registry = _FakeModelRegistry({"gpt-4o": (2.50, 10.00)})
    config = DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.7)

    metrics = build_model_cost_metrics(rows, costs, registry, config)

    # created_at spans day 1 to day 5 -> window_days = 4, sample_size = 20
    assert metrics[("gpt-4o", "complex")].requests_per_day == pytest.approx(20 / 4)


def test_skips_pair_when_model_unknown_to_registry():
    rows = [_row("retired-model", "complex", True, f"req-{i}", _dt(1)) for i in range(20)]
    costs = {f"req-{i}": 0.05 for i in range(20)}
    registry = _FakeModelRegistry({})  # empty -- get_model raises KeyError
    config = DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.7)

    metrics = build_model_cost_metrics(rows, costs, registry, config)

    assert metrics == {}


def test_excludes_non_completed_rows():
    rows = (
        [_row("gpt-4o", "complex", True, f"req-{i}", _dt(1)) for i in range(20)]
        + [
            _row("gpt-4o", "complex", None, f"req-fail-{i}", _dt(1), status=VerificationStatus.FAILED.value)
            for i in range(50)
        ]
    )
    costs = {f"req-{i}": 0.05 for i in range(20)}
    registry = _FakeModelRegistry({"gpt-4o": (2.50, 10.00)})
    config = DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.7)

    metrics = build_model_cost_metrics(rows, costs, registry, config)

    assert metrics[("gpt-4o", "complex")].pass_rate == pytest.approx(1.0)
