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
