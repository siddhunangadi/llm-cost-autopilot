from abc import ABC, abstractmethod


def calculate_linear_cost(
    input_tokens: int, output_tokens: int, input_cost: float, output_cost: float
) -> float:
    """Cost in dollars given per-million-token pricing."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must not be negative")
    if input_cost < 0 or output_cost < 0:
        raise ValueError("pricing must not be negative")
    return (input_tokens / 1_000_000) * input_cost + (output_tokens / 1_000_000) * output_cost


class BaseCostEstimator(ABC):
    @abstractmethod
    def estimate(
        self, input_tokens: int, output_tokens: int, input_cost: float, output_cost: float
    ) -> float: ...


class DefaultCostEstimator(BaseCostEstimator):
    def estimate(
        self, input_tokens: int, output_tokens: int, input_cost: float, output_cost: float
    ) -> float:
        return calculate_linear_cost(input_tokens, output_tokens, input_cost, output_cost)
