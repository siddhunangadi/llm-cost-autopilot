from collections import defaultdict
from typing import Protocol

from pydantic import BaseModel

from backend.database.models import VerificationRow
from backend.learning.rules import DetectionRuleConfig, eligible_verification_rows


class _ModelPricing(Protocol):
    input_cost: float
    output_cost: float


class _ModelRegistryLike(Protocol):
    def get_model(self, model_id: str) -> _ModelPricing: ...


class ModelCostMetrics(BaseModel):
    model: str
    complexity: str
    input_cost: float
    output_cost: float
    avg_cost_per_request: float
    requests_per_day: float
    pass_rate: float
    eligible_for_optimization: bool


def build_model_cost_metrics(
    verification_rows: list[VerificationRow],
    cost_by_request_id: dict[str, float],
    model_registry: _ModelRegistryLike,
    config: DetectionRuleConfig,
) -> dict[tuple[str, str], ModelCostMetrics]:
    groups: dict[tuple[str, str], list[VerificationRow]] = defaultdict(list)
    for row in eligible_verification_rows(verification_rows):
        groups[(row.routing_model, row.routing_complexity)].append(row)

    metrics: dict[tuple[str, str], ModelCostMetrics] = {}
    for (model, complexity), rows in groups.items():
        sample_size = len(rows)
        pass_rate = sum(1 for r in rows if r.passed) / sample_size

        costed_rows = [r for r in rows if r.request_id in cost_by_request_id]
        if not costed_rows:
            continue

        try:
            spec = model_registry.get_model(model)
        except KeyError:
            continue

        avg_cost_per_request = (
            sum(cost_by_request_id[r.request_id] for r in costed_rows) / len(costed_rows)
        )
        window_days = max(
            (max(r.created_at for r in costed_rows) - min(r.created_at for r in costed_rows)).days,
            1,
        )
        requests_per_day = sample_size / window_days

        metrics[(model, complexity)] = ModelCostMetrics(
            model=model,
            complexity=complexity,
            input_cost=spec.input_cost,
            output_cost=spec.output_cost,
            avg_cost_per_request=avg_cost_per_request,
            requests_per_day=requests_per_day,
            pass_rate=pass_rate,
            eligible_for_optimization=(
                sample_size >= config.min_samples and pass_rate >= config.pass_rate_threshold
            ),
        )
    return metrics
