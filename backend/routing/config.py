from pydantic import BaseModel, Field


class ClassifierPolicy(BaseModel):
    simple_max: int
    medium_max: int


class EligibilityPolicy(BaseModel):
    min_benchmark_score: float


class BalancedStrategyWeights(BaseModel):
    cost_weight: float = Field(default=1 / 3)
    latency_weight: float = Field(default=1 / 3)
    quality_weight: float = Field(default=1 / 3)


class RoutingConfig(BaseModel):
    classifier: ClassifierPolicy
    policy: dict[str, EligibilityPolicy]
    balanced_strategy: BalancedStrategyWeights
