from pydantic import BaseModel

from backend.analysis.prompt_analyzer import PromptFeatures
from backend.classifier.complexity_classifier import ComplexityTier
from backend.services.model_registry import ModelSpec


class RoutingContext(BaseModel):
    prompt: str
    features: PromptFeatures
    complexity: ComplexityTier
    candidates: list[ModelSpec]
