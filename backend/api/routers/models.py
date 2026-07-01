from fastapi import APIRouter

from backend.api.dependencies import ModelRegistryDep
from backend.services.model_registry import ModelSpec

router = APIRouter()


@router.get("/models", response_model=list[ModelSpec])
def list_models(model_registry: ModelRegistryDep) -> list[ModelSpec]:
    return model_registry.get_models()
