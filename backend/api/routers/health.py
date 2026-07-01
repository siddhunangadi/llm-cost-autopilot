import time

from fastapi import APIRouter
from sqlalchemy import text

from backend.api.dependencies import (
    AppStartTimeDep,
    AppVersionDep,
    ModelRegistryDep,
    ProviderManagerDep,
    SessionFactoryDep,
    SettingsDep,
)

router = APIRouter()


@router.get("/health")
def get_health(
    settings: SettingsDep,
    version: AppVersionDep,
    start_time: AppStartTimeDep,
    provider_manager: ProviderManagerDep,
    model_registry: ModelRegistryDep,
    session_factory: SessionFactoryDep,
):
    try:
        with session_factory() as session:
            session.execute(text("SELECT 1"))
        database_status = "healthy"
    except Exception:
        database_status = "unhealthy"

    return {
        "status": "healthy",
        "version": version,
        "environment": settings.environment,
        "database": database_status,
        "providers": provider_manager.list_providers(),
        "loaded_models": len(model_registry.get_models()),
        "uptime_seconds": round(time.time() - start_time, 1),
    }
