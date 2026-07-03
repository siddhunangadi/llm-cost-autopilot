from typing import Annotated

from fastapi import Depends, Request

from backend.chat.service import ChatService
from backend.config.settings import Settings
from backend.events.bus import EventBus
from backend.learning.service import LearningService
from backend.providers.executor import ProviderExecutor
from backend.providers.factory import ProviderFactory
from backend.providers.manager import ProviderManager
from backend.services.analytics_service import AnalyticsService
from backend.services.credential_store import CredentialStore
from backend.services.dashboard_service import DashboardService
from backend.services.model_registry import ModelRegistry


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_event_bus(request: Request) -> EventBus:
    return request.app.state.event_bus


def get_provider_manager(request: Request) -> ProviderManager:
    return request.app.state.provider_manager


def get_model_registry(request: Request) -> ModelRegistry:
    return request.app.state.model_registry


def get_session_factory(request: Request):
    return request.app.state.session_factory


def get_app_version(request: Request) -> str:
    return request.app.state.version


def get_app_start_time(request: Request) -> float:
    return request.app.state.start_time


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


def get_learning_service(request: Request) -> LearningService:
    return request.app.state.learning_service


def get_provider_executor(request: Request) -> ProviderExecutor:
    return request.app.state.provider_executor


def get_dashboard_service(request: Request) -> DashboardService:
    return request.app.state.dashboard_service


def get_analytics_service(request: Request) -> AnalyticsService:
    return request.app.state.analytics_service


def get_credential_store(request: Request) -> CredentialStore:
    return request.app.state.credential_store


def get_provider_factory(request: Request) -> ProviderFactory:
    return request.app.state.provider_factory


SettingsDep = Annotated[Settings, Depends(get_settings)]
EventBusDep = Annotated[EventBus, Depends(get_event_bus)]
ProviderManagerDep = Annotated[ProviderManager, Depends(get_provider_manager)]
ModelRegistryDep = Annotated[ModelRegistry, Depends(get_model_registry)]
SessionFactoryDep = Annotated[object, Depends(get_session_factory)]
AppVersionDep = Annotated[str, Depends(get_app_version)]
AppStartTimeDep = Annotated[float, Depends(get_app_start_time)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
LearningServiceDep = Annotated[LearningService, Depends(get_learning_service)]
ProviderExecutorDep = Annotated[ProviderExecutor, Depends(get_provider_executor)]
DashboardServiceDep = Annotated[DashboardService, Depends(get_dashboard_service)]
AnalyticsServiceDep = Annotated[AnalyticsService, Depends(get_analytics_service)]
CredentialStoreDep = Annotated[CredentialStore, Depends(get_credential_store)]
ProviderFactoryDep = Annotated[ProviderFactory, Depends(get_provider_factory)]
