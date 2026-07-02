from typing import Annotated

from fastapi import Depends, Request

from backend.chat.service import ChatService
from backend.config.settings import Settings
from backend.events.bus import EventBus
from backend.providers.manager import ProviderManager
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


SettingsDep = Annotated[Settings, Depends(get_settings)]
EventBusDep = Annotated[EventBus, Depends(get_event_bus)]
ProviderManagerDep = Annotated[ProviderManager, Depends(get_provider_manager)]
ModelRegistryDep = Annotated[ModelRegistry, Depends(get_model_registry)]
SessionFactoryDep = Annotated[object, Depends(get_session_factory)]
AppVersionDep = Annotated[str, Depends(get_app_version)]
AppStartTimeDep = Annotated[float, Depends(get_app_start_time)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
