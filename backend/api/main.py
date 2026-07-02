import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.analysis.prompt_analyzer import PromptAnalyzer
from backend.api.routers.chat import router as chat_router
from backend.api.routers.health import router as health_router
from backend.api.routers.models import router as models_router
from backend.chat.service import ChatService
from backend.classifier.complexity_classifier import HeuristicComplexityClassifier
from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.events.bus import EventBus
from backend.events.subscribers import register_logging_subscriber
from backend.providers.factory import ProviderFactory
from backend.providers.manager import ProviderManager
from backend.providers.mock_provider import MockProvider
from backend.providers.openai_provider import OpenAIProvider
from backend.routing.config_loader import RoutingConfigLoader
from backend.routing.engine import RoutingEngine
from backend.routing.explanation import ExplanationGenerator
from backend.routing.policy import RoutingPolicy
from backend.routing.strategies import (
    BalancedStrategy,
    CostOptimizedStrategy,
    LatencyOptimizedStrategy,
    QualityOptimizedStrategy,
)
from backend.services.cost_estimator import DefaultCostEstimator
from backend.services.model_registry import ModelRegistry
from backend.telemetry.logging import configure_logging

APP_VERSION = "0.1.0"


def _build_provider_factory() -> ProviderFactory:
    factory = ProviderFactory()
    factory.register("mock", MockProvider)
    factory.register("openai", OpenAIProvider)
    return factory


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    configure_logging(settings)

    event_bus = EventBus()
    register_logging_subscriber(event_bus)

    # No try/except around DB init: a bad DATABASE_URL must crash startup.
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    provider_manager = ProviderManager(_build_provider_factory(), settings)

    model_registry = ModelRegistry(
        provider_manager=provider_manager,
        event_bus=event_bus,
        cost_estimator=DefaultCostEstimator(),
        session_factory=session_factory,
        yaml_path=settings.models_yaml_path,
    )
    model_registry.reload()
    await model_registry.refresh_provider_status()

    routing_config = RoutingConfigLoader.load(settings.routing_config_path)
    classifier = HeuristicComplexityClassifier(routing_config.classifier)
    routing_policy = RoutingPolicy(routing_config.policy)
    strategies = {
        "cost": CostOptimizedStrategy(),
        "latency": LatencyOptimizedStrategy(),
        "quality": QualityOptimizedStrategy(),
        "balanced": BalancedStrategy(routing_config.balanced_strategy),
    }
    routing_engine = RoutingEngine(
        model_registry=model_registry,
        analyzer=PromptAnalyzer(),
        classifier=classifier,
        routing_policy=routing_policy,
        strategies=strategies,
        explanation_generator=ExplanationGenerator(),
    )
    chat_service = ChatService(
        routing_engine=routing_engine,
        provider_manager=provider_manager,
        model_registry=model_registry,
        session_factory=session_factory,
    )

    app.state.settings = settings
    app.state.event_bus = event_bus
    app.state.provider_manager = provider_manager
    app.state.model_registry = model_registry
    app.state.session_factory = session_factory
    app.state.chat_service = chat_service
    app.state.version = APP_VERSION
    app.state.start_time = time.time()

    yield


def create_app() -> FastAPI:
    app = FastAPI(title="LLM Cost Autopilot", version=APP_VERSION, lifespan=lifespan)
    app.include_router(health_router, prefix="/v1")
    app.include_router(models_router, prefix="/v1")
    app.include_router(chat_router, prefix="/v1")
    return app


app = create_app()
