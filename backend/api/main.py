import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.analysis.prompt_analyzer import PromptAnalyzer
from backend.api.paths import STATIC_DIR
from backend.api.routers.analytics import router as analytics_router
from backend.api.routers.chat import router as chat_router
from backend.api.routers.dashboard import router as dashboard_router
from backend.api.routers.dashboard_ui import router as dashboard_ui_router
from backend.api.routers.health import router as health_router
from backend.api.routers.learning import router as learning_router
from backend.api.routers.metrics import router as metrics_router
from backend.api.routers.models import router as models_router
from backend.api.routers.verification import router as verification_router
from backend.chat.service import ChatService
from backend.classifier.complexity_classifier import HeuristicComplexityClassifier
from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.events.bus import EventBus
from backend.events.subscribers import register_logging_subscriber
from backend.learning.detector import FailurePatternDetector
from backend.learning.generator import RecommendationGenerator
from backend.learning.rules import (
    ComplexityTierRule, DetectionRuleConfig, ModelComplexityRule, OverpoweredModelRule,
)
from backend.learning.service import LearningService
from backend.providers.circuit_breaker import CircuitBreaker
from backend.providers.executor import ProviderExecutor
from backend.providers.factory import ProviderFactory
from backend.providers.manager import KNOWN_PROVIDER_NAMES, ProviderManager
from backend.providers.mock_provider import MockProvider
from backend.providers.openai_provider import OpenAIProvider
from backend.providers.retry import ExponentialBackoffRetryPolicy
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
from backend.services.analytics_service import AnalyticsService
from backend.services.cost_estimator import DefaultCostEstimator
from backend.services.dashboard_repository import DashboardRepository
from backend.services.dashboard_service import DashboardService
from backend.services.model_registry import ModelRegistry
from backend.telemetry.logging import configure_logging, get_logger
from backend.verification.config_loader import VerificationConfigLoader
from backend.verification.engine import JudgeEngine
from backend.verification.judge import BaseJudge, JudgeVerdict, LLMJudge
from backend.verification.service import VerificationService

APP_VERSION = "0.8.0"


class _UnavailableJudge(BaseJudge):
    """Fallback judge used when the configured judge model's provider
    cannot be resolved at startup (e.g. missing API key). Every
    verification attempt fails cleanly through VerificationService's
    existing exception handling (-> FAILED, never raised to the caller)
    instead of crashing app startup entirely."""

    def __init__(self, reason: str) -> None:
        self._reason = reason

    async def evaluate(self, prompt: str, response: str) -> JudgeVerdict:
        raise RuntimeError(self._reason)


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
    verification_config = VerificationConfigLoader.load(settings.verification_config_path)
    try:
        judge_provider = provider_manager.get_provider(
            model_registry.get_model(verification_config.judge_model_id).provider
        )
        judge: BaseJudge = LLMJudge(
            provider=judge_provider,
            model=verification_config.judge_model_id,
            pass_threshold=verification_config.pass_threshold,
        )
    except Exception as exc:
        # The judge model's provider may be unavailable (e.g. no API key
        # configured) at startup. Verification is a best-effort side
        # effect of /v1/chat, never a prerequisite for it -- the app must
        # still boot and serve chat requests. Every verification attempt
        # will fail cleanly (FAILED status) via VerificationService's
        # own exception handling rather than crashing startup here.
        get_logger("verification").exception("judge_provider_unavailable_at_startup")
        judge = _UnavailableJudge(reason=str(exc))
    judge_engine = JudgeEngine(judge=judge, judge_model_id=verification_config.judge_model_id)
    verification_service = VerificationService(
        judge_engine=judge_engine,
        session_factory=session_factory,
        event_bus=event_bus,
        judge_prompt_version=verification_config.judge_prompt_version,
    )

    provider_executor = ProviderExecutor(
        provider_manager=provider_manager,
        retry_policy=ExponentialBackoffRetryPolicy(),
        circuit_breakers={name: CircuitBreaker() for name in KNOWN_PROVIDER_NAMES},
        event_bus=event_bus,
    )

    chat_service = ChatService(
        routing_engine=routing_engine,
        provider_manager=provider_manager,
        provider_executor=provider_executor,
        model_registry=model_registry,
        session_factory=session_factory,
        verification_service=verification_service,
    )

    cost_optimization_config = DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.75)
    detector = FailurePatternDetector(rules=[
        ModelComplexityRule(DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.6)),
        ComplexityTierRule(DetectionRuleConfig(min_samples=30, pass_rate_threshold=0.5)),
        OverpoweredModelRule(cost_optimization_config),
    ])
    learning_service = LearningService(
        detector=detector,
        generator=RecommendationGenerator(),
        session_factory=session_factory,
        model_registry=model_registry,
        cost_optimization_config=cost_optimization_config,
    )

    dashboard_repository = DashboardRepository(session_factory=session_factory)
    dashboard_service = DashboardService(
        provider_manager=provider_manager,
        provider_executor=provider_executor,
        learning_service=learning_service,
        dashboard_repository=dashboard_repository,
    )
    analytics_service = AnalyticsService(dashboard_repository=dashboard_repository)

    app.state.settings = settings
    app.state.event_bus = event_bus
    app.state.provider_manager = provider_manager
    app.state.model_registry = model_registry
    app.state.session_factory = session_factory
    app.state.chat_service = chat_service
    app.state.learning_service = learning_service
    app.state.provider_executor = provider_executor
    app.state.dashboard_service = dashboard_service
    app.state.analytics_service = analytics_service
    app.state.version = APP_VERSION
    app.state.start_time = time.time()

    yield


def create_app() -> FastAPI:
    app = FastAPI(title="LLM Cost Autopilot", version=APP_VERSION, lifespan=lifespan)
    app.include_router(health_router, prefix="/v1")
    app.include_router(models_router, prefix="/v1")
    app.include_router(chat_router, prefix="/v1")
    app.include_router(verification_router, prefix="/v1")
    app.include_router(metrics_router, prefix="/v1")
    app.include_router(learning_router, prefix="/v1")
    app.include_router(dashboard_router, prefix="/v1")
    app.include_router(dashboard_ui_router)
    app.include_router(analytics_router)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


app = create_app()
