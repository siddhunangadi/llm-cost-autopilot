import json
import textwrap
from unittest.mock import AsyncMock

import pytest
from fastapi import BackgroundTasks

from backend.analysis.prompt_analyzer import PromptAnalyzer
from backend.chat.service import ChatService
from backend.classifier.complexity_classifier import HeuristicComplexityClassifier
from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.database.models import RequestRow, ResponseRow, RoutingEventRow
from backend.events.bus import EventBus
from backend.providers.base import ProviderError
from backend.providers.circuit_breaker import CircuitBreaker
from backend.providers.executor import ProviderExecutor
from backend.providers.factory import ProviderFactory
from backend.providers.manager import ProviderManager
from backend.providers.mock_provider import MockProvider
from backend.providers.retry import ExponentialBackoffRetryPolicy
from backend.routing.config import BalancedStrategyWeights, ClassifierPolicy, EligibilityPolicy
from backend.routing.engine import NoEligibleModelError, RoutingEngine
from backend.routing.explanation import ExplanationGenerator
from backend.routing.policy import RoutingPolicy
from backend.routing.strategies import BalancedStrategy
from backend.services.cost_estimator import DefaultCostEstimator
from backend.services.credential_store import CredentialStore
from backend.services.model_registry import ModelRegistry
from backend.verification.engine import JudgeEngine
from backend.verification.judge import LLMJudge
from backend.verification.service import VerificationService

ONE_MODEL_YAML = textwrap.dedent("""
    models:
      - id: mock-model
        provider: mock
        model: mock-model
        pricing:
          input_cost: 0.15
          output_cost: 0.60
        limits:
          context_window: 128000
          max_output_tokens: 16384
        capabilities:
          supports_streaming: true
          supports_tools: true
          supports_json: true
          supports_vision: false
        metadata:
          benchmark_score: 0.82
          average_latency_ms: 450
""")


class _FailingVerificationService:
    async def verify(self, request_id, prompt, response):
        raise RuntimeError("verification exploded")


async def _no_op_sleep(delay: float) -> None:
    pass


def _make_chat_service(tmp_path, verification_service=None):
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(ONE_MODEL_YAML)

    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine_db = create_engine_from_settings(settings)
    init_db(engine_db)
    session_factory = create_session_factory(engine_db)

    factory = ProviderFactory()
    factory.register("mock", MockProvider)
    credential_store = CredentialStore(session_factory=session_factory, settings=settings)
    provider_manager = ProviderManager(factory, credential_store)
    provider_executor = ProviderExecutor(
        provider_manager=provider_manager,
        retry_policy=ExponentialBackoffRetryPolicy(max_attempts=3, sleep=_no_op_sleep),
        circuit_breakers={"mock": CircuitBreaker(failure_threshold=5)},
        event_bus=EventBus(),
    )

    model_registry = ModelRegistry(
        provider_manager=provider_manager,
        event_bus=EventBus(),
        cost_estimator=DefaultCostEstimator(),
        session_factory=session_factory,
        yaml_path=str(yaml_path),
    )
    model_registry.reload()

    routing_engine = RoutingEngine(
        model_registry=model_registry,
        analyzer=PromptAnalyzer(),
        classifier=HeuristicComplexityClassifier(ClassifierPolicy(simple_max=1, medium_max=3)),
        routing_policy=RoutingPolicy({
            "simple": EligibilityPolicy(min_benchmark_score=0.0),
            "medium": EligibilityPolicy(min_benchmark_score=0.75),
            "complex": EligibilityPolicy(min_benchmark_score=0.90),
        }),
        strategies={"balanced": BalancedStrategy(BalancedStrategyWeights())},
        explanation_generator=ExplanationGenerator(),
    )

    if verification_service is None:
        judge = LLMJudge(provider=MockProvider(response="{}"), model="mock", pass_threshold=0.7)
        judge_engine = JudgeEngine(judge=judge, judge_model_id="mock")
        verification_service = VerificationService(
            judge_engine=judge_engine, session_factory=session_factory,
            event_bus=EventBus(), judge_prompt_version="v1",
        )

    chat_service = ChatService(
        routing_engine=routing_engine,
        provider_manager=provider_manager,
        provider_executor=provider_executor,
        model_registry=model_registry,
        session_factory=session_factory,
        verification_service=verification_service,
    )
    return chat_service, session_factory


async def test_chat_returns_result_and_persists_rows(tmp_path):
    chat_service, session_factory = _make_chat_service(tmp_path)

    result = await chat_service.chat(
        "List three fruits.", strategy="balanced", background_tasks=BackgroundTasks()
    )

    assert result.response
    assert result.routing.selected_model == "mock-model"

    with session_factory() as session:
        request_row = session.query(RequestRow).filter_by(request_id=result.request_id).one()
        response_row = session.query(ResponseRow).filter_by(request_id=result.request_id).one()
        routing_event_row = (
            session.query(RoutingEventRow).filter_by(request_id=result.request_id).one()
        )

    assert request_row.prompt == "List three fruits."
    assert response_row.response_text == result.response
    assert response_row.error is None
    assert routing_event_row.selected_model == "mock-model"
    assert json.loads(routing_event_row.reasoning) == result.routing.reasoning


async def test_chat_persists_no_eligible_model_error_when_primary_fails_and_no_failover_candidate(
    tmp_path, mocker
):
    chat_service, session_factory = _make_chat_service(tmp_path)

    mocker.patch(
        "backend.providers.mock_provider.MockProvider.generate",
        new_callable=AsyncMock,
        side_effect=ProviderError("simulated failure"),
    )

    with pytest.raises(NoEligibleModelError):
        await chat_service.chat(
            "List three fruits.", strategy="balanced", background_tasks=BackgroundTasks()
        )

    with session_factory() as session:
        response_row = session.query(ResponseRow).one()

    assert response_row.response_text is None
    assert response_row.error_type == "no_eligible_model"
    assert response_row.error == "No available model meets the 'simple' complexity policy"


@pytest.mark.asyncio
async def test_chat_schedules_verification_background_task(tmp_path):
    chat_service, _ = _make_chat_service(tmp_path)
    background_tasks = BackgroundTasks()

    result = await chat_service.chat("Hello.", strategy="balanced", background_tasks=background_tasks)

    assert result.response
    assert len(background_tasks.tasks) == 1


@pytest.mark.asyncio
async def test_chat_succeeds_even_if_verification_service_would_fail(tmp_path):
    chat_service, _ = _make_chat_service(tmp_path, verification_service=_FailingVerificationService())
    background_tasks = BackgroundTasks()

    result = await chat_service.chat("Hello.", strategy="balanced", background_tasks=background_tasks)

    assert result.response
    # The scheduled background task itself would raise if awaited directly,
    # but chat() must return successfully regardless -- scheduling never fails
    # here since add_task() only registers the call, it doesn't invoke it.


TWO_PROVIDER_YAML = textwrap.dedent("""
    models:
      - id: primary-model
        provider: primary
        model: primary-model
        pricing:
          input_cost: 0.15
          output_cost: 0.60
        limits:
          context_window: 128000
          max_output_tokens: 16384
        capabilities:
          supports_streaming: true
          supports_tools: true
          supports_json: true
          supports_vision: false
        metadata:
          benchmark_score: 0.82
          average_latency_ms: 450
      - id: backup-model
        provider: backup
        model: backup-model
        pricing:
          input_cost: 0.15
          output_cost: 0.60
        limits:
          context_window: 128000
          max_output_tokens: 16384
        capabilities:
          supports_streaming: true
          supports_tools: true
          supports_json: true
          supports_vision: false
        metadata:
          benchmark_score: 0.82
          average_latency_ms: 450
""")


class _TwoProviderManager:
    """Duck-typed ProviderManager double exposing two independently
    named providers, each backed by its own MockProvider instance --
    ProviderManager itself only supports 'mock' and 'openai', so a test
    double is required to exercise real cross-provider failover."""

    def __init__(self, primary: MockProvider, backup: MockProvider) -> None:
        self._providers = {"primary": primary, "backup": backup}

    def get_provider(self, name: str):
        return self._providers[name]

    def is_provider_available(self, name: str) -> bool:
        return name in self._providers

    def list_providers(self):
        return {name: "available" for name in self._providers}


def _make_failover_chat_service(tmp_path):
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(TWO_PROVIDER_YAML)

    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine_db = create_engine_from_settings(settings)
    init_db(engine_db)
    session_factory = create_session_factory(engine_db)

    primary = MockProvider(response="primary-response")
    backup = MockProvider(response="backup-response")
    provider_manager = _TwoProviderManager(primary, backup)

    model_registry = ModelRegistry(
        provider_manager=provider_manager,
        event_bus=EventBus(),
        cost_estimator=DefaultCostEstimator(),
        session_factory=session_factory,
        yaml_path=str(yaml_path),
    )
    model_registry.reload()

    routing_engine = RoutingEngine(
        model_registry=model_registry,
        analyzer=PromptAnalyzer(),
        classifier=HeuristicComplexityClassifier(ClassifierPolicy(simple_max=1, medium_max=3)),
        routing_policy=RoutingPolicy({
            "simple": EligibilityPolicy(min_benchmark_score=0.0),
            "medium": EligibilityPolicy(min_benchmark_score=0.75),
            "complex": EligibilityPolicy(min_benchmark_score=0.90),
        }),
        strategies={"balanced": BalancedStrategy(BalancedStrategyWeights())},
        explanation_generator=ExplanationGenerator(),
    )

    provider_executor = ProviderExecutor(
        provider_manager=provider_manager,
        retry_policy=ExponentialBackoffRetryPolicy(max_attempts=3, sleep=_no_op_sleep),
        circuit_breakers={
            "primary": CircuitBreaker(failure_threshold=5),
            "backup": CircuitBreaker(failure_threshold=5),
        },
        event_bus=EventBus(),
    )

    judge = LLMJudge(provider=MockProvider(response="{}"), model="mock", pass_threshold=0.7)
    verification_service = VerificationService(
        judge_engine=JudgeEngine(judge=judge, judge_model_id="mock"),
        session_factory=session_factory, event_bus=EventBus(), judge_prompt_version="v1",
    )

    chat_service = ChatService(
        routing_engine=routing_engine,
        provider_manager=provider_manager,
        provider_executor=provider_executor,
        model_registry=model_registry,
        session_factory=session_factory,
        verification_service=verification_service,
    )
    return chat_service, session_factory, primary, backup


async def test_chat_fails_over_to_backup_provider_after_primary_exhausts_retries(tmp_path, mocker):
    chat_service, session_factory, primary, backup = _make_failover_chat_service(tmp_path)

    mocker.patch.object(
        primary, "generate", new_callable=AsyncMock, side_effect=ProviderError("primary down"),
    )

    result = await chat_service.chat("Hello.", strategy="balanced", background_tasks=BackgroundTasks())

    assert result.response == "backup-response"
    assert result.routing.selected_model == "backup-model"

    with session_factory() as session:
        routing_events = (
            session.query(RoutingEventRow)
            .filter_by(request_id=result.request_id)
            .order_by(RoutingEventRow.id)
            .all()
        )
        response_row = session.query(ResponseRow).filter_by(request_id=result.request_id).one()

    assert len(routing_events) == 2
    assert routing_events[0].selected_model == "primary-model"
    assert routing_events[1].selected_model == "backup-model"
    assert response_row.response_text == "backup-response"
    assert response_row.error is None


async def test_chat_persists_error_type_when_both_primary_and_failover_fail(tmp_path, mocker):
    chat_service, session_factory, primary, backup = _make_failover_chat_service(tmp_path)

    mocker.patch.object(
        primary, "generate", new_callable=AsyncMock, side_effect=ProviderError("primary down"),
    )
    mocker.patch.object(
        backup, "generate", new_callable=AsyncMock, side_effect=ProviderError("backup down"),
    )

    with pytest.raises(ProviderError):
        await chat_service.chat("Hello.", strategy="balanced", background_tasks=BackgroundTasks())

    with session_factory() as session:
        response_row = session.query(ResponseRow).one()

    assert response_row.response_text is None
    assert response_row.error == "backup down"
    assert response_row.error_type == "provider_error"


async def test_chat_failover_attempt_makes_exactly_one_call_no_retry(tmp_path, mocker):
    chat_service, session_factory, primary, backup = _make_failover_chat_service(tmp_path)

    mocker.patch.object(
        primary, "generate", new_callable=AsyncMock, side_effect=ProviderError("primary down"),
    )
    backup_spy = mocker.patch.object(
        backup, "generate", new_callable=AsyncMock, side_effect=ProviderError("backup down"),
    )

    with pytest.raises(ProviderError):
        await chat_service.chat("Hello.", strategy="balanced", background_tasks=BackgroundTasks())

    assert backup_spy.await_count == 1  # no retry on the failover attempt
