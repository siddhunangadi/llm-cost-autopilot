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
from backend.providers.factory import ProviderFactory
from backend.providers.manager import ProviderManager
from backend.providers.mock_provider import MockProvider
from backend.routing.config import BalancedStrategyWeights, ClassifierPolicy, EligibilityPolicy
from backend.routing.engine import RoutingEngine
from backend.routing.explanation import ExplanationGenerator
from backend.routing.policy import RoutingPolicy
from backend.routing.strategies import BalancedStrategy
from backend.services.cost_estimator import DefaultCostEstimator
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


def _make_chat_service(tmp_path, verification_service=None):
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(ONE_MODEL_YAML)

    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine_db = create_engine_from_settings(settings)
    init_db(engine_db)
    session_factory = create_session_factory(engine_db)

    factory = ProviderFactory()
    factory.register("mock", MockProvider)
    provider_manager = ProviderManager(factory, settings)

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


async def test_chat_persists_error_and_reraises_on_provider_failure(tmp_path, mocker):
    chat_service, session_factory = _make_chat_service(tmp_path)

    mocker.patch(
        "backend.providers.mock_provider.MockProvider.generate",
        new_callable=AsyncMock,
        side_effect=ProviderError("simulated failure"),
    )

    with pytest.raises(ProviderError):
        await chat_service.chat(
            "List three fruits.", strategy="balanced", background_tasks=BackgroundTasks()
        )

    with session_factory() as session:
        response_row = session.query(ResponseRow).one()

    assert response_row.response_text is None
    assert response_row.error == "simulated failure"


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
