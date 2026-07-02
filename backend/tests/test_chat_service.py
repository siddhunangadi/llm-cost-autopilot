import json
import textwrap
from unittest.mock import AsyncMock

import pytest

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


def _make_chat_service(tmp_path):
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

    chat_service = ChatService(
        routing_engine=routing_engine,
        provider_manager=provider_manager,
        model_registry=model_registry,
        session_factory=session_factory,
    )
    return chat_service, session_factory


async def test_chat_returns_result_and_persists_rows(tmp_path):
    chat_service, session_factory = _make_chat_service(tmp_path)

    result = await chat_service.chat("List three fruits.", strategy="balanced")

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
        await chat_service.chat("List three fruits.", strategy="balanced")

    with session_factory() as session:
        response_row = session.query(ResponseRow).one()

    assert response_row.response_text is None
    assert response_row.error == "simulated failure"
