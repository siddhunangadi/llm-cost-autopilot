import json
import textwrap

import pytest

from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.database.models import RequestRow, ResponseRow, RoutingEventRow, VerificationRow
from backend.events.bus import EventBus
from backend.events.types import EventType
from backend.providers.circuit_breaker import CircuitBreaker
from backend.providers.executor import ProviderExecutor
from backend.providers.factory import ProviderFactory
from backend.providers.manager import ProviderManager
from backend.providers.mock_provider import MockProvider
from backend.providers.retry import ExponentialBackoffRetryPolicy
from backend.services.cost_estimator import DefaultCostEstimator
from backend.services.credential_store import CredentialStore
from backend.services.model_registry import ModelRegistry
from backend.verification.engine import JudgeEngine
from backend.verification.judge import LLMJudge
from backend.verification.service import VerificationService
from backend.verification.status import VerificationStatus

MODELS_YAML = textwrap.dedent("""
    models:
      - id: gpt-4o-mini
        provider: mock
        model: gpt-4o-mini
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
      - id: gpt-4o
        provider: mock
        model: gpt-4o
        pricing:
          input_cost: 2.50
          output_cost: 10.00
        limits:
          context_window: 128000
          max_output_tokens: 16384
        capabilities:
          supports_streaming: true
          supports_tools: true
          supports_json: true
          supports_vision: true
        metadata:
          benchmark_score: 0.93
          average_latency_ms: 900
""")


async def _no_op_sleep(delay: float) -> None:
    pass


def _valid_judge_json() -> str:
    return json.dumps({
        "correctness": 0.9, "completeness": 0.9, "instruction_following": 0.9,
        "format_adherence": 0.9, "confidence": 0.9, "rationale": "Good answer.",
    })


def _low_score_judge_json() -> str:
    return json.dumps({
        "correctness": 0.2, "completeness": 0.2, "instruction_following": 0.2,
        "format_adherence": 0.2, "confidence": 0.9, "rationale": "Missed the point.",
    })


def _seed_request_and_routing_event(session_factory, request_id: str) -> None:
    with session_factory() as session:
        session.add(RequestRow(request_id=request_id, prompt="What is 2+2?", strategy="balanced"))
        session.add(RoutingEventRow(
            request_id=request_id, complexity="simple", confidence=0.9,
            selected_model="gpt-4o-mini", selected_strategy="balanced",
            estimated_cost=0.001, estimated_latency_ms=450, reasoning="[]",
        ))
        session.add(ResponseRow(
            request_id=request_id, response_text="4",
            actual_input_tokens=5, actual_output_tokens=1, actual_cost=0.001,
        ))
        session.commit()


def _make_service(tmp_path, provider_response: str, event_bus: EventBus | None = None):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(MODELS_YAML)

    factory = ProviderFactory()
    factory.register("mock", MockProvider)
    credential_store = CredentialStore(
        session_factory=session_factory, settings=settings, provider_names=("mock",),
    )
    provider_manager = ProviderManager(factory, credential_store)
    provider_manager.get_provider("mock")._response = provider_response

    model_registry = ModelRegistry(
        provider_manager=provider_manager,
        event_bus=EventBus(),
        cost_estimator=DefaultCostEstimator(),
        session_factory=session_factory,
        yaml_path=str(yaml_path),
    )
    model_registry.reload()

    provider_executor = ProviderExecutor(
        provider_manager=provider_manager,
        retry_policy=ExponentialBackoffRetryPolicy(max_attempts=3, sleep=_no_op_sleep),
        circuit_breakers={"mock": CircuitBreaker(failure_threshold=5)},
        event_bus=EventBus(),
    )

    judge = LLMJudge(provider=provider_manager.get_provider("mock"), model="gpt-4o", pass_threshold=0.7)
    judge_engine = JudgeEngine(judge=judge, judge_model_id="gpt-4o")

    service = VerificationService(
        judge_engine=judge_engine,
        session_factory=session_factory,
        event_bus=event_bus or EventBus(),
        judge_prompt_version="v1",
        pass_threshold=0.7,
        escalation_model_id="gpt-4o",
        provider_executor=provider_executor,
        provider_manager=provider_manager,
        model_registry=model_registry,
    )
    return service, session_factory


@pytest.mark.asyncio
async def test_verify_completes_and_snapshots_routing(tmp_path):
    service, session_factory = _make_service(tmp_path, _valid_judge_json())
    _seed_request_and_routing_event(session_factory, "req-1")

    await service.verify("req-1", "What is 2+2?", "4")

    with session_factory() as session:
        row = session.query(VerificationRow).filter_by(request_id="req-1").one()
        assert row.status == VerificationStatus.COMPLETED.value
        assert row.score == pytest.approx(0.9)
        assert row.passed is True
        assert row.escalated is False
        assert row.judge_model == "gpt-4o"
        assert row.judge_prompt_version == "v1"
        assert row.evaluation_duration_ms >= 0
        assert row.routing_model == "gpt-4o-mini"
        assert row.routing_strategy == "balanced"
        assert row.routing_complexity == "simple"
        assert row.started_at is not None
        assert row.completed_at is not None


@pytest.mark.asyncio
async def test_verify_persists_failed_row_on_malformed_judge_output(tmp_path):
    service, session_factory = _make_service(tmp_path, "not valid json")
    _seed_request_and_routing_event(session_factory, "req-2")

    await service.verify("req-2", "What is 2+2?", "4")

    with session_factory() as session:
        row = session.query(VerificationRow).filter_by(request_id="req-2").one()
        assert row.status == VerificationStatus.FAILED.value
        assert row.error_type == "ValidationError"
        assert row.error is not None
        assert row.score is None
        assert row.completed_at is not None


@pytest.mark.asyncio
async def test_verify_emits_events_in_order_after_persistence(tmp_path):
    events: list[tuple[str, dict]] = []
    bus = EventBus()
    bus.subscribe(EventType.VERIFICATION_STARTED, lambda p: events.append(("started", p)))
    bus.subscribe(EventType.VERIFICATION_COMPLETED, lambda p: events.append(("completed", p)))

    service, session_factory = _make_service(tmp_path, _valid_judge_json(), event_bus=bus)
    _seed_request_and_routing_event(session_factory, "req-3")

    await service.verify("req-3", "What is 2+2?", "4")

    assert [name for name, _ in events] == ["started", "completed"]
    assert events[1][1]["score"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_verify_never_raises_on_judge_failure(tmp_path):
    service, session_factory = _make_service(tmp_path, "not valid json")
    _seed_request_and_routing_event(session_factory, "req-4")

    await service.verify("req-4", "prompt", "response")  # must not raise


@pytest.mark.asyncio
async def test_verify_marks_escalated_and_emits_escalation_event_below_pass_threshold(tmp_path):
    events: list[tuple[str, dict]] = []
    bus = EventBus()
    bus.subscribe(EventType.ESCALATION_TRIGGERED, lambda p: events.append(("escalated", p)))

    service, session_factory = _make_service(tmp_path, _low_score_judge_json(), event_bus=bus)
    _seed_request_and_routing_event(session_factory, "req-5")

    await service.verify("req-5", "What is 2+2?", "purple")

    with session_factory() as session:
        row = session.query(VerificationRow).filter_by(request_id="req-5").one()
        assert row.passed is False
        assert row.escalated is True
        assert row.escalated_model == "gpt-4o"
        assert row.quality_gap == pytest.approx(0.7 - row.score)
        assert row.escalation_cost_delta is not None
        assert row.escalation_latency_ms is not None
        assert row.escalation_latency_ms >= 0

    assert len(events) == 1
    payload = events[0][1]
    assert payload["request_id"] == "req-5"
    assert payload["routing_model"] == "gpt-4o-mini"
    assert payload["reason"] == "verification_score_below_pass_threshold"
    assert payload["escalated_model"] == "gpt-4o"
    assert payload["cost_delta"] is not None
    assert payload["latency_ms"] is not None
    assert payload["quality_gap"] == pytest.approx(0.7 - payload["score"])


@pytest.mark.asyncio
async def test_verify_does_not_emit_escalation_event_when_passed(tmp_path):
    events: list[tuple[str, dict]] = []
    bus = EventBus()
    bus.subscribe(EventType.ESCALATION_TRIGGERED, lambda p: events.append(("escalated", p)))

    service, session_factory = _make_service(tmp_path, _valid_judge_json(), event_bus=bus)
    _seed_request_and_routing_event(session_factory, "req-6")

    await service.verify("req-6", "What is 2+2?", "4")

    assert events == []


@pytest.mark.asyncio
async def test_verify_records_quality_gap_and_still_emits_event_when_escalation_model_missing(
    tmp_path,
):
    """Escalation regeneration must never raise even if the configured
    escalation model isn't registered -- the quality_gap and event are
    still recorded so the failure remains observable."""
    events: list[tuple[str, dict]] = []
    bus = EventBus()
    bus.subscribe(EventType.ESCALATION_TRIGGERED, lambda p: events.append(("escalated", p)))

    service, session_factory = _make_service(tmp_path, _low_score_judge_json(), event_bus=bus)
    service._escalation_model_id = "does-not-exist"
    _seed_request_and_routing_event(session_factory, "req-7")

    await service.verify("req-7", "What is 2+2?", "purple")  # must not raise

    with session_factory() as session:
        row = session.query(VerificationRow).filter_by(request_id="req-7").one()
        assert row.escalated is True
        assert row.quality_gap == pytest.approx(0.7 - row.score)
        assert row.escalated_model is None
        assert row.escalation_cost_delta is None

    assert len(events) == 1
    assert events[0][1]["escalated_model"] is None
