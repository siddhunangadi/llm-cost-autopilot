import json

import pytest

from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.database.models import RequestRow, RoutingEventRow, VerificationRow
from backend.events.bus import EventBus
from backend.events.types import EventType
from backend.providers.mock_provider import MockProvider
from backend.verification.engine import JudgeEngine
from backend.verification.judge import LLMJudge
from backend.verification.service import VerificationService
from backend.verification.status import VerificationStatus


def _valid_judge_json() -> str:
    return json.dumps({
        "correctness": 0.9, "completeness": 0.9, "instruction_following": 0.9,
        "format_adherence": 0.9, "confidence": 0.9, "rationale": "Good answer.",
    })


def _seed_request_and_routing_event(session_factory, request_id: str) -> None:
    with session_factory() as session:
        session.add(RequestRow(request_id=request_id, prompt="What is 2+2?", strategy="balanced"))
        session.add(RoutingEventRow(
            request_id=request_id, complexity="simple", confidence=0.9,
            selected_model="gpt-4o-mini", selected_strategy="balanced",
            estimated_cost=0.001, estimated_latency_ms=450, reasoning="[]",
        ))
        session.commit()


def _make_service(tmp_path, provider_response: str, event_bus: EventBus | None = None):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    provider = MockProvider(response=provider_response)
    judge = LLMJudge(provider=provider, model="gpt-4o", pass_threshold=0.7)
    judge_engine = JudgeEngine(judge=judge, judge_model_id="gpt-4o")

    service = VerificationService(
        judge_engine=judge_engine,
        session_factory=session_factory,
        event_bus=event_bus or EventBus(),
        judge_prompt_version="v1",
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
