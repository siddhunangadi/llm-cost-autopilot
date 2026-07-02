from datetime import datetime, timezone

from sqlalchemy.orm import sessionmaker

from backend.database.models import RoutingEventRow, VerificationRow
from backend.events.bus import EventBus
from backend.events.types import EventType
from backend.verification.engine import JudgeEngine
from backend.verification.events import VerificationCompleted, VerificationFailed, VerificationStarted
from backend.verification.status import VerificationStatus


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class _RoutingSnapshot:
    def __init__(self, selected_model: str, strategy: str, complexity: str) -> None:
        self.selected_model = selected_model
        self.strategy = strategy
        self.complexity = complexity


class VerificationService:
    def __init__(
        self,
        judge_engine: JudgeEngine,
        session_factory: sessionmaker,
        event_bus: EventBus,
        judge_prompt_version: str,
    ) -> None:
        self._judge_engine = judge_engine
        self._session_factory = session_factory
        self._event_bus = event_bus
        self._judge_prompt_version = judge_prompt_version

    async def verify(self, request_id: str, prompt: str, response: str) -> None:
        routing = self._load_routing_snapshot(request_id)

        with self._session_factory() as session:
            session.add(VerificationRow(
                request_id=request_id,
                status=VerificationStatus.PENDING.value,
                routing_model=routing.selected_model,
                routing_strategy=routing.strategy,
                routing_complexity=routing.complexity,
            ))
            session.commit()

        with self._session_factory() as session:
            row = session.query(VerificationRow).filter_by(request_id=request_id).one()
            row.status = VerificationStatus.RUNNING.value
            row.started_at = _utcnow()
            session.commit()

        self._event_bus.emit(
            EventType.VERIFICATION_STARTED, VerificationStarted(request_id=request_id).model_dump()
        )

        try:
            verdict, duration_ms = await self._judge_engine.run(prompt, response)
        except Exception as exc:
            with self._session_factory() as session:
                row = session.query(VerificationRow).filter_by(request_id=request_id).one()
                row.status = VerificationStatus.FAILED.value
                row.error_type = type(exc).__name__
                row.error = str(exc)
                row.completed_at = _utcnow()
                session.commit()
            self._event_bus.emit(
                EventType.VERIFICATION_FAILED,
                VerificationFailed(
                    request_id=request_id, error_type=type(exc).__name__, error=str(exc)
                ).model_dump(),
            )
            return

        with self._session_factory() as session:
            row = session.query(VerificationRow).filter_by(request_id=request_id).one()
            row.status = VerificationStatus.COMPLETED.value
            row.score = verdict.score
            row.passed = verdict.passed
            row.confidence = verdict.confidence
            row.rationale = verdict.rationale
            row.dimensions = verdict.dimensions.model_dump()
            row.judge_model = self._judge_engine.judge_model_id
            row.judge_prompt_version = self._judge_prompt_version
            row.evaluation_duration_ms = duration_ms
            row.completed_at = _utcnow()
            session.commit()

        self._event_bus.emit(
            EventType.VERIFICATION_COMPLETED,
            VerificationCompleted(request_id=request_id, score=verdict.score).model_dump(),
        )

    def _load_routing_snapshot(self, request_id: str) -> _RoutingSnapshot:
        with self._session_factory() as session:
            event = session.query(RoutingEventRow).filter_by(request_id=request_id).one()
            return _RoutingSnapshot(
                selected_model=event.selected_model,
                strategy=event.selected_strategy,
                complexity=event.complexity,
            )
