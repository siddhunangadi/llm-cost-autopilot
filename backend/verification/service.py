import time
from datetime import datetime, timezone

from sqlalchemy.orm import sessionmaker

from backend.database.models import ResponseRow, RoutingEventRow, VerificationRow
from backend.events.bus import EventBus
from backend.events.types import EventType
from backend.providers.executor import ProviderExecutor
from backend.providers.manager import ProviderManager
from backend.services.model_registry import ModelRegistry
from backend.telemetry.logging import get_logger
from backend.verification.engine import JudgeEngine
from backend.verification.events import (
    EscalationTriggered,
    VerificationCompleted,
    VerificationFailed,
    VerificationStarted,
)
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
        pass_threshold: float,
        escalation_model_id: str,
        provider_executor: ProviderExecutor,
        provider_manager: ProviderManager,
        model_registry: ModelRegistry,
    ) -> None:
        self._judge_engine = judge_engine
        self._session_factory = session_factory
        self._event_bus = event_bus
        self._judge_prompt_version = judge_prompt_version
        self._pass_threshold = pass_threshold
        self._escalation_model_id = escalation_model_id
        self._provider_executor = provider_executor
        self._provider_manager = provider_manager
        self._model_registry = model_registry
        self._logger = get_logger("verification")

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

        escalated = not verdict.passed

        with self._session_factory() as session:
            row = session.query(VerificationRow).filter_by(request_id=request_id).one()
            row.status = VerificationStatus.COMPLETED.value
            row.score = verdict.score
            row.passed = verdict.passed
            row.escalated = escalated
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

        if escalated:
            await self._run_escalation(
                request_id=request_id, prompt=prompt, routing_model=routing.selected_model,
                score=verdict.score,
            )

    async def _run_escalation(
        self, *, request_id: str, prompt: str, routing_model: str, score: float
    ) -> None:
        quality_gap = round(self._pass_threshold - score, 4)
        event = EscalationTriggered(
            request_id=request_id, routing_model=routing_model, score=score,
            reason="verification_score_below_pass_threshold", quality_gap=quality_gap,
        )

        try:
            escalation_spec = self._model_registry.get_model(self._escalation_model_id)
            start = time.monotonic()
            escalated_response = await self._provider_executor.generate(
                escalation_spec.provider, prompt, escalation_spec.model, retry=False
            )
            latency_ms = round((time.monotonic() - start) * 1000, 1)

            escalation_provider = self._provider_manager.get_provider(escalation_spec.provider)
            input_tokens = escalation_provider.count_tokens(prompt)
            output_tokens = escalation_provider.count_tokens(escalated_response)
            escalated_cost = self._model_registry.estimate_cost(
                escalation_spec.id, input_tokens, output_tokens
            )

            with self._session_factory() as session:
                original_cost = (
                    session.query(ResponseRow)
                    .filter_by(request_id=request_id)
                    .one()
                    .actual_cost
                )
            cost_delta = (
                round(escalated_cost - original_cost, 6) if original_cost is not None else None
            )

            event = event.model_copy(update={
                "escalated_model": escalation_spec.id,
                "cost_delta": cost_delta,
                "latency_ms": latency_ms,
            })
            with self._session_factory() as session:
                row = session.query(VerificationRow).filter_by(request_id=request_id).one()
                row.escalated_model = escalation_spec.id
                row.escalation_cost_delta = cost_delta
                row.escalation_latency_ms = latency_ms
                row.quality_gap = quality_gap
                session.commit()
        except Exception as exc:
            self._logger.warning(
                "escalation_regeneration_failed",
                extra={"request_id": request_id, "error": str(exc)},
            )
            with self._session_factory() as session:
                row = session.query(VerificationRow).filter_by(request_id=request_id).one()
                row.quality_gap = quality_gap
                session.commit()

        self._event_bus.emit(EventType.ESCALATION_TRIGGERED, event.model_dump())

    def _load_routing_snapshot(self, request_id: str) -> _RoutingSnapshot:
        with self._session_factory() as session:
            event = session.query(RoutingEventRow).filter_by(request_id=request_id).one()
            return _RoutingSnapshot(
                selected_model=event.selected_model,
                strategy=event.selected_strategy,
                complexity=event.complexity,
            )
