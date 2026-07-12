import hashlib
import json
import time
import uuid

from fastapi import BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker

from backend.database.models import RequestRow, ResponseRow, RoutingEventRow
from backend.providers.base import ProviderError
from backend.providers.executor import CircuitOpenError, ProviderExecutor
from backend.providers.manager import ProviderManager
from backend.routing.engine import NoEligibleModelError, RoutingDecision, RoutingEngine
from backend.services.model_registry import ModelRegistry
from backend.telemetry.logging import get_logger
from backend.verification.service import VerificationService


class ChatResult(BaseModel):
    request_id: str
    response: str
    routing: RoutingDecision


class ChatService:
    def __init__(
        self,
        routing_engine: RoutingEngine,
        provider_manager: ProviderManager,
        provider_executor: ProviderExecutor,
        model_registry: ModelRegistry,
        session_factory: sessionmaker,
        verification_service: VerificationService,
    ) -> None:
        self._routing_engine = routing_engine
        self._provider_manager = provider_manager
        self._provider_executor = provider_executor
        self._model_registry = model_registry
        self._session_factory = session_factory
        self._verification_service = verification_service
        self._logger = get_logger("chat")

    async def chat(
        self, prompt: str, strategy: str, background_tasks: BackgroundTasks
    ) -> ChatResult:
        start = time.monotonic()
        request_id = str(uuid.uuid4())
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        decision = self._routing_engine.route(prompt, strategy_name=strategy)

        with self._session_factory() as session:
            session.add(RequestRow(request_id=request_id, prompt=prompt, strategy=strategy))
            session.commit()
        self._persist_routing_event(request_id, decision)

        spec = self._model_registry.get_model(decision.selected_model)

        try:
            response_text = await self._provider_executor.generate(
                spec.provider, prompt, spec.model, retry=True
            )
        except (ProviderError, CircuitOpenError) as exc:
            reason = "circuit_open" if isinstance(exc, CircuitOpenError) else "provider_error"
            try:
                failover_decision = self._routing_engine.route(
                    prompt, strategy_name=strategy,
                    exclude_providers=frozenset({spec.provider}),
                )
            except NoEligibleModelError as no_eligible:
                self._persist_error_response(
                    request_id, error_type="no_eligible_model", error=str(no_eligible)
                )
                raise

            self._persist_routing_event(request_id, failover_decision)
            new_spec = self._model_registry.get_model(failover_decision.selected_model)
            self._provider_executor.emit_failover_triggered(
                failed_provider=spec.provider, replacement_provider=new_spec.provider,
                original_model=spec.id, replacement_model=new_spec.id,
                reason=reason, attempt_number=2,
            )

            try:
                response_text = await self._provider_executor.generate(
                    new_spec.provider, prompt, new_spec.model, retry=False
                )
            except (ProviderError, CircuitOpenError) as exc2:
                error_type = "circuit_open" if isinstance(exc2, CircuitOpenError) else "provider_error"
                self._persist_error_response(request_id, error_type=error_type, error=str(exc2))
                raise

            decision, spec = failover_decision, new_spec

        try:
            provider = self._provider_manager.get_provider(spec.provider)
        except ProviderError:
            # Generation already succeeded -- this is only for tokenizing
            # the prompt/response we already have for cost accounting. If
            # the provider was disabled/deleted concurrently in the window
            # since generate() returned, fall back to "mock" (always
            # present) rather than turning an already-successful response
            # into an unhandled 500.
            self._logger.warning(
                "provider_unavailable_for_cost_accounting", extra={"provider": spec.provider}
            )
            provider = self._provider_manager.get_provider("mock")
        input_tokens = provider.count_tokens(prompt)
        output_tokens = provider.count_tokens(response_text)
        actual_cost = self._model_registry.estimate_cost(spec.id, input_tokens, output_tokens)

        with self._session_factory() as session:
            session.add(ResponseRow(
                request_id=request_id,
                response_text=response_text,
                actual_input_tokens=input_tokens,
                actual_output_tokens=output_tokens,
                actual_cost=actual_cost,
            ))
            session.commit()

        self._logger.info("chat_request_completed", extra={
            "request_id": request_id,
            "prompt_hash": prompt_hash,
            "complexity": decision.complexity.value,
            "final_model": spec.id,
            "provider": spec.provider,
            "latency_ms": round((time.monotonic() - start) * 1000, 1),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost": actual_cost,
            "routing_reason": decision.reasoning,
        })

        try:
            background_tasks.add_task(
                self._verification_service.verify, request_id, prompt, response_text
            )
        except Exception:
            self._logger.exception(
                "verification_scheduling_failed", extra={"request_id": request_id}
            )

        return ChatResult(request_id=request_id, response=response_text, routing=decision)

    def _persist_routing_event(self, request_id: str, decision: RoutingDecision) -> None:
        with self._session_factory() as session:
            session.add(RoutingEventRow(
                request_id=request_id,
                complexity=decision.complexity.value,
                confidence=decision.confidence,
                selected_model=decision.selected_model,
                selected_strategy=decision.strategy,
                estimated_cost=decision.estimated_cost,
                estimated_latency_ms=decision.estimated_latency_ms,
                reasoning=json.dumps(decision.reasoning),
            ))
            session.commit()

    def _persist_error_response(self, request_id: str, *, error_type: str, error: str) -> None:
        with self._session_factory() as session:
            session.add(ResponseRow(request_id=request_id, error_type=error_type, error=error))
            session.commit()
