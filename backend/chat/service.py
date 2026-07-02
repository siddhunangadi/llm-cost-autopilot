import json
import uuid

from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker

from backend.database.models import RequestRow, ResponseRow, RoutingEventRow
from backend.providers.base import ProviderError
from backend.providers.manager import ProviderManager
from backend.routing.engine import RoutingDecision, RoutingEngine
from backend.services.model_registry import ModelRegistry


class ChatResult(BaseModel):
    request_id: str
    response: str
    routing: RoutingDecision


class ChatService:
    def __init__(
        self,
        routing_engine: RoutingEngine,
        provider_manager: ProviderManager,
        model_registry: ModelRegistry,
        session_factory: sessionmaker,
    ) -> None:
        self._routing_engine = routing_engine
        self._provider_manager = provider_manager
        self._model_registry = model_registry
        self._session_factory = session_factory

    async def chat(self, prompt: str, strategy: str = "balanced") -> ChatResult:
        request_id = str(uuid.uuid4())
        decision = self._routing_engine.route(prompt, strategy_name=strategy)

        with self._session_factory() as session:
            session.add(RequestRow(request_id=request_id, prompt=prompt, strategy=strategy))
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

        model_spec = self._model_registry.get_model(decision.selected_model)
        provider = self._provider_manager.get_provider(model_spec.provider)

        try:
            response_text = await provider.generate(prompt, model=model_spec.model)
        except ProviderError as exc:
            with self._session_factory() as session:
                session.add(ResponseRow(request_id=request_id, error=str(exc)))
                session.commit()
            raise

        input_tokens = provider.count_tokens(prompt)
        output_tokens = provider.count_tokens(response_text)
        actual_cost = self._model_registry.estimate_cost(model_spec.id, input_tokens, output_tokens)

        with self._session_factory() as session:
            session.add(ResponseRow(
                request_id=request_id,
                response_text=response_text,
                actual_input_tokens=input_tokens,
                actual_output_tokens=output_tokens,
                actual_cost=actual_cost,
            ))
            session.commit()

        return ChatResult(request_id=request_id, response=response_text, routing=decision)
