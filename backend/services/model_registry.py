from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import datetime, timezone
from types import MappingProxyType

import yaml
from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker

from backend.database.models import ModelRow, ProviderRow
from backend.events.bus import EventBus
from backend.events.types import EventType
from backend.providers.manager import ProviderManager
from backend.services.cost_estimator import BaseCostEstimator


class _Pricing(BaseModel):
    input_cost: float
    output_cost: float


class _Limits(BaseModel):
    context_window: int
    max_output_tokens: int


class _Capabilities(BaseModel):
    supports_streaming: bool
    supports_tools: bool
    supports_json: bool
    supports_vision: bool


class _Metadata(BaseModel):
    benchmark_score: float
    average_latency_ms: float


class _ModelYamlEntry(BaseModel):
    id: str
    provider: str
    model: str
    pricing: _Pricing
    limits: _Limits
    capabilities: _Capabilities
    metadata: _Metadata


class ModelSpec(BaseModel):
    id: str
    provider: str
    model: str
    input_cost: float
    output_cost: float
    context_window: int
    max_output_tokens: int
    supports_streaming: bool
    supports_tools: bool
    supports_json: bool
    supports_vision: bool
    benchmark_score: float
    average_latency_ms: float
    available: bool = False

    @classmethod
    def from_yaml_entry(cls, entry: _ModelYamlEntry, available: bool) -> "ModelSpec":
        return cls(
            id=entry.id,
            provider=entry.provider,
            model=entry.model,
            input_cost=entry.pricing.input_cost,
            output_cost=entry.pricing.output_cost,
            context_window=entry.limits.context_window,
            max_output_tokens=entry.limits.max_output_tokens,
            supports_streaming=entry.capabilities.supports_streaming,
            supports_tools=entry.capabilities.supports_tools,
            supports_json=entry.capabilities.supports_json,
            supports_vision=entry.capabilities.supports_vision,
            benchmark_score=entry.metadata.benchmark_score,
            average_latency_ms=entry.metadata.average_latency_ms,
            available=available,
        )


class BaseRegistry(ABC):
    @abstractmethod
    def get_model(self, model_id: str) -> ModelSpec: ...

    @abstractmethod
    def get_models(self) -> list[ModelSpec]: ...

    @abstractmethod
    def get_available_models(self) -> list[ModelSpec]: ...

    @abstractmethod
    def get_provider_models(self, provider: str) -> list[ModelSpec]: ...

    @abstractmethod
    def reload(self) -> None: ...


class ModelRegistry(BaseRegistry):
    def __init__(
        self,
        provider_manager: ProviderManager,
        event_bus: EventBus,
        cost_estimator: BaseCostEstimator,
        session_factory: sessionmaker,
        yaml_path: str,
    ) -> None:
        self._provider_manager = provider_manager
        self._event_bus = event_bus
        self._cost_estimator = cost_estimator
        self._session_factory = session_factory
        self._yaml_path = yaml_path
        self._cache_data: dict[str, ModelSpec] = {}
        self._cache: Mapping[str, ModelSpec] = MappingProxyType(self._cache_data)
        self._provider_health: dict[str, bool] = {}

    def _is_available(self, provider: str) -> bool:
        return self._provider_manager.is_provider_available(
            provider
        ) and self._provider_health.get(provider, True)

    def reload(self) -> None:
        with open(self._yaml_path) as f:
            raw = yaml.safe_load(f)

        entries = [_ModelYamlEntry.model_validate(item) for item in raw["models"]]

        seen_ids: set[str] = set()
        for entry in entries:
            if entry.id in seen_ids:
                raise ValueError(f"Duplicate model id in {self._yaml_path}: '{entry.id}'")
            seen_ids.add(entry.id)

        cache: dict[str, ModelSpec] = {}

        with self._session_factory() as session:
            for entry in entries:
                spec = ModelSpec.from_yaml_entry(entry, available=self._is_available(entry.provider))
                cache[spec.id] = spec

                row = session.query(ModelRow).filter_by(model_id=spec.id).one_or_none()
                if row is None:
                    row = ModelRow(model_id=spec.id)
                    session.add(row)
                row.provider = spec.provider
                row.model_name = spec.model
                row.input_cost = spec.input_cost
                row.output_cost = spec.output_cost
                row.context_window = spec.context_window
                row.benchmark_score = spec.benchmark_score
                row.supports_streaming = spec.supports_streaming
                row.supports_tools = spec.supports_tools
                row.supports_json = spec.supports_json
                row.average_latency_ms = spec.average_latency_ms
                row.available = spec.available

                self._event_bus.emit(
                    EventType.MODEL_REGISTERED, {"model_id": spec.id, "provider": spec.provider}
                )

            session.commit()

        # Swap the whole cache atomically -- a failed reload (raised above,
        # before this point) never partially applies, and stale entries
        # removed from the YAML are dropped rather than lingering.
        self._cache_data = cache
        self._cache = MappingProxyType(self._cache_data)

    def get_model(self, model_id: str) -> ModelSpec:
        if model_id not in self._cache:
            raise KeyError(f"Unknown model_id '{model_id}'")
        return self._cache[model_id]

    def get_models(self) -> list[ModelSpec]:
        return list(self._cache.values())

    def get_available_models(self) -> list[ModelSpec]:
        return [spec for spec in self._cache.values() if spec.available]

    def get_provider_models(self, provider: str) -> list[ModelSpec]:
        return [spec for spec in self._cache.values() if spec.provider == provider]

    async def refresh_provider_status(self) -> None:
        with self._session_factory() as session:
            for provider_name in self._provider_manager.registered_names():
                row = session.query(ProviderRow).filter_by(name=provider_name).one_or_none()
                if row is None:
                    row = ProviderRow(name=provider_name)
                    session.add(row)

                if not self._provider_manager.is_provider_available(provider_name):
                    row.status = "disabled"
                    row.last_error = None
                    row.last_checked_at = datetime.now(timezone.utc)
                    self._event_bus.emit(EventType.PROVIDER_DISABLED, {"provider": provider_name})
                    continue

                provider = self._provider_manager.get_provider(provider_name)
                try:
                    healthy = await provider.health_check()
                    row.last_error = None
                except Exception as exc:
                    healthy = False
                    row.last_error = str(exc)

                self._provider_health[provider_name] = healthy
                row.status = "available" if healthy else "error"
                row.last_checked_at = datetime.now(timezone.utc)

                event_type = EventType.PROVIDER_AVAILABLE if healthy else EventType.PROVIDER_FAILED
                self._event_bus.emit(event_type, {"provider": provider_name, "status": row.status})

            session.commit()

        # Same snapshot-then-swap pattern as reload(): build a whole new
        # cache with updated availability, then swap the reference once.
        # Never mutate individual ModelSpec entries in self._cache_data in
        # place -- readers must always see a consistent snapshot.
        updated_cache: dict[str, ModelSpec] = {
            spec_id: spec.model_copy(update={"available": self._is_available(spec.provider)})
            for spec_id, spec in self._cache_data.items()
        }
        self._cache_data = updated_cache
        self._cache = MappingProxyType(self._cache_data)

    def estimate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        spec = self.get_model(model_id)
        return self._cost_estimator.estimate(
            input_tokens, output_tokens, spec.input_cost, spec.output_cost
        )
