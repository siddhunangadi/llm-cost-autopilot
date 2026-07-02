# LLM Cost Autopilot — Phase 6 Design: Operations Dashboard API

Status: **Approved — frozen as implementation contract**
Date: 2026-07-02

## 1. Purpose & Scope

Phase 6 answers: **can an operator see the whole system's health in one
request?** It is a pure aggregation layer over data Phases 1-5 already
produce — no new business logic, no new persistence, no changes to any
existing subsystem's internals.

```
GET /v1/dashboard/overview?days=7
        │
        ▼
DashboardService.get_overview(window: TimeWindow) -> DashboardOverview
        │
        ▼
asyncio.gather(
    ProviderManager.list_providers(),
    ProviderExecutor.circuit_states(),
    DashboardRepository.get_quality_aggregation(window),
    DashboardRepository.get_cost_trend(window),
    DashboardRepository.get_failover_summary(window),
    LearningService.get_recommendations(),   -- read-only, no recompute
)
        │
        ▼
DashboardOverview  -- pure composition, no calculation in the model itself
```

**In scope:**
- `TimeWindow` (one `days` -> `cutoff` calculation, reused by every
  windowed query)
- `DashboardRepository` (`backend/services/dashboard_repository.py`) —
  three read-only aggregation methods: `get_quality_aggregation`,
  `get_cost_trend`, `get_failover_summary`
- `LearningService.get_recommendations()` — new read-only method
  alongside the existing `refresh_recommendations()`, same ordering,
  never recomputes
- `DashboardService` — orchestrates the six independent reads via
  `asyncio.gather`, then assembles `DashboardOverview` by composition
- `GET /v1/dashboard/overview?days=<int>` (default 7)
- Response models: `ProviderDashboardStatus`, `CostBucket`,
  `FailoverSummary`, `DashboardOverview`

**Explicitly out of scope for Phase 6a** (this phase):
- Any frontend/UI (Phase 6b, separate spec/plan cycle)
- Persisting failover events (failed provider, reason) — the dashboard
  only reports *which* requests failed over (`request_id`s with two
  `RoutingEventRow`s in the window), not why; adding that detail would
  require re-touching Phase 5's `ProviderExecutor`/`ChatService` wiring,
  which is out of scope for a read-only aggregation phase
- Retry-attempt counts — Phase 5's `CircuitBreaker` only tracks final
  success/failure per request by design (the final-outcome-only
  invariant); the dashboard reports those existing counts as-is
- Hourly/sub-day bucketing — daily buckets only
- Caching/memoization of aggregation results — computed fresh per request
- Any endpoint that triggers a write (e.g. no dashboard-triggered
  recommendation refresh)

## 2. Directory Structure

```
backend/
  services/
    dashboard_repository.py   # TimeWindow, QualityAggregation, DashboardRepository
    dashboard_service.py      # DashboardService, DashboardOverview + sub-models
  api/
    routers/
      dashboard.py             # GET /v1/dashboard/overview
    dependencies.py             # + DashboardServiceDep (modify)
    main.py                      # wire DashboardService, mount router (modify)
  learning/
    service.py                    # + LearningService.get_recommendations() (modify)
```

## 3. `TimeWindow`

```python
# backend/services/dashboard_repository.py
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class TimeWindow:
    days: int

    @property
    def cutoff(self) -> datetime:
        return datetime.now(timezone.utc) - timedelta(days=self.days)
```

Every windowed query (`get_cost_trend`, `get_failover_summary`) takes a
`TimeWindow` and filters on `.cutoff` — one calculation, reused
everywhere, so cost and failover data are always computed over
*identical* time ranges even if a future query is added.

## 4. `DashboardRepository`

Owns every new SQL aggregation. Returns raw/internal data shapes, never
API response models — that's `DashboardService`'s job (§6). Read-only:
no method in this class ever calls `session.add`/`session.commit`.

```python
@dataclass(frozen=True)
class QualityAggregation:
    total_verified: int
    average_score: float
    average_confidence: float
    pass_rate: float
    average_queue_delay_ms: float
    average_evaluation_duration_ms: float
    average_total_verification_ms: float
    verification_failure_count: int
    by_model: dict[str, float]
    by_strategy: dict[str, float]
    by_complexity: dict[str, float]


@dataclass(frozen=True)
class CostBucketData:
    date: date
    request_count: int
    total_cost: float


@dataclass(frozen=True)
class FailoverData:
    request_ids: list[str]


class DashboardRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def get_quality_aggregation(self) -> QualityAggregation:
        # Identical query/grouping logic to GET /v1/metrics/quality
        # (backend/api/routers/metrics.py) -- same COMPLETED-status
        # VerificationRow query, same _avg/_group_avg helpers, moved
        # here so both the /v1/metrics/quality endpoint and the
        # dashboard compute quality the same way. Not time-windowed:
        # quality is reported over all verified requests, matching the
        # existing /v1/metrics/quality endpoint's behavior exactly.
        ...

    def get_cost_trend(self, window: TimeWindow) -> list[CostBucketData]:
        # Query ResponseRow where created_at >= window.cutoff and
        # actual_cost is not null, group by date(created_at) (UTC),
        # return one CostBucketData per day present in the data,
        # ordered by date ascending. Days with zero requests in the
        # window are omitted (not zero-filled) -- the dashboard renders
        # only days that actually had traffic.
        ...

    def get_failover_summary(self, window: TimeWindow) -> FailoverData:
        # Query RoutingEventRow where created_at >= window.cutoff,
        # group by request_id, keep only request_ids with count == 2
        # (per Phase 5's invariant: a request has either 1 row -- no
        # failover -- or exactly 2 -- failover occurred -- never more).
        # Return request_ids sorted for deterministic output.
        ...
```

## 5. `LearningService.get_recommendations()`

Add to the existing `LearningService` (`backend/learning/service.py`),
alongside `refresh_recommendations()`:

```python
class LearningService:
    ...  # existing __init__, refresh_recommendations() unchanged

    def get_recommendations(self) -> list[RecommendationRow]:
        with self._session_factory() as session:
            return (
                session.query(RecommendationRow)
                .order_by(
                    RecommendationRow.severity.desc(),
                    RecommendationRow.evidence_confidence.desc(),
                    RecommendationRow.updated_at.desc(),
                )
                .all()
            )
```

**Read-only invariant:** this method never calls `refresh_recommendations()`
and never inserts/updates a `RecommendationRow`. It returns whatever was
last persisted — possibly stale relative to the newest `VerificationRow`
data, exactly like any other dashboard field. Freshening recommendations
remains `GET /v1/learning/recommendations`'s job (Phase 4, unchanged).
Ordering is copied verbatim from `refresh_recommendations()`'s existing
`ORDER BY` (§6 of the Phase 4 spec) so every consumer — the Phase 4
endpoint and this new method — sees identical ordering forever, not two
independently-maintained sort rules that could drift apart.

## 6. `DashboardService` & Response Models

```python
# backend/services/dashboard_service.py
import asyncio
from datetime import date, datetime, timezone

from pydantic import BaseModel

from backend.database.models import RecommendationRow
from backend.learning.generator import RecommendationEvidence, RecommendationSource, Severity
from backend.learning.rules import RuleType
from backend.learning.service import LearningService
from backend.providers.executor import ProviderExecutor
from backend.providers.manager import ProviderManager
from backend.services.dashboard_repository import DashboardRepository, TimeWindow


class ProviderDashboardStatus(BaseModel):
    availability: str            # "available" | "disabled" (from ProviderManager.list_providers())
    circuit_state: str           # "closed" | "open" | "half_open"
    consecutive_failures: int


class CostBucket(BaseModel):
    date: date
    request_count: int
    total_cost: float
    average_cost: float


class FailoverSummary(BaseModel):
    total_failovers: int
    request_ids: list[str]


class QualityMetrics(BaseModel):
    total_verified: int
    average_score: float
    average_confidence: float
    pass_rate: float
    average_queue_delay_ms: float
    average_evaluation_duration_ms: float
    average_total_verification_ms: float
    verification_failure_count: int
    by_model: dict[str, float]
    by_strategy: dict[str, float]
    by_complexity: dict[str, float]


class LearningSummary(BaseModel):
    total_verified: int
    overall_pass_rate: float
    by_model: dict[str, float]
    by_strategy: dict[str, float]
    by_complexity: dict[str, float]


class RecommendationResponse(BaseModel):
    signature: str
    rule_type: RuleType
    subject: str
    text: str
    evidence_confidence: float
    severity: Severity
    evidence: RecommendationEvidence
    status: str
    source: RecommendationSource
    created_at: datetime
    updated_at: datetime


class DashboardOverview(BaseModel):
    generated_at: datetime
    providers: dict[str, ProviderDashboardStatus]
    quality: QualityMetrics
    cost_trend: list[CostBucket]
    failovers: FailoverSummary
    recommendations: list[RecommendationResponse]


class DashboardService:
    def __init__(
        self,
        provider_manager: ProviderManager,
        provider_executor: ProviderExecutor,
        learning_service: LearningService,
        dashboard_repository: DashboardRepository,
    ) -> None:
        self._provider_manager = provider_manager
        self._provider_executor = provider_executor
        self._learning_service = learning_service
        self._dashboard_repository = dashboard_repository

    async def get_overview(self, window: TimeWindow) -> DashboardOverview:
        (
            availability,
            circuits,
            quality_agg,
            cost_buckets,
            failover_data,
            recommendation_rows,
        ) = await asyncio.gather(
            asyncio.to_thread(self._provider_manager.list_providers),
            asyncio.to_thread(self._provider_executor.circuit_states),
            asyncio.to_thread(self._dashboard_repository.get_quality_aggregation),
            asyncio.to_thread(self._dashboard_repository.get_cost_trend, window),
            asyncio.to_thread(self._dashboard_repository.get_failover_summary, window),
            asyncio.to_thread(self._learning_service.get_recommendations),
        )

        return DashboardOverview(
            generated_at=datetime.now(timezone.utc),
            providers=self._merge_provider_status(availability, circuits),
            quality=QualityMetrics(**quality_agg.__dict__),
            cost_trend=[
                CostBucket(
                    date=b.date, request_count=b.request_count, total_cost=b.total_cost,
                    average_cost=b.total_cost / b.request_count if b.request_count else 0.0,
                )
                for b in cost_buckets
            ],
            failovers=FailoverSummary(
                total_failovers=len(failover_data.request_ids),
                request_ids=failover_data.request_ids,
            ),
            recommendations=[self._to_recommendation_response(r) for r in recommendation_rows],
        )

    def _merge_provider_status(
        self, availability: dict[str, str], circuits: dict[str, dict],
    ) -> dict[str, ProviderDashboardStatus]:
        return {
            name: ProviderDashboardStatus(
                availability=status,
                circuit_state=circuits[name]["state"],
                consecutive_failures=circuits[name]["consecutive_failures"],
            )
            for name, status in availability.items()
        }

    def _to_recommendation_response(self, r: RecommendationRow) -> RecommendationResponse:
        return RecommendationResponse(
            signature=r.signature, rule_type=RuleType(r.rule_type), subject=r.subject,
            text=r.recommendation_text, evidence_confidence=r.evidence_confidence,
            severity=Severity(r.severity), evidence=RecommendationEvidence(**r.evidence),
            status=r.status, source=RecommendationSource(r.source),
            created_at=r.created_at, updated_at=r.updated_at,
        )
```

**Pure composition invariant:** `get_overview()` performs no aggregation
of its own — every number in `DashboardOverview` already exists in full
before the model is constructed. The only computation done inline is
`average_cost = total_cost / request_count`, a per-bucket derived field
trivial enough that adding a fourth `DashboardRepository` method just to
compute a division would be over-engineering; everything else is a
direct field mapping.

**`LearningSummary` reuse note:** `LearningSummary` is redefined here
(not imported from `backend.api.routers.learning`) to avoid a
cross-router import; the two models are structurally identical by
design and must be kept in sync if either changes — this mirrors how
`QualityMetrics`/`RecommendationResponse` are also redefined here rather
than imported, keeping `dashboard_service.py` self-contained and not
dependent on other routers' internals.

**`asyncio.to_thread`:** the underlying `DashboardRepository`/
`ProviderManager`/`LearningService` methods are synchronous (SQLAlchemy
session calls, dict lookups) — wrapping each in `asyncio.to_thread`
inside `asyncio.gather` lets them run concurrently on the thread pool
rather than serially, so `get_overview()`'s wall-clock cost approaches
`max()` of the six calls rather than their `sum()`.

## 7. API Endpoint (`backend/api/routers/dashboard.py`)

```python
from fastapi import APIRouter

from backend.api.dependencies import DashboardServiceDep
from backend.services.dashboard_repository import TimeWindow
from backend.services.dashboard_service import DashboardOverview

router = APIRouter()


@router.get("/dashboard/overview", response_model=DashboardOverview)
async def get_dashboard_overview(
    dashboard_service: DashboardServiceDep, days: int = 7,
) -> DashboardOverview:
    return await dashboard_service.get_overview(TimeWindow(days=days))
```

`DashboardServiceDep` (`backend/api/dependencies.py`, modified) follows
the exact same `Depends()`-reading-`app.state` pattern as
`LearningServiceDep`/`ProviderExecutorDep`; `DashboardService` and
`DashboardRepository` are constructed once in `main.py`'s `lifespan`,
alongside everything else, reusing the already-constructed
`provider_manager`, `provider_executor`, `learning_service`, and
`session_factory`.

## 8. Testing

Same discipline as Phases 1-5:

- `TimeWindow.cutoff` tested with a fixed reference time (freeze via
  injected `datetime.now` is unnecessary here since the test only
  asserts the *difference* between `cutoff` and "now" equals `days`,
  which is stable without time injection).
- `DashboardRepository` tested against a real SQLite test database (same
  pattern as `LearningService`/`VerificationService` tests): seeded
  `ResponseRow`s across multiple days assert `get_cost_trend` buckets
  and orders correctly, including the average/zero-request-day-omitted
  behavior; seeded `RoutingEventRow` pairs (2 rows same `request_id`)
  vs. singles assert `get_failover_summary` only returns the paired
  ones; `get_quality_aggregation` output is asserted equal to what
  `GET /v1/metrics/quality` already returns for the same seed data (same
  numbers, different model class).
- `LearningService.get_recommendations()` tested for: returns persisted
  rows without recomputing (seed a `RecommendationRow` directly, assert
  it's returned unchanged even though matching `VerificationRow` data
  that would normally trigger a new finding is also present but never
  refreshed), and ordering matches `refresh_recommendations()`'s.
- `DashboardService.get_overview()` tested with stub/fake
  `ProviderManager`/`ProviderExecutor`/`LearningService`/
  `DashboardRepository` (following the `_FakeProviderManager`-style
  duck-typed test double pattern from Phase 5) asserting: all six inputs
  are correctly merged into `DashboardOverview`, `generated_at` is set,
  `average_cost` is computed correctly (including the zero-request-count
  edge case, which cannot occur given `get_cost_trend` only returns days
  with data, but the ternary guard is tested for defensiveness), and no
  method on any of the four collaborators is ever called more than once
  (asserting the read-only/no-recompute invariant directly).
- `GET /v1/dashboard/overview` endpoint test (`TestClient`, following
  the `test_learning_router.py`/`test_health_endpoint.py` pattern):
  seeds real data across the relevant tables, asserts response shape and
  a `days` query-param default of 7, and that passing `?days=1` narrows
  the cost/failover data appropriately.

## 9. Tooling

No new dependencies — `asyncio` (already used for the app's async
FastAPI handlers) is standard library; `pydantic`, SQLAlchemy remain the
only external dependencies, consistent with Phases 1-5.
