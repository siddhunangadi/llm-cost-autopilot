# Phase 6a Implementation Plan: Operations Dashboard API

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give operators one endpoint (`GET /v1/dashboard/overview`) that aggregates provider/circuit health, quality metrics, cost trend, failover history, and learning recommendations — a pure read-only composition layer over data Phases 1-5 already produce.

**Architecture:** A new `DashboardRepository` owns all SQL aggregation (quality, cost trend, failover detection); a new `LearningService.get_recommendations()` read-only method reuses Phase 4's existing persistence without recomputing; a new `DashboardService` composes six independent reads via `asyncio.gather` into one `DashboardOverview` response, with zero writes anywhere in the path.

**Tech Stack:** Same as Phases 1-5 — Python 3.11+, `uv`, FastAPI, Pydantic v2, SQLAlchemy 2.0, stdlib `asyncio`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-02-phase6-dashboard-design.md` (frozen — implement exactly).

## Global Constraints

- Same `uv`-managed Python 3.11+ project as Phases 1-5; no new dependencies.
- Two batches (Tasks 49-51, Tasks 52-53), one full regression run per batch, one manual end-to-end verification per batch, one commit per batch, then tag `v0.6.0` at the end.
- **Read-only invariant:** `DashboardRepository`, `DashboardService`, and `LearningService.get_recommendations()` never call `session.add`/`session.commit`/`refresh_recommendations()`. No method in this phase mutates the database.
- **Pure composition invariant:** `DashboardService.get_overview()` performs no aggregation of its own beyond the trivial per-bucket `average_cost = total_cost / request_count` division — every other number already exists in full before `DashboardOverview` is constructed.
- `TimeWindow` is the single source of the `days -> cutoff` calculation, reused by every windowed query (`get_cost_trend`, `get_failover_summary`) so both always compute over identical time ranges.
- `get_quality_aggregation()` is NOT time-windowed — it reports over all verified requests, matching `GET /v1/metrics/quality`'s existing (non-windowed) behavior exactly.
- `get_cost_trend()` omits days with zero requests in the window (not zero-filled).
- `get_failover_summary()` only counts `request_id`s with exactly 2 `RoutingEventRow`s (per Phase 5's invariant: 1 row = no failover, 2 rows = failover, never more).
- `LearningService.get_recommendations()` uses the exact same `ORDER BY` as `refresh_recommendations()`: `severity desc, evidence_confidence desc, updated_at desc`.
- Response models (`QualityMetrics`, `LearningSummary`, `RecommendationResponse`) are deliberately redefined in `dashboard_service.py` rather than imported cross-router — this is an intentional, spec-documented tradeoff, not an oversight.
- No placeholder code, no TODOs, no speculative abstractions.

---

## Batch 1: Repository, Learning Read Path, Service Composition (Tasks 49-51)

### Task 49: `TimeWindow` & `DashboardRepository`

**Files:**
- Create: `backend/services/dashboard_repository.py`
- Test: `backend/tests/test_dashboard_repository.py`

**Interfaces:**
- Produces: `TimeWindow(days: int)` with `.cutoff -> datetime` property; `QualityAggregation` (frozen dataclass, fields: `total_verified: int, average_score: float, average_confidence: float, pass_rate: float, average_queue_delay_ms: float, average_evaluation_duration_ms: float, average_total_verification_ms: float, verification_failure_count: int, by_model: dict[str, float], by_strategy: dict[str, float], by_complexity: dict[str, float]`); `CostBucketData` (frozen dataclass, fields: `date: date, request_count: int, total_cost: float`); `FailoverData` (frozen dataclass, field: `request_ids: list[str]`); `DashboardRepository(session_factory: sessionmaker)` with `get_quality_aggregation() -> QualityAggregation`, `get_cost_trend(window: TimeWindow) -> list[CostBucketData]`, `get_failover_summary(window: TimeWindow) -> FailoverData`. Consumed by `DashboardService` (Task 51).

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_dashboard_repository.py
from datetime import datetime, timedelta, timezone

import pytest

from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.database.models import RequestRow, ResponseRow, RoutingEventRow, VerificationRow
from backend.services.dashboard_repository import DashboardRepository, TimeWindow
from backend.verification.status import VerificationStatus


def _make_repository(tmp_path):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)
    return DashboardRepository(session_factory=session_factory), session_factory


def _routing_event(request_id: str, created_at: datetime, model: str = "gpt-4o-mini") -> RoutingEventRow:
    return RoutingEventRow(
        request_id=request_id, complexity="simple", confidence=0.9, selected_model=model,
        selected_strategy="balanced", estimated_cost=0.01, estimated_latency_ms=100,
        reasoning="[]", created_at=created_at,
    )


def test_time_window_cutoff_is_days_ago():
    window = TimeWindow(days=7)
    now = datetime.now(timezone.utc)

    delta = now - window.cutoff

    assert timedelta(days=6, hours=23) < delta < timedelta(days=7, hours=1)


def test_get_quality_aggregation_matches_verification_data(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-1", prompt="hi", strategy="balanced"))
        session.add(VerificationRow(
            request_id="req-1", status=VerificationStatus.COMPLETED.value,
            routing_model="gpt-4o-mini", routing_strategy="balanced", routing_complexity="simple",
            score=0.9, passed=True, confidence=0.8, evaluation_duration_ms=100,
        ))
        session.add(RequestRow(request_id="req-2", prompt="hi", strategy="balanced"))
        session.add(VerificationRow(
            request_id="req-2", status=VerificationStatus.FAILED.value,
            routing_model="gpt-4o-mini", routing_strategy="balanced", routing_complexity="simple",
            error_type="ValidationError", error="bad json",
        ))
        session.commit()

    result = repository.get_quality_aggregation()

    assert result.total_verified == 1
    assert result.verification_failure_count == 1
    assert result.pass_rate == pytest.approx(1.0)
    assert result.by_model["gpt-4o-mini"] == pytest.approx(0.9)


def test_get_cost_trend_buckets_by_day_and_omits_empty_days(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    now = datetime.now(timezone.utc)
    day0 = now
    day2 = now - timedelta(days=2)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-a", prompt="hi", strategy="balanced"))
        session.add(ResponseRow(
            request_id="req-a", response_text="ok", actual_cost=0.10, created_at=day0,
        ))
        session.add(RequestRow(request_id="req-b", prompt="hi", strategy="balanced"))
        session.add(ResponseRow(
            request_id="req-b", response_text="ok", actual_cost=0.20, created_at=day0,
        ))
        session.add(RequestRow(request_id="req-c", prompt="hi", strategy="balanced"))
        session.add(ResponseRow(
            request_id="req-c", response_text="ok", actual_cost=0.05, created_at=day2,
        ))
        session.commit()

    buckets = repository.get_cost_trend(TimeWindow(days=7))

    assert len(buckets) == 2
    assert buckets[0].date < buckets[1].date  # ascending order
    day0_bucket = buckets[1]
    assert day0_bucket.request_count == 2
    assert day0_bucket.total_cost == pytest.approx(0.30)


def test_get_cost_trend_excludes_data_outside_window(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    old = datetime.now(timezone.utc) - timedelta(days=30)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-old", prompt="hi", strategy="balanced"))
        session.add(ResponseRow(
            request_id="req-old", response_text="ok", actual_cost=0.10, created_at=old,
        ))
        session.commit()

    buckets = repository.get_cost_trend(TimeWindow(days=7))

    assert buckets == []


def test_get_cost_trend_excludes_error_responses_with_no_cost(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-err", prompt="hi", strategy="balanced"))
        session.add(ResponseRow(request_id="req-err", error_type="provider_error", error="boom"))
        session.commit()

    buckets = repository.get_cost_trend(TimeWindow(days=7))

    assert buckets == []


def test_get_failover_summary_only_counts_requests_with_two_routing_events(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    now = datetime.now(timezone.utc)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-single", prompt="hi", strategy="balanced"))
        session.add(_routing_event("req-single", now))

        session.add(RequestRow(request_id="req-failover", prompt="hi", strategy="balanced"))
        session.add(_routing_event("req-failover", now, model="gpt-4o-mini"))
        session.add(_routing_event("req-failover", now, model="gpt-4o"))
        session.commit()

    summary = repository.get_failover_summary(TimeWindow(days=7))

    assert summary.request_ids == ["req-failover"]


def test_get_failover_summary_excludes_events_outside_window(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    old = datetime.now(timezone.utc) - timedelta(days=30)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-old-failover", prompt="hi", strategy="balanced"))
        session.add(_routing_event("req-old-failover", old, model="gpt-4o-mini"))
        session.add(_routing_event("req-old-failover", old, model="gpt-4o"))
        session.commit()

    summary = repository.get_failover_summary(TimeWindow(days=7))

    assert summary.request_ids == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest backend/tests/test_dashboard_repository.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.services.dashboard_repository'`

- [ ] **Step 3: Write the implementation**

```python
# backend/services/dashboard_repository.py
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from sqlalchemy.orm import sessionmaker

from backend.database.models import ResponseRow, RoutingEventRow, VerificationRow
from backend.verification.status import VerificationStatus


@dataclass(frozen=True)
class TimeWindow:
    days: int

    @property
    def cutoff(self) -> datetime:
        return datetime.now(timezone.utc) - timedelta(days=self.days)


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


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _group_avg(rows: list[VerificationRow], key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(getattr(row, key), []).append(row.score)
    return {name: _avg(scores) for name, scores in grouped.items()}


class DashboardRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def get_quality_aggregation(self) -> QualityAggregation:
        with self._session_factory() as session:
            completed = (
                session.query(VerificationRow)
                .filter_by(status=VerificationStatus.COMPLETED.value)
                .all()
            )
            failure_count = (
                session.query(VerificationRow)
                .filter_by(status=VerificationStatus.FAILED.value)
                .count()
            )

        queue_delays = [
            (row.started_at - row.created_at).total_seconds() * 1000
            for row in completed
            if row.started_at is not None
        ]
        total_durations = [
            (row.completed_at - row.started_at).total_seconds() * 1000
            for row in completed
            if row.started_at is not None and row.completed_at is not None
        ]
        eval_durations = [
            row.evaluation_duration_ms for row in completed if row.evaluation_duration_ms is not None
        ]

        return QualityAggregation(
            total_verified=len(completed),
            average_score=_avg([row.score for row in completed]),
            average_confidence=_avg(
                [row.confidence for row in completed if row.confidence is not None]
            ),
            pass_rate=_avg([1.0 if row.passed else 0.0 for row in completed]),
            average_queue_delay_ms=_avg(queue_delays),
            average_evaluation_duration_ms=_avg(eval_durations),
            average_total_verification_ms=_avg(total_durations),
            verification_failure_count=failure_count,
            by_model=_group_avg(completed, "routing_model"),
            by_strategy=_group_avg(completed, "routing_strategy"),
            by_complexity=_group_avg(completed, "routing_complexity"),
        )

    def get_cost_trend(self, window: TimeWindow) -> list[CostBucketData]:
        with self._session_factory() as session:
            rows = (
                session.query(ResponseRow)
                .filter(ResponseRow.created_at >= window.cutoff)
                .filter(ResponseRow.actual_cost.isnot(None))
                .all()
            )

        buckets: dict[date, list[float]] = {}
        for row in rows:
            day = row.created_at.date()
            buckets.setdefault(day, []).append(row.actual_cost)

        return [
            CostBucketData(date=day, request_count=len(costs), total_cost=sum(costs))
            for day, costs in sorted(buckets.items())
        ]

    def get_failover_summary(self, window: TimeWindow) -> FailoverData:
        with self._session_factory() as session:
            rows = (
                session.query(RoutingEventRow)
                .filter(RoutingEventRow.created_at >= window.cutoff)
                .all()
            )

        counts: dict[str, int] = {}
        for row in rows:
            counts[row.request_id] = counts.get(row.request_id, 0) + 1

        return FailoverData(
            request_ids=sorted(rid for rid, count in counts.items() if count == 2)
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest backend/tests/test_dashboard_repository.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/services/dashboard_repository.py backend/tests/test_dashboard_repository.py
git commit -m "feat: add TimeWindow and DashboardRepository for dashboard aggregation"
```

### Task 50: `LearningService.get_recommendations()`

**Files:**
- Modify: `backend/learning/service.py`
- Test: `backend/tests/test_learning_service.py` (append)

**Interfaces:**
- Consumes: nothing new (uses `LearningService`'s existing `_session_factory`).
- Produces: `LearningService.get_recommendations() -> list[RecommendationRow]`. Consumed by `DashboardService` (Task 51).

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_learning_service.py`:

```python
def test_get_recommendations_returns_persisted_rows_without_recomputing(tmp_path):
    service, session_factory = _make_service(tmp_path)
    _seed_failing_model(session_factory)
    service.refresh_recommendations()

    with session_factory() as session:
        row = session.query(RecommendationRow).filter_by(
            signature="model_complexity:gpt-4o-mini:medium"
        ).one()
        row.recommendation_text = "manually edited, should not be overwritten"
        session.commit()

    # Seed more failing data that WOULD change the recommendation text if
    # get_recommendations() recomputed -- it must not.
    _seed_failing_model(session_factory, count=20, passed_count=2, prefix="req2")

    results = service.get_recommendations()

    assert len(results) == 1
    assert results[0].recommendation_text == "manually edited, should not be overwritten"


def test_get_recommendations_ordering_matches_refresh(tmp_path):
    service, session_factory = _make_service(tmp_path)
    _seed_failing_model(session_factory)
    refreshed = service.refresh_recommendations()

    results = service.get_recommendations()

    assert [r.signature for r in results] == [r.signature for r in refreshed]


def test_get_recommendations_returns_empty_list_when_none_persisted(tmp_path):
    service, _ = _make_service(tmp_path)

    results = service.get_recommendations()

    assert results == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest backend/tests/test_learning_service.py -v -k get_recommendations`
Expected: FAIL — `AttributeError: 'LearningService' object has no attribute 'get_recommendations'`

- [ ] **Step 3: Modify `backend/learning/service.py`**

Add, after the existing `refresh_recommendations()` method (keep everything else in the file unchanged):

```python
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

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest backend/tests/test_learning_service.py -v`
Expected: PASS (all tests, 3 new)

- [ ] **Step 5: Commit**

```bash
git add backend/learning/service.py backend/tests/test_learning_service.py
git commit -m "feat: add read-only LearningService.get_recommendations()"
```

### Task 51: `DashboardService` & Response Models

**Files:**
- Create: `backend/services/dashboard_service.py`
- Test: `backend/tests/test_dashboard_service.py`

**Interfaces:**
- Consumes: `DashboardRepository`, `TimeWindow`, `QualityAggregation`, `CostBucketData`, `FailoverData` (Task 49); `LearningService.get_recommendations()` (Task 50); `ProviderManager.list_providers() -> dict[str, str]` (Phase 1); `ProviderExecutor.circuit_states() -> dict[str, dict]` (Phase 5).
- Produces: `ProviderDashboardStatus`, `CostBucket`, `FailoverSummary`, `QualityMetrics`, `LearningSummary`, `RecommendationResponse`, `DashboardOverview` (all Pydantic `BaseModel`s), `DashboardService(provider_manager, provider_executor, learning_service, dashboard_repository)` with `async get_overview(window: TimeWindow) -> DashboardOverview`. Consumed by `backend/api/routers/dashboard.py` (Task 52), `main.py` (Task 53).

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_dashboard_service.py
from datetime import date, datetime, timezone

import pytest

from backend.database.models import RecommendationRow
from backend.services.dashboard_repository import CostBucketData, FailoverData, QualityAggregation, TimeWindow
from backend.services.dashboard_service import DashboardService


class _FakeProviderManager:
    def __init__(self):
        self.list_providers_calls = 0

    def list_providers(self):
        self.list_providers_calls += 1
        return {"openai": "available", "anthropic": "disabled", "ollama": "disabled"}


class _FakeProviderExecutor:
    def __init__(self):
        self.circuit_states_calls = 0

    def circuit_states(self):
        self.circuit_states_calls += 1
        return {
            "openai": {"state": "closed", "consecutive_failures": 0, "successes": 5, "failures": 1},
            "anthropic": {"state": "closed", "consecutive_failures": 0, "successes": 0, "failures": 0},
            "ollama": {"state": "closed", "consecutive_failures": 0, "successes": 0, "failures": 0},
        }


class _FakeLearningService:
    def __init__(self, rows):
        self._rows = rows
        self.get_recommendations_calls = 0

    def get_recommendations(self):
        self.get_recommendations_calls += 1
        return self._rows


class _FakeDashboardRepository:
    def __init__(self, quality, cost_buckets, failover_data):
        self._quality = quality
        self._cost_buckets = cost_buckets
        self._failover_data = failover_data
        self.get_quality_aggregation_calls = 0
        self.get_cost_trend_calls = 0
        self.get_failover_summary_calls = 0

    def get_quality_aggregation(self):
        self.get_quality_aggregation_calls += 1
        return self._quality

    def get_cost_trend(self, window):
        self.get_cost_trend_calls += 1
        return self._cost_buckets

    def get_failover_summary(self, window):
        self.get_failover_summary_calls += 1
        return self._failover_data


def _recommendation_row():
    now = datetime.now(timezone.utc)
    return RecommendationRow(
        signature="model_complexity:gpt-4o-mini:medium", rule_type="model_complexity",
        subject="gpt-4o-mini:medium", recommendation_text="text", evidence_confidence=0.6,
        severity="high", evidence={"sample_size": 20, "pass_rate": 0.35, "threshold": 0.6},
        status="new", source="verification", created_at=now, updated_at=now,
    )


def _quality_aggregation():
    return QualityAggregation(
        total_verified=10, average_score=0.8, average_confidence=0.7, pass_rate=0.9,
        average_queue_delay_ms=5.0, average_evaluation_duration_ms=100.0,
        average_total_verification_ms=105.0, verification_failure_count=1,
        by_model={"gpt-4o-mini": 0.8}, by_strategy={"balanced": 0.8}, by_complexity={"simple": 0.8},
    )


async def test_get_overview_merges_all_six_inputs():
    provider_manager = _FakeProviderManager()
    provider_executor = _FakeProviderExecutor()
    learning_service = _FakeLearningService([_recommendation_row()])
    repository = _FakeDashboardRepository(
        quality=_quality_aggregation(),
        cost_buckets=[CostBucketData(date=date(2026, 7, 1), request_count=2, total_cost=0.30)],
        failover_data=FailoverData(request_ids=["req-failover"]),
    )
    service = DashboardService(
        provider_manager=provider_manager, provider_executor=provider_executor,
        learning_service=learning_service, dashboard_repository=repository,
    )

    overview = await service.get_overview(TimeWindow(days=7))

    assert overview.providers["openai"].availability == "available"
    assert overview.providers["openai"].circuit_state == "closed"
    assert overview.providers["openai"].consecutive_failures == 0
    assert overview.quality.total_verified == 10
    assert overview.cost_trend[0].date == date(2026, 7, 1)
    assert overview.cost_trend[0].request_count == 2
    assert overview.cost_trend[0].total_cost == pytest.approx(0.30)
    assert overview.cost_trend[0].average_cost == pytest.approx(0.15)
    assert overview.failovers.total_failovers == 1
    assert overview.failovers.request_ids == ["req-failover"]
    assert len(overview.recommendations) == 1
    assert overview.recommendations[0].signature == "model_complexity:gpt-4o-mini:medium"


async def test_get_overview_sets_generated_at():
    before = datetime.now(timezone.utc)
    service = DashboardService(
        provider_manager=_FakeProviderManager(), provider_executor=_FakeProviderExecutor(),
        learning_service=_FakeLearningService([]),
        dashboard_repository=_FakeDashboardRepository(
            quality=_quality_aggregation(), cost_buckets=[], failover_data=FailoverData(request_ids=[]),
        ),
    )

    overview = await service.get_overview(TimeWindow(days=7))
    after = datetime.now(timezone.utc)

    assert before <= overview.generated_at <= after


async def test_get_overview_computes_average_cost_correctly():
    repository = _FakeDashboardRepository(
        quality=_quality_aggregation(),
        cost_buckets=[CostBucketData(date=date(2026, 7, 1), request_count=4, total_cost=1.00)],
        failover_data=FailoverData(request_ids=[]),
    )
    service = DashboardService(
        provider_manager=_FakeProviderManager(), provider_executor=_FakeProviderExecutor(),
        learning_service=_FakeLearningService([]), dashboard_repository=repository,
    )

    overview = await service.get_overview(TimeWindow(days=7))

    assert overview.cost_trend[0].average_cost == pytest.approx(0.25)


async def test_get_overview_calls_each_collaborator_exactly_once():
    provider_manager = _FakeProviderManager()
    provider_executor = _FakeProviderExecutor()
    learning_service = _FakeLearningService([])
    repository = _FakeDashboardRepository(
        quality=_quality_aggregation(), cost_buckets=[], failover_data=FailoverData(request_ids=[]),
    )
    service = DashboardService(
        provider_manager=provider_manager, provider_executor=provider_executor,
        learning_service=learning_service, dashboard_repository=repository,
    )

    await service.get_overview(TimeWindow(days=7))

    assert provider_manager.list_providers_calls == 1
    assert provider_executor.circuit_states_calls == 1
    assert learning_service.get_recommendations_calls == 1
    assert repository.get_quality_aggregation_calls == 1
    assert repository.get_cost_trend_calls == 1
    assert repository.get_failover_summary_calls == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest backend/tests/test_dashboard_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.services.dashboard_service'`

- [ ] **Step 3: Write the implementation**

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
    availability: str
    circuit_state: str
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
            quality=QualityMetrics(
                total_verified=quality_agg.total_verified,
                average_score=quality_agg.average_score,
                average_confidence=quality_agg.average_confidence,
                pass_rate=quality_agg.pass_rate,
                average_queue_delay_ms=quality_agg.average_queue_delay_ms,
                average_evaluation_duration_ms=quality_agg.average_evaluation_duration_ms,
                average_total_verification_ms=quality_agg.average_total_verification_ms,
                verification_failure_count=quality_agg.verification_failure_count,
                by_model=quality_agg.by_model,
                by_strategy=quality_agg.by_strategy,
                by_complexity=quality_agg.by_complexity,
            ),
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

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest backend/tests/test_dashboard_service.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Batch verification & commit**

Run the full suite:
```bash
.venv/bin/python -m pytest -v
```
Expected: all tests pass (248 existing + new tests from Tasks 49-51; verify against actual collected count rather than assuming an exact number).

```bash
git add backend/services/dashboard_service.py backend/tests/test_dashboard_service.py
git commit -m "feat: add DashboardService composing repository, provider, and learning reads"
```

---

## Batch 2: API Endpoint & Wiring, Tag `v0.6.0` (Tasks 52-53)

### Task 52: `GET /v1/dashboard/overview` Endpoint & Dependency Wiring

**Files:**
- Create: `backend/api/routers/dashboard.py`
- Modify: `backend/api/dependencies.py`
- Test: `backend/tests/test_dashboard_router.py`

**Interfaces:**
- Consumes: `DashboardService`, `DashboardOverview` (Task 51).
- Produces: `DashboardServiceDep`, `GET /v1/dashboard/overview?days=<int>`. Consumed by `main.py` (Task 53).

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_dashboard_router.py
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.database.models import RequestRow, ResponseRow, RoutingEventRow, VerificationRow
from backend.verification.status import VerificationStatus


def _seed(session_factory):
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=30)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-1", prompt="hi", strategy="balanced"))
        session.add(VerificationRow(
            request_id="req-1", status=VerificationStatus.COMPLETED.value,
            routing_model="gpt-4o-mini", routing_strategy="balanced", routing_complexity="simple",
            score=0.9, passed=True, confidence=0.8,
        ))
        session.add(ResponseRow(request_id="req-1", response_text="ok", actual_cost=0.10, created_at=now))

        session.add(RequestRow(request_id="req-old", prompt="hi", strategy="balanced"))
        session.add(ResponseRow(
            request_id="req-old", response_text="ok", actual_cost=99.0, created_at=old,
        ))
        session.commit()


def test_dashboard_overview_returns_expected_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()
    with TestClient(app) as client:
        _seed(app.state.session_factory)

        response = client.get("/v1/dashboard/overview")

        assert response.status_code == 200
        body = response.json()
        assert "generated_at" in body
        assert set(body["providers"].keys()) == {"openai", "anthropic", "ollama"}
        assert body["quality"]["total_verified"] == 1
        assert body["failovers"]["total_failovers"] == 0
        assert body["recommendations"] == []


def test_dashboard_overview_days_param_narrows_cost_trend(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()
    with TestClient(app) as client:
        _seed(app.state.session_factory)

        response_default = client.get("/v1/dashboard/overview")
        response_narrow = client.get("/v1/dashboard/overview?days=1")

        default_dates = {b["date"] for b in response_default.json()["cost_trend"]}
        narrow_dates = {b["date"] for b in response_narrow.json()["cost_trend"]}
        assert len(narrow_dates) <= len(default_dates)
        # the 30-day-old response must never appear within a 1-day window
        assert sum(
            b["total_cost"] for b in response_narrow.json()["cost_trend"]
        ) == pytest.approx(0.10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest backend/tests/test_dashboard_router.py -v`
Expected: FAIL — `404 Not Found` for `/v1/dashboard/overview`

- [ ] **Step 3: Write `backend/api/routers/dashboard.py`**

```python
# backend/api/routers/dashboard.py
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

- [ ] **Step 4: Modify `backend/api/dependencies.py`**

Change the import block:
```python
from backend.chat.service import ChatService
from backend.config.settings import Settings
from backend.events.bus import EventBus
from backend.learning.service import LearningService
from backend.providers.executor import ProviderExecutor
from backend.providers.manager import ProviderManager
from backend.services.model_registry import ModelRegistry
```

To:
```python
from backend.chat.service import ChatService
from backend.config.settings import Settings
from backend.events.bus import EventBus
from backend.learning.service import LearningService
from backend.providers.executor import ProviderExecutor
from backend.providers.manager import ProviderManager
from backend.services.dashboard_service import DashboardService
from backend.services.model_registry import ModelRegistry
```

Add, immediately after `get_provider_executor`:
```python
def get_dashboard_service(request: Request) -> DashboardService:
    return request.app.state.dashboard_service
```

Add, immediately after `ProviderExecutorDep`:
```python
DashboardServiceDep = Annotated[DashboardService, Depends(get_dashboard_service)]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest backend/tests/test_dashboard_router.py -v`
Expected: PASS (2 tests) — this requires Task 53's `main.py` wiring to also be done, since `create_app()` needs `app.state.dashboard_service` set; if this task is executed before Task 53, expect a `500`/`AttributeError` here instead and treat that as confirmation the test correctly detects the missing wiring — proceed to Task 53 to complete the wiring, then re-run this test as part of Task 53's Step 5.

- [ ] **Step 6: Commit**

```bash
git add backend/api/routers/dashboard.py backend/api/dependencies.py backend/tests/test_dashboard_router.py
git commit -m "feat: add GET /v1/dashboard/overview endpoint and DashboardServiceDep"
```

### Task 53: Wiring in `main.py`, Tag `v0.6.0`

**Files:**
- Modify: `backend/api/main.py`

**Interfaces:**
- Consumes: `DashboardRepository` (Task 49), `DashboardService` (Task 51), `dashboard_router` (Task 52).

- [ ] **Step 1: Add imports**

Add to the import block in `backend/api/main.py`, alongside the other `backend.api.routers.*` imports:
```python
from backend.api.routers.dashboard import router as dashboard_router
```

Add, alongside the other `backend.services.*`/`backend.learning.*` imports:
```python
from backend.services.dashboard_repository import DashboardRepository
from backend.services.dashboard_service import DashboardService
```

- [ ] **Step 2: Change `APP_VERSION`**

Change `APP_VERSION = "0.5.0"` to `APP_VERSION = "0.6.0"`.

- [ ] **Step 3: Construct `DashboardRepository`/`DashboardService` in `lifespan`**

In `lifespan`, immediately after the `learning_service = LearningService(...)` block, add:

```python
    dashboard_repository = DashboardRepository(session_factory=session_factory)
    dashboard_service = DashboardService(
        provider_manager=provider_manager,
        provider_executor=provider_executor,
        learning_service=learning_service,
        dashboard_repository=dashboard_repository,
    )
```

- [ ] **Step 4: Add to `app.state` and mount the router**

In the `app.state.*` assignment block, add:
```python
    app.state.dashboard_service = dashboard_service
```

In `create_app()`, add:
```python
    app.include_router(dashboard_router, prefix="/v1")
```

- [ ] **Step 5: Run the full regression suite**

Run: `.venv/bin/python -m pytest -v`
Expected: all tests pass, including the two `test_dashboard_router.py` tests deferred from Task 52 (248 pre-Phase-6 + all new Phase 6 tests; verify against actual collected count rather than assuming an exact number).

- [ ] **Step 6: Manual end-to-end verification**

```bash
.venv/bin/uvicorn backend.api.main:app --reload
```
```bash
curl -s http://localhost:8000/v1/health | python3 -m json.tool   # confirm version "0.6.0"
curl -s http://localhost:8000/v1/dashboard/overview | python3 -m json.tool
```
Expected: `generated_at` present, `providers` has all three known provider names, `quality`/`cost_trend`/`failovers`/`recommendations` all present with sensible empty/zero values on a fresh database (`quality.total_verified: 0`, `cost_trend: []`, `failovers.total_failovers: 0`, `recommendations: []`).

```bash
curl -s "http://localhost:8000/v1/dashboard/overview?days=1" | python3 -m json.tool
```
Expected: same shape, `days` query param accepted without error.

- [ ] **Step 7: Commit and tag**

```bash
git add backend/api/main.py
git commit -m "feat: wire DashboardService into app lifespan, bump to v0.6.0"
git tag v0.6.0
```
