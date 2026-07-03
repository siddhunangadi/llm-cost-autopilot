# Phase 8 Analytics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a historical analytics report (cost, quality, failover, routing distribution, recommendation trends over a rolling window) composed by a new read-only `AnalyticsService`, exposed as `GET /v1/analytics/report` (JSON) and `GET /dashboard/analytics` (single-render HTML page), on top of the existing Phase 6/6b operations dashboard and Phase 7 optimization engine.

**Architecture:** Extends `backend/services/dashboard_repository.py` with three new query methods (`get_failover_trend`, `get_routing_distribution`, `get_recommendation_trend`), each following the exact pattern already used by `get_cost_trend`/`get_quality_trend` (fetch filtered rows with SQLAlchemy, then group in Python dicts, return sorted dataclasses — do not introduce raw SQL `GROUP BY`; match the codebase's established convention). A new `backend/services/analytics_service.py` composes these plus the two existing trend methods into one `AnalyticsReport` Pydantic model, mirroring `DashboardService`'s `asyncio.gather`/`asyncio.to_thread` composition style. Two new router functions in a new `backend/api/routers/analytics.py` expose the same service as JSON and HTML — no HTMX polling on the HTML page. No new database tables, no schema migration.

**Tech Stack:** Python 3.13, SQLAlchemy 2.0 (ORM, session-per-call), Pydantic v2, FastAPI, Jinja2, Chart.js (already vendored at `backend/static/js/chart.min.js`), pytest, `fastapi.testclient.TestClient`.

## Global Constraints

- Frozen design doc: `docs/superpowers/specs/2026-07-03-phase8-analytics-design.md` — every task below implements one section of it; do not deviate without checking that doc first.
- `AnalyticsService` never writes — no calls to `LearningService` methods that refresh/regenerate recommendations, no writes to routing, no writes to the database. Read-only, same invariant as `DashboardService` (spec §6, §11).
- No new database tables or migrations (spec §2, §11).
- `/dashboard/analytics` renders once per request — no `hx-trigger="every ..."` polling attributes anywhere in its template or fragments (spec §8, §11). (Contrast: `/dashboard` intentionally does poll via `fragments/overview.html`'s `hx-get`/`hx-trigger="every 15s"` — do not copy that pattern here.)
- `get_failover_trend` must reuse the identical two-routing-events-per-request failover predicate that `get_failover_events`/`get_failover_summary` already use (a `request_id` with exactly 2 `RoutingEventRow`s), just grouped by day and counted instead of listed (spec §5). Do not build it on top of `get_failover_events` — it is an independent query over the same underlying data.
- `get_recommendation_trend`'s `open_count` is `status == "new"` (the default status set by `LearningService`, before a human sets `"acknowledged"` or anything else) — count of recommendations generated that day whose *current* status is still `"new"` as of report generation time. This is a stated simplification, not a true point-in-time daily count (spec §5) — add a one-line code comment noting this.
- `TimeWindow` (existing dataclass in `backend/services/dashboard_repository.py`) is reused as-is for all new queries — no new window abstraction.
- Repository query methods are synchronous (existing pattern: `session_factory()` context manager, no `async def`); the service wraps each in `asyncio.to_thread`, matching `DashboardService`.
- Version bump to `0.8.0` happens only in Batch 2, Task 8 (after everything else is verified) — do not bump early.

---

## Batch 1 — Repository, Service, Domain Models

### Task 1: Repository — `get_failover_trend`

**Files:**
- Modify: `backend/services/dashboard_repository.py`
- Test: `backend/tests/test_dashboard_repository.py`

**Interfaces:**
- Consumes: existing `TimeWindow`, `RoutingEventRow` (from `backend.database.models`), existing `_session_factory`.
- Produces: `FailoverTrendBucket` dataclass and `DashboardRepository.get_failover_trend(window: TimeWindow) -> list[FailoverTrendBucket]`, both consumed by Task 4 (`AnalyticsService`).

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/test_dashboard_repository.py` (this file already imports `datetime, timedelta, timezone`, `RequestRow, ResponseRow, RoutingEventRow, VerificationRow`, `DashboardRepository, TimeWindow`, and already defines `_make_repository` and `_routing_event` helpers — reuse them, do not redefine):

```python
def test_get_failover_trend_counts_failovers_per_day(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    now = datetime.now(timezone.utc)
    with session_factory() as session:
        # req-failover-1: two routing events same day -> 1 failover today
        session.add(RequestRow(request_id="req-failover-1", prompt="hi", strategy="balanced"))
        session.add(_routing_event("req-failover-1", now, model="gpt-4o-mini"))
        session.add(_routing_event("req-failover-1", now + timedelta(seconds=1), model="gpt-4o"))

        # req-single: only one routing event -> not a failover
        session.add(RequestRow(request_id="req-single", prompt="hi", strategy="balanced"))
        session.add(_routing_event("req-single", now))

        # req-failover-2: two routing events yesterday -> 1 failover yesterday
        yesterday = now - timedelta(days=1)
        session.add(RequestRow(request_id="req-failover-2", prompt="hi", strategy="balanced"))
        session.add(_routing_event("req-failover-2", yesterday, model="gpt-4o-mini"))
        session.add(_routing_event("req-failover-2", yesterday + timedelta(seconds=1), model="gpt-4o"))
        session.commit()

    buckets = repository.get_failover_trend(TimeWindow(days=7))

    by_date = {b.date: b.failover_count for b in buckets}
    assert by_date[now.date()] == 1
    assert by_date[yesterday.date()] == 1


def test_get_failover_trend_excludes_events_outside_window(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=30)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-old", prompt="hi", strategy="balanced"))
        session.add(_routing_event("req-old", old, model="gpt-4o-mini"))
        session.add(_routing_event("req-old", old + timedelta(seconds=1), model="gpt-4o"))
        session.commit()

    buckets = repository.get_failover_trend(TimeWindow(days=7))

    assert buckets == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest backend/tests/test_dashboard_repository.py -k get_failover_trend -v`
Expected: FAIL with `AttributeError: 'DashboardRepository' object has no attribute 'get_failover_trend'`

- [ ] **Step 3: Implement**

In `backend/services/dashboard_repository.py`, add the dataclass near the other `*TrendBucket`/`*Data` dataclasses (after `FailoverEvent`):

```python
@dataclass(frozen=True)
class FailoverTrendBucket:
    date: date
    failover_count: int
```

Add the method to `DashboardRepository`, directly below `get_failover_events`:

```python
    def get_failover_trend(self, window: TimeWindow) -> list[FailoverTrendBucket]:
        with self._session_factory() as session:
            rows = (
                session.query(RoutingEventRow)
                .filter(RoutingEventRow.created_at >= window.cutoff)
                .order_by(RoutingEventRow.request_id, RoutingEventRow.created_at)
                .all()
            )

        grouped: dict[str, list[RoutingEventRow]] = {}
        for row in rows:
            grouped.setdefault(row.request_id, []).append(row)

        # Same failover predicate as get_failover_events: exactly two
        # routing events for a request_id. Bucketed by the day of the
        # second (failover) event and counted, not listed.
        buckets: dict[date, int] = {}
        for group in grouped.values():
            if len(group) == 2:
                day = group[1].created_at.date()
                buckets[day] = buckets.get(day, 0) + 1

        return [
            FailoverTrendBucket(date=day, failover_count=count)
            for day, count in sorted(buckets.items())
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest backend/tests/test_dashboard_repository.py -k get_failover_trend -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `pytest backend/tests/ -q`
Expected: all tests pass, no regressions.

- [ ] **Step 6: Commit**

```bash
git add backend/services/dashboard_repository.py backend/tests/test_dashboard_repository.py
git commit -m "feat: add get_failover_trend to DashboardRepository"
```

---

### Task 2: Repository — `get_routing_distribution`

**Files:**
- Modify: `backend/services/dashboard_repository.py`
- Test: `backend/tests/test_dashboard_repository.py`

**Interfaces:**
- Consumes: existing `TimeWindow`, `RoutingEventRow`.
- Produces: `RoutingDistributionBucket` dataclass and `DashboardRepository.get_routing_distribution(window: TimeWindow) -> list[RoutingDistributionBucket]`, consumed by Task 4.

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/test_dashboard_repository.py`:

```python
def test_get_routing_distribution_groups_by_day_and_model(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    now = datetime.now(timezone.utc)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-1", prompt="hi", strategy="balanced"))
        session.add(_routing_event("req-1", now, model="gpt-4o-mini"))
        session.add(RequestRow(request_id="req-2", prompt="hi", strategy="balanced"))
        session.add(_routing_event("req-2", now, model="gpt-4o-mini"))
        session.add(RequestRow(request_id="req-3", prompt="hi", strategy="balanced"))
        session.add(_routing_event("req-3", now, model="gpt-4o"))
        session.commit()

    buckets = repository.get_routing_distribution(TimeWindow(days=7))

    counts = {(b.date, b.model): b.request_count for b in buckets}
    assert counts[(now.date(), "gpt-4o-mini")] == 2
    assert counts[(now.date(), "gpt-4o")] == 1


def test_get_routing_distribution_excludes_events_outside_window(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=30)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-old", prompt="hi", strategy="balanced"))
        session.add(_routing_event("req-old", old, model="gpt-4o-mini"))
        session.commit()

    buckets = repository.get_routing_distribution(TimeWindow(days=7))

    assert buckets == []
```

Note: `get_routing_distribution` counts every routing event (including the second event of a failover pair), matching how `get_cost_by_model` and the raw `RoutingEventRow` table already represent "which model was selected" — no failover de-duplication is applied here, that's what `get_failover_trend`/`get_failover_events` are for.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest backend/tests/test_dashboard_repository.py -k get_routing_distribution -v`
Expected: FAIL with `AttributeError: 'DashboardRepository' object has no attribute 'get_routing_distribution'`

- [ ] **Step 3: Implement**

Add the dataclass in `backend/services/dashboard_repository.py`, next to `FailoverTrendBucket`:

```python
@dataclass(frozen=True)
class RoutingDistributionBucket:
    date: date
    model: str
    request_count: int
```

Add the method to `DashboardRepository`, directly below `get_failover_trend`:

```python
    def get_routing_distribution(self, window: TimeWindow) -> list[RoutingDistributionBucket]:
        with self._session_factory() as session:
            rows = (
                session.query(RoutingEventRow)
                .filter(RoutingEventRow.created_at >= window.cutoff)
                .all()
            )

        buckets: dict[tuple[date, str], int] = {}
        for row in rows:
            key = (row.created_at.date(), row.selected_model)
            buckets[key] = buckets.get(key, 0) + 1

        return [
            RoutingDistributionBucket(date=day, model=model, request_count=count)
            for (day, model), count in sorted(buckets.items())
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest backend/tests/test_dashboard_repository.py -k get_routing_distribution -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `pytest backend/tests/ -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add backend/services/dashboard_repository.py backend/tests/test_dashboard_repository.py
git commit -m "feat: add get_routing_distribution to DashboardRepository"
```

---

### Task 3: Repository — `get_recommendation_trend`

**Files:**
- Modify: `backend/services/dashboard_repository.py`
- Test: `backend/tests/test_dashboard_repository.py`

**Interfaces:**
- Consumes: `RecommendationRow` (needs a new import from `backend.database.models` at the top of `dashboard_repository.py`, added alongside the existing `RequestRow, ResponseRow, RoutingEventRow, VerificationRow` import), existing `TimeWindow`.
- Produces: `RecommendationTrendBucket` dataclass and `DashboardRepository.get_recommendation_trend(window: TimeWindow) -> list[RecommendationTrendBucket]`, consumed by Task 4.

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/test_dashboard_repository.py`. First add `RecommendationRow` to the existing model import at the top of the file (change `from backend.database.models import RequestRow, ResponseRow, RoutingEventRow, VerificationRow` to also include `RecommendationRow`):

```python
def test_get_recommendation_trend_counts_generated_and_open_per_day(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    now = datetime.now(timezone.utc)
    with session_factory() as session:
        session.add(RecommendationRow(
            signature="sig-1", rule_type="model_complexity", subject="gpt-4o-mini",
            recommendation_text="text", evidence_confidence=0.9, severity="low",
            evidence={}, status="new", source="rule_based", created_at=now,
        ))
        session.add(RecommendationRow(
            signature="sig-2", rule_type="model_complexity", subject="gpt-4o",
            recommendation_text="text", evidence_confidence=0.9, severity="low",
            evidence={}, status="acknowledged", source="rule_based", created_at=now,
        ))
        session.commit()

    buckets = repository.get_recommendation_trend(TimeWindow(days=7))

    assert len(buckets) == 1
    assert buckets[0].date == now.date()
    assert buckets[0].generated_count == 2
    assert buckets[0].open_count == 1  # only sig-1 is still status="new"


def test_get_recommendation_trend_excludes_recommendations_outside_window(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=30)
    with session_factory() as session:
        session.add(RecommendationRow(
            signature="sig-old", rule_type="model_complexity", subject="gpt-4o-mini",
            recommendation_text="text", evidence_confidence=0.9, severity="low",
            evidence={}, status="new", source="rule_based", created_at=old,
        ))
        session.commit()

    buckets = repository.get_recommendation_trend(TimeWindow(days=7))

    assert buckets == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest backend/tests/test_dashboard_repository.py -k get_recommendation_trend -v`
Expected: FAIL with `AttributeError: 'DashboardRepository' object has no attribute 'get_recommendation_trend'`

- [ ] **Step 3: Implement**

In `backend/services/dashboard_repository.py`, update the model import line to:

```python
from backend.database.models import RecommendationRow, RequestRow, ResponseRow, RoutingEventRow, VerificationRow
```

Add the dataclass next to `RoutingDistributionBucket`:

```python
@dataclass(frozen=True)
class RecommendationTrendBucket:
    date: date
    generated_count: int
    open_count: int
```

Add the method to `DashboardRepository`, directly below `get_routing_distribution`:

```python
    def get_recommendation_trend(self, window: TimeWindow) -> list[RecommendationTrendBucket]:
        with self._session_factory() as session:
            rows = (
                session.query(RecommendationRow)
                .filter(RecommendationRow.created_at >= window.cutoff)
                .all()
            )

        buckets: dict[date, list[RecommendationRow]] = {}
        for row in rows:
            day = row.created_at.date()
            buckets.setdefault(day, []).append(row)

        # open_count is a simplification: recommendations generated that
        # day whose *current* status is still "new" as of report
        # generation time, not a true point-in-time daily open count.
        return [
            RecommendationTrendBucket(
                date=day,
                generated_count=len(group),
                open_count=sum(1 for r in group if r.status == "new"),
            )
            for day, group in sorted(buckets.items())
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest backend/tests/test_dashboard_repository.py -k get_recommendation_trend -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `pytest backend/tests/ -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add backend/services/dashboard_repository.py backend/tests/test_dashboard_repository.py
git commit -m "feat: add get_recommendation_trend to DashboardRepository"
```

---

### Task 4: `AnalyticsService` and `AnalyticsReport`

**Files:**
- Create: `backend/services/analytics_service.py`
- Test: `backend/tests/test_analytics_service.py`

**Interfaces:**
- Consumes: `DashboardRepository` (specifically `get_cost_trend`, `get_quality_trend`, `get_failover_trend`, `get_routing_distribution`, `get_recommendation_trend`, and the `TimeWindow` dataclass) from `backend.services.dashboard_repository`.
- Produces: `AnalyticsReport`, `CostTrendPoint`, `QualityTrendPoint`, `FailoverTrendPoint`, `RoutingDistributionPoint`, `RecommendationTrendPoint` (all Pydantic `BaseModel`), and `AnalyticsService.get_report(window: TimeWindow) -> AnalyticsReport`. Consumed by Task 5 (router) and Task 6 (dependency wiring).

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_analytics_service.py`:

```python
from datetime import datetime, timedelta, timezone

import pytest

from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.database.models import RecommendationRow, RequestRow, ResponseRow, RoutingEventRow, VerificationRow
from backend.services.analytics_service import AnalyticsService
from backend.services.dashboard_repository import DashboardRepository, TimeWindow
from backend.verification.status import VerificationStatus


def _make_service(tmp_path):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)
    repository = DashboardRepository(session_factory=session_factory)
    return AnalyticsService(dashboard_repository=repository), session_factory


@pytest.mark.asyncio
async def test_get_report_composes_all_five_trends(tmp_path):
    service, session_factory = _make_service(tmp_path)
    now = datetime.now(timezone.utc)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-1", prompt="hi", strategy="balanced"))
        session.add(RoutingEventRow(
            request_id="req-1", complexity="simple", confidence=0.9, selected_model="gpt-4o-mini",
            selected_strategy="balanced", estimated_cost=0.01, estimated_latency_ms=100,
            reasoning="[]", created_at=now,
        ))
        session.add(ResponseRow(request_id="req-1", response_text="ok", actual_cost=0.10, created_at=now))
        session.add(VerificationRow(
            request_id="req-1", status=VerificationStatus.COMPLETED.value,
            routing_model="gpt-4o-mini", routing_strategy="balanced", routing_complexity="simple",
            score=0.9, passed=True, confidence=0.8, created_at=now,
        ))
        session.add(RecommendationRow(
            signature="sig-1", rule_type="model_complexity", subject="gpt-4o-mini",
            recommendation_text="text", evidence_confidence=0.9, severity="low",
            evidence={}, status="new", source="rule_based", created_at=now,
        ))
        session.commit()

    report = await service.get_report(TimeWindow(days=7))

    assert report.window_days == 7
    assert len(report.cost_trend) == 1
    assert report.cost_trend[0].total_cost == pytest.approx(0.10)
    assert len(report.quality_trend) == 1
    assert report.quality_trend[0].pass_rate == pytest.approx(1.0)
    assert report.failover_trend == []
    assert len(report.routing_distribution) == 1
    assert report.routing_distribution[0].model == "gpt-4o-mini"
    assert report.routing_distribution[0].request_count == 1
    assert len(report.recommendation_trend) == 1
    assert report.recommendation_trend[0].generated_count == 1
    assert report.recommendation_trend[0].open_count == 1


@pytest.mark.asyncio
async def test_get_report_has_no_write_capable_dependencies(tmp_path):
    service, session_factory = _make_service(tmp_path)

    # AnalyticsService must hold only a DashboardRepository reference --
    # no LearningService/ProviderManager/ProviderExecutor dependency that
    # could be used to write, unlike DashboardService.
    assert not hasattr(service, "_learning_service")
    assert not hasattr(service, "_provider_manager")
    assert not hasattr(service, "_provider_executor")

    report = await service.get_report(TimeWindow(days=7))
    assert report.window_days == 7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest backend/tests/test_analytics_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.services.analytics_service'`

- [ ] **Step 3: Implement**

Create `backend/services/analytics_service.py`:

```python
import asyncio
from datetime import date, datetime, timezone

from pydantic import BaseModel

from backend.services.dashboard_repository import DashboardRepository, TimeWindow


class CostTrendPoint(BaseModel):
    date: date
    request_count: int
    total_cost: float
    average_cost: float


class QualityTrendPoint(BaseModel):
    date: date
    average_score: float
    pass_rate: float


class FailoverTrendPoint(BaseModel):
    date: date
    failover_count: int


class RoutingDistributionPoint(BaseModel):
    date: date
    model: str
    request_count: int


class RecommendationTrendPoint(BaseModel):
    date: date
    generated_count: int
    open_count: int


class AnalyticsReport(BaseModel):
    generated_at: datetime
    window_days: int
    cost_trend: list[CostTrendPoint]
    quality_trend: list[QualityTrendPoint]
    failover_trend: list[FailoverTrendPoint]
    routing_distribution: list[RoutingDistributionPoint]
    recommendation_trend: list[RecommendationTrendPoint]


class AnalyticsService:
    """Read-only historical analytics. Never writes: no recommendation
    refresh, no routing/learning mutation -- only reads DashboardRepository."""

    def __init__(self, dashboard_repository: DashboardRepository) -> None:
        self._dashboard_repository = dashboard_repository

    async def get_report(self, window: TimeWindow) -> AnalyticsReport:
        (
            cost_buckets, quality_buckets, failover_buckets,
            routing_buckets, recommendation_buckets,
        ) = await asyncio.gather(
            asyncio.to_thread(self._dashboard_repository.get_cost_trend, window),
            asyncio.to_thread(self._dashboard_repository.get_quality_trend, window),
            asyncio.to_thread(self._dashboard_repository.get_failover_trend, window),
            asyncio.to_thread(self._dashboard_repository.get_routing_distribution, window),
            asyncio.to_thread(self._dashboard_repository.get_recommendation_trend, window),
        )

        return AnalyticsReport(
            generated_at=datetime.now(timezone.utc),
            window_days=window.days,
            cost_trend=[
                CostTrendPoint(
                    date=b.date, request_count=b.request_count, total_cost=b.total_cost,
                    average_cost=b.total_cost / b.request_count if b.request_count else 0.0,
                )
                for b in cost_buckets
            ],
            quality_trend=[
                QualityTrendPoint(date=b.date, average_score=b.average_score, pass_rate=b.pass_rate)
                for b in quality_buckets
            ],
            failover_trend=[
                FailoverTrendPoint(date=b.date, failover_count=b.failover_count)
                for b in failover_buckets
            ],
            routing_distribution=[
                RoutingDistributionPoint(date=b.date, model=b.model, request_count=b.request_count)
                for b in routing_buckets
            ],
            recommendation_trend=[
                RecommendationTrendPoint(
                    date=b.date, generated_count=b.generated_count, open_count=b.open_count,
                )
                for b in recommendation_buckets
            ],
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest backend/tests/test_analytics_service.py -v`
Expected: PASS (2 tests). If `pytest-asyncio` is not configured project-wide, check how `backend/tests/test_chat_service.py` marks its async tests (it uses `@pytest.mark.asyncio` per the project's existing async test convention — reuse the identical marker style found there; no new test config needed).

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `pytest backend/tests/ -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add backend/services/analytics_service.py backend/tests/test_analytics_service.py
git commit -m "feat: add AnalyticsService composing five historical trends into AnalyticsReport"
```

---

### Task 5: Batch 1 regression checkpoint

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `pytest backend/tests/ -q`
Expected: all tests pass, including the 6 new repository tests (Tasks 1–3) and 2 new service tests (Task 4), no regressions in existing Phase 1–7 tests.

- [ ] **Step 2: Confirm no accidental scope creep**

Run: `git diff main --stat` (or `git log --oneline main..HEAD` if working on a branch) and confirm only these files changed since Batch 1 started: `backend/services/dashboard_repository.py`, `backend/services/analytics_service.py`, `backend/tests/test_dashboard_repository.py`, `backend/tests/test_analytics_service.py`. No router, template, or `main.py` changes yet — those belong to Batch 2.

---

## Batch 2 — REST API, HTML Page, Wiring, Verification, Release

### Task 6: Dependency wiring — `AnalyticsService` in `main.py`/`dependencies.py`

**Files:**
- Modify: `backend/api/dependencies.py`
- Modify: `backend/api/main.py`

**Interfaces:**
- Consumes: `AnalyticsService` from Task 4, existing `DashboardRepository` instance already constructed in `backend/api/main.py`'s `lifespan`.
- Produces: `AnalyticsServiceDep` (an `Annotated[AnalyticsService, Depends(get_analytics_service)]`), consumed by Task 7 (router).

- [ ] **Step 1: Modify `backend/api/dependencies.py`**

Add the import at the top (alongside the existing `DashboardService` import):

```python
from backend.services.analytics_service import AnalyticsService
```

Add the getter function (below `get_dashboard_service`):

```python
def get_analytics_service(request: Request) -> AnalyticsService:
    return request.app.state.analytics_service
```

Add the `Annotated` alias (below `DashboardServiceDep`):

```python
AnalyticsServiceDep = Annotated[AnalyticsService, Depends(get_analytics_service)]
```

- [ ] **Step 2: Modify `backend/api/main.py`**

Add the import (alongside the existing `DashboardRepository`/`DashboardService` imports):

```python
from backend.services.analytics_service import AnalyticsService
```

In `lifespan`, immediately after the existing `dashboard_service = DashboardService(...)` block, add:

```python
    analytics_service = AnalyticsService(dashboard_repository=dashboard_repository)
```

In the same function, add to `app.state` assignments (alongside `app.state.dashboard_service = dashboard_service`):

```python
    app.state.analytics_service = analytics_service
```

- [ ] **Step 3: Verify the app still boots**

Run: `pytest backend/tests/ -q`
Expected: all existing tests still pass (this task adds no new tests of its own — it's pure wiring, verified indirectly by Task 7's router tests, which will fail at import/construction time if this wiring is wrong).

- [ ] **Step 4: Commit**

```bash
git add backend/api/dependencies.py backend/api/main.py
git commit -m "feat: wire AnalyticsService into app state and dependency injection"
```

---

### Task 7: `GET /v1/analytics/report` and `GET /dashboard/analytics` router + template

**Files:**
- Create: `backend/api/routers/analytics.py`
- Modify: `backend/api/main.py` (register the new router)
- Create: `backend/templates/analytics.html`
- Modify: `backend/templates/dashboard.html` (add nav link)
- Test: `backend/tests/test_analytics_router.py`

**Interfaces:**
- Consumes: `AnalyticsServiceDep` (Task 6), `AnalyticsReport` (Task 4), `TimeWindow` (existing), `AppVersionDep` (existing, from `backend.api.dependencies`), `TEMPLATES_DIR` (existing, from `backend.api.paths`).
- Produces: `router` (FastAPI `APIRouter`) exported from `backend/api/routers/analytics.py`, registered in `create_app()`.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_analytics_router.py`:

```python
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.database.models import RecommendationRow, RequestRow, ResponseRow, RoutingEventRow, VerificationRow
from backend.verification.status import VerificationStatus


def _seed(session_factory):
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=30)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-1", prompt="hi", strategy="balanced"))
        session.add(RoutingEventRow(
            request_id="req-1", complexity="simple", confidence=0.9, selected_model="gpt-4o-mini",
            selected_strategy="balanced", estimated_cost=0.01, estimated_latency_ms=100,
            reasoning="[]", created_at=now,
        ))
        session.add(ResponseRow(request_id="req-1", response_text="ok", actual_cost=0.10, created_at=now))
        session.add(VerificationRow(
            request_id="req-1", status=VerificationStatus.COMPLETED.value,
            routing_model="gpt-4o-mini", routing_strategy="balanced", routing_complexity="simple",
            score=0.9, passed=True, confidence=0.8, created_at=now,
        ))
        session.add(RecommendationRow(
            signature="sig-1", rule_type="model_complexity", subject="gpt-4o-mini",
            recommendation_text="text", evidence_confidence=0.9, severity="low",
            evidence={}, status="new", source="rule_based", created_at=now,
        ))

        session.add(RequestRow(request_id="req-old", prompt="hi", strategy="balanced"))
        session.add(ResponseRow(
            request_id="req-old", response_text="ok", actual_cost=99.0, created_at=old,
        ))
        session.commit()


def test_analytics_report_returns_expected_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()
    with TestClient(app) as client:
        _seed(app.state.session_factory)

        response = client.get("/v1/analytics/report")

        assert response.status_code == 200
        body = response.json()
        assert body["window_days"] == 30
        assert len(body["cost_trend"]) == 1
        assert len(body["routing_distribution"]) == 1
        assert len(body["recommendation_trend"]) == 1


def test_analytics_report_days_param_narrows_window(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()
    with TestClient(app) as client:
        _seed(app.state.session_factory)

        response = client.get("/v1/analytics/report?days=1")

        assert response.status_code == 200
        assert response.json()["window_days"] == 1
        # the 30-day-old response must never appear within a 1-day window
        assert sum(b["total_cost"] for b in response.json()["cost_trend"]) == 0.0


def test_analytics_page_renders_200_with_no_polling_attributes(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()
    with TestClient(app) as client:
        _seed(app.state.session_factory)

        response = client.get("/dashboard/analytics")

        assert response.status_code == 200
        assert "hx-trigger" not in response.text
        assert 'id="cost-trend-chart"' in response.text
        assert 'id="routing-distribution-chart"' in response.text
        assert 'id="failover-trend-chart"' in response.text
        assert 'id="recommendation-trend-chart"' in response.text


def test_analytics_page_handles_empty_database(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/dashboard/analytics")

        assert response.status_code == 200
        assert "No cost data yet" in response.text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest backend/tests/test_analytics_router.py -v`
Expected: FAIL with `404 Not Found` for both routes (router not yet registered) or import errors.

- [ ] **Step 3: Implement the router**

Create `backend/api/routers/analytics.py`:

```python
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from backend.api.dependencies import AnalyticsServiceDep, AppVersionDep
from backend.api.paths import TEMPLATES_DIR
from backend.services.analytics_service import AnalyticsReport
from backend.services.dashboard_repository import TimeWindow

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


@router.get("/v1/analytics/report", response_model=AnalyticsReport)
async def get_analytics_report(
    analytics_service: AnalyticsServiceDep, days: int = 30,
) -> AnalyticsReport:
    return await analytics_service.get_report(TimeWindow(days=days))


@router.get("/dashboard/analytics")
async def analytics_page(
    request: Request, analytics_service: AnalyticsServiceDep, app_version: AppVersionDep, days: int = 30,
):
    report = await analytics_service.get_report(TimeWindow(days=days))
    return templates.TemplateResponse(request, "analytics.html", {
        "report": report,
        "app_version": app_version,
        "now": _now_str(),
    })
```

- [ ] **Step 4: Register the router in `backend/api/main.py`**

Add the import (alongside the other router imports, keeping alphabetical order with the existing block):

```python
from backend.api.routers.analytics import router as analytics_router
```

In `create_app()`, add registration (alongside the other `app.include_router(...)` calls — this router mixes a `/v1`-prefixed route and a `/dashboard`-prefixed route defined with full paths inside the router itself, so register it **without** a prefix, same as `dashboard_ui_router`):

```python
    app.include_router(analytics_router)
```

- [ ] **Step 5: Create the template**

Create `backend/templates/analytics.html`:

```html
{% extends "base.html" %}
{% block title %}LLM Cost Autopilot — Analytics{% endblock %}
{% block content %}
<main class="dashboard">
  <nav class="dashboard-nav">
    <a href="/dashboard">Operations Dashboard</a> | <a href="/dashboard/analytics">Analytics</a>
  </nav>
  <p class="last-updated">Report window: last {{ report.window_days }} days. Generated at {{ report.generated_at }}. <a href="/dashboard/analytics?days={{ report.window_days }}">Refresh</a></p>

  <section class="dashboard-section" id="section-cost-trend">
    <h2>Cost Trend</h2>
    {% if not report.cost_trend %}
    <p class="empty-state">No cost data yet.</p>
    {% else %}
    <canvas id="cost-trend-chart"></canvas>
    <script>
      new Chart(document.getElementById('cost-trend-chart'), {
        type: 'line',
        data: {
          labels: {{ report.cost_trend | map(attribute='date') | map('string') | list | tojson }},
          datasets: [{ label: 'Total cost/day', data: {{ report.cost_trend | map(attribute='total_cost') | list | tojson }} }],
        },
      });
    </script>
    {% endif %}
  </section>

  <section class="dashboard-section" id="section-quality-trend">
    <h2>Quality Trend</h2>
    {% if not report.quality_trend %}
    <p class="empty-state">No verification results available.</p>
    {% else %}
    <canvas id="quality-trend-chart"></canvas>
    <script>
      new Chart(document.getElementById('quality-trend-chart'), {
        type: 'line',
        data: {
          labels: {{ report.quality_trend | map(attribute='date') | map('string') | list | tojson }},
          datasets: [
            { label: 'Pass rate', data: {{ report.quality_trend | map(attribute='pass_rate') | list | tojson }} },
            { label: 'Average score', data: {{ report.quality_trend | map(attribute='average_score') | list | tojson }} },
          ],
        },
      });
    </script>
    {% endif %}
  </section>

  <section class="dashboard-section" id="section-routing-distribution">
    <h2>Routing Distribution</h2>
    {% if not report.routing_distribution %}
    <p class="empty-state">No routing data yet.</p>
    {% else %}
    <canvas id="routing-distribution-chart"></canvas>
    <script>
      new Chart(document.getElementById('routing-distribution-chart'), {
        type: 'bar',
        data: {
          labels: {{ report.routing_distribution | map(attribute='date') | map('string') | unique | list | tojson }},
          datasets: [
            {% for model in report.routing_distribution | map(attribute='model') | unique | list %}
            {
              label: {{ model | tojson }},
              data: {{ report.routing_distribution | selectattr('model', 'equalto', model) | map(attribute='request_count') | list | tojson }},
            },
            {% endfor %}
          ],
        },
      });
    </script>
    {% endif %}
  </section>

  <section class="dashboard-section" id="section-failover-trend">
    <h2>Failover Trend</h2>
    {% if not report.failover_trend %}
    <p class="empty-state">No failovers recorded.</p>
    {% else %}
    <canvas id="failover-trend-chart"></canvas>
    <script>
      new Chart(document.getElementById('failover-trend-chart'), {
        type: 'bar',
        data: {
          labels: {{ report.failover_trend | map(attribute='date') | map('string') | list | tojson }},
          datasets: [{ label: 'Failovers/day', data: {{ report.failover_trend | map(attribute='failover_count') | list | tojson }} }],
        },
      });
    </script>
    {% endif %}
  </section>

  <section class="dashboard-section" id="section-recommendation-trend">
    <h2>Recommendation Trend</h2>
    {% if not report.recommendation_trend %}
    <p class="empty-state">No recommendations yet.</p>
    {% else %}
    <canvas id="recommendation-trend-chart"></canvas>
    <script>
      new Chart(document.getElementById('recommendation-trend-chart'), {
        type: 'line',
        data: {
          labels: {{ report.recommendation_trend | map(attribute='date') | map('string') | list | tojson }},
          datasets: [
            { label: 'Generated/day', data: {{ report.recommendation_trend | map(attribute='generated_count') | list | tojson }} },
            { label: 'Still open', data: {{ report.recommendation_trend | map(attribute='open_count') | list | tojson }} },
          ],
        },
      });
    </script>
    {% endif %}
  </section>
</main>
{% endblock %}
```

Note: this template does not `{% include %}` any fragment with `hx-trigger` — it is a fully standalone, non-polling page, satisfying the Global Constraint. `base.html` (unmodified) already loads `htmx.min.js` and `chart.min.js` globally; `htmx.min.js` being loaded is fine since nothing on this page uses `hx-*` attributes to trigger it.

- [ ] **Step 6: Add the nav link on the existing dashboard page**

Modify `backend/templates/dashboard.html`: add one line right after the opening `<main class="dashboard">` tag:

```html
<main class="dashboard">
  <nav class="dashboard-nav">
    <a href="/dashboard">Operations Dashboard</a> | <a href="/dashboard/analytics">Analytics</a>
  </nav>
  {% include "fragments/overview.html" %}
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest backend/tests/test_analytics_router.py -v`
Expected: PASS (4 tests)

- [ ] **Step 8: Run the existing dashboard UI test to confirm the nav-link change didn't break it**

Run: `pytest backend/tests/test_dashboard_ui.py -v`
Expected: PASS (no assertions there check for absence of a nav element, so adding one is safe — confirm by reading the existing test's marker list first if this fails).

- [ ] **Step 9: Run full test suite to check for regressions**

Run: `pytest backend/tests/ -q`
Expected: all tests pass.

- [ ] **Step 10: Commit**

```bash
git add backend/api/routers/analytics.py backend/api/main.py backend/templates/analytics.html backend/templates/dashboard.html backend/tests/test_analytics_router.py
git commit -m "feat: add /v1/analytics/report and /dashboard/analytics"
```

---

### Task 8: Manual verification, version bump, tag v0.8.0

**Files:**
- Modify: `pyproject.toml` (version)
- Modify: `backend/api/main.py` (`APP_VERSION`)

- [ ] **Step 1: Start the app locally**

Run: `uvicorn backend.api.main:app --reload`

- [ ] **Step 2: Manual verification checklist**

Work through each item and check it off:

- [ ] `curl http://localhost:8000/v1/analytics/report` returns valid JSON with `cost_trend`, `quality_trend`, `failover_trend`, `routing_distribution`, `recommendation_trend` keys.
- [ ] `curl "http://localhost:8000/v1/analytics/report?days=7"` returns `"window_days": 7`.
- [ ] Open `http://localhost:8000/dashboard/analytics` in a browser — page renders successfully (200, no template errors in server logs).
- [ ] With no data seeded, `/dashboard/analytics` shows empty-state text for each section (`No cost data yet.` etc.) instead of erroring.
- [ ] After sending a few requests through `/v1/chat` (seeding real data), reload `/dashboard/analytics` and confirm all charts that have data render as canvases (cost trend, quality trend, routing distribution, and — if any failover happened — failover trend; recommendation trend requires the learning pipeline to have generated at least one recommendation).
- [ ] Click the "Analytics" nav link from `http://localhost:8000/dashboard` — lands on `/dashboard/analytics` successfully.
- [ ] Click "Operations Dashboard" from `/dashboard/analytics` — lands back on `/dashboard` successfully, and `/dashboard` still shows all of its original sections (overview, providers, cost, quality, recommendations, failovers, circuits, recent requests) exactly as before this phase.
- [ ] View page source on `/dashboard/analytics` and confirm no `hx-trigger` attribute appears anywhere.
- [ ] Run `pytest backend/tests/ -q` one final time — full regression suite green.

- [ ] **Step 3: Bump the version**

In `pyproject.toml`, change:
```toml
version = "0.1.0"
```
to:
```toml
version = "0.8.0"
```

(Note: `pyproject.toml`'s version has drifted from `APP_VERSION` in `backend/api/main.py` since at least Phase 7 — bump both here so they're consistent going forward, but do not treat reconciling historical drift as in-scope beyond this one line.)

In `backend/api/main.py`, change:
```python
APP_VERSION = "0.7.0"
```
to:
```python
APP_VERSION = "0.8.0"
```

- [ ] **Step 4: Run full suite once more after the version bump**

Run: `pytest backend/tests/ -q`
Expected: all tests pass (version string isn't asserted anywhere, but this confirms the edit didn't break syntax).

- [ ] **Step 5: Commit the version bump**

```bash
git add pyproject.toml backend/api/main.py
git commit -m "chore: bump to v0.8.0 - Phase 8 analytics complete"
```

- [ ] **Step 6: Tag the release**

```bash
git tag -a v0.8.0 -m "Phase 8: Advanced Analytics & Reporting"
```

Do not push the tag or push to any remote without explicit user confirmation — this step only creates the local tag.

---

## End state

After both batches: `main` has cost, quality, failover, and routing-distribution trends plus recommendation-generation trends, exposed via `GET /v1/analytics/report` and rendered at `/dashboard/analytics`, fully covered by tests, tagged `v0.8.0`. This is the last phase on the frozen roadmap (Foundation → Routing → Verification → Learning → Resilience → Dashboard → Optimization → Analytics) — no further phases are scheduled.
