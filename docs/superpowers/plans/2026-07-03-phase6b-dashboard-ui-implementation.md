# Phase 6b: Operations Dashboard UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a server-rendered operations dashboard UI at `GET /dashboard`, backed by three new read-only `DashboardRepository` queries and per-fragment `DashboardService` methods, so operators can visually monitor providers, costs, quality, recommendations, failovers, and recent requests without hitting the JSON API directly.

**Architecture:** FastAPI serves Jinja2 templates (`backend/templates/`) and static assets (`backend/static/`) from the existing app process — no new service, no build step. A new HTML router (`dashboard_ui.py`) exposes `/dashboard` (full page) and `/dashboard/fragments/{section}` (HTMX-polled partials for the four live sections). Charts render client-side with a vendored Chart.js build from server-provided JSON.

**Tech Stack:** FastAPI + Jinja2 (new dependency) + vendored htmx 1.9.12 + vendored Chart.js 4.4.4 (UMD, minified). No Node, no React, no CDN dependency at runtime.

## Global Constraints

- No changes to existing JSON endpoint contracts (`/v1/dashboard/overview`, `/v1/metrics/quality`) or Phase 5 write paths (`ChatService`, `ProviderExecutor`).
- Recent Requests fields: Request ID, Model, Cost, Score/Pass, Complexity, Strategy, Timestamp only — no Latency, no Retry Count (not tracked anywhere in the system).
- Fragment endpoints (`overview`, `providers`, `circuits`, `recent-requests`) fetch only their own data — never rebuild the whole page.
- Charts, recommendations, and the failover timeline render once per full page load — never re-fetched on the 15s poll cycle.
- Static asset URLs are versioned with `?v={APP_VERSION}` for cache-busting.
- Every section handles the empty-data case with a friendly placeholder, not a blank chart/table.
- App version bumps to `0.6.1` as part of this work; tag `v0.6.1` once complete.

---

## Batch 1: Backend additions (repository + service)

### Task 1: DashboardRepository — quality trend

**Files:**
- Modify: `backend/services/dashboard_repository.py`
- Test: `backend/tests/test_dashboard_repository.py`

**Interfaces:**
- Produces: `QualityTrendBucket(date: date, average_score: float, pass_rate: float)` dataclass; `DashboardRepository.get_quality_trend(window: TimeWindow) -> list[QualityTrendBucket]`, ascending by date, one bucket per day that has at least one COMPLETED verification in the window.

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_dashboard_repository.py`:

```python
def test_get_quality_trend_buckets_by_day(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    now = datetime.now(timezone.utc)
    day2 = now - timedelta(days=2)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-a", prompt="hi", strategy="balanced"))
        session.add(VerificationRow(
            request_id="req-a", status=VerificationStatus.COMPLETED.value,
            routing_model="gpt-4o-mini", routing_strategy="balanced", routing_complexity="simple",
            score=0.9, passed=True, confidence=0.8, created_at=now,
        ))
        session.add(RequestRow(request_id="req-b", prompt="hi", strategy="balanced"))
        session.add(VerificationRow(
            request_id="req-b", status=VerificationStatus.COMPLETED.value,
            routing_model="gpt-4o-mini", routing_strategy="balanced", routing_complexity="simple",
            score=0.5, passed=False, confidence=0.6, created_at=now,
        ))
        session.add(RequestRow(request_id="req-c", prompt="hi", strategy="balanced"))
        session.add(VerificationRow(
            request_id="req-c", status=VerificationStatus.COMPLETED.value,
            routing_model="gpt-4o-mini", routing_strategy="balanced", routing_complexity="simple",
            score=1.0, passed=True, confidence=0.9, created_at=day2,
        ))
        session.add(RequestRow(request_id="req-failed", prompt="hi", strategy="balanced"))
        session.add(VerificationRow(
            request_id="req-failed", status=VerificationStatus.FAILED.value,
            routing_model="gpt-4o-mini", routing_strategy="balanced", routing_complexity="simple",
            error_type="ValidationError", error="bad json", created_at=now,
        ))
        session.commit()

    buckets = repository.get_quality_trend(TimeWindow(days=7))

    assert len(buckets) == 2
    assert buckets[0].date < buckets[1].date
    today_bucket = buckets[1]
    assert today_bucket.average_score == pytest.approx(0.7)
    assert today_bucket.pass_rate == pytest.approx(0.5)


def test_get_quality_trend_excludes_data_outside_window(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    old = datetime.now(timezone.utc) - timedelta(days=30)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-old", prompt="hi", strategy="balanced"))
        session.add(VerificationRow(
            request_id="req-old", status=VerificationStatus.COMPLETED.value,
            routing_model="gpt-4o-mini", routing_strategy="balanced", routing_complexity="simple",
            score=0.9, passed=True, confidence=0.8, created_at=old,
        ))
        session.commit()

    buckets = repository.get_quality_trend(TimeWindow(days=7))

    assert buckets == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_dashboard_repository.py -k quality_trend -v`
Expected: FAIL with `AttributeError: 'DashboardRepository' object has no attribute 'get_quality_trend'`

- [ ] **Step 3: Write minimal implementation**

In `backend/services/dashboard_repository.py`, add the dataclass next to `CostBucketData`:

```python
@dataclass(frozen=True)
class QualityTrendBucket:
    date: date
    average_score: float
    pass_rate: float
```

Add the method to `DashboardRepository`, next to `get_quality_aggregation`:

```python
def get_quality_trend(self, window: TimeWindow) -> list[QualityTrendBucket]:
    with self._session_factory() as session:
        rows = (
            session.query(VerificationRow)
            .filter(VerificationRow.created_at >= window.cutoff)
            .filter(VerificationRow.status == VerificationStatus.COMPLETED.value)
            .all()
        )

    buckets: dict[date, list[VerificationRow]] = {}
    for row in rows:
        day = row.created_at.date()
        buckets.setdefault(day, []).append(row)

    return [
        QualityTrendBucket(
            date=day,
            average_score=_avg([r.score for r in group]),
            pass_rate=_avg([1.0 if r.passed else 0.0 for r in group]),
        )
        for day, group in sorted(buckets.items())
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend/tests/test_dashboard_repository.py -k quality_trend -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/services/dashboard_repository.py backend/tests/test_dashboard_repository.py
git commit -m "feat: add DashboardRepository.get_quality_trend for daily pass-rate/score buckets"
```

---

### Task 2: DashboardRepository — failover events with timestamps

**Files:**
- Modify: `backend/services/dashboard_repository.py`
- Test: `backend/tests/test_dashboard_repository.py`

**Interfaces:**
- Consumes: `_routing_event(request_id, created_at, model)` test helper already defined in the test file.
- Produces: `FailoverEvent(request_id: str, from_model: str, to_model: str, occurred_at: datetime)` dataclass; `DashboardRepository.get_failover_events(window: TimeWindow) -> list[FailoverEvent]`, ascending by `occurred_at`. `get_failover_summary` is unchanged (existing JSON API keeps its shape).

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_dashboard_repository.py`:

```python
def test_get_failover_events_returns_from_to_and_timestamp(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    now = datetime.now(timezone.utc)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-single", prompt="hi", strategy="balanced"))
        session.add(_routing_event("req-single", now))

        session.add(RequestRow(request_id="req-failover", prompt="hi", strategy="balanced"))
        session.add(_routing_event("req-failover", now, model="gpt-4o-mini"))
        session.add(_routing_event("req-failover", now + timedelta(seconds=1), model="gpt-4o"))
        session.commit()

    events = repository.get_failover_events(TimeWindow(days=7))

    assert len(events) == 1
    assert events[0].request_id == "req-failover"
    assert events[0].from_model == "gpt-4o-mini"
    assert events[0].to_model == "gpt-4o"


def test_get_failover_events_excludes_events_outside_window(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    old = datetime.now(timezone.utc) - timedelta(days=30)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-old-failover", prompt="hi", strategy="balanced"))
        session.add(_routing_event("req-old-failover", old, model="gpt-4o-mini"))
        session.add(_routing_event("req-old-failover", old, model="gpt-4o"))
        session.commit()

    events = repository.get_failover_events(TimeWindow(days=7))

    assert events == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_dashboard_repository.py -k failover_events -v`
Expected: FAIL with `AttributeError: 'DashboardRepository' object has no attribute 'get_failover_events'`

- [ ] **Step 3: Write minimal implementation**

Add the dataclass next to `FailoverData`:

```python
@dataclass(frozen=True)
class FailoverEvent:
    request_id: str
    from_model: str
    to_model: str
    occurred_at: datetime
```

Add the method to `DashboardRepository`, next to `get_failover_summary`:

```python
def get_failover_events(self, window: TimeWindow) -> list[FailoverEvent]:
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

    events = [
        FailoverEvent(
            request_id=request_id,
            from_model=group[0].selected_model,
            to_model=group[1].selected_model,
            occurred_at=group[1].created_at,
        )
        for request_id, group in grouped.items()
        if len(group) == 2
    ]
    return sorted(events, key=lambda e: e.occurred_at)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend/tests/test_dashboard_repository.py -k failover_events -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/services/dashboard_repository.py backend/tests/test_dashboard_repository.py
git commit -m "feat: add DashboardRepository.get_failover_events with from/to/timestamp detail"
```

---

### Task 3: DashboardRepository — recent requests

**Files:**
- Modify: `backend/services/dashboard_repository.py`
- Test: `backend/tests/test_dashboard_repository.py`

**Interfaces:**
- Produces: `RecentRequestRow(request_id: str, model: str, strategy: str, complexity: str, cost: float | None, score: float | None, passed: bool | None, created_at: datetime)` dataclass; `DashboardRepository.get_recent_requests(limit: int = 50) -> list[RecentRequestRow]`, most-recent-first.

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_dashboard_repository.py`:

```python
def test_get_recent_requests_returns_most_recent_first_with_joined_data(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    now = datetime.now(timezone.utc)
    earlier = now - timedelta(minutes=5)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-old", prompt="hi", strategy="balanced", created_at=earlier))
        session.add(_routing_event("req-old", earlier, model="gpt-4o-mini"))
        session.add(ResponseRow(request_id="req-old", response_text="ok", actual_cost=0.05, created_at=earlier))

        session.add(RequestRow(request_id="req-new", prompt="hi", strategy="balanced", created_at=now))
        session.add(_routing_event("req-new", now, model="gpt-4o"))
        session.add(ResponseRow(request_id="req-new", response_text="ok", actual_cost=0.20, created_at=now))
        session.add(VerificationRow(
            request_id="req-new", status=VerificationStatus.COMPLETED.value,
            routing_model="gpt-4o", routing_strategy="balanced", routing_complexity="simple",
            score=0.95, passed=True, confidence=0.9, created_at=now,
        ))
        session.commit()

    requests = repository.get_recent_requests(limit=10)

    assert [r.request_id for r in requests] == ["req-new", "req-old"]
    newest = requests[0]
    assert newest.model == "gpt-4o"
    assert newest.cost == pytest.approx(0.20)
    assert newest.score == pytest.approx(0.95)
    assert newest.passed is True
    oldest = requests[1]
    assert oldest.model == "gpt-4o-mini"
    assert oldest.score is None
    assert oldest.passed is None


def test_get_recent_requests_respects_limit(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    now = datetime.now(timezone.utc)
    with session_factory() as session:
        for i in range(5):
            session.add(RequestRow(request_id=f"req-{i}", prompt="hi", strategy="balanced", created_at=now))
        session.commit()

    requests = repository.get_recent_requests(limit=3)

    assert len(requests) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_dashboard_repository.py -k recent_requests -v`
Expected: FAIL with `AttributeError: 'DashboardRepository' object has no attribute 'get_recent_requests'`

- [ ] **Step 3: Write minimal implementation**

Add the dataclass next to `CostBucketData`:

```python
@dataclass(frozen=True)
class RecentRequestRow:
    request_id: str
    model: str
    strategy: str
    complexity: str
    cost: float | None
    score: float | None
    passed: bool | None
    created_at: datetime
```

Add the method to `DashboardRepository`:

```python
def get_recent_requests(self, limit: int = 50) -> list[RecentRequestRow]:
    with self._session_factory() as session:
        requests = (
            session.query(RequestRow)
            .order_by(RequestRow.created_at.desc())
            .limit(limit)
            .all()
        )
        request_ids = [r.request_id for r in requests]
        routing_events = (
            session.query(RoutingEventRow)
            .filter(RoutingEventRow.request_id.in_(request_ids))
            .order_by(RoutingEventRow.created_at)
            .all()
        )
        responses = (
            session.query(ResponseRow)
            .filter(ResponseRow.request_id.in_(request_ids))
            .all()
        )
        verifications = (
            session.query(VerificationRow)
            .filter(VerificationRow.request_id.in_(request_ids))
            .all()
        )

    latest_routing: dict[str, RoutingEventRow] = {}
    for row in routing_events:
        latest_routing[row.request_id] = row  # ascending order, last write wins
    response_by_request = {row.request_id: row for row in responses}
    verification_by_request = {row.request_id: row for row in verifications}

    result = []
    for req in requests:
        routing = latest_routing.get(req.request_id)
        response = response_by_request.get(req.request_id)
        verification = verification_by_request.get(req.request_id)
        result.append(RecentRequestRow(
            request_id=req.request_id,
            model=routing.selected_model if routing else "unknown",
            strategy=routing.selected_strategy if routing else req.strategy,
            complexity=routing.complexity if routing else "unknown",
            cost=response.actual_cost if response else None,
            score=verification.score if verification else None,
            passed=verification.passed if verification else None,
            created_at=req.created_at,
        ))
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend/tests/test_dashboard_repository.py -k recent_requests -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/services/dashboard_repository.py backend/tests/test_dashboard_repository.py
git commit -m "feat: add DashboardRepository.get_recent_requests"
```

---

### Task 4: DashboardRepository — cost-by-model breakdown

**Files:**
- Modify: `backend/services/dashboard_repository.py`
- Test: `backend/tests/test_dashboard_repository.py`

**Interfaces:**
- Produces: `DashboardRepository.get_cost_by_model(window: TimeWindow) -> dict[str, float]` — total cost per model, in the window.

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_dashboard_repository.py`:

```python
def test_get_cost_by_model_sums_cost_per_model(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    now = datetime.now(timezone.utc)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-a", prompt="hi", strategy="balanced"))
        session.add(_routing_event("req-a", now, model="gpt-4o-mini"))
        session.add(ResponseRow(request_id="req-a", response_text="ok", actual_cost=0.10, created_at=now))

        session.add(RequestRow(request_id="req-b", prompt="hi", strategy="balanced"))
        session.add(_routing_event("req-b", now, model="gpt-4o-mini"))
        session.add(ResponseRow(request_id="req-b", response_text="ok", actual_cost=0.05, created_at=now))

        session.add(RequestRow(request_id="req-c", prompt="hi", strategy="balanced"))
        session.add(_routing_event("req-c", now, model="gpt-4o"))
        session.add(ResponseRow(request_id="req-c", response_text="ok", actual_cost=0.30, created_at=now))
        session.commit()

    totals = repository.get_cost_by_model(TimeWindow(days=7))

    assert totals["gpt-4o-mini"] == pytest.approx(0.15)
    assert totals["gpt-4o"] == pytest.approx(0.30)


def test_get_cost_by_model_excludes_data_outside_window(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    old = datetime.now(timezone.utc) - timedelta(days=30)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-old", prompt="hi", strategy="balanced"))
        session.add(_routing_event("req-old", old, model="gpt-4o-mini"))
        session.add(ResponseRow(request_id="req-old", response_text="ok", actual_cost=0.10, created_at=old))
        session.commit()

    totals = repository.get_cost_by_model(TimeWindow(days=7))

    assert totals == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_dashboard_repository.py -k cost_by_model -v`
Expected: FAIL with `AttributeError: 'DashboardRepository' object has no attribute 'get_cost_by_model'`

- [ ] **Step 3: Write minimal implementation**

Add the method to `DashboardRepository`, next to `get_cost_trend`:

```python
def get_cost_by_model(self, window: TimeWindow) -> dict[str, float]:
    with self._session_factory() as session:
        responses = (
            session.query(ResponseRow)
            .filter(ResponseRow.created_at >= window.cutoff)
            .filter(ResponseRow.actual_cost.isnot(None))
            .all()
        )
        request_ids = [r.request_id for r in responses]
        routing_events = (
            session.query(RoutingEventRow)
            .filter(RoutingEventRow.request_id.in_(request_ids))
            .order_by(RoutingEventRow.created_at)
            .all()
        )

    latest_model: dict[str, str] = {}
    for row in routing_events:
        latest_model[row.request_id] = row.selected_model

    totals: dict[str, float] = {}
    for response in responses:
        model = latest_model.get(response.request_id, "unknown")
        totals[model] = totals.get(model, 0.0) + response.actual_cost
    return totals
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend/tests/test_dashboard_repository.py -k cost_by_model -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/services/dashboard_repository.py backend/tests/test_dashboard_repository.py
git commit -m "feat: add DashboardRepository.get_cost_by_model breakdown"
```

---

### Task 5: DashboardService — per-fragment and full-page methods

**Files:**
- Modify: `backend/services/dashboard_service.py`
- Test: `backend/tests/test_dashboard_service.py`

**Interfaces:**
- Consumes: `DashboardRepository.get_quality_trend`, `.get_failover_events`, `.get_recent_requests`, `.get_cost_by_model` (Task 1-4); existing `.get_quality_aggregation`, `.get_cost_trend`, `.get_failover_summary`; `ProviderManager.list_providers()`; `ProviderExecutor.circuit_states()`; `LearningService.get_recommendations()`.
- Produces:
  - `DashboardService.get_overview_fragment(window: TimeWindow) -> dict` — keys: `total_requests`, `total_cost`, `average_quality_score`, `pass_rate`, `active_providers`, `open_circuits`, `failovers_today`.
  - `DashboardService.get_provider_fragment() -> dict` — key: `providers` (dict of `ProviderDashboardStatus`, same shape as existing `DashboardOverview.providers`).
  - `DashboardService.get_circuit_fragment() -> dict` — key: `circuits` (raw dict from `circuit_states()`).
  - `DashboardService.get_recent_requests_fragment(limit: int = 50) -> dict` — key: `requests` (list of `RecentRequestRow`).
  - `DashboardService.get_dashboard_page(window: TimeWindow) -> dict` — keys: `overview`, `providers`, `circuits`, `requests`, `cost_trend`, `quality_trend`, `cost_by_model`, `failover_events`, `recommendations`.

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_dashboard_service.py`. First check the existing fakes at the top of the file (`_FakeProviderManager`, `_FakeProviderExecutor`, `_FakeLearningService`, `_FakeDashboardRepository`) and extend `_FakeDashboardRepository` with the four new methods:

```python
class _FakeDashboardRepository:
    def __init__(self, quality, cost_buckets, failover_data,
                 quality_trend=None, failover_events=None, recent_requests=None, cost_by_model=None):
        self._quality = quality
        self._cost_buckets = cost_buckets
        self._failover_data = failover_data
        self._quality_trend = quality_trend or []
        self._failover_events = failover_events or []
        self._recent_requests = recent_requests or []
        self._cost_by_model = cost_by_model or {}
        self.get_quality_aggregation_calls = 0
        self.get_cost_trend_calls = 0
        self.get_failover_summary_calls = 0
        self.get_quality_trend_calls = 0
        self.get_failover_events_calls = 0
        self.get_recent_requests_calls = 0
        self.get_cost_by_model_calls = 0

    def get_quality_aggregation(self):
        self.get_quality_aggregation_calls += 1
        return self._quality

    def get_cost_trend(self, window):
        self.get_cost_trend_calls += 1
        return self._cost_buckets

    def get_failover_summary(self, window):
        self.get_failover_summary_calls += 1
        return self._failover_data

    def get_quality_trend(self, window):
        self.get_quality_trend_calls += 1
        return self._quality_trend

    def get_failover_events(self, window):
        self.get_failover_events_calls += 1
        return self._failover_events

    def get_recent_requests(self, limit=50):
        self.get_recent_requests_calls += 1
        return self._recent_requests

    def get_cost_by_model(self, window):
        self.get_cost_by_model_calls += 1
        return self._cost_by_model
```

(This replaces the existing `_FakeDashboardRepository` class — same constructor name, superset of methods.)

Then add the new tests:

```python
@pytest.mark.asyncio
async def test_get_overview_fragment_computes_aggregate_stats():
    quality = QualityAggregation(
        total_verified=10, average_score=0.8, average_confidence=0.7, pass_rate=0.9,
        average_queue_delay_ms=5.0, average_evaluation_duration_ms=50.0,
        average_total_verification_ms=60.0, verification_failure_count=1,
        by_model={}, by_strategy={}, by_complexity={},
    )
    cost_buckets = [
        CostBucketData(date=date(2026, 7, 3), request_count=4, total_cost=1.20),
    ]
    failover_data = FailoverData(request_ids=["req-1"])
    repository = _FakeDashboardRepository(quality, cost_buckets, failover_data)
    service = DashboardService(
        provider_manager=_FakeProviderManager(),
        provider_executor=_FakeProviderExecutor(),
        learning_service=_FakeLearningService([]),
        dashboard_repository=repository,
    )

    result = await service.get_overview_fragment(TimeWindow(days=7))

    assert result["total_requests"] == 4
    assert result["total_cost"] == pytest.approx(1.20)
    assert result["average_quality_score"] == pytest.approx(0.8)
    assert result["pass_rate"] == pytest.approx(0.9)
    assert result["active_providers"] == 1
    assert result["open_circuits"] == 0
    assert result["failovers_today"] == 1


@pytest.mark.asyncio
async def test_get_provider_fragment_returns_only_providers_key():
    repository = _FakeDashboardRepository(
        QualityAggregation(0, 0, 0, 0, 0, 0, 0, 0, {}, {}, {}), [], FailoverData([]),
    )
    service = DashboardService(
        provider_manager=_FakeProviderManager(),
        provider_executor=_FakeProviderExecutor(),
        learning_service=_FakeLearningService([]),
        dashboard_repository=repository,
    )

    result = await service.get_provider_fragment()

    assert set(result.keys()) == {"providers"}
    assert set(result["providers"].keys()) == {"openai", "anthropic", "ollama"}


@pytest.mark.asyncio
async def test_get_circuit_fragment_returns_only_circuits_key():
    repository = _FakeDashboardRepository(
        QualityAggregation(0, 0, 0, 0, 0, 0, 0, 0, {}, {}, {}), [], FailoverData([]),
    )
    service = DashboardService(
        provider_manager=_FakeProviderManager(),
        provider_executor=_FakeProviderExecutor(),
        learning_service=_FakeLearningService([]),
        dashboard_repository=repository,
    )

    result = await service.get_circuit_fragment()

    assert set(result.keys()) == {"circuits"}
    assert result["circuits"]["openai"]["state"] == "closed"


@pytest.mark.asyncio
async def test_get_recent_requests_fragment_delegates_to_repository():
    repository = _FakeDashboardRepository(
        QualityAggregation(0, 0, 0, 0, 0, 0, 0, 0, {}, {}, {}), [], FailoverData([]),
        recent_requests=["fake-row"],
    )
    service = DashboardService(
        provider_manager=_FakeProviderManager(),
        provider_executor=_FakeProviderExecutor(),
        learning_service=_FakeLearningService([]),
        dashboard_repository=repository,
    )

    result = await service.get_recent_requests_fragment()

    assert result == {"requests": ["fake-row"]}
    assert repository.get_recent_requests_calls == 1


@pytest.mark.asyncio
async def test_get_dashboard_page_assembles_all_sections():
    quality = QualityAggregation(
        total_verified=1, average_score=0.9, average_confidence=0.8, pass_rate=1.0,
        average_queue_delay_ms=0, average_evaluation_duration_ms=0,
        average_total_verification_ms=0, verification_failure_count=0,
        by_model={}, by_strategy={}, by_complexity={},
    )
    repository = _FakeDashboardRepository(
        quality, [], FailoverData([]),
        quality_trend=["trend-bucket"], failover_events=["failover-event"],
        recent_requests=["request-row"], cost_by_model={"gpt-4o": 1.0},
    )
    service = DashboardService(
        provider_manager=_FakeProviderManager(),
        provider_executor=_FakeProviderExecutor(),
        learning_service=_FakeLearningService([]),
        dashboard_repository=repository,
    )

    result = await service.get_dashboard_page(TimeWindow(days=7))

    assert set(result.keys()) == {
        "overview", "providers", "circuits", "requests",
        "cost_trend", "quality_trend", "cost_by_model", "failover_events", "recommendations",
    }
    assert result["quality_trend"] == ["trend-bucket"]
    assert result["failover_events"] == ["failover-event"]
    assert result["requests"] == ["request-row"]
    assert result["cost_by_model"] == {"gpt-4o": 1.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_dashboard_service.py -k "fragment or dashboard_page" -v`
Expected: FAIL with `AttributeError: 'DashboardService' object has no attribute 'get_overview_fragment'` (and similar for the other new methods)

- [ ] **Step 3: Write minimal implementation**

In `backend/services/dashboard_service.py`, add these methods to `DashboardService` (after `get_overview`):

```python
async def get_overview_fragment(self, window: TimeWindow) -> dict:
    availability, circuits, quality_agg, cost_buckets, failover_today = await asyncio.gather(
        asyncio.to_thread(self._provider_manager.list_providers),
        asyncio.to_thread(self._provider_executor.circuit_states),
        asyncio.to_thread(self._dashboard_repository.get_quality_aggregation),
        asyncio.to_thread(self._dashboard_repository.get_cost_trend, window),
        asyncio.to_thread(self._dashboard_repository.get_failover_summary, TimeWindow(days=1)),
    )
    return {
        "total_requests": sum(b.request_count for b in cost_buckets),
        "total_cost": sum(b.total_cost for b in cost_buckets),
        "average_quality_score": quality_agg.average_score,
        "pass_rate": quality_agg.pass_rate,
        "active_providers": sum(1 for status in availability.values() if status == "available"),
        "open_circuits": sum(1 for c in circuits.values() if c.get("state") == "open"),
        "failovers_today": len(failover_today.request_ids),
    }

async def get_provider_fragment(self) -> dict:
    availability, circuits = await asyncio.gather(
        asyncio.to_thread(self._provider_manager.list_providers),
        asyncio.to_thread(self._provider_executor.circuit_states),
    )
    return {"providers": self._merge_provider_status(availability, circuits)}

async def get_circuit_fragment(self) -> dict:
    circuits = await asyncio.to_thread(self._provider_executor.circuit_states)
    return {"circuits": circuits}

async def get_recent_requests_fragment(self, limit: int = 50) -> dict:
    requests = await asyncio.to_thread(self._dashboard_repository.get_recent_requests, limit)
    return {"requests": requests}

async def get_dashboard_page(self, window: TimeWindow) -> dict:
    (
        overview, provider_data, circuit_data, recent_requests_data,
        cost_trend, quality_trend, cost_by_model, failover_events, recommendation_rows,
    ) = await asyncio.gather(
        self.get_overview_fragment(window),
        self.get_provider_fragment(),
        self.get_circuit_fragment(),
        self.get_recent_requests_fragment(),
        asyncio.to_thread(self._dashboard_repository.get_cost_trend, window),
        asyncio.to_thread(self._dashboard_repository.get_quality_trend, window),
        asyncio.to_thread(self._dashboard_repository.get_cost_by_model, window),
        asyncio.to_thread(self._dashboard_repository.get_failover_events, window),
        asyncio.to_thread(self._learning_service.get_recommendations),
    )
    return {
        "overview": overview,
        "providers": provider_data["providers"],
        "circuits": circuit_data["circuits"],
        "requests": recent_requests_data["requests"],
        "cost_trend": cost_trend,
        "quality_trend": quality_trend,
        "cost_by_model": cost_by_model,
        "failover_events": failover_events,
        "recommendations": [self._to_recommendation_response(r) for r in recommendation_rows],
    }
```

Note: `TimeWindow` is already imported in this file (from `backend.services.dashboard_repository import DashboardRepository, TimeWindow`); no new import needed for that. `_merge_provider_status` and `_to_recommendation_response` already exist as private methods on the class.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend/tests/test_dashboard_service.py -v`
Expected: PASS (all tests in the file, including the pre-existing ones — confirms the `_FakeDashboardRepository` replacement didn't break anything)

- [ ] **Step 5: Commit**

```bash
git add backend/services/dashboard_service.py backend/tests/test_dashboard_service.py
git commit -m "feat: add DashboardService per-fragment and full-page methods for dashboard UI"
```

---

### Batch 1 checkpoint

- [ ] Run the full regression suite: `python -m pytest -q`
  Expected: all tests pass (265 existing + ~14 new = ~279)

---

## Batch 2: UI (templates, static assets, routes)

### Task 6: Vendor static assets and shared path constants

**Files:**
- Create: `backend/static/js/htmx.min.js`
- Create: `backend/static/js/chart.min.js`
- Create: `backend/static/css/dashboard.css`
- Create: `backend/api/paths.py`
- Test: `backend/tests/test_static_assets.py`

**Interfaces:**
- Produces: `backend.api.paths.BASE_DIR`, `TEMPLATES_DIR`, `STATIC_DIR` (all `pathlib.Path`), used by Task 8 (`dashboard_ui.py`) and Task 9 (`main.py`).

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_static_assets.py`:

```python
from backend.api.paths import STATIC_DIR


def test_htmx_asset_is_vendored():
    path = STATIC_DIR / "js" / "htmx.min.js"
    assert path.exists()
    assert path.stat().st_size > 10_000
    assert "htmx" in path.read_text()[:2000].lower()


def test_chartjs_asset_is_vendored():
    path = STATIC_DIR / "js" / "chart.min.js"
    assert path.exists()
    assert path.stat().st_size > 50_000


def test_dashboard_css_exists():
    path = STATIC_DIR / "css" / "dashboard.css"
    assert path.exists()
    assert path.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_static_assets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.api.paths'`

- [ ] **Step 3: Write minimal implementation**

Create `backend/api/paths.py`:

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
```

Vendor htmx 1.9.12 and Chart.js 4.4.4 (pinned versions, minified UMD builds):

```bash
mkdir -p backend/static/js backend/static/css
curl -s -L https://unpkg.com/htmx.org@1.9.12/dist/htmx.min.js -o backend/static/js/htmx.min.js
curl -s -L https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js -o backend/static/js/chart.min.js
```

Create `backend/static/css/dashboard.css`:

```css
:root {
  --bg: #0f1115;
  --panel: #171a21;
  --border: #2a2e38;
  --text: #e6e8eb;
  --muted: #9098a8;
  --accent: #4f8cff;
  --ok: #3ecf8e;
  --warn: #f0a93d;
  --err: #ef5a5a;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--text);
}

header.dashboard-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 1rem 1.5rem;
  border-bottom: 1px solid var(--border);
}

.last-updated { color: var(--muted); font-size: 0.85rem; }

main.dashboard {
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

section.dashboard-section {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem 1.25rem;
}

section.dashboard-section h2 {
  margin-top: 0;
  font-size: 1rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.stat-row { display: flex; flex-wrap: wrap; gap: 1.5rem; }
.stat { min-width: 8rem; }
.stat .value { font-size: 1.5rem; font-weight: 600; }
.stat .label { color: var(--muted); font-size: 0.8rem; }

.card-grid { display: flex; flex-wrap: wrap; gap: 1rem; }
.card {
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.75rem 1rem;
  min-width: 12rem;
}

.state-closed { color: var(--ok); }
.state-open { color: var(--err); }
.state-half_open { color: var(--warn); }

table.recent-requests { width: 100%; border-collapse: collapse; }
table.recent-requests th, table.recent-requests td {
  text-align: left;
  padding: 0.4rem 0.6rem;
  border-bottom: 1px solid var(--border);
  font-size: 0.9rem;
}

.empty-state { color: var(--muted); font-style: italic; padding: 0.5rem 0; }

canvas { max-width: 100%; }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest backend/tests/test_static_assets.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/api/paths.py backend/static backend/tests/test_static_assets.py
git commit -m "feat: vendor htmx 1.9.12 and Chart.js 4.4.4, add dashboard CSS and shared path constants"
```

---

### Task 7: Templates — base layout, dashboard page, fragment partials

**Files:**
- Create: `backend/templates/base.html`
- Create: `backend/templates/dashboard.html`
- Create: `backend/templates/fragments/overview.html`
- Create: `backend/templates/fragments/providers.html`
- Create: `backend/templates/fragments/circuits.html`
- Create: `backend/templates/fragments/recent_requests.html`

**Interfaces:**
- Consumes: context dicts produced by `DashboardService.get_dashboard_page` (Task 5) and the four fragment methods, plus `app_version: str` and `now: str` passed by the router (Task 8).
- Produces: rendered HTML consumed by Task 8's `Jinja2Templates.TemplateResponse` calls.

This task has no automated test of its own — templates are exercised by Task 8's route tests. Write all six files, then proceed.

- [ ] **Step 1: Create `backend/templates/base.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{% block title %}LLM Cost Autopilot — Operations Dashboard{% endblock %}</title>
  <link rel="stylesheet" href="/static/css/dashboard.css?v={{ app_version }}">
  <script src="/static/js/htmx.min.js?v={{ app_version }}"></script>
  <script src="/static/js/chart.min.js?v={{ app_version }}"></script>
</head>
<body>
  <header class="dashboard-header">
    <h1>Operations Dashboard</h1>
    <span class="last-updated" id="last-updated">Last updated: {{ now }} UTC</span>
  </header>
  {% block content %}{% endblock %}
</body>
</html>
```

- [ ] **Step 2: Create `backend/templates/fragments/overview.html`**

```html
{% if standalone %}
<div id="last-updated" hx-swap-oob="true" class="last-updated">Last updated: {{ now }} UTC</div>
{% endif %}
<section class="dashboard-section" id="section-overview">
  <h2>System Overview</h2>
  <div class="stat-row">
    <div class="stat"><div class="value">{{ overview.total_requests }}</div><div class="label">Total Requests</div></div>
    <div class="stat"><div class="value">${{ "%.2f"|format(overview.total_cost) }}</div><div class="label">Total Cost</div></div>
    <div class="stat"><div class="value">{{ "%.2f"|format(overview.average_quality_score) }}</div><div class="label">Avg Quality Score</div></div>
    <div class="stat"><div class="value">{{ "%.0f"|format(overview.pass_rate * 100) }}%</div><div class="label">Pass Rate</div></div>
    <div class="stat"><div class="value">{{ overview.active_providers }}</div><div class="label">Active Providers</div></div>
    <div class="stat"><div class="value">{{ overview.open_circuits }}</div><div class="label">Open Circuits</div></div>
    <div class="stat"><div class="value">{{ overview.failovers_today }}</div><div class="label">Failovers Today</div></div>
  </div>
</section>
```

- [ ] **Step 3: Create `backend/templates/fragments/providers.html`**

```html
{% if standalone %}
<div id="last-updated" hx-swap-oob="true" class="last-updated">Last updated: {{ now }} UTC</div>
{% endif %}
<section class="dashboard-section" id="section-providers">
  <h2>Provider Health</h2>
  {% if not providers %}
  <p class="empty-state">No providers configured.</p>
  {% else %}
  <div class="card-grid">
    {% for name, status in providers.items() %}
    <div class="card">
      <strong>{{ name }}</strong><br>
      {{ "Available" if status.availability == "available" else "Disabled" }}<br>
      Circuit: <span class="state-{{ status.circuit_state }}">{{ status.circuit_state }}</span><br>
      Failures: {{ status.consecutive_failures }}
    </div>
    {% endfor %}
  </div>
  {% endif %}
</section>
```

- [ ] **Step 4: Create `backend/templates/fragments/circuits.html`**

```html
{% if standalone %}
<div id="last-updated" hx-swap-oob="true" class="last-updated">Last updated: {{ now }} UTC</div>
{% endif %}
<section class="dashboard-section" id="section-circuits">
  <h2>Circuit Breakers</h2>
  {% if not circuits %}
  <p class="empty-state">No circuit breaker data yet.</p>
  {% else %}
  <div class="card-grid">
    {% for name, state in circuits.items() %}
    <div class="card">
      <strong>{{ name }}</strong><br>
      <span class="state-{{ state.state }}">{{ state.state | capitalize }}</span><br>
      Consecutive failures: {{ state.consecutive_failures }}
    </div>
    {% endfor %}
  </div>
  {% endif %}
</section>
```

- [ ] **Step 5: Create `backend/templates/fragments/recent_requests.html`**

```html
{% if standalone %}
<div id="last-updated" hx-swap-oob="true" class="last-updated">Last updated: {{ now }} UTC</div>
{% endif %}
<section class="dashboard-section" id="section-recent-requests">
  <h2>Recent Requests</h2>
  {% if not requests %}
  <p class="empty-state">No requests yet.</p>
  {% else %}
  <table class="recent-requests">
    <thead>
      <tr><th>Request ID</th><th>Model</th><th>Cost</th><th>Score</th><th>Passed</th><th>Complexity</th><th>Strategy</th><th>Time</th></tr>
    </thead>
    <tbody>
      {% for r in requests %}
      <tr>
        <td>{{ r.request_id }}</td>
        <td>{{ r.model }}</td>
        <td>{{ "$%.4f"|format(r.cost) if r.cost is not none else "—" }}</td>
        <td>{{ "%.2f"|format(r.score) if r.score is not none else "—" }}</td>
        <td>{{ "Yes" if r.passed else ("No" if r.passed is not none else "—") }}</td>
        <td>{{ r.complexity }}</td>
        <td>{{ r.strategy }}</td>
        <td>{{ r.created_at.strftime("%Y-%m-%d %H:%M:%S") }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% endif %}
</section>
```

- [ ] **Step 6: Create `backend/templates/dashboard.html`**

```html
{% extends "base.html" %}
{% block content %}
<main class="dashboard">
  {% include "fragments/overview.html" %}
  {% include "fragments/providers.html" %}

  <section class="dashboard-section" id="section-cost">
    <h2>Cost Analytics</h2>
    {% if not cost_trend %}
    <p class="empty-state">No cost data yet.</p>
    {% else %}
    <canvas id="cost-trend-chart"></canvas>
    <canvas id="cost-by-model-chart"></canvas>
    <script>
      new Chart(document.getElementById('cost-trend-chart'), {
        type: 'line',
        data: {
          labels: {{ cost_trend | map(attribute='date') | map('string') | list | tojson }},
          datasets: [{ label: 'Total cost/day', data: {{ cost_trend | map(attribute='total_cost') | list | tojson }} }],
        },
      });
      new Chart(document.getElementById('cost-by-model-chart'), {
        type: 'bar',
        data: {
          labels: {{ cost_by_model.keys() | list | tojson }},
          datasets: [{ label: 'Cost by model', data: {{ cost_by_model.values() | list | tojson }} }],
        },
      });
    </script>
    {% endif %}
  </section>

  <section class="dashboard-section" id="section-quality">
    <h2>Quality Analytics</h2>
    {% if not quality_trend %}
    <p class="empty-state">No verification results available.</p>
    {% else %}
    <canvas id="quality-trend-chart"></canvas>
    <script>
      new Chart(document.getElementById('quality-trend-chart'), {
        type: 'line',
        data: {
          labels: {{ quality_trend | map(attribute='date') | map('string') | list | tojson }},
          datasets: [
            { label: 'Pass rate', data: {{ quality_trend | map(attribute='pass_rate') | list | tojson }} },
            { label: 'Average score', data: {{ quality_trend | map(attribute='average_score') | list | tojson }} },
          ],
        },
      });
    </script>
    {% endif %}
  </section>

  <section class="dashboard-section" id="section-recommendations">
    <h2>Learning Recommendations</h2>
    {% if not recommendations %}
    <p class="empty-state">No recommendations yet.</p>
    {% else %}
    <div class="card-grid">
      {% for rec in recommendations %}
      <div class="card">
        <strong>{{ rec.subject }}</strong><br>
        {{ rec.text }}<br>
        Confidence: {{ "%.0f"|format(rec.evidence_confidence * 100) }}%
      </div>
      {% endfor %}
    </div>
    {% endif %}
  </section>

  <section class="dashboard-section" id="section-failovers">
    <h2>Failover Timeline</h2>
    {% if not failover_events %}
    <p class="empty-state">No failovers recorded.</p>
    {% else %}
    <ul>
      {% for event in failover_events %}
      <li>{{ event.occurred_at.strftime("%H:%M:%S") }} — {{ event.from_model }} → {{ event.to_model }} ({{ event.request_id }})</li>
      {% endfor %}
    </ul>
    {% endif %}
  </section>

  {% include "fragments/circuits.html" %}
  {% include "fragments/recent_requests.html" %}
</main>
{% endblock %}
```

- [ ] **Step 7: Commit**

```bash
git add backend/templates
git commit -m "feat: add dashboard Jinja2 templates with empty states"
```

---

### Task 8: Dashboard UI router

**Files:**
- Create: `backend/api/routers/dashboard_ui.py`
- Test: `backend/tests/test_dashboard_ui.py`

**Interfaces:**
- Consumes: `DashboardServiceDep`, `AppVersionDep` (`backend/api/dependencies.py`); `TimeWindow` (`backend/services/dashboard_repository.py`); `TEMPLATES_DIR` (`backend/api/paths.py`, Task 6); `DashboardService.get_dashboard_page`/`get_overview_fragment`/`get_provider_fragment`/`get_circuit_fragment`/`get_recent_requests_fragment` (Task 5).
- Produces: `router` (FastAPI `APIRouter`) with `GET /dashboard` and `GET /dashboard/fragments/{section}`, imported by `main.py` (Task 9).

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_dashboard_ui.py`:

```python
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.database.models import RequestRow, ResponseRow, RoutingEventRow, VerificationRow
from backend.verification.status import VerificationStatus


def _seed(session_factory):
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
        session.commit()


def test_dashboard_page_renders_200_with_all_sections(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()
    with TestClient(app) as client:
        _seed(app.state.session_factory)

        response = client.get("/dashboard")

        assert response.status_code == 200
        body = response.text
        for marker in [
            "section-overview", "section-providers", "section-cost", "section-quality",
            "section-recommendations", "section-failovers", "section-circuits",
            "section-recent-requests",
        ]:
            assert marker in body


def test_dashboard_page_empty_database_shows_empty_states(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/empty.db")
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/dashboard")

        assert response.status_code == 200
        assert "No requests yet." in response.text
        assert "No verification results available." in response.text
        assert "No recommendations yet." in response.text
        assert "No failovers recorded." in response.text


def test_dashboard_fragment_overview_returns_partial(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()
    with TestClient(app) as client:
        _seed(app.state.session_factory)

        response = client.get("/dashboard/fragments/overview")

        assert response.status_code == 200
        assert "section-overview" in response.text
        assert "section-providers" not in response.text


def test_dashboard_fragment_unknown_section_returns_404(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/dashboard/fragments/nonexistent")

        assert response.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest backend/tests/test_dashboard_ui.py -v`
Expected: FAIL (route not registered yet)

- [ ] **Step 3: Write minimal implementation**

Create `backend/api/routers/dashboard_ui.py`:

```python
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from fastapi.templating import Jinja2Templates

from backend.api.dependencies import AppVersionDep, DashboardServiceDep
from backend.api.paths import TEMPLATES_DIR
from backend.services.dashboard_repository import TimeWindow

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

FRAGMENT_TEMPLATES = {
    "overview": "fragments/overview.html",
    "providers": "fragments/providers.html",
    "circuits": "fragments/circuits.html",
    "recent-requests": "fragments/recent_requests.html",
}


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


@router.get("/dashboard")
async def dashboard_page(
    request: Request, dashboard_service: DashboardServiceDep, app_version: AppVersionDep, days: int = 7,
):
    data = await dashboard_service.get_dashboard_page(TimeWindow(days=days))
    return templates.TemplateResponse(request, "dashboard.html", {
        **data,
        "app_version": app_version,
        "now": _now_str(),
        "standalone": False,
    })


@router.get("/dashboard/fragments/{section}")
async def dashboard_fragment(
    section: str, request: Request, dashboard_service: DashboardServiceDep, days: int = 7,
):
    if section not in FRAGMENT_TEMPLATES:
        raise HTTPException(status_code=404, detail=f"Unknown dashboard section: {section}")

    window = TimeWindow(days=days)
    if section == "overview":
        data = await dashboard_service.get_overview_fragment(window)
    elif section == "providers":
        data = await dashboard_service.get_provider_fragment()
    elif section == "circuits":
        data = await dashboard_service.get_circuit_fragment()
    else:
        data = await dashboard_service.get_recent_requests_fragment()

    return templates.TemplateResponse(request, FRAGMENT_TEMPLATES[section], {
        **data,
        "now": _now_str(),
        "standalone": True,
    })
```

- [ ] **Step 4: Run test to verify it passes**

This requires Task 9's wiring (router registration + static mount) to be done first for the full page to render — proceed to Task 9, then return here and run:

Run: `python -m pytest backend/tests/test_dashboard_ui.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/api/routers/dashboard_ui.py backend/tests/test_dashboard_ui.py
git commit -m "feat: add dashboard_ui router for /dashboard and /dashboard/fragments/{section}"
```

---

### Task 9: Wire dashboard UI into the app

**Files:**
- Modify: `backend/api/main.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `dashboard_ui_router` (Task 8); `STATIC_DIR` (`backend/api/paths.py`, Task 6).
- Produces: `/dashboard`, `/dashboard/fragments/{section}`, and `/static/*` all served by the running app; `APP_VERSION` bumped to `"0.6.1"`.

- [ ] **Step 1: Add the `jinja2` dependency**

In `pyproject.toml`, add to the `dependencies` list (after `"openai>=1.54.0",`):

```toml
    "jinja2>=3.1.4",
```

Install it:

```bash
uv sync
```

- [ ] **Step 2: Wire the router and static mount into `main.py`**

In `backend/api/main.py`, add imports near the other router imports:

```python
from fastapi.staticfiles import StaticFiles

from backend.api.paths import STATIC_DIR
from backend.api.routers.dashboard_ui import router as dashboard_ui_router
```

Bump the version constant:

```python
APP_VERSION = "0.6.1"
```

In `create_app()`, add the new router (no `/v1` prefix — this serves HTML pages, not JSON API) and mount static files, after the existing `include_router` calls:

```python
    app.include_router(dashboard_ui_router)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
```

- [ ] **Step 3: Run the dashboard UI tests (from Task 8) now that wiring is complete**

Run: `python -m pytest backend/tests/test_dashboard_ui.py -v`
Expected: PASS (4 tests)

- [ ] **Step 4: Run the full regression suite**

Run: `python -m pytest -q`
Expected: all tests pass (no regressions in existing `/v1/*` routes or app startup)

- [ ] **Step 5: Commit**

```bash
git add backend/api/main.py pyproject.toml uv.lock
git commit -m "feat: wire dashboard UI router and static files into the app, bump to v0.6.1"
```

---

### Task 10: Manual verification and release

**Files:** None (verification + tagging only)

- [ ] **Step 1: Start the app locally**

```bash
source .venv/bin/activate
uvicorn backend.api.main:app --reload --port 8000
```

- [ ] **Step 2: Verify empty-state rendering on a fresh database**

Open `http://localhost:8000/dashboard` in a browser (or `curl -s http://localhost:8000/dashboard`). Confirm every section shows its empty-state placeholder ("No requests yet.", "No verification results available.", "No recommendations yet.", "No failovers recorded.") rather than a blank/broken chart or table.

- [ ] **Step 3: Verify live data rendering**

Send a few requests through `POST /v1/chat` (or reuse the app's existing seeding/testing flow) to populate the database, then reload `/dashboard`. Confirm: Provider Health cards show real availability/circuit state, Cost Analytics and Quality Analytics charts render with Chart.js (open browser devtools console — no JS errors), Recent Requests table shows the new rows.

- [ ] **Step 4: Verify HTMX polling and the last-updated indicator**

With devtools Network tab open, confirm `GET /dashboard/fragments/overview`, `/providers`, `/circuits`, `/recent-requests` each fire every 15 seconds. Confirm the "Last updated" timestamp in the header updates after each poll (via the `hx-swap-oob` swap) without a full page reload, and that charts do NOT re-render on the poll cycle.

- [ ] **Step 5: Run the full regression suite one final time**

Run: `python -m pytest -q`
Expected: all tests pass

- [ ] **Step 6: Tag the release**

```bash
git tag v0.6.1
```

- [ ] **Step 7: Update the progress ledger**

Append to `.superpowers/sdd/progress.md`:

```markdown

## Phase 6b: Operations Dashboard UI
- Backend additions (Batch 1): DashboardRepository.get_quality_trend, .get_failover_events,
  .get_recent_requests, .get_cost_by_model; DashboardService per-fragment/page methods.
  All read-only, no changes to existing endpoint contracts or Phase 5 write paths.
- UI (Batch 2): vendored htmx 1.9.12 + Chart.js 4.4.4, Jinja2 templates (base + dashboard +
  4 fragment partials), dashboard_ui router (/dashboard, /dashboard/fragments/{section}),
  empty states for fresh installs, last-updated indicator via hx-swap-oob.
- Manually verified: empty-state rendering, live data rendering, HTMX polling (15s),
  chart rendering, no console errors.
- Phase 6b COMPLETE, tag v0.6.1. Phase 6 (backend API + operator UI) is now finished
  end-to-end.
```

```bash
git add .superpowers/sdd/progress.md
git commit -m "docs: record Phase 6b completion in progress ledger"
```

---

## Post-plan: merge to main

Once all tasks are complete and verified, merge `phase6-dashboard` into `main` (same pattern used for Phase 6a) and confirm `v0.6.1` is reachable from `main`.
