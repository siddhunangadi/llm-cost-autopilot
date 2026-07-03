# Phase 8: Advanced Analytics & Reporting — Design Spec (v0.8.0)

## 1. Goal

Add historical, trend-oriented analytics on top of the existing Phase 6/6b operations dashboard and Phase 7 optimization engine. Where the operations dashboard answers "what is happening right now," Phase 8 answers "how has the platform evolved over the last N days." This is the final planned phase (roadmap frozen after v0.8.0).

## 2. Scope

In scope:
- Daily trend for cost (reuse existing)
- Daily trend for quality (reuse existing)
- Daily trend for failover count (new, direct query)
- Daily distribution of routing decisions by model (new, direct query)
- Daily trend of recommendations generated / still open (new, direct query)
- One new read-only service (`AnalyticsService`) producing one domain object (`AnalyticsReport`)
- One new REST endpoint returning `AnalyticsReport` as JSON
- One new dashboard page rendering the same `AnalyticsReport`

Out of scope (explicitly deferred, not part of v0.8.0):
- Spend forecasting / predictive modeling
- Provider-performance-over-time (latency/availability trends) — no existing data source distinguishes provider-level daily aggregates today
- Recommendation *impact* tracking (dollars actually saved after a recommendation was acted on) — would require tracking acceptance/adoption, which doesn't exist
- Any auto-refreshing/polling UI — analytics is historical, not operational
- Any new database tables or migrations — all new queries read existing tables

## 3. Architecture

```
DashboardRepository (extended)
        │
        ├── get_cost_trend(window)              [existing, reused]
        ├── get_quality_trend(window)            [existing, reused]
        ├── get_failover_trend(window)           [new]
        ├── get_routing_distribution(window)     [new]
        └── get_recommendation_trend(window)     [new]

                        │
                        ▼
                 AnalyticsService
                        │
                        ▼
                 AnalyticsReport (Pydantic domain object)
                    │           │
                    ▼           ▼
        GET /v1/analytics/report   GET /dashboard/analytics
              (JSON)                  (Jinja2, single render)
```

`AnalyticsService` is the single producer of truth. It has no knowledge of HTTP, JSON serialization, or HTML — it returns one `AnalyticsReport` Pydantic model. The API router serializes it as JSON; the UI router passes it into a Jinja2 template. Neither consumer re-derives data — both read the same object.

This mirrors the existing `DashboardService` → `DashboardOverview` pattern exactly, so Phase 8 introduces no new layering concept.

## 4. Data model

### 4.1 Shared base

```python
class DailyBucket(BaseModel):
    date: date
```

All new trend buckets subclass this for a consistent repository return shape. Existing `CostBucketData` / `QualityTrendBucket` (dataclasses in `dashboard_repository.py`) are left as-is — Phase 8 does not touch Phase 6 code. New buckets for Phase 8 follow the same `date`-first field convention for consistency but are defined as their own dataclasses in `dashboard_repository.py`, matching the existing dataclass style there (not Pydantic — the repository layer uses plain dataclasses throughout; Pydantic is used only at the service/API boundary, exactly as `DashboardService` does today for `CostBucket` vs repository's `CostBucketData`).

### 4.2 New repository dataclasses (`backend/services/dashboard_repository.py`)

```python
@dataclass(frozen=True)
class FailoverTrendBucket:
    date: date
    failover_count: int

@dataclass(frozen=True)
class RoutingDistributionBucket:
    date: date
    model: str
    request_count: int

@dataclass(frozen=True)
class RecommendationTrendBucket:
    date: date
    generated_count: int
    open_count: int
```

### 4.3 New service/API Pydantic models (`backend/services/analytics_service.py`)

```python
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
```

`AnalyticsService` maps repository dataclasses → these Pydantic models, same translation step `DashboardService.get_overview` already performs for cost/quality data.

## 5. Repository queries

All three new methods take `window: TimeWindow` (existing dataclass, reused as-is — no new window abstraction) and execute a single grouped SQL query each. No Python-side re-bucketing of already-fetched rows.

- **`get_failover_trend(window)`**: groups by `date(created_at)` using the identical failover predicate that `get_failover_events`/`get_failover_summary` already use (implementation must read that existing predicate and reuse it verbatim, just grouped/counted by day instead of listed), returns count per day.
- **`get_routing_distribution(window)`**: groups `RoutingEventRow` by `(date(created_at), selected_model)`, returns count per group.
- **`get_recommendation_trend(window)`**: groups `RecommendationRow` by `date(created_at)` for `generated_count`. `open_count` is a simplification: count of recommendations still open **as of report generation time**, attributed to the day they were generated — not a true point-in-time daily open count (that would require a status-change history table, which doesn't exist and is out of scope). This simplification is called out in a code comment and in the implementation plan.

`get_failover_events(window)` (existing, Phase 6b) is left completely untouched and continues to serve the failover timeline UI. `get_failover_trend` is a new, independent, directly-grouped query — not built on top of `get_failover_events`.

## 6. Service

```python
class AnalyticsService:
    def __init__(self, dashboard_repository: DashboardRepository) -> None:
        self._dashboard_repository = dashboard_repository

    async def get_report(self, window: TimeWindow) -> AnalyticsReport:
        ...
```

Invariant, enforced by never importing a write-capable dependency: `AnalyticsService` never writes. It never calls `LearningService` methods that regenerate/refresh recommendations, never touches `ProviderManager`/`ProviderExecutor`, never modifies routing or learning state. It only reads through `DashboardRepository`, exactly as `DashboardService` does for its existing reads.

Repository calls run via `asyncio.to_thread` / `asyncio.gather` (matching the existing pattern in `DashboardService.get_overview`).

## 7. API

- `GET /v1/analytics/report?days=30` — new router, `backend/api/routers/analytics.py`. Query param `days` (default `30`, reused as `TimeWindow(days=days)`) — makes the API range-flexible from day one even though the UI defaults to 30 and doesn't expose a selector yet. Returns `AnalyticsReport` JSON.

No mutation endpoints. No auth changes (matches existing router auth posture — none of the other dashboard/analytics endpoints have auth today).

## 8. UI

- `GET /dashboard/analytics?days=30` — new route, same `analytics.py` router (owns both JSON and HTML since both share `AnalyticsService`).
- Renders `backend/templates/analytics.html` once per request from a single `AnalyticsReport` fetch. **No HTMX polling, no fragment endpoints.** Refresh = reload the page.
- Reuses the shared static assets/layout established in Phase 6b (Task 6: static assets and shared path constants) rather than duplicating CSS.
- Charts: reuse whatever charting approach Phase 6/6b already uses for the operations dashboard (confirm during implementation — if none exists yet, use minimal server-rendered tables/sparkline-style bars rather than introducing a new JS charting dependency, to keep this phase's footprint small).
- Add a nav link from the existing `/dashboard` page to `/dashboard/analytics`.

## 9. Testing

TDD, matching every prior phase's pattern: failing tests first, then implementation, per method/layer.

- `backend/tests/test_dashboard_repository.py` (existing file, extended): new tests for `get_failover_trend`, `get_routing_distribution`, `get_recommendation_trend`.
- `backend/tests/test_analytics_service.py` (new): tests `AnalyticsService.get_report` composes all five trends correctly, and asserts it never calls any write-capable method (verifies the read-only invariant via a mock repository/dependencies that fail the test if invoked).
- `backend/tests/test_analytics_router.py` (new): tests `GET /v1/analytics/report` (JSON shape, `days` query param honored) and `GET /dashboard/analytics` (200, renders expected content, no polling attributes present in HTML — guards requirement §8 directly).

## 10. Versioning

Bump `pyproject.toml` version to `0.8.0` on completion (matching the version-bump pattern used at the end of every prior phase, e.g. Phase 7's bump to `0.7.0`).

## 11. Non-goals / explicit invariants (restated)

- `AnalyticsService` never writes, never refreshes recommendations, never modifies routing, never modifies learning state — read-only, exactly like `DashboardService`.
- No new database tables or migrations.
- No polling/auto-refresh on the analytics page.
- This is the last phase per the frozen roadmap — no Phase 9 scope creep (e.g. no forecasting) is to be added here even if it would be "almost free."
