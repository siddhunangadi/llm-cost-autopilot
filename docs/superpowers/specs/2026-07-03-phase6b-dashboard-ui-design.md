# Phase 6b: Operations Dashboard UI — Design Spec

Status: Frozen
Depends on: Phase 6a (Operations Dashboard API, v0.6.0, merged to main at commit b3d0142)
Target release: v0.6.1

## Purpose

Phase 6a shipped a read-only JSON API (`GET /v1/dashboard/overview`) for operational
state: provider health, circuit breakers, cost trend, quality metrics, failovers,
learning recommendations. No human-facing surface consumes it yet. Phase 6b adds
a server-rendered operator dashboard so a person can visually monitor the platform
without hitting the API directly.

This completes Phase 6 end-to-end: operators can both query the backend and watch
the platform live.

## Non-goals (deferred to v0.6.2)

- Dark mode
- Search / filter by provider or model
- Date-range picker
- CSV export
- Any charting on the 15s poll cycle (charts render once per page load, not live)
- Per-request latency and retry-count tracking (would require instrumenting
  Phase 5's `ChatService`/`ProviderExecutor` write path — out of scope for a
  dashboard-only feature; see "Recent Requests" below)

## Architecture

Server-rendered HTML, no new process, no build step, no new dependency ecosystem:

- New router `backend/api/routers/dashboard_ui.py` (HTML responses), separate
  from the existing JSON `backend/api/routers/dashboard.py`.
- Jinja2 templates under `backend/templates/`.
- Static assets under `backend/static/`: hand-written CSS, a vendored copy of
  Chart.js (no CDN dependency at runtime), and a small vanilla-JS file for
  HTMX wiring.
- New dependencies: `jinja2`, vendored `chart.js` static file. No React, no
  Node toolchain, no new frontend package ecosystem.
- No authentication is added — the dashboard matches the rest of the API,
  which has no auth today.

## Routes

- `GET /dashboard` — full page, all 8 sections, initial render.
- `GET /dashboard/fragments/{section}` — HTMX partial re-render for polled
  sections only: `overview`, `providers`, `circuits`, `recent-requests`.
  Polled client-side every 15s via `hx-trigger="every 15s"`.

Charts, recommendations, and the failover timeline are rendered once as part
of the full-page load and are **not** re-fetched on the poll cycle — redrawing
a chart every 15s adds complexity for data that doesn't change second-to-second.

## Backend additions (read-only, additive — no changes to Phase 5 write paths or existing endpoint contracts)

Three new methods on `DashboardRepository`, following the existing patterns in
`backend/services/dashboard_repository.py`:

1. **`get_quality_trend(window: TimeWindow) -> list[QualityTrendBucket]`**
   Daily buckets of `{date, average_score, pass_rate}` from `VerificationRow`,
   grouped the same way `get_cost_trend` groups `ResponseRow` by day.

2. **`get_failover_events(window: TimeWindow) -> list[FailoverEvent]`**
   Extends today's `get_failover_summary` (which only returns a list of
   request IDs that failed over). New method returns
   `{request_id, from_model, to_model, occurred_at}` per failover, derived by
   joining the two `RoutingEventRow` rows sharing a `request_id` within the
   window, ordered by `created_at`. `get_failover_summary` stays as-is (JSON
   API keeps its existing shape); this is a new method for the UI's timeline.

3. **`get_recent_requests(limit: int = 50) -> list[RecentRequestRow]`**
   Left-joins `RequestRow` + `ResponseRow` + `VerificationRow`, most recent
   first, returning `{request_id, model, cost, score, passed, complexity,
   strategy, created_at}`.

4. **Cost-by-model breakdown**: extend cost aggregation with a `by_model`
   split analogous to the existing `by_model`/`by_strategy`/`by_complexity`
   pattern already used in `QualityAggregation`, for the "provider/model cost
   split" chart.

`DashboardService` gains a corresponding method to assemble page data from
these (or the router calls the repository directly for read-only page
rendering — implementation detail for the plan phase).

### Recent Requests: field scope

The original UI concept included Latency and Retry Count columns. Neither is
currently measured or stored anywhere in the system — `ResponseRow` has no
latency field, and no code tracks per-request retry counts. Adding them would
mean instrumenting Phase 5's `ChatService`/`ProviderExecutor` write path, which
is a materially bigger and riskier change than a dashboard read layer.

**Decision**: v0.6.1's Recent Requests table shows only fields backed by
existing data: Request ID, Model (`RoutingEventRow.selected_model`), Cost,
Score/Pass, Complexity, Strategy, Timestamp. Latency and Retry Count are
dropped from this release.

## Page layout (`/dashboard`, single scrolling page)

In order:

1. **System Overview** — stat row: total requests, total cost, average quality
   score, pass rate, active providers, open circuits, failovers today.
   Polled fragment.
2. **Provider Health** — cards per provider: availability, circuit state,
   consecutive failures. Sourced from existing `DashboardOverview.providers`.
   Polled fragment.
3. **Cost Analytics** — Chart.js charts: cost/day, requests/day, average
   cost/request (from existing `cost_trend`), plus cost-by-model split (new).
   Rendered once per page load.
4. **Quality Analytics** — charts: pass rate/score over time (new
   `quality_trend`), quality by model/strategy/complexity (existing
   `by_model`/`by_strategy`/`by_complexity`). Rendered once per page load.
5. **Learning Recommendations** — cards from existing `recommendations`.
   Rendered once per page load.
6. **Failover Timeline** — list from new `failover_events` (from → to model,
   timestamp). Rendered once per page load.
7. **Circuit Breakers** — live per-provider state (closed/open/half-open).
   Polled fragment.
8. **Recent Requests** — table from new `get_recent_requests`. Polled fragment.

## Testing

- Route-level tests (pytest + FastAPI `TestClient`) for `GET /dashboard`
  (200, contains expected section markers) and each
  `GET /dashboard/fragments/{section}` endpoint.
- Repository unit tests for the 3 new `DashboardRepository` methods, following
  the existing style in `backend/tests/test_dashboard_repository.py`.
- No JavaScript test framework is introduced. HTMX polling and Chart.js
  rendering are verified manually in a browser as part of implementation
  sign-off, not by an automated test.

## Release plan

- **v0.6.1**: this spec — full dashboard UI, all 8 sections, backend
  extensions included.
- **v0.6.2** (later, separate spec): dark mode, search/filter, date-range
  picker, CSV export.
