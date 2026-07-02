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
- Jinja2 templates under `backend/templates/`, structured for reuse rather
  than one large file:

  ```text
  templates/
      base.html                    # shared shell: <head>, nav, static asset links
      dashboard.html               # extends base.html; full-page layout, all 8 sections
      fragments/
          overview.html
          providers.html
          circuits.html
          recent_requests.html
  ```

  `dashboard.html` extends `base.html` via Jinja2 template inheritance
  (`{% extends "base.html" %}`), and includes each fragment template both for
  its initial render and reuses the same fragment templates for the
  `/dashboard/fragments/{section}` HTMX responses — one template per polled
  section, no duplication between full-page and fragment rendering.

- Static assets under `backend/static/`: hand-written CSS, a vendored copy of
  Chart.js (no CDN dependency at runtime), and a small vanilla-JS file for
  HTMX wiring.
- **Static asset versioning**: asset URLs carry a cache-busting query param
  tied to the app version, e.g. `<link href="/static/css/dashboard.css?v=0.6.1">`,
  `<script src="/static/js/dashboard.js?v=0.6.1">`. The version string comes
  from the existing app version constant (same one used for the `v0.6.x` tags)
  so it's bumped automatically each release rather than hand-maintained per file.
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

**Service methods are split per-page and per-fragment, not one method that
returns everything.** A fragment endpoint polled every 15s should only fetch
the data that fragment needs, not rebuild the whole dashboard on every poll:

```python
DashboardService.get_dashboard_page(window)        # full page: all 8 sections' data
DashboardService.get_overview_fragment(window)      # stat row only
DashboardService.get_provider_fragment()            # provider health cards only
DashboardService.get_circuit_fragment()             # circuit breaker states only
DashboardService.get_recent_requests_fragment()     # recent requests table only
```

`get_dashboard_page` composes the full initial render (used once, by
`GET /dashboard`); each `get_*_fragment` method is used both by the initial
page render (to avoid a second round-trip) and by its corresponding
`GET /dashboard/fragments/{section}` poll — same method, two callers.

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

### Empty states

A fresh install has no requests, no verification data, no recommendations, no
failovers. Every section that renders a list/chart from possibly-empty data
must render a friendly placeholder instead of a blank chart or empty table:

- Recent Requests / Provider Health / Circuit Breakers: "No requests yet."
- Quality Analytics / Cost Analytics charts: skip the chart, show
  "No verification results available." / "No cost data yet."
- Learning Recommendations: "No recommendations yet."
- Failover Timeline: "No failovers recorded."

Each template checks for an empty collection before invoking Chart.js or
rendering a table, so first-run and low-traffic deployments look intentional
rather than broken.

### Last updated indicator

The page header shows a "Last updated: `<UTC timestamp>`" line. Each polled
fragment response updates this timestamp client-side (via a small HTMX
`hx-swap-oob` out-of-band swap included in every fragment response) so an
operator can tell at a glance whether polling is still working, without
needing to watch the data itself change.

## Testing

- Route-level tests (pytest + FastAPI `TestClient`) for `GET /dashboard`
  (200, contains expected section markers) and each
  `GET /dashboard/fragments/{section}` endpoint.
- Repository unit tests for the 3 new `DashboardRepository` methods, following
  the existing style in `backend/tests/test_dashboard_repository.py`.
- Unit tests for each `DashboardService.get_*_fragment`/`get_dashboard_page`
  method, including the empty-data case (no requests/verifications/
  recommendations/failovers) to confirm empty-state data is returned rather
  than an error.
- No JavaScript test framework is introduced. HTMX polling and Chart.js
  rendering are verified manually in a browser as part of implementation
  sign-off, not by an automated test.

## Release plan

- **v0.6.1**: this spec — full dashboard UI, all 8 sections, backend
  extensions included.
- **v0.6.2** (later, separate spec): dark mode, search/filter, date-range
  picker, CSV export.

## Implementation workflow

Following the same subagent-driven, batch-and-review workflow used for
Phase 6a:

1. Write the Phase 6b implementation plan.
2. Implement in two batches:
   - Batch 1: backend additions (repository methods, service fragment/page
     methods) — read-only, no template/route work yet.
   - Batch 2: UI (templates, static assets, `dashboard_ui.py` routes, HTMX
     wiring, empty states, last-updated indicator).
3. Run the full regression suite after each batch.
4. Manual browser verification: HTMX polling, chart rendering, empty states
   on a fresh database.
5. Tag v0.6.1 once complete.
