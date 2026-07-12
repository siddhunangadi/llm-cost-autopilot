# Benchmarks

Run:

```bash
python -m benchmarks.run_benchmarks
```

Writes `benchmarks/report.md` and prints the same report to stdout.

Validates the claims in `CLAUDE.md`'s Performance Targets section:

- **Routing latency** (< 50ms target) and **classifier latency**
  (< 10ms target), measured by timing the real `RoutingEngine`/
  `HeuristicComplexityClassifier` directly.
- **Load test**: 500 requests through the routing engine, reporting cost
  savings vs. the highest-cost model, quality parity, and routing
  distribution.
- **Provider failover**: drives a real `CircuitBreaker` through
  closed -> open -> half-open -> closed against a provider scripted to
  fail 3 times then recover, asserting each transition actually happens.

Uses the production `backend/config/models.yaml` and `routing.yaml` with
`MockProvider` swapped in for every provider -- no network calls, no real
API keys required, fully reproducible.

A pytest smoke test (`backend/tests/test_benchmarks.py`) runs the same
functions at low volume to catch import/signature drift between full runs.
