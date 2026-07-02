# Phase 2 Retrospective: Intelligent Routing Engine

Date: 2026-07-02
Tag: `v0.2.0`

## What went well

- **Layered pipeline held up exactly as designed.** `PromptAnalyzer -> HeuristicComplexityClassifier -> RoutingPolicy -> RoutingStrategy -> ExplanationGenerator -> RoutingEngine` never needed a shortcut or backchannel between layers. Each stage only knew the layer directly below it.
- **`RoutingEngine` staying provider- and DB-ignorant paid off immediately.** It made `RoutingEngine` trivially testable with an in-memory `ModelRegistry` and no mocked HTTP calls, and kept `ChatService` as the single, obvious place to look for "what actually calls OpenAI and writes to the DB."
- **Single `RoutingConfigLoader` avoided three-parser drift.** Once `ClassifierPolicy`, `EligibilityPolicy`, and `BalancedStrategyWeights` all came from one already-validated `RoutingConfig`, none of `HeuristicComplexityClassifier`, `RoutingPolicy`, or `BalancedStrategy` needed to know about YAML or file paths at all.
- **`ClassificationResult.signals` + `ExplanationGenerator` separation worked as intended.** The explanation layer never recomputed why a prompt was scored a certain way — it just rendered signals the classifier had already produced. `RoutingEngine` stayed free of string-building logic.
- **String-keyed `RoutingConfig.policy` (not `ComplexityTier`-keyed) avoided a real circular import**, confirming this was a genuine constraint and not premature caution.
- **Test-first batching (one commit per 2-3 tasks) kept velocity high without losing safety** — 64 new tests landed across 5 feature commits with no regressions in the 97 Phase 1 tests at any point.

## What changed from the original spec

- **Integration test isolation had to extend one level deeper than planned.** The spec/plan only anticipated mocking `OpenAIProvider.generate()` for the end-to-end chat test. In reality, `ModelRegistry.refresh_provider_status()` — invoked during the real app `lifespan` — also calls `OpenAIProvider.health_check()`, which makes its own live network call. With a fake `sk-test` key this made the model appear unavailable and the test got a 503 instead of the expected 200. Fix: mock `health_check()` to return `True` alongside `generate()` in both integration tests. This is a test-boundary fix, not an architecture change — the provider abstraction itself was correct, the plan just hadn't enumerated every method that startup touches.

## Problems encountered

- No functional bugs surfaced in the core routing logic (classifier scoring, policy filtering, strategy selection, balanced-weight normalization all matched their test tables on the first implementation pass).
- The only friction was the `health_check()` network call above — an environmental/test-isolation issue, not a design flaw.

## Technical debt intentionally carried into Phase 3+

- **No quality verification of routing decisions.** Nothing currently checks whether the selected model's response was actually adequate for the prompt's complexity — Phase 2 only measures *predicted* cost/latency/quality, never *actual* output quality.
- **No streaming support** (`provider.stream()` unused) — `/v1/chat` is request/response only.
- **No `quality_profile` (per-category benchmark scores)** — `ModelSpec.benchmark_score` is still a single global number, not broken out by task category (code, reasoning, creative, etc.).
- **No runtime reload of `routing.yaml`** — loaded once at startup, same as `ModelRegistry` in Phase 1. A future phase could add a `reload()` following that precedent.
- **No background workers / async queues** — everything in the request path is synchronous within `ChatService.chat()`.
- **No retry policies, semantic caching, or rate limiting.**

## Lessons learned

- Writing the full implementation plan with exact file contents up front (rather than sketching and improvising during implementation) made delegation to a subagent nearly frictionless — it executed 6 batches with only one minor, well-contained deviation.
- Enumerating "explicitly out of scope" items in the spec (as Phase 2 did) is worth doing every phase — it made the one deviation easy to classify as "test isolation fix" rather than "scope creep," because the actual scope boundary was already written down.
- Next time, the plan should trace *every* method a component's constructor-time or startup-time code path touches (not just the "main" call like `generate()`) before writing integration test mocks — `health_check()` was a one-line miss that cost real debugging time.
