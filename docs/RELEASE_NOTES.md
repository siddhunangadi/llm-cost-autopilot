# LLM Cost Autopilot — Release Notes (v0.1.0 → v0.3.0)

An intelligent, cost-aware LLM routing platform: it analyzes a prompt,
classifies its complexity, routes it to the cheapest model that can
handle it under a chosen strategy, and — as of v0.3.0 — verifies whether
that routing decision actually produced a good answer.

## Architecture

```
FastAPI
                    │
      ┌─────────────┼─────────────┬─────────────┐
      │             │             │             │
 Health API     Models API     Chat API    Metrics API
      │             │             │             │
      └─────────────┴─────────────┼─────────────┘
                                   │
                             ChatService
                                   │
                 ┌─────────────────┴─────────────────┐
                 │                                     │
          Routing Engine                  Background Verification
                 │                                     │
     Prompt Analysis                              Judge Engine
     Complexity Classification                    LLM Judge
     Routing Policy + Strategies                  Verification DB
                 │
          Provider Manager
                 │
          OpenAI / Mock
```

Every layer talks only to the layer directly below it. `RoutingEngine`
never calls a provider or touches the database; `BaseJudge`/`LLMJudge`
never touch the database, retry, or emit events. `ChatService` is the
sole component that spans routing, providers, persistence, and
verification scheduling — everything else stays single-purpose and
independently testable.

## v0.1.0 — Project Skeleton & Provider Foundation

The foundation: a provider abstraction (`OpenAIProvider`/`MockProvider`
behind `BaseProvider`), a `ModelRegistry` backed by YAML config and
persisted to SQLite, an in-process `EventBus`, and structured JSON
logging. `ProviderManager` treats `mock` as mandatory (crashes startup if
broken) and `openai` as optional (degrades to "disabled" if
unconfigured) — establishing the "fail loudly on what must work, degrade
gracefully on what's optional" discipline the rest of the project follows.

**97 tests.**

## v0.2.0 — Intelligent Routing Engine

A full heuristic routing pipeline: `PromptAnalyzer` extracts deterministic
features from a prompt (code/reasoning/comparison/analysis/math signals,
constraint counts, an output-length heuristic distinct from input
length); `HeuristicComplexityClassifier` scores those features into a
`simple`/`medium`/`complex` tier with human-readable `signals` and a
confidence score; `RoutingPolicy` filters eligible models per tier;
one of four pluggable strategies (`cost`, `latency`, `quality`,
`balanced`) picks the model; `ExplanationGenerator` turns the classifier's
own signals into readable reasoning. `POST /v1/chat` ties it together and
persists every request, routing decision, and response.

Configuration for the classifier, eligibility policy, and balanced-
strategy weights all comes from one YAML file through a single
`RoutingConfigLoader` — no component parses YAML itself.

**161 tests (+64).**

## v0.3.0 — Quality Verification & Evaluation

Answers the question Phase 2 couldn't: *was the routing decision actually
good?* After `ChatService` returns a response to the client, it schedules
an in-process background task that asks an LLM judge to score the
response on four dimensions (correctness, completeness, instruction
following, format adherence), computes an overall score as their mean,
and persists the verdict — all without adding latency to `/v1/chat` or
risking its availability if the judge is unavailable.

`VerificationService` owns a strict state machine
(`PENDING → RUNNING → COMPLETED | FAILED`), always persisting to the
database before emitting an event, and always snapshotting which routing
decision (model, strategy, complexity) is being verified so the result
stays meaningful even if routing configuration changes later.
`GET /v1/chat/{request_id}/verification` exposes a single result;
`GET /v1/metrics/quality` exposes aggregate pass rate, average score, and
timing/breakdown metrics by model, strategy, and complexity.

**190 tests (+29).**

## Engineering Process

Every phase followed the same cycle: brainstorm → frozen design spec →
implementation plan with exact file contents and TDD steps → batched
implementation with a full regression run and one manual end-to-end
verification per batch → tagged release. Specs and plans live under
`docs/superpowers/specs/` and `docs/superpowers/plans/`; a written
retrospective after Phase 2 fed directly into sharper constraints for
Phase 3's plan (e.g. tracing every method a component's startup path
touches, not just its "main" call, before writing integration test
mocks).

## What's Explicitly Not Built Yet

By design — each deferred to keep every phase's scope provable:
ML-based classification, LLM-as-judge auto-escalation, classifier
retraining or online learning, a feedback loop from verification back
into routing, streaming responses, semantic caching, rate limiting, and
distributed background workers.
