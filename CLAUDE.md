# CLAUDE.md

# LLM Cost Autopilot

## Purpose

This repository implements an intelligent LLM routing platform that minimizes inference cost while maintaining response quality.

This project is **already partially implemented**.

Your job is NOT to rebuild the project.

Your job is to:

- inspect the repository
- understand the architecture
- compare implementation against the project specification
- implement only missing functionality
- improve existing implementations
- verify correctness
- continue iterating until all acceptance criteria are satisfied

Never duplicate existing code.

---

# Product Vision

LLM Cost Autopilot is not merely an LLM router. It is an AI Cost
Optimization platform.

Every feature should help the user answer one or more of these questions:

- How much money am I spending?
- How much money am I saving?
- Why was this model selected?
- Can I trust this routing decision?
- How can I reduce my AI bill further?

Features that improve engineering quality but do not improve one of these
user outcomes should have lower priority than features that do.

## User Journey

1. **First visit**: "How much money did I save?" (Savings vs Baseline KPI)
2. **Second question**: "Which model handled my traffic?" (routing distribution)
3. **Third**: "Why did you choose that model?" (explainable routing decision card)
4. **Fourth**: "Can I save even more?" (waste detection, optimization recommendations)

This journey is the design philosophy: build in this order, not by
engineering convenience.

## Product Acceptance

In addition to the engineering Acceptance Checklist below, a feature is
not done until:

- User immediately understands savings
- Routing decisions are explainable
- Dashboard surfaces actionable insights
- Users can identify waste
- Cost reduction is obvious
- Product builds confidence
- Product demonstrates business value

## Decision Hierarchy

When implementing features, answer in order and stop at the first "no":

1. Does this improve user value?
2. Does this preserve architecture?
3. Can existing code be extended?
4. Can configuration solve it?
5. Is this the smallest implementation?
6. Are tests added?
7. Does it improve the portfolio?

This hierarchy exists to prevent chasing technically interesting work that
doesn't improve the product. Before building a new subsystem, inspect
whether an existing one (`analysis/`, `routing/`, `learning/`,
`verification/`, `dashboard/`) already does it -- e.g. "waste detection"
turned out to already exist as `OverpoweredModelRule` +
`RecommendationGenerator` in `backend/learning`, needing only a
presentation fix, not a new service.

---

# Engineering Philosophy

Always prefer:

- extending existing code
- small incremental changes
- reusable abstractions
- strongly typed models
- production-ready implementations
- maintainability over cleverness

Avoid:

- unnecessary rewrites
- duplicate provider implementations
- duplicate API routes
- duplicate database models
- large refactors without need
- introducing technical debt

---

# Repository Architecture

Read these modules before making changes.

backend/

    analysis/
        prompt analysis
        feature extraction

    api/
        FastAPI routes

    chat/
        request handling

    classifier/
        complexity prediction

    config/
        YAML configuration
        settings

    database/
        SQLAlchemy models

    events/
        internal event bus

    learning/
        routing feedback
        metrics
        learning pipeline

    providers/
        provider abstraction
        retry
        circuit breaker

    routing/
        routing engine

    telemetry/
        logging
        metrics

    verification/
        async quality verification

tests/

Always understand how these modules interact before modifying them.

---

# Project Objectives

The completed system should provide:

✓ Unified provider interface

✓ Cost-aware routing

✓ Complexity classifier

✓ Configurable routing rules

✓ Async quality verification

✓ Automatic escalation

✓ Feedback learning loop

✓ Complete logging

✓ Cost analytics

✓ Dashboard

✓ FastAPI API

✓ Docker deployment

---

# Ground Rules

## 1. Inspect Before Coding

Before implementing anything:

- inspect repository
- search for similar implementation
- understand execution flow
- identify reusable abstractions

Never assume something is missing.

---

## 2. Minimal Changes

Modify as few files as possible.

Reuse existing architecture.

Prefer extending over replacing.

---

## 3. Preserve Architecture

Provider logic belongs only inside:

backend/providers

Routing logic belongs only inside:

backend/routing

Verification belongs only inside:

backend/verification

Learning belongs only inside:

backend/learning

Never mix responsibilities.

---

## 4. Configuration First

Never hardcode:

- model names
- prices
- thresholds
- routing rules
- provider settings

Configuration belongs in:

backend/config

---

## 5. Strong Typing

Prefer:

Pydantic

Dataclasses

Enums

Typed return values

Avoid dictionaries for internal APIs.

---

# Provider Rules

Every provider should expose a consistent interface.

Provider SDKs should never leak outside provider modules.

Retry logic must reuse existing retry utilities.

Circuit breaker must reuse existing implementation.

Never duplicate pricing calculations.

---

# Routing Rules

Routing decisions must be explainable.

Every routing decision should include:

- predicted complexity
- selected model
- routing reason
- estimated cost
- estimated latency

Routing should remain configurable via YAML.

---

# Verification Rules

Verification must always run asynchronously.

User latency must not increase because of verification.

Verifier responsibilities:

- compare outputs
- score agreement
- detect routing failures
- log quality
- trigger escalation
- feed learning pipeline

---

# Learning Rules

Routing failures become training examples.

Learning pipeline should:

collect failures

↓

store dataset

↓

retrain classifier

↓

improve routing

Never discard useful failure data.

---

# Logging

Every request should capture:

timestamp

request id

prompt hash

complexity

selected model

provider

latency

token usage

estimated cost

verification score

escalation

final model

routing reason

Never log raw prompts unless explicitly configured.

---

# Testing

Every new feature requires tests.

Always verify:

unit tests

integration tests

API tests

routing tests

verification tests

provider tests

If a bug is fixed:

add a regression test.

---

# Performance Targets

Classifier

<10ms

Routing overhead

<50ms

Verification

fully async

Provider calls

async

Avoid blocking the event loop.

---

# Code Style

Prefer:

small functions

dependency injection

pure functions

composition

clear naming

Avoid:

nested conditionals

magic numbers

long functions

global state

duplicate logic

---

# Definition of Done

A feature is NOT complete until:

✓ implementation finished

✓ tests added

✓ tests passing

✓ lint passing

✓ type checking passing

✓ documentation updated

✓ architecture preserved

✓ configuration updated if necessary

✓ no TODO placeholders

✓ no duplicate logic

---

# Autonomous Engineering Workflow

For every task execute the following loop.

## Phase 1

Inspect repository.

Read relevant modules.

Understand architecture.

---

## Phase 2

Compare current implementation against the project specification.

Produce a gap analysis.

Determine exactly what is missing.

---

## Phase 3

Create an implementation plan.

Identify:

- files to modify
- tests to update
- risks
- dependencies

---

## Phase 4

Implement the smallest working solution.

---

## Phase 5

Run verification.

Execute:

pytest

ruff

black --check

mypy

or project equivalents.

Fix failures automatically.

---

## Phase 6

Review implementation.

Check for:

duplicate code

dead code

architecture violations

missing logging

missing tests

poor naming

Fix any issues.

---

## Phase 7

Compare against acceptance criteria.

If anything remains incomplete:

return to Phase 3.

Repeat until complete.

---

# Stop Conditions

Only stop when ALL are true.

✓ Tests pass

✓ Lint passes

✓ Type checking passes

✓ No failing verification

✓ Acceptance criteria satisfied

✓ Documentation updated

✓ No incomplete implementations

---

# Acceptance Checklist

The completed project should satisfy:

□ Unified provider abstraction

□ Provider failover

□ Cost-aware routing

□ Complexity classifier

□ Configurable routing

□ Async verification

□ Agreement scoring

□ Auto escalation

□ Feedback dataset generation

□ Continuous learning

□ Cost metrics

□ Dashboard metrics

□ API endpoints

□ Database logging

□ Structured telemetry

□ Docker support

□ End-to-end tests

---

# Verification Before Completion

A feature is not considered complete merely because code exists.

Claude must verify every documented claim.

Examples:

If documentation claims:

"Routing overhead <50ms"

Claude should:

- locate benchmark
- create benchmark if absent
- execute benchmark
- record results

If documentation claims:

"Classifier latency <10ms"

Claude should:

- measure inference time
- verify average latency
- report results

If documentation claims:

"Provider failover"

Claude should:

- simulate provider failure
- verify fallback behavior
- verify logging
- verify recovery

If verification is impossible because external infrastructure is unavailable, explain why.

---

# Specification Interpretation

The project specification defines the minimum expected functionality.

It is NOT a restriction on improvements.

When comparing the implementation against the specification:

- Never remove a feature simply because it is not mentioned.
- Never replace a superior implementation with a simpler one solely to match the specification.
- Treat the specification as a baseline.

If the repository contains an implementation that is objectively better, retain it.

---

# End User Perspective

Always evaluate changes from the perspective of the user.

Ask:

Would I enjoy using this product?

Would this feature make the experience better?

Would this interface be intuitive?

Would this response inspire confidence?

Would this dashboard help me understand costs?

Would this API be pleasant to integrate?

If the answer is no, improve it.

---

# Dashboard Evaluation

The specification suggests Streamlit or Grafana.

This repository may use another frontend.

Do NOT replace an existing dashboard simply because it differs from the suggested implementation.

Instead evaluate:

- usability
- responsiveness
- accessibility
- clarity
- performance
- maintainability
- deployment simplicity

Only replace a dashboard if the replacement is objectively superior.

---

# Preserve Existing Improvements

This repository may contain features beyond the original specification.

Examples:

- improved dashboard
- richer analytics
- better UI
- better routing
- additional providers
- caching
- telemetry
- monitoring
- retries
- circuit breakers
- authentication

Treat these as project assets.

Never remove them merely to conform to the original specification.

Only replace an existing implementation if the new implementation is demonstrably better.

---

# Multi-Perspective Review

Before completing any feature, evaluate it from four perspectives.

1. End User

Can someone use this product without confusion?

Is it intuitive?

Is the experience pleasant?

---

2. Backend Engineer

Is the code maintainable?

Is it testable?

Is it scalable?

---

3. DevOps Engineer

Can it be deployed?

Is it observable?

Is it reliable?

---

4. Hiring Manager

Would this implementation impress an interviewer?

Does it demonstrate engineering maturity?

Would it strengthen the portfolio?

If improvements are obvious, implement them.

---

# Product Quality

Do not stop at functional correctness.

Evaluate:

- UX
- consistency
- naming
- documentation
- error messages
- loading states
- empty states
- accessibility
- responsiveness
- visual polish
- API ergonomics

The project should feel like a polished product, not merely a completed assignment.

---

# Innovation

The specification is a starting point.

Reasonable improvements are encouraged.

When adding improvements:

- preserve compatibility
- document the change
- explain the benefit
- avoid unnecessary complexity

Innovation is preferred over strict conformity.

---

# Behavior Expectations

When working autonomously:

Never invent architecture without inspecting.

Never replace working implementations.

Always reuse existing abstractions.

Always explain why a change is required.

Always make the smallest safe modification.

Prefer fixing over rewriting.

Think like a senior backend engineer maintaining a production system.
