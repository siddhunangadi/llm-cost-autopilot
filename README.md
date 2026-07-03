# LLM Cost Autopilot

Phase 1: project skeleton, provider foundation, model registry, and event
bus for an intelligent cost-aware LLM routing layer.

## What exists today

- `GET /v1/health` — service, database, and provider status
- `GET /v1/models` — full model registry (pricing, limits, capabilities, benchmark info)
- `POST /v1/chat` — routes a prompt through prompt analysis, heuristic
  complexity classification, and a configurable strategy (`cost`,
  `latency`, `quality`, `balanced`) to select a model, then returns the
  response plus a full routing explanation (complexity, confidence,
  estimated cost/latency, human-readable reasoning)
- `ModelRegistry` — memory-first, backed by `backend/config/models.yaml` and persisted to SQLite; immutable read-only cache, atomic `reload()`/`refresh_provider_status()`, fails fast on malformed/invalid/duplicate config
- `OpenAIProvider` and `MockProvider` behind a shared `BaseProvider` interface; SDK exceptions are translated into `ProviderError`, never leaked to the rest of the app
- `ProviderManager` — the mandatory `mock` provider crashes startup if it fails to construct; the optional `openai` provider degrades gracefully to "disabled" (logged) if its construction fails
- In-process event bus (`PROVIDER_AVAILABLE`, `PROVIDER_DISABLED`, `PROVIDER_FAILED`, `MODEL_REGISTERED`), with per-subscriber exception isolation
- Structured JSON logging (console + rotating file) with `contextvars`-based request context

Routing, classification, verification, and the dashboard are not built yet
— see `docs/superpowers/specs/` for the full roadmap.

## Setup

```bash
uv sync
cp .env.example .env   # add OPENAI_API_KEY to enable the OpenAI provider
```

## Run tests

```bash
uv run pytest
```

## Run the API

```bash
uv run uvicorn backend.api.main:app --reload
```

Then:

```bash
curl http://127.0.0.1:8000/v1/health
curl http://127.0.0.1:8000/v1/models
```

### Example responses

`GET /v1/health`:

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "environment": "development",
  "database": "healthy",
  "providers": {"openai": "disabled", "anthropic": "disabled", "ollama": "disabled", "gemini": "disabled", "nvidia_nim": "disabled", "openrouter": "disabled", "groq": "disabled", "mistral": "disabled"},
  "loaded_models": 2,
  "uptime_seconds": 14.4
}
```

`GET /v1/models` (one entry shown):

```json
[
  {
    "id": "gpt-4o-mini",
    "provider": "openai",
    "model": "gpt-4o-mini",
    "input_cost": 0.15,
    "output_cost": 0.6,
    "context_window": 128000,
    "max_output_tokens": 16384,
    "supports_streaming": true,
    "supports_tools": true,
    "supports_json": true,
    "supports_vision": false,
    "benchmark_score": 0.82,
    "average_latency_ms": 450.0,
    "available": false
  }
]
```

`providers` shows `"disabled"` and `available` is `false` above because no
`OPENAI_API_KEY` was configured when this was captured — set one in `.env`
to see `"available"`/`true`.

`POST /v1/chat`:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain why the sky is blue.", "strategy": "balanced"}'
```

```json
{
  "request_id": "b3f1...",
  "response": "...",
  "routing": {
    "selected_model": "gpt-4o-mini",
    "strategy": "balanced",
    "complexity": "simple",
    "confidence": 0.66,
    "estimated_cost": 0.00013,
    "estimated_latency_ms": 450.0,
    "reasoning": [
      "Classified as simple (confidence 0.66): reasoning keywords detected.",
      "Strategy 'balanced' evaluated 2 eligible model(s).",
      "Selected 'gpt-4o-mini'."
    ]
  }
}
```

### Phase 3: Quality Verification

Every `/v1/chat` call schedules an in-process, best-effort background
task that scores the response with an LLM-as-judge and persists the
verdict — this never adds latency to `/v1/chat` or risks its
availability. Poll the result once it completes:

```bash
curl http://127.0.0.1:8000/v1/chat/<request_id>/verification
```

```json
{
  "request_id": "b3f1...",
  "status": "completed",
  "score": 0.9,
  "passed": true,
  "confidence": 0.9,
  "rationale": "The response correctly and completely answers the prompt.",
  "dimensions": {
    "correctness": 0.9,
    "completeness": 0.9,
    "instruction_following": 0.9,
    "format_adherence": 0.9
  },
  "judge_model": "gpt-4o",
  "judge_prompt_version": "v1",
  "evaluation_duration_ms": 812,
  "error_type": null,
  "error": null,
  "created_at": "2026-07-02T05:10:00Z",
  "started_at": "2026-07-02T05:10:00Z",
  "completed_at": "2026-07-02T05:10:01Z"
}
```

`GET /v1/metrics/quality` aggregates every completed/failed verification
(average score, pass rate, timing, and per-model / per-strategy /
per-complexity breakdowns):

```bash
curl http://127.0.0.1:8000/v1/metrics/quality
```
