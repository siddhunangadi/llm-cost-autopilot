# LLM Cost Autopilot

An intelligent cost-aware LLM routing layer: it classifies prompt
complexity, routes each request to the best model under a configurable
strategy (cost, latency, quality, balanced), verifies response quality
with an LLM-as-judge, learns from outcomes, and exposes an operations
dashboard for analytics, failover events, and provider configuration.

Currently at v0.9.1, with 10 phases shipped — see `docs/superpowers/specs/`
and `CHANGELOG.md` for the full history:

1. Project skeleton, provider foundation, model registry, event bus
2. Routing engine — complexity classification and strategy-based model selection
3. Quality verification — LLM-as-judge scoring of responses
4. Learning — outcome-driven recommendations (including cost optimization)
5. Resilience — circuit breakers and failover across providers
6. Operations dashboard (backend + UI overhaul)
7. Cost/latency optimization
8. Analytics — quality trends, recent requests, per-model cost breakdowns
9. Live provider configuration (encrypted, hot-reloadable credentials)
10. Provider expansion — Anthropic, Gemini, Groq, Mistral, NVIDIA NIM, OpenRouter, Ollama

## What exists today

- `GET /v1/health` — service, database, and provider status
- `GET /v1/models` — full model registry (pricing, limits, capabilities, benchmark info)
- `POST /v1/chat` — routes a prompt through prompt analysis, heuristic
  complexity classification, and a configurable strategy (`cost`,
  `latency`, `quality`, `balanced`) to select a model, then returns the
  response plus a full routing explanation (complexity, confidence,
  estimated cost/latency, human-readable reasoning)
- `ModelRegistry` — memory-first, backed by `backend/config/models.yaml` and persisted to SQLite; immutable read-only cache, atomic `reload()`/`refresh_provider_status()`, fails fast on malformed/invalid/duplicate config
- Providers behind a shared `BaseProvider` interface — `openai`, `anthropic`, `ollama`, `gemini`, `nvidia_nim`, `openrouter`, `groq`, `mistral`, plus `MockProvider`; SDK exceptions are translated into `ProviderError`, never leaked to the rest of the app
- `ProviderManager` — the mandatory `mock` provider crashes startup if it fails to construct; optional providers degrade gracefully to "disabled" (logged) if construction fails, with per-provider circuit breakers for failover
- Live, encrypted provider credential configuration via `/v1/providers/config` — hot-reloadable, no restart required
- In-process event bus (`PROVIDER_AVAILABLE`, `PROVIDER_DISABLED`, `PROVIDER_FAILED`, `MODEL_REGISTERED`), with per-subscriber exception isolation
- Structured JSON logging (console + rotating file) with `contextvars`-based request context
- Operations dashboard at `/dashboard` — quality trends, failover events, recent requests, per-model cost, cost-optimization recommendations

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
