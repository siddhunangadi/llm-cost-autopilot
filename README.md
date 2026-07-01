# LLM Cost Autopilot

Phase 1: project skeleton, provider foundation, model registry, and event
bus for an intelligent cost-aware LLM routing layer.

## What exists today

- `GET /v1/health` — service, database, and provider status
- `GET /v1/models` — full model registry (pricing, limits, capabilities, benchmark info)
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
  "providers": {"openai": "disabled", "anthropic": "disabled", "ollama": "disabled"},
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
