<div align="center">

# 🚀 LLM Cost Autopilot

### Stop overpaying to talk to AI.

**One API. Eight AI providers. Every request automatically routed to the model that gets the job done for the least money — without sacrificing quality.**

[![Version](https://img.shields.io/badge/version-0.9.1-blue)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](#license)

[Why this exists](#why-this-exists) • [What it does](#what-it-does-for-you) • [See it live](#see-it-live-in-5-minutes) • [How it works](#how-it-works)

</div>

---

## Why this exists

If you're building anything with AI, you already know the dirty secret:
you're probably paying for a $10-per-million-token model to answer
questions a $0.15 model could handle just fine — because nobody has
time to hand-pick the "right" model for every single request.

**LLM Cost Autopilot does that picking for you, automatically, every
time.** Send it a prompt. It figures out how hard the question actually
is, checks live pricing and speed across every provider you've
connected, and routes the request to the cheapest model that can
answer it well — then double-checks the answer was actually good, and
learns from the result.

You get one endpoint to call. It gets you the receipts.

## What it does for you

- 💸 **Cuts your AI bill automatically** — easy questions go to cheap,
  fast models; hard questions go to your best model. No manual
  model-picking, no overpaying by default.
- 🔌 **Works with the provider you already use** — OpenAI, Anthropic
  (Claude), Google Gemini, Groq, Mistral, NVIDIA NIM, OpenRouter, and
  local models via Ollama. Plug in the API keys you already have.
- ✅ **Grades its own work** — every response is automatically scored
  by an AI judge for correctness and completeness, so you can see
  quality, not just guess at it.
- 🛡️ **Never goes down because one provider does** — if a provider
  errors out or gets slow, it automatically fails over to another one,
  no code changes required.
- 📊 **Shows you exactly where your money goes** — a built-in dashboard
  breaks down cost, quality, and reliability per model, so you can see
  the savings, not just trust they're happening.
- 🔑 **Add or swap providers without restarting anything** — credentials
  are managed live through the API, encrypted at rest.

## See it live in 5 minutes

You don't need to read any code to try this.

```bash
# 1. Get the project
git clone https://github.com/siddhunangadi/llm-cost-autopilot.git
cd llm-cost-autopilot

# 2. Install dependencies (uv is a fast Python package manager)
uv sync

# 3. Add at least one AI provider key
cp .env.example .env
# open .env and paste in an OPENAI_API_KEY or ANTHROPIC_API_KEY

# 4. Start it up
uv run uvicorn backend.api.main:app --reload
```

Now open **http://127.0.0.1:8000/dashboard** in your browser — that's
the live operations dashboard: cost trends, quality scores, failover
events, all in one view.

Or just ask it something directly:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain why the sky is blue.", "strategy": "balanced"}'
```

It responds with the answer **and** a full explanation of why it chose
the model it did:

```json
{
  "response": "The sky appears blue because...",
  "routing": {
    "selected_model": "gpt-4o-mini",
    "strategy": "balanced",
    "complexity": "simple",
    "estimated_cost": 0.00013,
    "reasoning": [
      "Classified as simple: no advanced reasoning required.",
      "Strategy 'balanced' evaluated 2 eligible model(s).",
      "Selected 'gpt-4o-mini' — cheapest model that meets quality bar."
    ]
  }
}
```

## How it works

```
your prompt
    │
    ▼
┌─────────────────────┐     "how hard is this, really?"
│ Complexity classifier│
└─────────────────────┘
    │
    ▼
┌─────────────────────┐     "given cost/speed/quality priorities,
│   Routing strategy   │      which model wins?"
└─────────────────────┘
    │
    ▼
┌─────────────────────┐     tries your chosen provider first,
│  Provider + failover │      falls back automatically if it fails
└─────────────────────┘
    │
    ▼
┌─────────────────────┐     an AI judge scores the response —
│  Quality verification│      correctness, completeness, format
└─────────────────────┘
    │
    ▼
   your answer, plus a full paper trail of every decision made
```

Everything above is visible and explained back to you — nothing is a
black box.

## What's under the hood (for the technical reader)

- **FastAPI** backend, **SQLite** for persistence, **Jinja2** dashboard
- 8 providers behind one shared interface, so adding a 9th is a small,
  contained change — not a rewrite
- Per-provider circuit breakers for automatic failover
- Encrypted, hot-reloadable provider credentials (no restart to add a key)
- Full test suite; every feature shipped test-first — see
  [`CHANGELOG.md`](CHANGELOG.md) for the complete build history across
  10 shipped phases (routing → verification → learning → resilience →
  dashboard → analytics → live provider config → provider expansion)

## Run the tests

```bash
uv run pytest
```

## License

MIT — use it, fork it, ship it.
