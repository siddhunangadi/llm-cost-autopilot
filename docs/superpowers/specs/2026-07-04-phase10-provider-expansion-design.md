# Phase 10: Provider Expansion — Design Spec

**Status:** Frozen
**Date:** 2026-07-04
**Target version:** v0.9.1 (or next minor if implementation warrants)

## Goal

Add five new LLM providers — Google Gemini, NVIDIA NIM, OpenRouter, Groq, and
Mistral AI — reusing the existing provider architecture (`BaseProvider`,
`ProviderFactory`, `ProviderManager`, credential store, circuit breaker,
routing, dashboard). Eliminate duplicated provider-name lists so
`ProviderFactory` becomes the single source of truth.

## Background

All five target providers expose an OpenAI-compatible chat-completions API
(same request/response shape as `POST /chat/completions`, differing only in
`base_url` and API key). The existing `OpenAIProvider` already wraps the
OpenAI SDK (`AsyncOpenAI`) directly. Extracting its logic into a
parameterized base class lets each new provider become a ~10-line subclass
instead of a bespoke adapter.

`AnthropicProvider` and `OllamaProvider` use different protocols (native
Anthropic SDK; raw Ollama HTTP API) and are explicitly out of scope for this
refactor — they keep their current implementations unchanged.

## Architecture

### `OpenAIProtocolProvider` (new: `backend/providers/openai_protocol_provider.py`)

```python
class OpenAIProtocolProvider(BaseProvider):
    """Shared adapter for any provider exposing an OpenAI-compatible
    chat-completions API. Subclasses declare `_NAME` and `_BASE_URL`;
    everything else (generate/stream/health_check/count_tokens/
    estimate_cost) is inherited. If a provider later needs capabilities
    beyond the OpenAI-compatible protocol, its subclass may override
    the relevant method(s) while still satisfying BaseProvider."""

    _NAME: str  # subclass-defined
    _BASE_URL: str | None  # subclass-defined; None = SDK default (openai.com)

    def __init__(self, credential, client=None):
        self._client = client or AsyncOpenAI(
            api_key=credential.api_key if credential else None,
            base_url=self._BASE_URL,
        )

    @property
    def name(self) -> str:
        return self._NAME

    # generate / stream / health_check / count_tokens / estimate_cost:
    # ported verbatim from today's OpenAIProvider, unchanged in behavior.
```

This is extracted from `OpenAIProvider`'s current implementation, not
rewritten — `generate`, `stream`, `health_check` (`client.models.list()`
connectivity probe), `count_tokens` (`len(text) // 4` heuristic), and
`estimate_cost` (`calculate_linear_cost`) all move here verbatim.

`OpenAIProvider` becomes:

```python
class OpenAIProvider(OpenAIProtocolProvider):
    _NAME = "openai"
    _BASE_URL = None
```

Behavior is identical to today (SDK default base_url), so this is a
zero-behavior-change refactor for the existing provider.

### Five new provider classes

Each is a `_NAME`/`_BASE_URL` pair, one file per provider under
`backend/providers/`, mirroring the existing one-file-per-provider layout:

| File | `_NAME` | `_BASE_URL` |
|---|---|---|
| `gemini_provider.py` | `gemini` | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| `nvidia_nim_provider.py` | `nvidia_nim` | `https://integrate.api.nvidia.com/v1` |
| `openrouter_provider.py` | `openrouter` | `https://openrouter.ai/api/v1` |
| `groq_provider.py` | `groq` | `https://api.groq.com/openai/v1` |
| `mistral_provider.py` | `mistral` | `https://api.mistral.ai/v1` |

No provider-specific logic (custom headers, response parsing, retry
tuning) is introduced outside these thin declarations unless a provider's
API genuinely deviates from the OpenAI-compatible contract during
implementation — if that happens, the deviation is isolated to that
provider's subclass via a method override, never leaked into shared code.

### Single source of truth: `ProviderFactory`

`ProviderFactory.register()` gains a `user_configurable: bool = True` flag:

```python
def register(self, name: str, provider_cls: type[BaseProvider], *, user_configurable: bool = True) -> None:
    self._registry[name] = provider_cls
    if user_configurable:
        self._user_configurable.append(name)

def registered_names(self) -> tuple[str, ...]:
    return tuple(self._user_configurable)
```

`main.py` registers `"mock"` with `user_configurable=False` (unchanged
behavior — mock has never been in `KNOWN_PROVIDER_NAMES`) and all eight real
providers (3 existing + 5 new) with the default `True`.

The two duplicated `KNOWN_PROVIDER_NAMES = ("openai", "anthropic", "ollama")`
tuples in `backend/providers/manager.py` and `backend/services/
credential_store.py` are deleted. Every current usage
(`ProviderManager.__init__`, `ProviderManager.list_providers`,
`CredentialStore` env-fallback iteration, `providers_config.py` validation,
`model_registry.py` lookups) switches to `factory.registered_names()`,
threading the already-constructed `ProviderFactory` instance through where
it isn't already available (`CredentialStore` currently doesn't hold a
factory reference — it gains one via constructor injection).

Registration order in `main.py` becomes the only place provider names are
declared:

```python
factory.register("mock", MockProvider, user_configurable=False)
factory.register("openai", OpenAIProvider)
factory.register("anthropic", AnthropicProvider)
factory.register("ollama", OllamaProvider)
factory.register("gemini", GeminiProvider)
factory.register("nvidia_nim", NvidiaNimProvider)
factory.register("openrouter", OpenRouterProvider)
factory.register("groq", GroqProvider)
factory.register("mistral", MistralProvider)
```

### Credential store: env fallback + Settings

`backend/config/settings.py` gains five optional fields, following the
existing `openai_api_key` pattern:

```python
gemini_api_key: str | None = None
nvidia_nim_api_key: str | None = None
openrouter_api_key: str | None = None
groq_api_key: str | None = None
mistral_api_key: str | None = None
```

`backend/services/credential_store.py`'s `_ENV_FALLBACK` dict gains five
matching entries (API-key-only, no base_url — see below):

```python
"gemini": lambda s: ProviderCredential("gemini", s.gemini_api_key, None) if s.gemini_api_key else None,
"nvidia_nim": lambda s: ProviderCredential("nvidia_nim", s.nvidia_nim_api_key, None) if s.nvidia_nim_api_key else None,
"openrouter": lambda s: ProviderCredential("openrouter", s.openrouter_api_key, None) if s.openrouter_api_key else None,
"groq": lambda s: ProviderCredential("groq", s.groq_api_key, None) if s.groq_api_key else None,
"mistral": lambda s: ProviderCredential("mistral", s.mistral_api_key, None) if s.mistral_api_key else None,
```

### Provider Configuration: no base_url field for new providers

Unlike Ollama (self-hosted, base_url is the primary identifier), the five
new providers are hosted cloud APIs with one canonical endpoint each —
matching how OpenAI/Anthropic are configured today. The Provider
Configuration UI/API (`providers_config.py`, `providers.html`) requires
only an API key for these five; `base_url` stays `None` in their
`ProviderCredential` and the hardcoded `_BASE_URL` class constant is what's
actually used by the SDK client. No new UI field types are needed — the
existing "API key only" card layout (already used for OpenAI/Anthropic)
is reused as-is for all five.

### Model registry seeding

`models.yaml` gets a small curated set per provider (1–3 flagship models,
not an exhaustive catalog):

- **Gemini:** `gemini-2.5-pro`, `gemini-2.5-flash`
- **Groq:** `llama-3.3-70b-versatile`
- **Mistral:** `mistral-large-latest`
- **NVIDIA NIM:** `meta/llama-3.3-70b-instruct`
- **OpenRouter:** `openai/gpt-4.1-mini`

Pricing (`input_cost`/`output_cost` per million tokens), `context_window`,
and `capabilities` are filled from each vendor's current public pricing
page at implementation time. **Model identifiers are stored exactly as
required by the provider's API — no normalization or translation layer is
introduced.** This matters most for OpenRouter (`vendor/model` slugs) and
NVIDIA NIM (`org/model` slugs), whose IDs look unlike OpenAI's flat
`gpt-4o` style but must be passed through unchanged to the API.

### Integration surface (all pre-existing, data-driven — no code changes needed beyond provider registration)

- **Dashboard / provider status cards:** iterate `ProviderManager.list_providers()` / `factory.registered_names()` generically today; new providers appear automatically once registered.
- **Health checks:** inherited `OpenAIProtocolProvider.health_check()` (`client.models.list()`) — identical contract for all 5, verified once via the shared contract test, not per-provider.
- **Circuit breakers:** `main.py`'s `{name: CircuitBreaker() for name in KNOWN_PROVIDER_NAMES}` becomes `{name: CircuitBreaker() for name in factory.registered_names()}` — automatically covers new providers.
- **Routing:** `routing.yaml` and the routing engine select by model id / provider name already present in `model_registry`; no schema change.
- **Model Registry:** consumes `models.yaml` + `factory.registered_names()` for validation (`model_registry.py:188` already loops `KNOWN_PROVIDER_NAMES` to cross-check config); switches to the factory-backed call.

## Testing

**Shared contract tests** (`backend/tests/providers/test_openai_protocol_contract.py`):
a pytest class parametrized over all `OpenAIProtocolProvider` subclasses
(`OpenAIProvider`, `GeminiProvider`, `NvidiaNimProvider`, `OpenRouterProvider`,
`GroqProvider`, `MistralProvider`), using a mocked `AsyncOpenAI` client per
the existing provider test pattern (see current `test_openai_provider.py`).
Covers: `generate` success/error mapping to `ProviderError`, `stream`
chunk yielding, `health_check` true/false, `count_tokens`, `estimate_cost`,
and that `_BASE_URL` is actually passed to the constructed client.

**Per-provider tests:** one thin test file per new provider asserting
`name` and `_BASE_URL`/`_NAME` wiring only — the behavioral contract is
already covered by the shared suite, so these stay minimal (no duplicated
behavior tests).

**Existing test suites** (credential_store, manager, factory,
providers_config router, model_registry, dashboard) get new cases for the
five additional provider names wherever they currently special-case or
enumerate `KNOWN_PROVIDER_NAMES`.

## Backward compatibility

- `OpenAIProvider`'s public behavior (constructor signature, `name`,
  generate/stream/health_check output) is unchanged — the refactor to
  inherit from `OpenAIProtocolProvider` is behavior-preserving.
- Existing `.env` variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
  `OLLAMA_BASE_URL`) and stored `provider_credentials` rows are unaffected.
- `KNOWN_PROVIDER_NAMES` is removed as a name, but every call site is
  updated in the same change — no dangling references, no deprecated
  shim (this is an internal-only symbol, not a public API).

## Out of scope

- Native vendor SDKs (google-genai, mistralai) — explicitly deferred; the
  `OpenAIProtocolProvider` base class makes this a future non-breaking
  option per-provider if a subclass ever needs capabilities the
  OpenAI-compatible protocol doesn't expose.
- User-configurable `base_url` for the five new providers.
- Exhaustive model catalogs beyond the curated 1–3 per provider.
- Streaming/tool-call feature-parity auditing beyond what the shared
  contract test already covers.
