# Phase 10: Provider Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Gemini, NVIDIA NIM, OpenRouter, Groq, and Mistral as providers via a shared `OpenAIProtocolProvider` base class, and make `ProviderFactory` the single source of truth for provider names (eliminating the two duplicated `KNOWN_PROVIDER_NAMES` tuples).

**Architecture:** Extract `OpenAIProvider`'s existing SDK-adapter logic into `OpenAIProtocolProvider(BaseProvider)`, parameterized by `_NAME`/`_BASE_URL` class attributes. `OpenAIProvider` and 5 new provider classes become one-line subclasses. `ProviderFactory.register()` gains a `user_configurable` flag and a `registered_names()` accessor; every current consumer of `KNOWN_PROVIDER_NAMES` (in `manager.py`, `credential_store.py`, `providers_config.py`, `model_registry.py`) switches to reading names from the factory (directly, or via `ProviderManager.registered_names()`/`CredentialStore`'s injected `provider_names`).

**Tech Stack:** Python 3.13, FastAPI, SQLAlchemy, `openai` SDK (`AsyncOpenAI`), pytest + pytest-asyncio + pytest-mock.

## Global Constraints

- `AnthropicProvider` and `OllamaProvider` are unchanged — do not touch `backend/providers/anthropic_provider.py` or `backend/providers/ollama_provider.py`.
- No provider-specific logic (headers, response parsing) outside the provider layer; if a new provider's API deviates from the OpenAI-compatible contract, isolate the deviation to that provider's subclass via a method override — never in shared/router/service code.
- Model identifiers in `models.yaml` are stored exactly as the vendor's API requires — no normalization.
- `base_url` for the 5 new providers is hardcoded per-provider (a class constant), not user-configurable; Provider Configuration only collects an API key for them.
- Every existing test must keep passing; every new file/behavior gets test coverage in the same task that introduces it (TDD: failing test first).
- Follow existing code conventions: `ProviderError` translation of SDK exceptions, `mask_key`/encryption boundaries in `CredentialStore`, thin-adapter provider files.

---

### Task 1: `ProviderFactory` — `user_configurable` flag and `registered_names()`

**Files:**
- Modify: `backend/providers/factory.py`
- Test: `backend/tests/test_provider_factory.py`

**Interfaces:**
- Produces: `ProviderFactory.register(name: str, provider_cls: type[BaseProvider], *, user_configurable: bool = True) -> None`; `ProviderFactory.registered_names() -> tuple[str, ...]` (returns only names registered with `user_configurable=True`, in registration order).

- [ ] **Step 1: Write the failing tests**

```python
# append to backend/tests/test_provider_factory.py

def test_registered_names_excludes_non_user_configurable():
    factory = ProviderFactory()
    factory.register("mock", MockProvider, user_configurable=False)
    factory.register("openai_alias", MockProvider)

    assert factory.registered_names() == ("openai_alias",)


def test_registered_names_preserves_registration_order():
    factory = ProviderFactory()
    factory.register("b", MockProvider)
    factory.register("a", MockProvider)

    assert factory.registered_names() == ("b", "a")


def test_registered_names_defaults_to_user_configurable_true():
    factory = ProviderFactory()
    factory.register("mock", MockProvider)

    assert factory.registered_names() == ("mock",)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest backend/tests/test_provider_factory.py -v`
Expected: FAIL — `register() got an unexpected keyword argument 'user_configurable'` / `AttributeError: 'ProviderFactory' object has no attribute 'registered_names'`

- [ ] **Step 3: Implement**

```python
# backend/providers/factory.py
from backend.providers.base import BaseProvider
from backend.services.credential_store import ProviderCredential


class ProviderFactory:
    def __init__(self) -> None:
        self._registry: dict[str, type[BaseProvider]] = {}
        self._user_configurable: list[str] = []

    def register(
        self, name: str, provider_cls: type[BaseProvider], *, user_configurable: bool = True,
    ) -> None:
        self._registry[name] = provider_cls
        if user_configurable:
            self._user_configurable.append(name)

    def create(self, name: str, credential: ProviderCredential | None) -> BaseProvider:
        if name not in self._registry:
            raise KeyError(f"No provider registered under name '{name}'")
        return self._registry[name](credential)

    def registered_names(self) -> tuple[str, ...]:
        """Every provider name registered with user_configurable=True (the
        default), in registration order. The single source of truth for
        which providers are user-facing (configurable via Provider
        Configuration, listed on the dashboard, etc.) -- callers must
        never maintain their own copy of this set."""
        return tuple(self._user_configurable)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest backend/tests/test_provider_factory.py -v`
Expected: PASS (5 tests: 2 existing + 3 new)

- [ ] **Step 5: Commit**

```bash
git add backend/providers/factory.py backend/tests/test_provider_factory.py
git commit -m "feat: add user_configurable flag and registered_names() to ProviderFactory"
```

---

### Task 2: Extract `OpenAIProtocolProvider`, refactor `OpenAIProvider`

**Files:**
- Create: `backend/providers/openai_protocol_provider.py`
- Modify: `backend/providers/openai_provider.py`
- Test: `backend/tests/test_openai_provider.py` (unchanged — verifies the refactor is behavior-preserving)
- Test: `backend/tests/test_openai_protocol_provider.py` (new — base_url wiring)

**Interfaces:**
- Produces: `OpenAIProtocolProvider(BaseProvider)` with class attributes `_NAME: str`, `_BASE_URL: str | None`, constructor `__init__(self, credential: ProviderCredential | None, client: AsyncOpenAI | None = None)`, and concrete `generate`/`stream`/`health_check`/`count_tokens`/`estimate_cost`/`name` — identical contract to today's `OpenAIProvider`.
- Consumes: `backend.services.cost_estimator.calculate_linear_cost`, `backend.services.credential_store.ProviderCredential` (same as today).

- [ ] **Step 1: Write the failing test for base_url wiring**

```python
# backend/tests/test_openai_protocol_provider.py
from backend.providers.openai_protocol_provider import OpenAIProtocolProvider
from backend.services.credential_store import ProviderCredential


class _FakeProvider(OpenAIProtocolProvider):
    _NAME = "fake"
    _BASE_URL = "https://fake.example.com/v1"


def test_name_returns_class_constant():
    provider = _FakeProvider(ProviderCredential("fake", "key", None))
    assert provider.name == "fake"


def test_base_url_is_passed_to_client():
    provider = _FakeProvider(ProviderCredential("fake", "key", None))
    assert str(provider._client.base_url) == "https://fake.example.com/v1/"


def test_none_base_url_uses_sdk_default():
    class _DefaultProvider(OpenAIProtocolProvider):
        _NAME = "default"
        _BASE_URL = None

    provider = _DefaultProvider(ProviderCredential("default", "key", None))
    assert str(provider._client.base_url) == "https://api.openai.com/v1/"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_openai_protocol_provider.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.providers.openai_protocol_provider'`

- [ ] **Step 3: Create `OpenAIProtocolProvider` (logic moved verbatim from `OpenAIProvider`)**

```python
# backend/providers/openai_protocol_provider.py
from collections.abc import AsyncIterator

from openai import AsyncOpenAI, OpenAIError

from backend.providers.base import BaseProvider, ProviderError
from backend.services.cost_estimator import calculate_linear_cost
from backend.services.credential_store import ProviderCredential


class OpenAIProtocolProvider(BaseProvider):
    """Shared adapter for any provider exposing an OpenAI-compatible
    chat-completions API. Subclasses declare `_NAME` and `_BASE_URL`;
    everything else is inherited. If a provider later needs capabilities
    beyond the OpenAI-compatible protocol, its subclass may override the
    relevant method(s) while still satisfying BaseProvider. No retries,
    caching, logging policy, budgeting, or failover -- those belong above
    this layer."""

    _NAME: str
    _BASE_URL: str | None = None

    def __init__(
        self, credential: ProviderCredential | None, client: AsyncOpenAI | None = None,
    ) -> None:
        self._client = client or AsyncOpenAI(
            api_key=credential.api_key if credential else None,
            base_url=self._BASE_URL,
        )

    @property
    def name(self) -> str:
        return self._NAME

    async def generate(self, prompt: str, model: str, **kwargs) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
        except OpenAIError as exc:
            raise ProviderError(f"{self._NAME} generate failed: {exc}") from exc
        return response.choices[0].message.content or ""

    async def stream(self, prompt: str, model: str, **kwargs) -> AsyncIterator[str]:
        try:
            stream = await self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except OpenAIError as exc:
            raise ProviderError(f"{self._NAME} stream failed: {exc}") from exc

    async def health_check(self) -> bool:
        # Cheap connectivity probe (list models) rather than a completion
        # request. A health probe reports status, it doesn't raise -- any
        # failure here just means "not available", not a bug to propagate.
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def estimate_cost(
        self, input_tokens: int, output_tokens: int, input_cost: float, output_cost: float
    ) -> float:
        return calculate_linear_cost(input_tokens, output_tokens, input_cost, output_cost)
```

- [ ] **Step 4: Refactor `OpenAIProvider` to a one-line subclass**

```python
# backend/providers/openai_provider.py (entire file replaced)
from backend.providers.openai_protocol_provider import OpenAIProtocolProvider


class OpenAIProvider(OpenAIProtocolProvider):
    """OpenAI, using the SDK's default base_url (api.openai.com)."""

    _NAME = "openai"
    _BASE_URL = None
```

- [ ] **Step 5: Run both test files to verify everything passes**

Run: `pytest backend/tests/test_openai_provider.py backend/tests/test_openai_protocol_provider.py -v`
Expected: PASS — all of `test_openai_provider.py`'s existing 12 tests pass unchanged (proves the refactor is behavior-preserving) plus 3 new tests.

- [ ] **Step 6: Commit**

```bash
git add backend/providers/openai_protocol_provider.py backend/providers/openai_provider.py backend/tests/test_openai_protocol_provider.py
git commit -m "refactor: extract OpenAIProtocolProvider base class from OpenAIProvider"
```

---

### Task 3: Five new provider classes

**Files:**
- Create: `backend/providers/gemini_provider.py`
- Create: `backend/providers/nvidia_nim_provider.py`
- Create: `backend/providers/openrouter_provider.py`
- Create: `backend/providers/groq_provider.py`
- Create: `backend/providers/mistral_provider.py`
- Test: `backend/tests/test_new_providers.py`

**Interfaces:**
- Consumes: `OpenAIProtocolProvider` from Task 2.
- Produces: `GeminiProvider`, `NvidiaNimProvider`, `OpenRouterProvider`, `GroqProvider`, `MistralProvider` — all `OpenAIProtocolProvider` subclasses, importable from their respective modules.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_new_providers.py
import pytest

from backend.providers.gemini_provider import GeminiProvider
from backend.providers.groq_provider import GroqProvider
from backend.providers.mistral_provider import MistralProvider
from backend.providers.nvidia_nim_provider import NvidiaNimProvider
from backend.providers.openai_protocol_provider import OpenAIProtocolProvider
from backend.providers.openrouter_provider import OpenRouterProvider
from backend.services.credential_store import ProviderCredential

_CASES = [
    (GeminiProvider, "gemini", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    (NvidiaNimProvider, "nvidia_nim", "https://integrate.api.nvidia.com/v1"),
    (OpenRouterProvider, "openrouter", "https://openrouter.ai/api/v1"),
    (GroqProvider, "groq", "https://api.groq.com/openai/v1"),
    (MistralProvider, "mistral", "https://api.mistral.ai/v1"),
]


@pytest.mark.parametrize("provider_cls,expected_name,expected_base_url", _CASES)
def test_provider_is_openai_protocol_subclass(provider_cls, expected_name, expected_base_url):
    assert issubclass(provider_cls, OpenAIProtocolProvider)


@pytest.mark.parametrize("provider_cls,expected_name,expected_base_url", _CASES)
def test_provider_name(provider_cls, expected_name, expected_base_url):
    provider = provider_cls(ProviderCredential(expected_name, "key", None))
    assert provider.name == expected_name


@pytest.mark.parametrize("provider_cls,expected_name,expected_base_url", _CASES)
def test_provider_base_url(provider_cls, expected_name, expected_base_url):
    provider = provider_cls(ProviderCredential(expected_name, "key", None))
    assert str(provider._client.base_url) == expected_base_url
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_new_providers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.providers.gemini_provider'`

- [ ] **Step 3: Implement the five provider classes**

```python
# backend/providers/gemini_provider.py
from backend.providers.openai_protocol_provider import OpenAIProtocolProvider


class GeminiProvider(OpenAIProtocolProvider):
    """Google Gemini, via its OpenAI-compatible endpoint."""

    _NAME = "gemini"
    _BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
```

```python
# backend/providers/nvidia_nim_provider.py
from backend.providers.openai_protocol_provider import OpenAIProtocolProvider


class NvidiaNimProvider(OpenAIProtocolProvider):
    """NVIDIA NIM, via its OpenAI-compatible endpoint."""

    _NAME = "nvidia_nim"
    _BASE_URL = "https://integrate.api.nvidia.com/v1"
```

```python
# backend/providers/openrouter_provider.py
from backend.providers.openai_protocol_provider import OpenAIProtocolProvider


class OpenRouterProvider(OpenAIProtocolProvider):
    """OpenRouter, via its OpenAI-compatible endpoint."""

    _NAME = "openrouter"
    _BASE_URL = "https://openrouter.ai/api/v1"
```

```python
# backend/providers/groq_provider.py
from backend.providers.openai_protocol_provider import OpenAIProtocolProvider


class GroqProvider(OpenAIProtocolProvider):
    """Groq, via its OpenAI-compatible endpoint."""

    _NAME = "groq"
    _BASE_URL = "https://api.groq.com/openai/v1"
```

```python
# backend/providers/mistral_provider.py
from backend.providers.openai_protocol_provider import OpenAIProtocolProvider


class MistralProvider(OpenAIProtocolProvider):
    """Mistral AI, via its OpenAI-compatible endpoint."""

    _NAME = "mistral"
    _BASE_URL = "https://api.mistral.ai/v1"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_new_providers.py -v`
Expected: PASS (15 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/providers/gemini_provider.py backend/providers/nvidia_nim_provider.py backend/providers/openrouter_provider.py backend/providers/groq_provider.py backend/providers/mistral_provider.py backend/tests/test_new_providers.py
git commit -m "feat: add Gemini, NVIDIA NIM, OpenRouter, Groq, Mistral providers"
```

---

### Task 4: Shared OpenAI-protocol contract test suite

**Files:**
- Create: `backend/tests/test_openai_protocol_contract.py`

**Interfaces:**
- Consumes: all 6 `OpenAIProtocolProvider` subclasses (`OpenAIProvider`, `GeminiProvider`, `NvidiaNimProvider`, `OpenRouterProvider`, `GroqProvider`, `MistralProvider`).

This task has no separate "implementation" — it's a test-only task verifying Tasks 2-3's classes share one behavioral contract (mirrors `test_openai_provider.py`'s cases, parametrized).

- [ ] **Step 1: Write the contract test file**

```python
# backend/tests/test_openai_protocol_contract.py
from unittest.mock import AsyncMock

import pytest
from openai import OpenAIError

from backend.providers.base import ProviderError
from backend.providers.gemini_provider import GeminiProvider
from backend.providers.groq_provider import GroqProvider
from backend.providers.mistral_provider import MistralProvider
from backend.providers.nvidia_nim_provider import NvidiaNimProvider
from backend.providers.openai_provider import OpenAIProvider
from backend.providers.openrouter_provider import OpenRouterProvider
from backend.services.credential_store import ProviderCredential

ALL_PROTOCOL_PROVIDERS = [
    OpenAIProvider, GeminiProvider, NvidiaNimProvider,
    OpenRouterProvider, GroqProvider, MistralProvider,
]


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


def _make_provider(provider_cls):
    return provider_cls(ProviderCredential(provider_cls._NAME, "test-key", None))


@pytest.mark.parametrize("provider_cls", ALL_PROTOCOL_PROVIDERS)
async def test_generate_returns_completion_content(provider_cls, mocker):
    provider = _make_provider(provider_cls)
    mocker.patch.object(
        provider._client.chat.completions, "create",
        new_callable=AsyncMock, return_value=_FakeCompletion("hello world"),
    )

    assert await provider.generate("hi", model="some-model") == "hello world"


@pytest.mark.parametrize("provider_cls", ALL_PROTOCOL_PROVIDERS)
async def test_generate_translates_sdk_errors_into_provider_error(provider_cls, mocker):
    provider = _make_provider(provider_cls)
    mocker.patch.object(
        provider._client.chat.completions, "create",
        new_callable=AsyncMock, side_effect=OpenAIError("boom"),
    )

    with pytest.raises(ProviderError):
        await provider.generate("hi", model="some-model")


@pytest.mark.parametrize("provider_cls", ALL_PROTOCOL_PROVIDERS)
async def test_stream_translates_sdk_errors_into_provider_error(provider_cls, mocker):
    provider = _make_provider(provider_cls)
    mocker.patch.object(
        provider._client.chat.completions, "create",
        new_callable=AsyncMock, side_effect=OpenAIError("boom"),
    )

    with pytest.raises(ProviderError):
        async for _ in provider.stream("hi", model="some-model"):
            pass


@pytest.mark.parametrize("provider_cls", ALL_PROTOCOL_PROVIDERS)
async def test_health_check_true_when_models_list_succeeds(provider_cls, mocker):
    provider = _make_provider(provider_cls)
    mocker.patch.object(
        provider._client.models, "list", new_callable=AsyncMock, return_value=None,
    )

    assert await provider.health_check() is True


@pytest.mark.parametrize("provider_cls", ALL_PROTOCOL_PROVIDERS)
async def test_health_check_false_when_models_list_raises(provider_cls, mocker):
    provider = _make_provider(provider_cls)
    mocker.patch.object(
        provider._client.models, "list",
        new_callable=AsyncMock, side_effect=RuntimeError("down"),
    )

    assert await provider.health_check() is False


@pytest.mark.parametrize("provider_cls", ALL_PROTOCOL_PROVIDERS)
def test_count_tokens_is_positive(provider_cls):
    provider = _make_provider(provider_cls)
    assert provider.count_tokens("abcdefgh") == 2


@pytest.mark.parametrize("provider_cls", ALL_PROTOCOL_PROVIDERS)
def test_estimate_cost_matches_linear_formula(provider_cls):
    provider = _make_provider(provider_cls)
    cost = provider.estimate_cost(1_000_000, 1_000_000, 1.0, 2.0)
    assert cost == pytest.approx(3.0)


@pytest.mark.parametrize("provider_cls", ALL_PROTOCOL_PROVIDERS)
def test_base_url_reaches_the_constructed_client(provider_cls):
    provider = _make_provider(provider_cls)
    expected = provider_cls._BASE_URL or "https://api.openai.com/v1"
    assert str(provider._client.base_url).rstrip("/") == expected.rstrip("/")
```

- [ ] **Step 2: Run to verify it passes (no implementation change needed -- Tasks 2-3 already satisfy the contract)**

Run: `pytest backend/tests/test_openai_protocol_contract.py -v`
Expected: PASS (48 tests: 8 cases x 6 providers)

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_openai_protocol_contract.py
git commit -m "test: add shared contract suite for OpenAI-protocol providers"
```

---

### Task 5: `ProviderManager` reads names from the factory

**Files:**
- Modify: `backend/providers/manager.py`
- Modify: `backend/tests/test_provider_manager.py`

**Interfaces:**
- Consumes: `ProviderFactory.registered_names()` (Task 1).
- Produces: `ProviderManager.registered_names() -> tuple[str, ...]` (delegates to `self._factory.registered_names()`); `KNOWN_PROVIDER_NAMES` module constant removed from this file.

- [ ] **Step 1: Update `_make_factory()` in the test file to register all 3 existing providers (matches production `_build_provider_factory`), and add a new test for `registered_names()`**

```python
# backend/tests/test_provider_manager.py
# Replace the existing _make_factory() with:

from backend.providers.anthropic_provider import AnthropicProvider
from backend.providers.ollama_provider import OllamaProvider


def _make_factory():
    factory = ProviderFactory()
    factory.register("mock", MockProvider, user_configurable=False)
    factory.register("openai", OpenAIProvider)
    factory.register("anthropic", AnthropicProvider)
    factory.register("ollama", OllamaProvider)
    return factory
```

```python
# append to backend/tests/test_provider_manager.py

def test_registered_names_delegates_to_factory(tmp_path):
    credential_store = _make_credential_store(tmp_path)
    manager = ProviderManager(_make_factory(), credential_store)

    assert manager.registered_names() == ("openai", "anthropic", "ollama")
```

Also update the two tests that build a bare 2-provider factory
(`test_optional_provider_initialization_failure_is_recorded_as_unavailable`,
`test_optional_provider_initialization_failure_is_logged`,
`test_mandatory_mock_provider_initialization_failure_is_not_swallowed`) to
pass `user_configurable=False` when registering `"mock"`:

```python
factory.register("mock", MockProvider, user_configurable=False)
```

And update `test_list_providers_covers_known_providers` (now covers exactly
what the factory registers, not a hardcoded module tuple — behavior is
unchanged since `_make_factory()` now registers all 3, matching before):

```python
def test_list_providers_covers_known_providers(tmp_path):
    credential_store = _make_credential_store(tmp_path, openai_api_key="sk-test")
    manager = ProviderManager(_make_factory(), credential_store)

    assert manager.list_providers() == {
        "openai": "available",
        "anthropic": "disabled",
        "ollama": "disabled",
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest backend/tests/test_provider_manager.py -v`
Expected: FAIL — `test_registered_names_delegates_to_factory` fails with `AttributeError`; other tests still pass since `list_providers()`/`__init__` still use the (about-to-be-removed) module constant, which happens to equal the same 3 names.

- [ ] **Step 3: Implement**

```python
# backend/providers/manager.py
from backend.providers.base import BaseProvider, ProviderUnavailableError
from backend.providers.factory import ProviderFactory
from backend.services.credential_store import CredentialStore
from backend.telemetry.logging import get_logger, request_context


class ProviderManager:
    def __init__(self, factory: ProviderFactory, credential_store: CredentialStore) -> None:
        self._logger = get_logger("providers")
        self._factory = factory
        self._credential_store = credential_store

        # "mock" is mandatory -- if it fails to construct there is no
        # sensible degraded mode, so the exception propagates and crashes
        # startup rather than being swallowed here.
        self._providers: dict[str, BaseProvider] = {"mock": factory.create("mock", None)}

        for name in self._factory.registered_names():
            credential = credential_store.get(name)
            if credential is not None:
                self._try_build(name, credential)

    def _try_build(self, name: str, credential) -> bool:
        try:
            self._providers[name] = self._factory.create(name, credential)
            return True
        except Exception:
            self._providers.pop(name, None)
            with request_context(provider=name):
                self._logger.exception("provider_initialization_failed")
            return False

    def reload_provider(self, name: str) -> bool:
        """Rebuilds exactly one provider from CredentialStore's current
        value for it, live -- no restart. Called after activation (a
        passing save/enable/disable/delete), never for validation."""
        credential = self._credential_store.get(name)
        if credential is None:
            self._providers.pop(name, None)
            return False
        return self._try_build(name, credential)

    def get_provider(self, name: str) -> BaseProvider:
        if name not in self._providers:
            raise ProviderUnavailableError(f"Provider '{name}' is not available")
        return self._providers[name]

    def is_provider_available(self, name: str) -> bool:
        return name in self._providers

    def registered_names(self) -> tuple[str, ...]:
        return self._factory.registered_names()

    def list_providers(self) -> dict[str, str]:
        return {
            name: ("available" if name in self._providers else "disabled")
            for name in self._factory.registered_names()
        }
```

- [ ] **Step 4: Run full manager test file to verify it passes**

Run: `pytest backend/tests/test_provider_manager.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add backend/providers/manager.py backend/tests/test_provider_manager.py
git commit -m "refactor: ProviderManager reads provider names from ProviderFactory"
```

---

### Task 6: `CredentialStore` takes injected `provider_names`; add 5 new `_ENV_FALLBACK` entries and `Settings` fields

**Files:**
- Modify: `backend/services/credential_store.py`
- Modify: `backend/config/settings.py`
- Modify: `backend/tests/test_credential_store.py`
- Modify: `backend/tests/test_provider_manager.py` (its `_make_credential_store` helper)
- Modify: `backend/tests/test_providers_config_router.py` (its credential-store construction helper, if any)

**Interfaces:**
- Produces: `CredentialStore.__init__(self, session_factory, settings, provider_names: tuple[str, ...])`; `KNOWN_PROVIDER_NAMES` module constant removed from this file.
- Consumes: nothing new from other tasks (constructor caller passes `ProviderFactory.registered_names()`).

- [ ] **Step 1: Update `Settings` with the 5 new optional fields**

```python
# backend/config/settings.py -- add after ollama_base_url
    gemini_api_key: str | None = None
    nvidia_nim_api_key: str | None = None
    openrouter_api_key: str | None = None
    groq_api_key: str | None = None
    mistral_api_key: str | None = None
```

- [ ] **Step 2: Write the failing tests**

```python
# append to backend/tests/test_credential_store.py

def test_list_status_covers_injected_provider_names(tmp_path):
    store = _make_store(tmp_path, provider_names=("openai", "gemini", "groq"))

    names = [status.provider for status in store.list_status(lambda name: False)]

    assert names == ["openai", "gemini", "groq"]


def test_new_provider_env_fallback_resolves_from_settings(tmp_path):
    store = _make_store(tmp_path, gemini_api_key="gm-test")

    credential = store.get("gemini")

    assert credential.api_key == "gm-test"
    assert credential.base_url is None
```

Check the existing helper name in `test_credential_store.py` (likely
`_make_store` or similar) and update every call site in that file to pass
`provider_names=("openai", "anthropic", "ollama")` as the new required
argument, preserving current behavior for pre-existing tests.

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest backend/tests/test_credential_store.py -v`
Expected: FAIL — `TypeError: CredentialStore.__init__() missing 1 required positional argument: 'provider_names'` (once the constructor signature changes) or `KeyError`/`AttributeError` for `gemini_api_key` before Step 1 lands. Confirm the two new tests specifically fail before Step 4.

- [ ] **Step 4: Implement**

```python
# backend/services/credential_store.py
# Remove: KNOWN_PROVIDER_NAMES = ("openai", "anthropic", "ollama")

_ENV_FALLBACK = {
    "openai": lambda s: ProviderCredential("openai", s.openai_api_key, None)
    if s.openai_api_key
    else None,
    "anthropic": lambda s: ProviderCredential("anthropic", s.anthropic_api_key, None)
    if s.anthropic_api_key
    else None,
    "ollama": lambda s: ProviderCredential("ollama", None, s.ollama_base_url)
    if s.ollama_base_url
    else None,
    "gemini": lambda s: ProviderCredential("gemini", s.gemini_api_key, None)
    if s.gemini_api_key
    else None,
    "nvidia_nim": lambda s: ProviderCredential("nvidia_nim", s.nvidia_nim_api_key, None)
    if s.nvidia_nim_api_key
    else None,
    "openrouter": lambda s: ProviderCredential("openrouter", s.openrouter_api_key, None)
    if s.openrouter_api_key
    else None,
    "groq": lambda s: ProviderCredential("groq", s.groq_api_key, None)
    if s.groq_api_key
    else None,
    "mistral": lambda s: ProviderCredential("mistral", s.mistral_api_key, None)
    if s.mistral_api_key
    else None,
}
```

```python
# backend/services/credential_store.py -- CredentialStore.__init__
    def __init__(
        self, session_factory: sessionmaker, settings: Settings, provider_names: tuple[str, ...],
    ) -> None:
        self._session_factory = session_factory
        self._settings = settings
        self._provider_names = provider_names
        self._fernet = (
            Fernet(settings.provider_credential_encryption_key.encode())
            if settings.provider_credential_encryption_key
            else None
        )
        with self._session_factory() as session:
            has_rows = session.query(ProviderCredentialRow).first() is not None
        if has_rows and self._fernet is None:
            raise RuntimeError(
                "PROVIDER_CREDENTIAL_ENCRYPTION_KEY is required: "
                "provider_credentials rows exist but no encryption key is configured"
            )
```

```python
# backend/services/credential_store.py -- CredentialStore.list_status, replace the loop target
        for name in self._provider_names:
```

- [ ] **Step 5: Fix every other constructor call site**

`backend/tests/test_provider_manager.py`'s `_make_credential_store` helper:

```python
def _make_credential_store(tmp_path, **settings_kwargs):
    settings = Settings(
        _env_file=None, database_url=f"sqlite:///{tmp_path}/test.db",
        provider_credential_encryption_key=Fernet.generate_key().decode(),
        **settings_kwargs,
    )
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)
    return CredentialStore(
        session_factory=session_factory, settings=settings,
        provider_names=("openai", "anthropic", "ollama"),
    )
```

Grep for every remaining `CredentialStore(` call site across `backend/`
(including `backend/tests/test_providers_config_router.py` and any
dashboard/analytics test fixtures) and add `provider_names=(...)` matching
that test's registered provider set. Do not skip any — a missing argument
is a hard `TypeError` at collection time, so `pytest --collect-only` will
surface every remaining site.

- [ ] **Step 6: Run the full test suite to verify it passes**

Run: `pytest backend/tests -v`
Expected: PASS, 0 failures (this is the first task touching a widely-used constructor, so run the whole suite, not just the two files above)

- [ ] **Step 7: Commit**

```bash
git add backend/services/credential_store.py backend/config/settings.py backend/tests/test_credential_store.py backend/tests/test_provider_manager.py backend/tests/test_providers_config_router.py
git commit -m "feat: inject provider_names into CredentialStore; add env fallback for 5 new providers"
```

---

### Task 7: `providers_config.py` router reads names from the factory/manager

**Files:**
- Modify: `backend/api/routers/providers_config.py`
- Modify: `backend/tests/test_providers_config_router.py`

**Interfaces:**
- Consumes: `ProviderFactoryDep`, `ProviderManagerDep` (already available in this router's dependencies).

- [ ] **Step 1: Update/add failing tests**

Locate the existing test(s) asserting a 404 for an unknown provider name
(e.g. `test_save_config_for_unknown_provider_returns_404`) and add a
parametrized case for one of the 5 new providers to confirm it is now
*known* once registered in the test's factory fixture:

```python
# append to backend/tests/test_providers_config_router.py

async def test_new_provider_is_known(client, ...):  # match existing fixture names in this file
    response = await client.get("/v1/providers/config")
    provider_names = {status["provider"] for status in response.json()}
    assert "gemini" in provider_names
```

Inspect this test file's app/client fixture (it likely builds its own
`ProviderFactory`/`CredentialStore` similar to `test_provider_manager.py`)
and register `GeminiProvider` (plus the other 4, or at least one
representative) there, matching the production registration set from
Task 8.

- [ ] **Step 2: Run to verify it fails**

Run: `pytest backend/tests/test_providers_config_router.py -v`
Expected: FAIL — `gemini` not present, since the router still imports the old `KNOWN_PROVIDER_NAMES` and the fixture doesn't register it.

- [ ] **Step 3: Implement**

```python
# backend/api/routers/providers_config.py
# Remove: from backend.providers.manager import KNOWN_PROVIDER_NAMES

def _require_known(name: str, known_names: tuple[str, ...]) -> None:
    if name not in known_names:
        raise HTTPException(status_code=404, detail=f"Unknown provider '{name}'")
```

Update every call site of `_require_known(name)` to pass the names
explicitly, sourced from whichever dependency that handler already has:

```python
@router.post("/{name}/config", response_model=ProviderConfigResult)
async def save_provider_config(
    name: str, body: ProviderConfigRequest,
    credential_store: CredentialStoreDep, provider_manager: ProviderManagerDep,
    provider_factory: ProviderFactoryDep,
) -> ProviderConfigResult:
    _require_known(name, provider_manager.registered_names())
    ...

@router.delete("/{name}/config", response_model=ProviderConfigResult)
async def delete_provider_config(
    name: str, credential_store: CredentialStoreDep, provider_manager: ProviderManagerDep,
) -> ProviderConfigResult:
    _require_known(name, provider_manager.registered_names())
    ...

@router.post("/{name}/enable", response_model=ProviderConfigResult)
async def enable_provider(
    name: str, credential_store: CredentialStoreDep, provider_manager: ProviderManagerDep,
) -> ProviderConfigResult:
    _require_known(name, provider_manager.registered_names())
    ...

@router.post("/{name}/disable", response_model=ProviderConfigResult)
async def disable_provider(
    name: str, credential_store: CredentialStoreDep, provider_manager: ProviderManagerDep,
) -> ProviderConfigResult:
    _require_known(name, provider_manager.registered_names())
    ...

@router.post("/{name}/test", response_model=ProviderConfigResult)
async def test_provider_config(
    name: str, body: ProviderConfigRequest,
    credential_store: CredentialStoreDep, provider_factory: ProviderFactoryDep,
) -> ProviderConfigResult:
    _require_known(name, provider_factory.registered_names())
    ...
```

`list_provider_config` and `providers_page` are unaffected — they already
delegate to `credential_store.list_status(...)`, which (as of Task 6) is
driven by its injected `provider_names`.

- [ ] **Step 4: Run to verify it passes**

Run: `pytest backend/tests/test_providers_config_router.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/api/routers/providers_config.py backend/tests/test_providers_config_router.py
git commit -m "refactor: providers_config router reads known provider names from dependencies"
```

---

### Task 8: `model_registry.py` reads names from `ProviderManager`

**Files:**
- Modify: `backend/services/model_registry.py`
- Modify: `backend/tests/test_model_registry.py` (or equivalent — locate via `grep -rl KNOWN_PROVIDER_NAMES backend/tests`)

**Interfaces:**
- Consumes: `ProviderManager.registered_names()` (Task 5).

- [ ] **Step 1: Update the relevant test's provider-manager fixture/factory to register the same provider set expected after this change, and add a case asserting `refresh_provider_status` covers a newly-registered provider**

Locate `refresh_provider_status`'s existing test coverage (grep
`refresh_provider_status` in `backend/tests/`) and confirm its
`ProviderManager` fixture is built via a `ProviderFactory` (not a raw
mock) so `registered_names()` resolves correctly — if it currently uses a
test double for `ProviderManager`, add a `registered_names()` method to
that double returning the same tuple its other methods are stubbed for.

- [ ] **Step 2: Run to verify it fails (if the fixture needed updating) or confirm it already passes**

Run: `pytest backend/tests/test_model_registry.py -v`

- [ ] **Step 3: Implement**

```python
# backend/services/model_registry.py
# Remove: from backend.providers.manager import KNOWN_PROVIDER_NAMES, ProviderManager
# Replace with: from backend.providers.manager import ProviderManager

    async def refresh_provider_status(self) -> None:
        with self._session_factory() as session:
            for provider_name in self._provider_manager.registered_names():
                ...  # body unchanged
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest backend/tests/test_model_registry.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/services/model_registry.py backend/tests/test_model_registry.py
git commit -m "refactor: ModelRegistry.refresh_provider_status reads names from ProviderManager"
```

---

### Task 9: Wire everything in `main.py` — register 5 new providers, reorder factory/credential-store construction, circuit breakers

**Files:**
- Modify: `backend/api/main.py`
- Test: `backend/tests/test_main_wiring.py` (new, or extend an existing app-startup smoke test if one exists — check `backend/tests/` for a file that boots the app via `TestClient`/`ASGITransport` first)

**Interfaces:**
- Consumes: everything from Tasks 1-8.

- [ ] **Step 1: Write/extend a startup smoke test**

Check whether a test already boots the full app (search for
`TestClient(app)` or `lifespan` in `backend/tests/`). If one exists,
extend it; otherwise add:

```python
# backend/tests/test_main_wiring.py
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app


def test_app_boots_and_lists_all_nine_providers(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/wiring.db")
    monkeypatch.setenv("MODELS_YAML_PATH", "backend/config/models.yaml")
    with TestClient(app) as client:
        response = client.get("/v1/providers/config")
        names = {status["provider"] for status in response.json()}
        assert names == {
            "openai", "anthropic", "ollama",
            "gemini", "nvidia_nim", "openrouter", "groq", "mistral",
        }
```

If an existing app-boot fixture/pattern already exists in this codebase
(check `backend/tests/test_health_router.py` or similar for how the app is
instantiated in tests, including any required env vars like
`PROVIDER_CREDENTIAL_ENCRYPTION_KEY`), follow that exact pattern instead
of introducing a second one.

- [ ] **Step 2: Run to verify it fails**

Run: `pytest backend/tests/test_main_wiring.py -v`
Expected: FAIL — only `{"openai", "anthropic", "ollama"}` present.

- [ ] **Step 3: Implement**

```python
# backend/api/main.py -- imports: add the 5 new provider classes
from backend.providers.gemini_provider import GeminiProvider
from backend.providers.groq_provider import GroqProvider
from backend.providers.mistral_provider import MistralProvider
from backend.providers.nvidia_nim_provider import NvidiaNimProvider
from backend.providers.openrouter_provider import OpenRouterProvider
# Remove: from backend.providers.manager import KNOWN_PROVIDER_NAMES, ProviderManager
from backend.providers.manager import ProviderManager
```

```python
# backend/api/main.py -- _build_provider_factory
def _build_provider_factory() -> ProviderFactory:
    factory = ProviderFactory()
    factory.register("mock", MockProvider, user_configurable=False)
    factory.register("openai", OpenAIProvider)
    factory.register("anthropic", AnthropicProvider)
    factory.register("ollama", OllamaProvider)
    factory.register("gemini", GeminiProvider)
    factory.register("nvidia_nim", NvidiaNimProvider)
    factory.register("openrouter", OpenRouterProvider)
    factory.register("groq", GroqProvider)
    factory.register("mistral", MistralProvider)
    return factory
```

```python
# backend/api/main.py -- lifespan(): build provider_factory before
# credential_store, and pass provider_names through. This replaces the
# current block that builds credential_store before provider_factory --
# swap the order so provider_factory is built first, then reuse it.

    provider_factory = _build_provider_factory()
    credential_store = CredentialStore(
        session_factory=session_factory, settings=settings,
        provider_names=provider_factory.registered_names(),
    )
    provider_manager = ProviderManager(provider_factory, credential_store)
```

```python
# backend/api/main.py -- ProviderExecutor construction
    provider_executor = ProviderExecutor(
        provider_manager=provider_manager,
        retry_policy=ExponentialBackoffRetryPolicy(),
        circuit_breakers={name: CircuitBreaker() for name in provider_factory.registered_names()},
        event_bus=event_bus,
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest backend/tests/test_main_wiring.py -v`
Expected: PASS

- [ ] **Step 5: Run the entire suite**

Run: `pytest backend/tests -v`
Expected: PASS, 0 failures — this task touches the most shared wiring in the app, so a full-suite run is required before moving on.

- [ ] **Step 6: Commit**

```bash
git add backend/api/main.py backend/tests/test_main_wiring.py
git commit -m "feat: register 5 new providers in main.py; ProviderFactory is now the single source of truth for provider names"
```

---

### Task 10: Seed `models.yaml` with curated models for the 5 new providers

**Files:**
- Modify: `backend/config/models.yaml`
- Test: `backend/tests/test_model_registry.py` (extend, or a dedicated fixture-loading test if one already parses `models.yaml` directly)

**Interfaces:**
- Consumes: `ModelRegistry`'s existing YAML schema (`id`, `provider`, `model`, `pricing.{input_cost,output_cost}`, `limits.{context_window,max_output_tokens}`, `capabilities.{supports_streaming,supports_tools,supports_json,supports_vision}`, `metadata.{benchmark_score,average_latency_ms}`) — unchanged, no schema migration.

- [ ] **Step 1: Write the failing test**

```python
# append to backend/tests/test_model_registry.py (adjust fixture wiring to
# match this file's existing pattern for constructing a ModelRegistry
# against the real backend/config/models.yaml)

def test_new_provider_models_are_registered(model_registry_loaded_from_real_yaml):
    ids = {spec.id for spec in model_registry_loaded_from_real_yaml.get_models()}
    assert {
        "gemini-2.5-pro", "gemini-2.5-flash", "llama-3.3-70b-versatile",
        "mistral-large-latest", "meta/llama-3.3-70b-instruct", "openai/gpt-4.1-mini",
    } <= ids
```

Use whatever fixture name/helper this test file already has for loading
the real `backend/config/models.yaml` (grep `models_yaml_path` in this
test file) rather than inventing a new one.

- [ ] **Step 2: Run to verify it fails**

Run: `pytest backend/tests/test_model_registry.py -v`
Expected: FAIL — new model ids absent.

- [ ] **Step 3: Append entries to `models.yaml`**

```yaml
  - id: gemini-2.5-pro
    provider: gemini
    model: gemini-2.5-pro
    pricing:
      input_cost: 1.25
      output_cost: 10.00
    limits:
      context_window: 1048576
      max_output_tokens: 65536
    capabilities:
      supports_streaming: true
      supports_tools: true
      supports_json: true
      supports_vision: true
    metadata:
      benchmark_score: 0.90
      average_latency_ms: 1100
  - id: gemini-2.5-flash
    provider: gemini
    model: gemini-2.5-flash
    pricing:
      input_cost: 0.30
      output_cost: 2.50
    limits:
      context_window: 1048576
      max_output_tokens: 65536
    capabilities:
      supports_streaming: true
      supports_tools: true
      supports_json: true
      supports_vision: true
    metadata:
      benchmark_score: 0.85
      average_latency_ms: 500
  - id: llama-3.3-70b-versatile
    provider: groq
    model: llama-3.3-70b-versatile
    pricing:
      input_cost: 0.59
      output_cost: 0.79
    limits:
      context_window: 128000
      max_output_tokens: 32768
    capabilities:
      supports_streaming: true
      supports_tools: true
      supports_json: true
      supports_vision: false
    metadata:
      benchmark_score: 0.86
      average_latency_ms: 250
  - id: mistral-large-latest
    provider: mistral
    model: mistral-large-latest
    pricing:
      input_cost: 2.00
      output_cost: 6.00
    limits:
      context_window: 128000
      max_output_tokens: 32768
    capabilities:
      supports_streaming: true
      supports_tools: true
      supports_json: true
      supports_vision: false
    metadata:
      benchmark_score: 0.88
      average_latency_ms: 700
  - id: meta/llama-3.3-70b-instruct
    provider: nvidia_nim
    model: meta/llama-3.3-70b-instruct
    pricing:
      input_cost: 0.0  # Free tier via NVIDIA NIM developer API
      output_cost: 0.0
    limits:
      context_window: 128000
      max_output_tokens: 4096
    capabilities:
      supports_streaming: true
      supports_tools: true
      supports_json: true
      supports_vision: false
    metadata:
      benchmark_score: 0.86
      average_latency_ms: 600
  - id: openai/gpt-4.1-mini
    provider: openrouter
    model: openai/gpt-4.1-mini
    pricing:
      input_cost: 0.40
      output_cost: 1.60
    limits:
      context_window: 1047576
      max_output_tokens: 32768
    capabilities:
      supports_streaming: true
      supports_tools: true
      supports_json: true
      supports_vision: true
    metadata:
      benchmark_score: 0.87
      average_latency_ms: 550
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest backend/tests/test_model_registry.py -v`
Expected: PASS

- [ ] **Step 5: Run the entire suite to confirm no regression (e.g. optimization/routing tests that assert exact model counts)**

Run: `pytest backend/tests -v`
Expected: PASS. If any test asserts an exact total model count (grep `get_models()) ==` or similar in `backend/tests/`), update that assertion's expected count to include the 6 new entries.

- [ ] **Step 6: Commit**

```bash
git add backend/config/models.yaml backend/tests/test_model_registry.py
git commit -m "feat: seed models.yaml with curated Gemini/Groq/Mistral/NVIDIA NIM/OpenRouter models"
```

---

### Task 11: CHANGELOG, version bump, README (if applicable)

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `pyproject.toml`
- Modify: `backend/api/main.py` (`APP_VERSION`)
- Modify: `README.md` (only if it lists supported providers — check first)
- Modify: `uv.lock` (regenerate)

- [ ] **Step 1: Check whether README lists providers**

Run: `grep -n "openai\|anthropic\|ollama" README.md`

If it lists supported providers, add the 5 new ones to that list, matching
the existing format exactly (no other README changes).

- [ ] **Step 2: Bump version**

```toml
# pyproject.toml
version = "0.9.1"
```

```python
# backend/api/main.py
APP_VERSION = "0.9.1"
```

- [ ] **Step 3: Add CHANGELOG entry (prepend, following the exact structure of the v0.9.0 entry above it)**

```markdown
## v0.9.1 — Provider Expansion (2026-07-04)

Adds five new LLM providers -- Google Gemini, NVIDIA NIM, OpenRouter, Groq,
and Mistral AI -- all via a shared `OpenAIProtocolProvider` base class,
since each exposes an OpenAI-compatible chat-completions API differing
only in base_url and API key. `OpenAIProvider` is refactored to inherit
from it with no behavior change. `ProviderFactory` becomes the single
source of truth for which provider names exist: the two previously
duplicated `KNOWN_PROVIDER_NAMES` tuples (in `manager.py` and
`credential_store.py`) are removed, and every consumer now reads names
from `ProviderFactory.registered_names()` (directly, or via
`ProviderManager.registered_names()` / an injected `provider_names` tuple).

**Added**
- `OpenAIProtocolProvider` -- shared adapter for any OpenAI-compatible
  provider; subclasses declare only `_NAME` and `_BASE_URL`
- `GeminiProvider`, `NvidiaNimProvider`, `OpenRouterProvider`,
  `GroqProvider`, `MistralProvider` -- registered in `ProviderFactory`
  alongside the existing 4; configurable via Provider Configuration
  (API key only -- base_url is fixed per provider, these are hosted
  cloud APIs with one canonical endpoint) with `.env` fallback
  (`GEMINI_API_KEY`, `NVIDIA_NIM_API_KEY`, `OPENROUTER_API_KEY`,
  `GROQ_API_KEY`, `MISTRAL_API_KEY`)
- `ProviderFactory.registered_names()` / `register(..., user_configurable=)`
  -- the single source of truth for the user-facing provider set; `mock`
  is registered `user_configurable=False` and stays internal-only
- Six curated models across the new providers in `models.yaml`
  (`gemini-2.5-pro`, `gemini-2.5-flash`, `llama-3.3-70b-versatile`,
  `mistral-large-latest`, `meta/llama-3.3-70b-instruct`,
  `openai/gpt-4.1-mini`) -- model ids stored exactly as each vendor's API
  requires, no normalization layer

**Changed**
- `OpenAIProvider` now inherits from `OpenAIProtocolProvider`
  (`_NAME="openai"`, `_BASE_URL=None`) -- behavior-preserving refactor,
  verified by the full pre-existing `OpenAIProvider` test suite passing
  unchanged
- `CredentialStore.__init__` takes an explicit `provider_names` argument
  instead of reading a module-level constant
- `main.py`'s provider circuit-breaker map and `ProviderManager`
  construction now derive their provider set from
  `ProviderFactory.registered_names()`

**Stats:** <fill in from final test run> tests passing (<N> new), 0
regressions.
```

Fill in the actual final test count from Task 12's full-suite run before
committing (do not commit with the placeholder still present).

- [ ] **Step 4: Regenerate the lockfile**

Run: `uv lock`

- [ ] **Step 5: Commit**

```bash
git add CHANGELOG.md pyproject.toml backend/api/main.py uv.lock README.md
git commit -m "chore: bump to v0.9.1, document provider expansion in CHANGELOG"
```

(Omit `README.md` from the `git add` if Step 1 found nothing to change.)

---

### Task 12: Whole-branch review, full regression, manual verification, merge, tag

This task is process, not new code — it closes out the phase per the
project's established workflow (see `docs/superpowers/specs/2026-07-04-phase10-provider-expansion-design.md`
and prior phases' plans for precedent).

- [ ] **Step 1: Run the complete test suite one final time**

Run: `pytest backend/tests -v`
Expected: 100% pass, 0 failures, 0 skips introduced by this phase. Record
the exact passing count for the CHANGELOG's `**Stats:**` line (go back and
fill in Task 11 Step 3 if not already done).

- [ ] **Step 2: Whole-branch review**

Invoke `/code-review` (or the equivalent whole-branch reviewer subagent
per `superpowers:subagent-driven-development`) against the full diff from
this phase's first commit to `HEAD`. Fix every Critical and Important
finding; re-run the full test suite after each fix. Do not merge with any
open Critical/Important finding.

- [ ] **Step 3: Manual verification against a running server**

```bash
source .venv/bin/activate
uvicorn backend.api.main:app --port 8091 &
sleep 2
curl -s http://localhost:8091/v1/providers/config | python3 -m json.tool
# Expected: 8 entries (openai, anthropic, ollama, gemini, nvidia_nim,
# openrouter, groq, mistral), each configured=false (no keys set in this
# shell) but present and not 404.
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8091/dashboard/providers
# Expected: 200 -- confirms the 5 new providers render on the existing
# card-grid template without any template changes needed.
kill %1
```

If any endpoint fails, treat it as a Critical finding and fix before
proceeding — do not merge with a broken manual verification pass.

- [ ] **Step 4: Merge to main and tag**

Follow `superpowers:finishing-a-development-branch` → local merge option
(matches every prior phase's precedent per project memory). After a clean
fast-forward or merge commit onto `main`:

```bash
git tag v0.9.1
```

- [ ] **Step 5: Clean up the worktree**

Per `superpowers:using-git-worktrees`, remove the implementation worktree
and its branch once merged and tagged.

- [ ] **Step 6: Print the final summary**

Report: number of tasks completed (12), commits created, final test count
(pass/total), manual verification results, review findings fixed (count +
one-line description each), final tag (`v0.9.1`), and explicit
confirmation the feature is complete.
