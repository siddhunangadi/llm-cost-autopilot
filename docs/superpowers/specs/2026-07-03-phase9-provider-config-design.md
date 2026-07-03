# v0.9.0: Provider Configuration — Design Spec

## 1. Goal

Replace `.env`-only provider API keys with dashboard-managed, encrypted, live-reloadable credentials, scoped to the three providers already in `KNOWN_PROVIDER_NAMES` (`openai`, `anthropic`, `ollama`). This moves the platform from a single-operator, restart-required configuration model toward a runtime-managed one, without expanding the model registry or routing surface.

## 2. Scope

In scope:
- `AnthropicProvider` and `OllamaProvider` classes (currently only `OpenAIProvider`/`MockProvider` exist), implementing `BaseProvider` identically to `OpenAIProvider`'s pattern
- Encrypted credential storage (`provider_credentials` table), replacing `Settings.openai_api_key`/`anthropic_api_key` as the primary source, with env vars as a fallback
- `CredentialStore` — the only layer that knows encryption exists
- `ProviderManager.reload_provider(name)` — live, zero-downtime credential swap
- Save-then-health-check-then-activate flow (invalid credentials never take down a working provider)
- `POST /v1/providers/{name}/config`, `DELETE /v1/providers/{name}/config`, `POST /v1/providers/{name}/test`, `GET /v1/providers/config`
- `/dashboard/providers` page — one form per provider, masked keys, Test/Save/Delete/Disable
- `is_enabled` flag (disable without deleting)

Explicitly out of scope (deferred):
- Gemini, Groq, OpenRouter provider classes — no code exists for them today; a separate feature
- Populating `models.yaml` with new Anthropic/Ollama model entries so chat requests actually route to them — this feature makes providers *connectable and health-checkable*, not *routable*. Routing remains OpenAI-only until model registry work happens separately.
- Multi-organization/multi-tenant credential scoping — the `organization_id` column is added now (nullable, unused) purely to avoid a future schema migration; no organization logic is implemented
- Credential rotation history, audit log of who changed what
- Any UI/API auth — matches the existing dashboard/API's current posture (none)

## 3. Architecture

```
Dashboard UI → Provider API (routes)
                  ↓
             CredentialStore   (owns Fernet encryption + provider_credentials
                                 CRUD; returns plain ProviderCredential value
                                 objects — the only layer that knows
                                 encryption exists)
                  ↓
             ProviderManager   (never decrypts, never touches SQL; receives
                                 plain credentials from CredentialStore,
                                 builds providers via ProviderFactory)
                  ↓
             ProviderFactory → BaseProvider → health_check()
```

`ProviderManager` is extended, not replaced: at startup it still builds `mock` unconditionally, and now asks `CredentialStore` (rather than reading `Settings` fields directly) for each of `openai`/`anthropic`/`ollama`'s credential, building whichever providers have one. `CredentialStore.get(name)` internally falls back to the existing `Settings` env-var fields when no DB row exists — so a deployment with only `.env` configured behaves exactly as it does today, with zero migration required.

## 4. Data model

### 4.1 `provider_credentials` table (`backend/database/models.py`)

```python
class ProviderCredentialRow(Base):
    __tablename__ = "provider_credentials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider_name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    encrypted_api_key: Mapped[str | None] = mapped_column(String, nullable=True)
    base_url: Mapped[str | None] = mapped_column(String, nullable=True)
    organization_id: Mapped[str | None] = mapped_column(String, nullable=True)  # unused, reserved
    is_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_successful_health_check: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_failure_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
```

No migration tooling needed — `init_db`'s existing `Base.metadata.create_all(engine)` picks up the new table automatically, same as every prior phase's tables.

### 4.2 `ProviderCredential` (plain value object, `backend/services/credential_store.py`)

```python
@dataclass(frozen=True)
class ProviderCredential:
    provider_name: str
    api_key: str | None
    base_url: str | None
```

This is what `CredentialStore` hands to `ProviderManager` — plaintext in memory only, never serialized, never logged. `ProviderManager`/`ProviderFactory` never see `ProviderCredentialRow` or anything encrypted.

### 4.3 `ProviderState` (derived, not stored)

```python
class ProviderState(str, Enum):
    UNCONFIGURED = "unconfigured"        # no credential row, no env var
    CONNECTING = "connecting"             # health check in flight (save/test in progress)
    HEALTHY = "healthy"                   # active provider, last health check passed
    INVALID_CREDENTIALS = "invalid_credentials"  # credential saved, health check failed
    UNAVAILABLE = "unavailable"           # configured + enabled but not currently reachable
    DISABLED = "disabled"                 # is_enabled = False
```

Computed by the dashboard service layer from: credential presence, `is_enabled`, whether `ProviderManager.is_provider_available(name)` is true, and whether `last_failure_reason` is set. `CONNECTING` only appears in the synchronous request/response cycle of a save/test call, never persisted.

## 5. `CredentialStore`

```python
class CredentialStore:
    def __init__(self, session_factory: sessionmaker, settings: Settings, fernet: Fernet) -> None:
        ...

    def get(self, provider_name: str) -> ProviderCredential | None:
        """DB row (if is_enabled) -> Settings env-var fallback -> None."""

    def save(
        self, provider_name: str, api_key: str | None, base_url: str | None,
    ) -> ProviderCredentialRow:
        """Encrypts api_key if present, upserts the row. Does not touch
        is_enabled, last_successful_health_check, or last_failure_reason."""

    def record_health_check_result(
        self, provider_name: str, success: bool, failure_reason: str | None,
    ) -> None:
        """Sets last_successful_health_check (on success) or
        last_failure_reason (on failure). Never both in one call."""

    def set_enabled(self, provider_name: str, enabled: bool) -> None: ...

    def delete(self, provider_name: str) -> None: ...

    def list_status(self) -> list[ProviderCredentialStatus]:
        """Returns masked, non-secret status for every known provider name
        (including ones with no row at all) -- backs GET /v1/providers/config."""
```

Encryption: `cryptography.fernet.Fernet`, keyed by a new required `Settings.provider_credential_encryption_key: str | None` field. `CredentialStore.__init__` raises immediately if any `provider_credentials` row exists at startup but the key is missing or malformed — same "crash loud, don't swallow config errors" pattern as `DATABASE_URL`. If the table is empty, a missing key is not an error (nothing to decrypt yet).

## 6. `ProviderManager` changes

```python
class ProviderManager:
    def __init__(self, factory: ProviderFactory, credential_store: CredentialStore) -> None:
        # was: def __init__(self, factory: ProviderFactory, settings: Settings)
        ...
        self._providers = {"mock": factory.create("mock", None)}
        for name in KNOWN_PROVIDER_NAMES:
            credential = credential_store.get(name)
            if credential is not None:
                self._try_build(name, credential)

    def reload_provider(self, name: str) -> bool:
        """Rebuilds exactly one provider from CredentialStore's current
        value for it. Returns True if the resulting provider is now
        available, False otherwise (old provider, if any, is left
        untouched on failure -- this method is only called AFTER a
        successful health check by the caller, per the save flow in
        section 7, so failure here would indicate a race, not the normal
        path)."""
```

`ProviderFactory.create` signature changes from `create(name, settings: Settings)` to `create(name, credential: ProviderCredential | None)` — `mock` is constructed with `None` (it takes no credential). This is the one ripple into existing code: `OpenAIProvider.__init__` changes from reading `settings.openai_api_key` to reading `credential.api_key`.

## 7. Save flow (the safety-critical path)

```
POST /v1/providers/{name}/config  { api_key?, base_url? }
        │
        ▼
CredentialStore.save(name, api_key, base_url)   -- persisted immediately,
        │                                          encrypted, NOT yet active
        ▼
build a throwaway BaseProvider from the just-saved credential
        │
        ▼
await provider.health_check()
        │
   ┌────┴────┐
 success    failure
   │            │
   ▼            ▼
CredentialStore    CredentialStore.record_health_check_result(
  .record_...        name, success=False, failure_reason=str(exc))
  (success=True)      │
   │                  ▼
   ▼              response: {"saved": true, "activated": false,
ProviderManager             "reason": "..."} -- OLD PROVIDER, IF ANY,
  .reload_provider(name)     IS UNTOUCHED. No downtime.
   │
   ▼
response: {"saved": true, "activated": true}
```

Saving always persists the credential (so an operator can fix a typo without re-entering everything), but only a passing health check activates it in `ProviderManager`. This is the one significant deviation from the original proposal, per explicit user direction.

## 8. API

- `POST /v1/providers/{name}/config` — body `{api_key?: str, base_url?: str}`, implements the flow in §7. `name` must be one of `KNOWN_PROVIDER_NAMES`; 404 otherwise.
- `DELETE /v1/providers/{name}/config` — deletes the row, calls `reload_provider(name)` (which will now fall through to env-var fallback or become unavailable).
- `POST /v1/providers/{name}/test` — body `{api_key?: str, base_url?: str}`. Builds a throwaway provider from the request body (never from stored credentials — always exactly what was submitted), calls `health_check()`, discards the provider. **No DB write, no `reload_provider` call, ever.** Returns `{"healthy": bool, "reason": str | None}`.
- `GET /v1/providers/config` — returns `list[ProviderConfigStatus]`:
  ```python
  class ProviderConfigStatus(BaseModel):
      provider: str
      configured: bool
      masked_key: str | None     # e.g. "sk-********ABCD"; null if no key (Ollama) or unconfigured
      base_url: str | None
      state: ProviderState
      last_successful_health_check: datetime | None
      last_failure_reason: str | None
  ```
  Never returns `encrypted_api_key` or a plaintext key under any field name.

Masking rule: first 3 + last 4 characters visible, rest replaced with `*` (minimum key length assumed sufficient; keys shorter than 8 chars are fully masked).

## 9. UI

`/dashboard/providers` — single-render page (same pattern as `/dashboard/analytics`: no HTMX polling, refresh via reload), one card per provider in `KNOWN_PROVIDER_NAMES`:
- OpenAI / Anthropic: masked key field, `[Save]`, `[Test Connection]`, `[Disable]`/`[Enable]`, `[Delete]`, state badge, "last checked X ago" / "last failure: ..."
- Ollama: Base URL field (no key field), same buttons minus key-specific ones
- Nav link added alongside the existing `/dashboard` ↔ `/dashboard/analytics` links (three-way nav)

Forms POST via a small amount of vanilla JS (`fetch`) to the JSON endpoints above and re-render their own card's status from the response — no full-page HTMX polling, but this page does need *some* interactivity beyond a static render (Test/Save must show a result without a full page reload). This is a deliberate, scoped exception to "no JS beyond Chart.js" established in Phase 8 — confirmed necessary because credential forms are inherently interactive, unlike analytics' read-only charts.

## 10. Testing

TDD throughout, matching every prior phase:
- `backend/tests/test_credential_store.py` — encryption round-trip, env-var fallback, save/delete/enable-disable, masking
- `backend/tests/test_anthropic_provider.py`, `test_ollama_provider.py` — construction, `health_check()` success/failure, `generate`/`stream` against a mocked SDK/HTTP client (same style as any existing `OpenAIProvider` tests, if present — confirm pattern during planning)
- `backend/tests/test_provider_manager.py` (extend or create) — `reload_provider` swaps only the named provider, leaves others untouched
- `backend/tests/test_providers_router.py` — all four endpoints, including the save-fails-old-provider-survives case and the test-endpoint-never-persists case
- `backend/tests/test_dashboard_providers_ui.py` — page renders, masked keys only, three-way nav

## 11. Non-goals / explicit invariants

- `ProviderManager` never imports `cryptography` or touches SQL — encryption and persistence are entirely `CredentialStore`'s responsibility.
- `POST /v1/providers/{name}/test` never writes to the database and never calls `reload_provider`.
- No endpoint or template ever renders `encrypted_api_key` or an unmasked key.
- A failed save-time health check never disables or replaces a currently-working provider.
- No new model registry entries, no routing changes — Anthropic/Ollama become health-checkable, not chat-routable, in this release.
