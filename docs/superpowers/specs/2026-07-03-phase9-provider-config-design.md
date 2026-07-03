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
- `POST /v1/providers/{name}/config`, `DELETE /v1/providers/{name}/config`, `POST /v1/providers/{name}/test`, `POST /v1/providers/{name}/enable`, `POST /v1/providers/{name}/disable`, `GET /v1/providers/config`
- `/dashboard/providers` page — one form per provider, masked keys, Test/Save/Delete/Disable
- `is_enabled` flag (disable without deleting)

Explicitly out of scope (deferred):
- Gemini, Groq, OpenRouter provider classes — no code exists for them today; a separate feature
- Populating `models.yaml` with new Anthropic/Ollama model entries so chat requests actually route to them — this feature makes providers *connectable and health-checkable*, not *routable*. Routing remains OpenAI-only until model registry work happens separately.
- Multi-organization/multi-tenant credential scoping — the `organization_id` column is added now (nullable, unused) purely to avoid a future schema migration; no organization logic is implemented
- Credential rotation history, audit log of who changed what (an `updated_by` nullable column is added now, unpopulated, for the same forward-compatibility reason as `organization_id` — no auth/identity system exists yet to populate it)
- Any UI/API auth — matches the existing dashboard/API's current posture (none)
- Capability/metadata discovery (model count, supports_chat/embeddings/images per provider) — a distinct feature with its own caching and refresh-cadence questions; not part of credential lifecycle
- Encryption key rotation — changing `PROVIDER_CREDENTIAL_ENCRYPTION_KEY` after credentials exist requires re-encrypting every stored row; this release does not implement that migration, so rotating the key without a manual re-save of every credential will make existing rows undecryptable

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
    updated_by: Mapped[str | None] = mapped_column(String, nullable=True)  # unused, reserved
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
    HEALTHY = "healthy"                   # active provider, last health check passed
    INVALID_CREDENTIALS = "invalid_credentials"  # last save attempt failed health check
    UNAVAILABLE = "unavailable"           # configured + enabled but not currently reachable
    DISABLED = "disabled"                 # is_enabled = False
```

Computed by the dashboard service layer from: credential presence, `is_enabled`, whether `ProviderManager.is_provider_available(name)` is true, and whether `last_failure_reason` is set. There is no `CONNECTING`/in-flight backend state — health checks are synchronous (typically well under a second) and the backend response is always terminal (success or failure) by the time it returns. Any "testing…" indicator is purely an ephemeral frontend UI state while awaiting the API response, never returned by the backend or persisted.

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
        """Encrypts api_key if present, upserts the row, sets
        last_successful_health_check to now. Only ever called AFTER a
        passing health check (see section 7) -- CredentialStore has no
        method that persists an unvalidated credential, so there is no
        code path that can write a row this class itself considers
        invalid."""

    def record_health_check_failure(
        self, provider_name: str, failure_reason: str,
    ) -> None:
        """Records a failed validation attempt WITHOUT touching the
        stored credential row -- if provider_name has no existing row,
        this is a no-op beyond logging/observability; if it has one, that
        row (and whatever is currently active in ProviderManager) is left
        exactly as it was. failure_reason is surfaced to the UI in the
        API response directly, not persisted, since it describes an
        input that was never saved."""

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

Validate-before-persist: nothing is written to the database until the submitted credential has already passed a health check. This closes the restart failure mode of a save-then-validate design (where a bad key, once persisted, would reload and activate on the next process restart even though the live provider was correctly protected in the moment) — there is no code path that can leave an unvalidated or failing credential sitting in the table.

```
POST /v1/providers/{name}/config  { api_key?, base_url? }
        │
        ▼
build a throwaway BaseProvider from the SUBMITTED credential
(nothing persisted yet)
        │
        ▼
await provider.health_check()
        │
   ┌────┴────┐
 success    failure
   │            │
   ▼            ▼
CredentialStore    CredentialStore.record_health_check_failure(
  .save(name,         name, failure_reason=str(exc))
   api_key,            │        -- no row written/changed
   base_url)           ▼
   │              response: {"saved": false, "activated": false,
   ▼                          "reason": "..."} -- the submitted value is
ProviderManager              NOT persisted; the UI keeps what the operator
  .reload_provider(name)     typed in the form so it isn't lost, but a
   │                          page reload will show the previous
   ▼                          (still-active) state, not the failed input.
response: {"saved": true, "activated": true}
```

A credential is only ever written once it is already known to work, so a fresh process restart can never load a row that fails its own health check. The UI is responsible for not losing the operator's typed input across a failed submit — that's a client-side concern, not a server-side persistence one.

## 8. API

- `POST /v1/providers/{name}/config` — body `{api_key?: str, base_url?: str}`, implements the flow in §7. `name` must be one of `KNOWN_PROVIDER_NAMES`; 404 otherwise.
- `DELETE /v1/providers/{name}/config` — deterministic delete semantics: (1) remove the `provider_credentials` row for `name`; (2) call `reload_provider(name)`; (3) `reload_provider` asks `CredentialStore.get(name)` again — since the row is now gone, this returns the `Settings` env-var fallback if one exists (that provider becomes/stays active), or `None` if no env var is set either (the provider is unregistered from `ProviderManager`, `is_provider_available(name)` becomes `False`).
- `POST /v1/providers/{name}/enable`, `POST /v1/providers/{name}/disable` — dedicated endpoints (not overloaded onto save) that call `CredentialStore.set_enabled(name, True|False)` then `reload_provider(name)`. Disabling makes `CredentialStore.get(name)` return `None` for that provider regardless of a stored key (per §5's `get` semantics: "DB row (if `is_enabled`)"), so `reload_provider` unregisters it — same end state as delete, but the row and key remain stored for re-enabling later.
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
- `backend/tests/test_providers_router.py` — all six endpoints (config POST/DELETE, enable, disable, test, config GET), including the save-with-failing-health-check-persists-nothing case, the delete-falls-back-to-env-var case, and the test-endpoint-never-persists case
- `backend/tests/test_dashboard_providers_ui.py` — page renders, masked keys only, three-way nav

## 11. Failure recovery

- **Bad credential submitted:** never persisted (§7) — cannot cause a post-restart outage. The only trace is `last_failure_reason` on whatever row already existed for that provider, if any.
- **Server restart:** `ProviderManager.__init__` re-derives every provider from `CredentialStore.get(name)` exactly as it does at any other startup. Since only validated credentials are ever stored, a restart cannot newly break a provider that was working before it — the same key that passed health_check() when saved is what gets loaded.
- **Database unavailable at startup:** unchanged from existing behavior — `create_engine_from_settings`/`init_db` failures already crash startup loudly (no try/except), and `CredentialStore` sits on the same `session_factory`, so this failure mode is inherited, not new.
- **`PROVIDER_CREDENTIAL_ENCRYPTION_KEY` missing or malformed:** crashes startup immediately if any `provider_credentials` row exists (§5) — the app never starts in a state where it silently can't decrypt a stored key. If the table is empty, startup proceeds normally (nothing to decrypt).
- **Ollama offline:** `OllamaProvider.health_check()` fails like any other provider's — `test`/`save` report the failure normally; an already-active Ollama provider going offline mid-session is a runtime failure the existing `ProviderExecutor`/circuit breaker layer (Phase 5) already handles, unchanged by this feature.

## 12. Non-goals / explicit invariants

- `ProviderManager` never imports `cryptography` or touches SQL — encryption and persistence are entirely `CredentialStore`'s responsibility.
- `POST /v1/providers/{name}/test` never writes to the database and never calls `reload_provider`.
- No endpoint or template ever renders `encrypted_api_key` or an unmasked key.
- A failing health check — at save time or test time — never persists the submitted credential and never disables or replaces a currently-working provider.
- No new model registry entries, no routing changes — Anthropic/Ollama become health-checkable, not chat-routable, in this release.
- Capability/metadata discovery (model_count, supports_chat/embeddings/images) is deferred — a distinct feature with its own caching questions.
- Encryption key rotation is not supported — changing `PROVIDER_CREDENTIAL_ENCRYPTION_KEY` after credentials exist makes existing rows undecryptable; there is no re-encryption migration in this release.
- `organization_id` and `updated_by` columns exist on `provider_credentials` but are unpopulated and unused — reserved purely to avoid a future schema migration, not functioning fields in v0.9.0.
