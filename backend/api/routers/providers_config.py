from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from backend.api.dependencies import (
    AppVersionDep, CredentialStoreDep, ProviderFactoryDep, ProviderManagerDep,
)
from backend.api.paths import TEMPLATES_DIR
from backend.services.credential_store import ProviderConfigStatus, ProviderCredential

router = APIRouter(prefix="/v1/providers")
page_router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class ProviderConfigRequest(BaseModel):
    api_key: str | None = None
    base_url: str | None = None


class ProviderConfigResult(BaseModel):
    saved: bool
    activated: bool
    reason: str | None = None


def _require_known(name: str, known_names: tuple[str, ...]) -> None:
    if name not in known_names:
        raise HTTPException(status_code=404, detail=f"Unknown provider '{name}'")


def _resolve_candidate(
    name: str, body: ProviderConfigRequest, credential_store: CredentialStoreDep,
) -> ProviderCredential:
    """A blank field in the request means "unchanged", not "clear it" -- the
    UI only ever shows a masked existing key as a placeholder, never as a
    real value, so an omitted field must fall back to what's actually
    stored rather than being sent through as None and wiping/failing on a
    working credential."""
    stored = credential_store.get_stored(name)
    return ProviderCredential(
        provider_name=name,
        api_key=body.api_key if body.api_key is not None else (stored.api_key if stored else None),
        base_url=body.base_url if body.base_url is not None else (stored.base_url if stored else None),
    )


@router.post("/{name}/config", response_model=ProviderConfigResult)
async def save_provider_config(
    name: str, body: ProviderConfigRequest,
    credential_store: CredentialStoreDep, provider_manager: ProviderManagerDep,
    provider_factory: ProviderFactoryDep,
) -> ProviderConfigResult:
    _require_known(name, provider_manager.registered_names())
    candidate = _resolve_candidate(name, body, credential_store)
    provider = provider_factory.create(name, candidate)
    healthy = await provider.health_check()
    if not healthy:
        credential_store.record_health_check_failure(name, "health check failed")
        return ProviderConfigResult(saved=False, activated=False, reason="health check failed")
    credential_store.save(name, api_key=candidate.api_key, base_url=candidate.base_url)
    provider_manager.reload_provider(name)
    return ProviderConfigResult(saved=True, activated=True)


@router.delete("/{name}/config", response_model=ProviderConfigResult)
async def delete_provider_config(
    name: str, credential_store: CredentialStoreDep, provider_manager: ProviderManagerDep,
) -> ProviderConfigResult:
    _require_known(name, provider_manager.registered_names())
    credential_store.delete(name)
    activated = provider_manager.reload_provider(name)
    return ProviderConfigResult(saved=True, activated=activated)


@router.post("/{name}/enable", response_model=ProviderConfigResult)
async def enable_provider(
    name: str, credential_store: CredentialStoreDep, provider_manager: ProviderManagerDep,
) -> ProviderConfigResult:
    _require_known(name, provider_manager.registered_names())
    credential_store.set_enabled(name, True)
    activated = provider_manager.reload_provider(name)
    return ProviderConfigResult(saved=True, activated=activated)


@router.post("/{name}/disable", response_model=ProviderConfigResult)
async def disable_provider(
    name: str, credential_store: CredentialStoreDep, provider_manager: ProviderManagerDep,
) -> ProviderConfigResult:
    _require_known(name, provider_manager.registered_names())
    credential_store.set_enabled(name, False)
    activated = provider_manager.reload_provider(name)
    return ProviderConfigResult(saved=True, activated=activated)


@router.post("/{name}/test", response_model=ProviderConfigResult)
async def test_provider_config(
    name: str, body: ProviderConfigRequest,
    credential_store: CredentialStoreDep, provider_factory: ProviderFactoryDep,
) -> ProviderConfigResult:
    _require_known(name, provider_factory.registered_names())
    candidate = _resolve_candidate(name, body, credential_store)
    provider = provider_factory.create(name, candidate)
    healthy = await provider.health_check()
    return ProviderConfigResult(
        saved=False, activated=False, reason=None if healthy else "health check failed",
    )


@router.get("/config", response_model=list[ProviderConfigStatus])
async def list_provider_config(
    credential_store: CredentialStoreDep, provider_manager: ProviderManagerDep,
) -> list[ProviderConfigStatus]:
    return credential_store.list_status(provider_manager.is_provider_available)


@page_router.get("/dashboard/providers")
async def providers_page(
    request: Request, credential_store: CredentialStoreDep,
    provider_manager: ProviderManagerDep, app_version: AppVersionDep,
):
    statuses = credential_store.list_status(provider_manager.is_provider_available)
    return templates.TemplateResponse(request, "providers.html", {
        "statuses": statuses, "app_version": app_version, "now": _now_str(),
    })
