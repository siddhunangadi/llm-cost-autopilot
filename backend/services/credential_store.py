from dataclasses import dataclass
from datetime import datetime, timezone

from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker

from backend.config.settings import Settings
from backend.database.models import ProviderCredentialRow

KNOWN_PROVIDER_NAMES = ("openai", "anthropic", "ollama")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def mask_key(key: str | None) -> str | None:
    if key is None:
        return None
    if len(key) < 8:
        return "*" * len(key)
    return key[:3] + "*" * (len(key) - 7) + key[-4:]


@dataclass(frozen=True)
class ProviderCredential:
    provider_name: str
    api_key: str | None
    base_url: str | None


class ProviderConfigStatus(BaseModel):
    provider: str
    configured: bool
    masked_key: str | None
    base_url: str | None
    is_enabled: bool
    healthy: bool
    last_successful_health_check: datetime | None
    last_failure_reason: str | None


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
}


class CredentialStore:
    """Owns Fernet encryption and provider_credentials CRUD. The only
    layer in the system that knows encryption exists -- callers always
    receive/pass plain ProviderCredential value objects."""

    def __init__(self, session_factory: sessionmaker, settings: Settings) -> None:
        self._session_factory = session_factory
        self._settings = settings
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

    def _decrypt(self, encrypted: str | None) -> str | None:
        if encrypted is None:
            return None
        if self._fernet is None:
            raise RuntimeError("PROVIDER_CREDENTIAL_ENCRYPTION_KEY is not configured")
        try:
            return self._fernet.decrypt(encrypted.encode()).decode()
        except InvalidToken as exc:
            raise RuntimeError(f"Stored credential could not be decrypted: {exc}") from exc

    def _encrypt(self, plaintext: str | None) -> str | None:
        if plaintext is None:
            return None
        if self._fernet is None:
            raise RuntimeError("PROVIDER_CREDENTIAL_ENCRYPTION_KEY is not configured")
        return self._fernet.encrypt(plaintext.encode()).decode()

    def get(self, provider_name: str) -> ProviderCredential | None:
        with self._session_factory() as session:
            row = (
                session.query(ProviderCredentialRow)
                .filter_by(provider_name=provider_name)
                .first()
            )
        # is_enabled=False is an explicit user action (disable, or the
        # tombstone left by delete -- see delete()/set_enabled()) and must
        # never be silently overridden by an environment-variable
        # credential, regardless of whether stored key material exists.
        if row is not None and not row.is_enabled:
            return None
        has_credential_material = row is not None and (
            row.encrypted_api_key is not None or row.base_url is not None
        )
        if has_credential_material:
            return ProviderCredential(
                provider_name=provider_name,
                api_key=self._decrypt(row.encrypted_api_key),
                base_url=row.base_url,
            )
        fallback = _ENV_FALLBACK.get(provider_name)
        return fallback(self._settings) if fallback else None

    def save(
        self, provider_name: str, api_key: str | None, base_url: str | None,
    ) -> ProviderCredentialRow:
        with self._session_factory() as session:
            row = (
                session.query(ProviderCredentialRow)
                .filter_by(provider_name=provider_name)
                .first()
            )
            if row is None:
                row = ProviderCredentialRow(provider_name=provider_name)
                session.add(row)
            row.encrypted_api_key = self._encrypt(api_key)
            row.base_url = base_url
            row.is_enabled = True
            row.last_successful_health_check = _utcnow()
            row.last_failure_reason = None
            session.commit()
            session.refresh(row)
            return row

    def record_health_check_failure(self, provider_name: str, failure_reason: str) -> None:
        with self._session_factory() as session:
            row = (
                session.query(ProviderCredentialRow)
                .filter_by(provider_name=provider_name)
                .first()
            )
            if row is None:
                # No credential has ever been saved for this provider -- track
                # the failure anyway so a first-attempt failure isn't silently
                # dropped. is_enabled stays True (the default): this is a
                # failure note, not a user disable action, so it must not
                # suppress env-var fallback for an otherwise-working provider.
                row = ProviderCredentialRow(provider_name=provider_name)
                session.add(row)
            row.last_failure_reason = failure_reason
            session.commit()

    def set_enabled(self, provider_name: str, enabled: bool) -> None:
        with self._session_factory() as session:
            row = (
                session.query(ProviderCredentialRow)
                .filter_by(provider_name=provider_name)
                .first()
            )
            if row is None:
                if not enabled:
                    # No stored credential exists (provider is env-only, or
                    # was never configured) but the user explicitly asked to
                    # disable it -- persist a tombstone so get() stops
                    # resolving an environment-variable credential for it.
                    session.add(ProviderCredentialRow(provider_name=provider_name, is_enabled=False))
                    session.commit()
                return
            row.is_enabled = enabled
            session.commit()

    def delete(self, provider_name: str) -> None:
        with self._session_factory() as session:
            row = (
                session.query(ProviderCredentialRow)
                .filter_by(provider_name=provider_name)
                .first()
            )
            if row is None:
                row = ProviderCredentialRow(provider_name=provider_name)
                session.add(row)
            # Clear credential material but keep (or create) the row,
            # disabled, as a tombstone -- otherwise get() would see no row
            # at all and silently fall back to an environment-variable
            # credential, undoing the delete.
            row.encrypted_api_key = None
            row.base_url = None
            row.is_enabled = False
            row.last_failure_reason = None
            session.commit()

    def list_status(self, is_healthy_fn) -> list[ProviderConfigStatus]:
        with self._session_factory() as session:
            rows = {r.provider_name: r for r in session.query(ProviderCredentialRow).all()}
        result = []
        for name in KNOWN_PROVIDER_NAMES:
            row = rows.get(name)
            # A row can exist purely to record a health-check failure from a
            # save attempt that never actually persisted credential material
            # (see record_health_check_failure) -- that must not read as
            # "configured".
            has_credential_material = row is not None and (
                row.encrypted_api_key is not None or row.base_url is not None
            )
            if has_credential_material:
                result.append(ProviderConfigStatus(
                    provider=name,
                    configured=True,
                    masked_key=mask_key(self._decrypt(row.encrypted_api_key)),
                    base_url=row.base_url,
                    is_enabled=row.is_enabled,
                    healthy=is_healthy_fn(name),
                    last_successful_health_check=row.last_successful_health_check,
                    last_failure_reason=row.last_failure_reason,
                ))
            else:
                env_credential = self.get(name)
                result.append(ProviderConfigStatus(
                    provider=name,
                    configured=env_credential is not None,
                    masked_key=mask_key(env_credential.api_key) if env_credential else None,
                    base_url=env_credential.base_url if env_credential else None,
                    is_enabled=row.is_enabled if row is not None else True,
                    healthy=is_healthy_fn(name),
                    last_successful_health_check=None,
                    last_failure_reason=row.last_failure_reason if row is not None else None,
                ))
        return result
