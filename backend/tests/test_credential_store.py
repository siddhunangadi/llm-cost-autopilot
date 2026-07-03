import pytest
from cryptography.fernet import Fernet

from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.database.models import ProviderCredentialRow
from backend.services.credential_store import CredentialStore, mask_key


def _make_store(tmp_path, **settings_kwargs):
    key = Fernet.generate_key().decode()
    settings = Settings(
        _env_file=None, database_url=f"sqlite:///{tmp_path}/test.db",
        provider_credential_encryption_key=key, **settings_kwargs,
    )
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)
    return CredentialStore(session_factory=session_factory, settings=settings), settings


def test_save_then_get_round_trips_the_api_key(tmp_path):
    store, _ = _make_store(tmp_path)
    store.save("openai", api_key="sk-secret", base_url=None)

    credential = store.get("openai")

    assert credential.api_key == "sk-secret"


def test_get_falls_back_to_env_var_when_no_row_exists(tmp_path):
    store, _ = _make_store(tmp_path, openai_api_key="sk-env")

    credential = store.get("openai")

    assert credential.api_key == "sk-env"


def test_get_returns_none_when_no_row_and_no_env_var(tmp_path):
    store, _ = _make_store(tmp_path)

    assert store.get("anthropic") is None


def test_save_clears_previous_failure_reason(tmp_path):
    store, _ = _make_store(tmp_path)
    store.save("openai", api_key="sk-bad", base_url=None)
    store.record_health_check_failure("openai", "auth failed")
    store.save("openai", api_key="sk-good", base_url=None)

    with store._session_factory() as session:
        row = session.query(ProviderCredentialRow).filter_by(provider_name="openai").first()
        assert row.last_failure_reason is None
        assert row.last_successful_health_check is not None


def test_record_health_check_failure_does_not_touch_stored_key(tmp_path):
    store, _ = _make_store(tmp_path)
    store.save("openai", api_key="sk-good", base_url=None)

    store.record_health_check_failure("openai", "timeout")

    assert store.get("openai").api_key == "sk-good"


def test_set_enabled_false_falls_back_to_env(tmp_path):
    store, _ = _make_store(tmp_path, openai_api_key="sk-env")
    store.save("openai", api_key="sk-db", base_url=None)

    store.set_enabled("openai", False)

    assert store.get("openai").api_key == "sk-env"


def test_delete_removes_the_row(tmp_path):
    store, _ = _make_store(tmp_path)
    store.save("openai", api_key="sk-db", base_url=None)

    store.delete("openai")

    assert store.get("openai") is None


def test_ollama_credential_has_no_api_key_only_base_url(tmp_path):
    store, _ = _make_store(tmp_path)
    store.save("ollama", api_key=None, base_url="http://localhost:11434")

    credential = store.get("ollama")

    assert credential.api_key is None
    assert credential.base_url == "http://localhost:11434"


def test_mask_key_long_key():
    assert mask_key("sk-1234567890ABCD") == "sk-**********ABCD"


def test_mask_key_short_key_fully_masked():
    assert mask_key("short") == "*****"


def test_mask_key_none_returns_none():
    assert mask_key(None) is None


def test_missing_encryption_key_with_existing_row_raises(tmp_path):
    settings = Settings(
        _env_file=None, database_url=f"sqlite:///{tmp_path}/test.db",
        provider_credential_encryption_key=Fernet.generate_key().decode(),
    )
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)
    with session_factory() as session:
        session.add(ProviderCredentialRow(provider_name="openai", encrypted_api_key="x"))
        session.commit()

    settings_no_key = Settings(
        _env_file=None, database_url=settings.database_url, provider_credential_encryption_key=None,
    )
    with pytest.raises(RuntimeError):
        CredentialStore(session_factory=session_factory, settings=settings_no_key)


def test_missing_encryption_key_with_no_rows_does_not_raise(tmp_path):
    settings = Settings(
        _env_file=None, database_url=f"sqlite:///{tmp_path}/test.db",
        provider_credential_encryption_key=None,
    )
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    CredentialStore(session_factory=session_factory, settings=settings)


def test_list_status_reflects_configured_and_unconfigured_providers(tmp_path):
    store, _ = _make_store(tmp_path)
    store.save("openai", api_key="sk-good", base_url=None)

    statuses = store.list_status(is_healthy_fn=lambda name: name == "openai")

    by_provider = {s.provider: s for s in statuses}
    assert by_provider["openai"].configured is True
    assert by_provider["openai"].masked_key == mask_key("sk-good")
    assert by_provider["openai"].healthy is True
    assert by_provider["anthropic"].configured is False
    assert by_provider["anthropic"].masked_key is None
