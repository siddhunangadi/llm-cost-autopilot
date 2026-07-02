import pytest
from pydantic import ValidationError

from backend.config.settings import Settings


def test_settings_successful_load_with_defaults():
    settings = Settings(_env_file=None)
    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.database_url == "sqlite:///./llm_cost_autopilot.db"
    assert settings.models_yaml_path == "backend/config/models.yaml"
    assert settings.openai_api_key is None
    assert settings.anthropic_api_key is None


def test_settings_successful_load_with_explicit_values():
    settings = Settings(
        _env_file=None,
        environment="production",
        log_level="ERROR",
        database_url="sqlite:///./prod.db",
        models_yaml_path="config/models.yaml",
        openai_api_key="sk-live",
    )
    assert settings.environment == "production"
    assert settings.log_level == "ERROR"
    assert settings.database_url == "sqlite:///./prod.db"
    assert settings.models_yaml_path == "config/models.yaml"
    assert settings.openai_api_key == "sk-live"


def test_settings_rejects_invalid_environment():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="production-ish")


def test_settings_rejects_invalid_log_level():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, log_level="VERBOSE")


def test_settings_rejects_blank_database_url():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, database_url="")


def test_settings_rejects_blank_models_yaml_path():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, models_yaml_path="")


def test_settings_optional_provider_keys_default_to_none_when_missing():
    # Missing provider keys are intentionally not an error -- Phase 1 must
    # work with zero provider keys configured (ProviderManager decides what
    # "no key" means later, not Settings).
    settings = Settings(_env_file=None)
    assert settings.openai_api_key is None
    assert settings.anthropic_api_key is None


def test_settings_reads_env_var_overrides(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

    settings = Settings(_env_file=None)

    assert settings.environment == "test"
    assert settings.log_level == "DEBUG"
    assert settings.openai_api_key == "sk-from-env"


def test_settings_models_yaml_path_is_a_plain_path_string_not_parsed():
    # Settings only carries the path -- ModelRegistry (Task 13) owns reading
    # and validating the YAML content itself, per the frozen design's split
    # between config-loading and registry concerns.
    settings = Settings(_env_file=None, models_yaml_path="some/nonexistent/models.yaml")
    assert settings.models_yaml_path == "some/nonexistent/models.yaml"


def test_settings_routing_config_path_default():
    settings = Settings(_env_file=None)
    assert settings.routing_config_path == "backend/config/routing.yaml"


def test_settings_rejects_blank_routing_config_path():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, routing_config_path="")
