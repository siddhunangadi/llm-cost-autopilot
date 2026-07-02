import textwrap

import pytest
import yaml
from pydantic import ValidationError

from backend.verification.config import VerificationConfig
from backend.verification.config_loader import VerificationConfigLoader

VALID_YAML = textwrap.dedent("""
    judge_model_id: gpt-4o
    pass_threshold: 0.7
    judge_prompt_version: v1
""")


def test_load_valid_verification_config(tmp_path):
    yaml_path = tmp_path / "verification.yaml"
    yaml_path.write_text(VALID_YAML)

    config = VerificationConfigLoader.load(str(yaml_path))

    assert isinstance(config, VerificationConfig)
    assert config.judge_model_id == "gpt-4o"
    assert config.pass_threshold == 0.7
    assert config.judge_prompt_version == "v1"


def test_load_raises_on_malformed_yaml(tmp_path):
    yaml_path = tmp_path / "verification.yaml"
    yaml_path.write_text("judge_model_id:\n\t- bad indentation\n")

    with pytest.raises(yaml.YAMLError):
        VerificationConfigLoader.load(str(yaml_path))


def test_load_raises_on_invalid_schema_missing_field(tmp_path):
    yaml_path = tmp_path / "verification.yaml"
    yaml_path.write_text("judge_model_id: gpt-4o\npass_threshold: 0.7\n")

    with pytest.raises(ValidationError):
        VerificationConfigLoader.load(str(yaml_path))


def test_real_verification_yaml_loads_successfully():
    config = VerificationConfigLoader.load("backend/config/verification.yaml")
    assert config.judge_model_id == "gpt-4o"
    assert config.pass_threshold == 0.7
    assert config.judge_prompt_version == "v1"
