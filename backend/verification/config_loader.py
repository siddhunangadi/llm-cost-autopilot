import yaml

from backend.verification.config import VerificationConfig


class VerificationConfigLoader:
    @staticmethod
    def load(yaml_path: str) -> VerificationConfig:
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        return VerificationConfig.model_validate(raw)
