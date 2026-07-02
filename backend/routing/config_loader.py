import yaml

from backend.routing.config import RoutingConfig


class RoutingConfigLoader:
    @staticmethod
    def load(yaml_path: str) -> RoutingConfig:
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        return RoutingConfig.model_validate(raw)
