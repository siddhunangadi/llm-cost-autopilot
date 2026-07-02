import textwrap

import pytest
import yaml
from pydantic import ValidationError

from backend.routing.config import BalancedStrategyWeights, RoutingConfig
from backend.routing.config_loader import RoutingConfigLoader

VALID_YAML = textwrap.dedent("""
    classifier:
      simple_max: 1
      medium_max: 3

    policy:
      simple:
        min_benchmark_score: 0.0
      medium:
        min_benchmark_score: 0.75
      complex:
        min_benchmark_score: 0.90

    balanced_strategy:
      cost_weight: 0.4
      latency_weight: 0.2
      quality_weight: 0.4
""")


def test_load_valid_routing_config(tmp_path):
    yaml_path = tmp_path / "routing.yaml"
    yaml_path.write_text(VALID_YAML)

    config = RoutingConfigLoader.load(str(yaml_path))

    assert isinstance(config, RoutingConfig)
    assert config.classifier.simple_max == 1
    assert config.classifier.medium_max == 3
    assert config.policy["simple"].min_benchmark_score == 0.0
    assert config.policy["medium"].min_benchmark_score == 0.75
    assert config.policy["complex"].min_benchmark_score == 0.90
    assert config.balanced_strategy.cost_weight == 0.4
    assert config.balanced_strategy.latency_weight == 0.2
    assert config.balanced_strategy.quality_weight == 0.4


def test_balanced_strategy_weights_default_to_equal_thirds():
    weights = BalancedStrategyWeights()
    assert weights.cost_weight == pytest.approx(1 / 3)
    assert weights.latency_weight == pytest.approx(1 / 3)
    assert weights.quality_weight == pytest.approx(1 / 3)


def test_load_raises_on_malformed_yaml(tmp_path):
    yaml_path = tmp_path / "routing.yaml"
    yaml_path.write_text("classifier:\n\t- bad indentation\n")

    with pytest.raises(yaml.YAMLError):
        RoutingConfigLoader.load(str(yaml_path))


def test_load_raises_on_invalid_schema_missing_classifier(tmp_path):
    yaml_path = tmp_path / "routing.yaml"
    yaml_path.write_text(
        textwrap.dedent("""
            policy:
              simple:
                min_benchmark_score: 0.0
            balanced_strategy:
              cost_weight: 0.4
              latency_weight: 0.2
              quality_weight: 0.4
        """)
    )

    with pytest.raises(ValidationError):
        RoutingConfigLoader.load(str(yaml_path))


def test_real_routing_yaml_loads_successfully():
    config = RoutingConfigLoader.load("backend/config/routing.yaml")
    assert config.classifier.simple_max == 1
    assert config.classifier.medium_max == 3
    assert set(config.policy.keys()) == {"simple", "medium", "complex"}
