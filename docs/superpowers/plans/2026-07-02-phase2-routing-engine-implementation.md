# LLM Cost Autopilot — Phase 2 Implementation Plan: Intelligent Routing Engine

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a heuristic routing pipeline (prompt analysis → complexity classification → policy-filtered strategy selection → explanation) exposed via `POST /v1/chat`, with request/response/routing persistence.

**Architecture:** Five new top-level packages (`analysis`, `classifier`, `routing`, `chat`, plus a new API router) layered on top of the existing Phase 1 provider/registry/DI stack. `RoutingEngine` never calls a provider; `ChatService` is the only component that calls both `RoutingEngine` and a provider.

**Tech Stack:** Same as Phase 1 — Python 3.11+, `uv`, FastAPI, Pydantic v2, SQLAlchemy 2.0, PyYAML. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-02-phase2-routing-engine-design.md` (frozen — implement exactly).

## Global Constraints

- Same `uv`-managed Python 3.11+ project as Phase 1; no new dependencies.
- **Execution model differs from Phase 1**: tasks are grouped into 6 batches (2-3 tasks each). Within a batch, follow the test-first steps per task, but there is **one test run + one commit per batch**, not per task. Only revert to single-task commits if a batch hits an unexpected architectural issue or repeated test failures.
- `RoutingConfig.policy` is `dict[str, EligibilityPolicy]` (keyed by plain strings `"simple"`/`"medium"`/`"complex"`), **not** `dict[ComplexityTier, EligibilityPolicy]` — keying by the enum would create a circular import between `backend/routing/config.py` and `backend/classifier/complexity_classifier.py`. `RoutingPolicy.filter_candidates()` is the one place that converts `complexity.value` to look up this dict.
- `RoutingEngine` never calls a provider or touches the database. `ChatService` is the only component that does both.
- `POST /v1/chat` uses `provider.generate()` only — no streaming (`provider.stream()`) in Phase 2.
- No ML classifier, no LLM-as-judge, no auto-escalation, no background workers, no retries/caching/rate-limiting, no `quality_profile` schema change — all explicitly out of scope per the spec.
- `RoutingConfigLoader` is the only component that opens/parses `routing.yaml` — `ClassifierPolicy`, `RoutingPolicy`, and `BalancedStrategyWeights` all receive already-parsed, already-validated sub-models.
- All new API routes mounted under `/v1`, matching Phase 1.
- No placeholder code, no TODOs, no unused abstractions (e.g. no `BasePromptAnalyzer` — see spec §4.2 for why).

---

## Batch 1: Prompt Analysis

### Task 19: PromptAnalyzer & PromptFeatures

**Files:**
- Create: `backend/analysis/__init__.py` (empty)
- Create: `backend/analysis/prompt_analyzer.py`
- Test: `backend/tests/test_prompt_analyzer.py`

**Interfaces:**
- Produces: `PromptFeatures` (Pydantic model, 14 fields per spec §4.1), `PromptAnalyzer.analyze(prompt: str) -> PromptFeatures`. Consumed by `HeuristicComplexityClassifier` (Task 21), `RoutingEngine` (Task 26).

- [ ] **Step 1: Create the package directory**

Run:
```bash
mkdir -p backend/analysis
touch backend/analysis/__init__.py
```

- [ ] **Step 2: Write the failing test**

```python
# backend/tests/test_prompt_analyzer.py
from backend.analysis.prompt_analyzer import PromptAnalyzer


def test_prompt_length_and_estimated_tokens():
    analyzer = PromptAnalyzer()
    features = analyzer.analyze("hello world")
    assert features.prompt_length == 11
    assert features.estimated_tokens == max(1, 11 // 4)


def test_constraint_count_detects_multiple_constraints():
    analyzer = PromptAnalyzer()
    features = analyzer.analyze("You must include a summary and should ensure clarity.")
    assert features.constraint_count >= 2


def test_has_code_detects_code_fence():
    analyzer = PromptAnalyzer()
    features = analyzer.analyze("Here is code: ```python\nprint('hi')\n```")
    assert features.has_code is True


def test_has_code_false_for_plain_prompt():
    analyzer = PromptAnalyzer()
    features = analyzer.analyze("Tell me a joke.")
    assert features.has_code is False


def test_requested_language_detected_when_has_code():
    analyzer = PromptAnalyzer()
    features = analyzer.analyze(
        "Write a python function that adds two numbers. ```def add(a, b): return a + b```"
    )
    assert features.has_code is True
    assert features.requested_language == "python"


def test_requested_language_none_without_code():
    analyzer = PromptAnalyzer()
    features = analyzer.analyze("Tell me about python snakes.")
    assert features.has_code is False
    assert features.requested_language is None


def test_has_json_detects_json_mention():
    analyzer = PromptAnalyzer()
    features = analyzer.analyze("Return the result as JSON.")
    assert features.has_json is True


def test_has_reasoning_keywords():
    analyzer = PromptAnalyzer()
    features = analyzer.analyze("Explain why the sky is blue.")
    assert features.has_reasoning_keywords is True


def test_has_comparison_keywords():
    analyzer = PromptAnalyzer()
    features = analyzer.analyze("Compare Python versus JavaScript.")
    assert features.has_comparison_keywords is True


def test_has_analysis_keywords():
    analyzer = PromptAnalyzer()
    features = analyzer.analyze("Analyze this dataset for trends.")
    assert features.has_analysis_keywords is True


def test_has_creative_keywords():
    analyzer = PromptAnalyzer()
    features = analyzer.analyze("Write a short story about a dragon.")
    assert features.has_creative_keywords is True


def test_has_math_indicators_from_keyword():
    analyzer = PromptAnalyzer()
    features = analyzer.analyze("Calculate the total cost.")
    assert features.has_math_indicators is True


def test_has_math_indicators_from_operator():
    analyzer = PromptAnalyzer()
    features = analyzer.analyze("What is 5 + 7?")
    assert features.has_math_indicators is True


def test_has_chain_of_thought_indicators():
    analyzer = PromptAnalyzer()
    features = analyzer.analyze("Walk me through this step by step.")
    assert features.has_chain_of_thought_indicators is True


def test_requires_output_formatting():
    analyzer = PromptAnalyzer()
    features = analyzer.analyze("Format as bullet points.")
    assert features.requires_output_formatting is True


def test_estimated_output_tokens_brief_phrase():
    analyzer = PromptAnalyzer()
    features = analyzer.analyze("Answer briefly.")
    assert features.estimated_output_tokens == 20


def test_estimated_output_tokens_explicit_word_count():
    analyzer = PromptAnalyzer()
    features = analyzer.analyze(
        "Write a 500 word essay about the ocean, but keep style simple."
    )
    assert features.estimated_output_tokens == round(500 * 1.3)


def test_estimated_output_tokens_long_form_keyword():
    analyzer = PromptAnalyzer()
    features = analyzer.analyze("Write a comprehensive essay about climate change.")
    assert features.estimated_output_tokens == max(800, features.estimated_tokens * 2)


def test_estimated_output_tokens_default_scales_with_input():
    analyzer = PromptAnalyzer()
    features = analyzer.analyze("What is the capital of France?")
    assert features.estimated_output_tokens == max(50, min(features.estimated_tokens, 500))


def test_default_features_are_false_for_neutral_prompt():
    analyzer = PromptAnalyzer()
    features = analyzer.analyze("List three fruits.")
    assert features.has_code is False
    assert features.has_json is False
    assert features.has_reasoning_keywords is False
    assert features.has_comparison_keywords is False
    assert features.has_analysis_keywords is False
    assert features.has_creative_keywords is False
    assert features.has_math_indicators is False
    assert features.has_chain_of_thought_indicators is False
    assert features.requires_output_formatting is False
    assert features.constraint_count == 0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_prompt_analyzer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.analysis.prompt_analyzer'`

- [ ] **Step 4: Write the implementation**

```python
# backend/analysis/prompt_analyzer.py
import re

from pydantic import BaseModel


class PromptFeatures(BaseModel):
    prompt_length: int
    estimated_tokens: int
    estimated_output_tokens: int
    constraint_count: int
    has_code: bool
    has_json: bool
    has_reasoning_keywords: bool
    has_comparison_keywords: bool
    has_analysis_keywords: bool
    has_creative_keywords: bool
    has_math_indicators: bool
    has_chain_of_thought_indicators: bool
    requires_output_formatting: bool
    requested_language: str | None = None


_CONSTRAINT_PATTERN = re.compile(
    r"\b(must|should|need to|ensure|require[sd]?|make sure)\b", re.IGNORECASE
)
_CODE_KEYWORDS_PATTERN = re.compile(
    r"```|\bdef \b|\bfunction \b|\bclass \b|\bimport \b", re.IGNORECASE
)
_JSON_PATTERN = re.compile(r"\bjson\b", re.IGNORECASE)
_JSON_STRUCTURE_PATTERN = re.compile(r"\{[^{}]*:[^{}]*\}")
_REASONING_PATTERN = re.compile(r"\b(why|explain|reasoning|because)\b", re.IGNORECASE)
_COMPARISON_PATTERN = re.compile(r"\b(compare|versus|vs\.?|difference between)\b", re.IGNORECASE)
_ANALYSIS_PATTERN = re.compile(r"\b(analyze|analysis|evaluate|assess|review)\b", re.IGNORECASE)
_CREATIVE_PATTERN = re.compile(r"\b(story|poem|creative|imagine|brainstorm)\b", re.IGNORECASE)
_MATH_KEYWORD_PATTERN = re.compile(r"\b(calculate|solve|equation)\b", re.IGNORECASE)
_MATH_OPERATOR_PATTERN = re.compile(r"\d\s*[+\-*/=]\s*\d")
_CHAIN_OF_THOUGHT_PATTERN = re.compile(
    r"\b(step by step|walk me through|first.*then)\b", re.IGNORECASE
)
_OUTPUT_FORMAT_PATTERN = re.compile(
    r"\b(format as|return as|bullet points|as a table|in json)\b", re.IGNORECASE
)
_WORD_COUNT_PATTERN = re.compile(r"(\d+)\s*word", re.IGNORECASE)
_BRIEF_PATTERN = re.compile(
    r"\b(one sentence|briefly|short answer|one word|yes or no)\b", re.IGNORECASE
)
_LONG_FORM_PATTERN = re.compile(
    r"\b(essay|comprehensive|detailed|in-depth|thorough|elaborate)\b", re.IGNORECASE
)
_LANGUAGE_PATTERN = re.compile(
    r"\b(python|javascript|typescript|go|rust|java|c\+\+|sql)\b", re.IGNORECASE
)


class PromptAnalyzer:
    def analyze(self, prompt: str) -> PromptFeatures:
        prompt_length = len(prompt)
        estimated_tokens = max(1, prompt_length // 4)
        has_code = bool(_CODE_KEYWORDS_PATTERN.search(prompt))

        return PromptFeatures(
            prompt_length=prompt_length,
            estimated_tokens=estimated_tokens,
            estimated_output_tokens=self._estimate_output_tokens(prompt, estimated_tokens),
            constraint_count=len(_CONSTRAINT_PATTERN.findall(prompt)),
            has_code=has_code,
            has_json=bool(_JSON_PATTERN.search(prompt) or _JSON_STRUCTURE_PATTERN.search(prompt)),
            has_reasoning_keywords=bool(_REASONING_PATTERN.search(prompt)),
            has_comparison_keywords=bool(_COMPARISON_PATTERN.search(prompt)),
            has_analysis_keywords=bool(_ANALYSIS_PATTERN.search(prompt)),
            has_creative_keywords=bool(_CREATIVE_PATTERN.search(prompt)),
            has_math_indicators=bool(
                _MATH_KEYWORD_PATTERN.search(prompt) or _MATH_OPERATOR_PATTERN.search(prompt)
            ),
            has_chain_of_thought_indicators=bool(_CHAIN_OF_THOUGHT_PATTERN.search(prompt)),
            requires_output_formatting=bool(_OUTPUT_FORMAT_PATTERN.search(prompt)),
            requested_language=self._detect_language(prompt) if has_code else None,
        )

    def _estimate_output_tokens(self, prompt: str, estimated_tokens: int) -> int:
        if _BRIEF_PATTERN.search(prompt):
            return 20

        word_count_match = _WORD_COUNT_PATTERN.search(prompt)
        if word_count_match:
            return max(20, round(int(word_count_match.group(1)) * 1.3))

        if _LONG_FORM_PATTERN.search(prompt):
            return max(800, estimated_tokens * 2)

        return max(50, min(estimated_tokens, 500))

    def _detect_language(self, prompt: str) -> str | None:
        match = _LANGUAGE_PATTERN.search(prompt)
        return match.group(1).lower() if match else None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_prompt_analyzer.py -v`
Expected: PASS (20 tests)

- [ ] **Batch 1 verification & commit**

Run the full suite to check for regressions:
```bash
uv run pytest -v
```
Expected: all tests pass (97 from Phase 1 + 20 new = 117).

Commit:
```bash
git add backend/analysis backend/tests/test_prompt_analyzer.py
git commit -m "feat: add PromptAnalyzer and PromptFeatures"
```

---

## Batch 2: Complexity Classification

### Task 20: RoutingConfig, RoutingConfigLoader & Settings

**Files:**
- Create: `backend/routing/__init__.py` (empty)
- Create: `backend/routing/config.py`
- Create: `backend/routing/config_loader.py`
- Create: `backend/config/routing.yaml`
- Modify: `backend/config/settings.py`
- Modify: `backend/tests/test_settings.py`
- Test: `backend/tests/test_routing_config.py`

**Interfaces:**
- Produces: `ClassifierPolicy(simple_max: int, medium_max: int)`, `EligibilityPolicy(min_benchmark_score: float)`, `BalancedStrategyWeights(cost_weight, latency_weight, quality_weight)` (each defaulting to `1/3`), `RoutingConfig(classifier, policy: dict[str, EligibilityPolicy], balanced_strategy)`, `RoutingConfigLoader.load(yaml_path: str) -> RoutingConfig`. `Settings.routing_config_path: str` (new field, default `"backend/config/routing.yaml"`). Consumed by `HeuristicComplexityClassifier` (Task 21, `config.classifier`), `RoutingPolicy` (Task 22, `config.policy`), `BalancedStrategy` (Task 24, `config.balanced_strategy`), `main.py` (Task 29, loads once at startup).

- [ ] **Step 1: Create the package directory**

Run:
```bash
mkdir -p backend/routing
touch backend/routing/__init__.py
```

- [ ] **Step 2: Write `backend/config/routing.yaml`**

```yaml
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
  cost_weight: 0.3333333333333333
  latency_weight: 0.3333333333333333
  quality_weight: 0.3333333333333334
```

- [ ] **Step 3: Write the failing tests**

```python
# backend/tests/test_routing_config.py
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
```

Add to `backend/tests/test_settings.py` (append at the end of the file):

```python
def test_settings_routing_config_path_default():
    settings = Settings(_env_file=None)
    assert settings.routing_config_path == "backend/config/routing.yaml"


def test_settings_rejects_blank_routing_config_path():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, routing_config_path="")
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest backend/tests/test_routing_config.py backend/tests/test_settings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.routing.config'` (and the two new Settings tests fail with `AttributeError`)

- [ ] **Step 5: Write `backend/routing/config.py`**

```python
# backend/routing/config.py
from pydantic import BaseModel, Field


class ClassifierPolicy(BaseModel):
    simple_max: int
    medium_max: int


class EligibilityPolicy(BaseModel):
    min_benchmark_score: float


class BalancedStrategyWeights(BaseModel):
    cost_weight: float = Field(default=1 / 3)
    latency_weight: float = Field(default=1 / 3)
    quality_weight: float = Field(default=1 / 3)


class RoutingConfig(BaseModel):
    classifier: ClassifierPolicy
    policy: dict[str, EligibilityPolicy]
    balanced_strategy: BalancedStrategyWeights
```

- [ ] **Step 6: Write `backend/routing/config_loader.py`**

```python
# backend/routing/config_loader.py
import yaml

from backend.routing.config import RoutingConfig


class RoutingConfigLoader:
    @staticmethod
    def load(yaml_path: str) -> RoutingConfig:
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        return RoutingConfig.model_validate(raw)
```

- [ ] **Step 7: Modify `backend/config/settings.py`**

Change:
```python
    database_url: str = Field(default="sqlite:///./llm_cost_autopilot.db", min_length=1)
    models_yaml_path: str = Field(default="backend/config/models.yaml", min_length=1)

    openai_api_key: str | None = None
```

To:
```python
    database_url: str = Field(default="sqlite:///./llm_cost_autopilot.db", min_length=1)
    models_yaml_path: str = Field(default="backend/config/models.yaml", min_length=1)
    routing_config_path: str = Field(default="backend/config/routing.yaml", min_length=1)

    openai_api_key: str | None = None
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest backend/tests/test_routing_config.py backend/tests/test_settings.py -v`
Expected: PASS (5 + 11 = 16 tests)

### Task 21: Complexity Classifier

**Files:**
- Create: `backend/classifier/__init__.py` (empty)
- Create: `backend/classifier/complexity_classifier.py`
- Test: `backend/tests/test_complexity_classifier.py`

**Interfaces:**
- Consumes: `PromptFeatures` (Task 19), `ClassifierPolicy` (Task 20).
- Produces: `ComplexityTier(str, Enum)` (`SIMPLE`, `MEDIUM`, `COMPLEX`), `ClassificationResult(tier, score, confidence, signals: list[str])`, `BaseComplexityClassifier` ABC (`classify(features) -> ClassificationResult`), `HeuristicComplexityClassifier(policy: ClassifierPolicy)`. Consumed by `RoutingPolicy` (Task 22, via `ComplexityTier`), `RoutingContext` (Task 23), `RoutingEngine` (Task 26).

- [ ] **Step 1: Create the package directory**

Run:
```bash
mkdir -p backend/classifier
touch backend/classifier/__init__.py
```

- [ ] **Step 2: Write the failing test**

```python
# backend/tests/test_complexity_classifier.py
import pytest

from backend.analysis.prompt_analyzer import PromptFeatures
from backend.classifier.complexity_classifier import (
    BaseComplexityClassifier,
    ComplexityTier,
    HeuristicComplexityClassifier,
)
from backend.routing.config import ClassifierPolicy


def _features(**overrides) -> PromptFeatures:
    defaults = dict(
        prompt_length=10,
        estimated_tokens=10,
        estimated_output_tokens=50,
        constraint_count=0,
        has_code=False,
        has_json=False,
        has_reasoning_keywords=False,
        has_comparison_keywords=False,
        has_analysis_keywords=False,
        has_creative_keywords=False,
        has_math_indicators=False,
        has_chain_of_thought_indicators=False,
        requires_output_formatting=False,
        requested_language=None,
    )
    defaults.update(overrides)
    return PromptFeatures(**defaults)


def _classifier() -> HeuristicComplexityClassifier:
    return HeuristicComplexityClassifier(ClassifierPolicy(simple_max=1, medium_max=3))


def test_base_complexity_classifier_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseComplexityClassifier()


def test_zero_signals_classifies_as_simple():
    result = _classifier().classify(_features())
    assert result.tier == ComplexityTier.SIMPLE
    assert result.score == 0
    assert result.signals == []


def test_one_signal_still_simple():
    result = _classifier().classify(_features(has_reasoning_keywords=True))
    assert result.tier == ComplexityTier.SIMPLE
    assert result.score == 1
    assert result.signals == ["reasoning keywords detected"]


def test_two_signals_classifies_as_medium():
    result = _classifier().classify(_features(has_reasoning_keywords=True, has_code=True))
    assert result.tier == ComplexityTier.MEDIUM
    assert result.score == 2


def test_four_signals_classifies_as_complex():
    result = _classifier().classify(
        _features(
            has_reasoning_keywords=True,
            has_code=True,
            has_analysis_keywords=True,
            has_math_indicators=True,
        )
    )
    assert result.tier == ComplexityTier.COMPLEX
    assert result.score == 4


def test_all_nine_signals_present():
    result = _classifier().classify(
        _features(
            estimated_tokens=250,
            constraint_count=2,
            has_code=True,
            has_reasoning_keywords=True,
            has_comparison_keywords=True,
            has_analysis_keywords=True,
            has_math_indicators=True,
            has_chain_of_thought_indicators=True,
            requires_output_formatting=True,
        )
    )
    assert result.score == 9
    assert result.tier == ComplexityTier.COMPLEX
    assert len(result.signals) == 9


def test_confidence_is_low_at_tier_boundary():
    result = _classifier().classify(_features(has_reasoning_keywords=True))  # score=1
    assert result.confidence == 0.5


def test_confidence_is_high_deep_in_a_tier():
    result = _classifier().classify(
        _features(
            estimated_tokens=250,
            constraint_count=2,
            has_code=True,
            has_reasoning_keywords=True,
            has_comparison_keywords=True,
            has_analysis_keywords=True,
            has_math_indicators=True,
            has_chain_of_thought_indicators=True,
            requires_output_formatting=True,
        )
    )  # score=9
    assert result.confidence == 0.99
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_complexity_classifier.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.classifier.complexity_classifier'`

- [ ] **Step 4: Write the implementation**

```python
# backend/classifier/complexity_classifier.py
from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel

from backend.analysis.prompt_analyzer import PromptFeatures
from backend.routing.config import ClassifierPolicy


class ComplexityTier(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class ClassificationResult(BaseModel):
    tier: ComplexityTier
    score: int
    confidence: float
    signals: list[str]


class BaseComplexityClassifier(ABC):
    @abstractmethod
    def classify(self, features: PromptFeatures) -> ClassificationResult: ...


class HeuristicComplexityClassifier(BaseComplexityClassifier):
    def __init__(self, policy: ClassifierPolicy) -> None:
        self._policy = policy

    def classify(self, features: PromptFeatures) -> ClassificationResult:
        score = 0
        signals: list[str] = []

        if features.estimated_tokens > 200:
            score += 1
            signals.append("prompt exceeds 200 estimated tokens")
        if features.constraint_count >= 2:
            score += 1
            signals.append("multiple constraints detected")
        if features.has_code:
            score += 1
            signals.append("code content detected")
        if features.has_reasoning_keywords:
            score += 1
            signals.append("reasoning keywords detected")
        if features.has_comparison_keywords:
            score += 1
            signals.append("comparison keywords detected")
        if features.has_analysis_keywords:
            score += 1
            signals.append("analysis keywords detected")
        if features.has_math_indicators:
            score += 1
            signals.append("math indicators detected")
        if features.has_chain_of_thought_indicators:
            score += 1
            signals.append("chain-of-thought indicators detected")
        if features.requires_output_formatting:
            score += 1
            signals.append("output formatting requested")

        return ClassificationResult(
            tier=self._tier_for_score(score),
            score=score,
            confidence=self._confidence_for_score(score),
            signals=signals,
        )

    def _tier_for_score(self, score: int) -> ComplexityTier:
        if score <= self._policy.simple_max:
            return ComplexityTier.SIMPLE
        if score <= self._policy.medium_max:
            return ComplexityTier.MEDIUM
        return ComplexityTier.COMPLEX

    def _confidence_for_score(self, score: int) -> float:
        boundaries = (self._policy.simple_max, self._policy.medium_max)
        nearest_distance = min(abs(score - boundary) for boundary in boundaries)
        confidence = 0.5 + min(nearest_distance / 3, 1.0) * 0.49
        return round(confidence, 2)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_complexity_classifier.py -v`
Expected: PASS (9 tests)

- [ ] **Batch 2 verification & commit**

```bash
uv run pytest -v
```
Expected: all tests pass (117 + 16 + 9 = 142).

```bash
git add backend/routing backend/classifier backend/config/routing.yaml backend/config/settings.py backend/tests/test_routing_config.py backend/tests/test_settings.py backend/tests/test_complexity_classifier.py
git commit -m "feat: add RoutingConfig/RoutingConfigLoader and HeuristicComplexityClassifier"
```

---

## Batch 3: Routing Policy, Context & Strategies

### Task 22: RoutingPolicy

**Files:**
- Create: `backend/routing/policy.py`
- Test: `backend/tests/test_routing_policy.py`

**Interfaces:**
- Consumes: `ComplexityTier` (Task 21), `EligibilityPolicy` (Task 20), `ModelSpec` (Phase 1).
- Produces: `RoutingPolicy(policies: dict[str, EligibilityPolicy])`, `filter_candidates(complexity: ComplexityTier, candidates: list[ModelSpec]) -> list[ModelSpec]`. Consumed by `RoutingEngine` (Task 26).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_routing_policy.py
from backend.classifier.complexity_classifier import ComplexityTier
from backend.routing.config import EligibilityPolicy
from backend.routing.policy import RoutingPolicy
from backend.services.model_registry import ModelSpec


def _model(id: str, benchmark_score: float) -> ModelSpec:
    return ModelSpec(
        id=id, provider="openai", model=id, input_cost=0.15, output_cost=0.60,
        context_window=128000, max_output_tokens=16384, supports_streaming=True,
        supports_tools=True, supports_json=True, supports_vision=False,
        benchmark_score=benchmark_score, average_latency_ms=450, available=True,
    )


def _policy() -> RoutingPolicy:
    return RoutingPolicy({
        "simple": EligibilityPolicy(min_benchmark_score=0.0),
        "medium": EligibilityPolicy(min_benchmark_score=0.75),
        "complex": EligibilityPolicy(min_benchmark_score=0.90),
    })


def test_simple_allows_all_models():
    candidates = [_model("a", 0.5), _model("b", 0.95)]
    result = _policy().filter_candidates(ComplexityTier.SIMPLE, candidates)
    assert {m.id for m in result} == {"a", "b"}


def test_complex_excludes_low_benchmark_models():
    candidates = [_model("a", 0.82), _model("b", 0.93)]
    result = _policy().filter_candidates(ComplexityTier.COMPLEX, candidates)
    assert {m.id for m in result} == {"b"}


def test_medium_boundary_is_inclusive():
    candidates = [_model("a", 0.75)]
    result = _policy().filter_candidates(ComplexityTier.MEDIUM, candidates)
    assert {m.id for m in result} == {"a"}


def test_no_eligible_candidates_returns_empty_list():
    candidates = [_model("a", 0.5)]
    result = _policy().filter_candidates(ComplexityTier.COMPLEX, candidates)
    assert result == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_routing_policy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.routing.policy'`

- [ ] **Step 3: Write the implementation**

```python
# backend/routing/policy.py
from backend.classifier.complexity_classifier import ComplexityTier
from backend.routing.config import EligibilityPolicy
from backend.services.model_registry import ModelSpec


class RoutingPolicy:
    def __init__(self, policies: dict[str, EligibilityPolicy]) -> None:
        self._policies = policies

    def filter_candidates(
        self, complexity: ComplexityTier, candidates: list[ModelSpec]
    ) -> list[ModelSpec]:
        policy = self._policies[complexity.value]
        return [c for c in candidates if c.benchmark_score >= policy.min_benchmark_score]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_routing_policy.py -v`
Expected: PASS (4 tests)

### Task 23: RoutingContext

**Files:**
- Create: `backend/routing/context.py`
- Test: `backend/tests/test_routing_context.py`

**Interfaces:**
- Consumes: `PromptFeatures` (Task 19), `ComplexityTier` (Task 21), `ModelSpec` (Phase 1).
- Produces: `RoutingContext(prompt, features, complexity, candidates)`. Consumed by `BaseRoutingStrategy` implementations (Task 24), `ExplanationGenerator` (Task 25), `RoutingEngine` (Task 26).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_routing_context.py
from backend.analysis.prompt_analyzer import PromptAnalyzer
from backend.classifier.complexity_classifier import ComplexityTier
from backend.routing.context import RoutingContext
from backend.services.model_registry import ModelSpec


def test_routing_context_holds_all_fields():
    features = PromptAnalyzer().analyze("Explain why the sky is blue.")
    model = ModelSpec(
        id="gpt-4o-mini", provider="openai", model="gpt-4o-mini", input_cost=0.15,
        output_cost=0.60, context_window=128000, max_output_tokens=16384,
        supports_streaming=True, supports_tools=True, supports_json=True,
        supports_vision=False, benchmark_score=0.82, average_latency_ms=450, available=True,
    )

    context = RoutingContext(
        prompt="Explain why the sky is blue.",
        features=features,
        complexity=ComplexityTier.SIMPLE,
        candidates=[model],
    )

    assert context.prompt == "Explain why the sky is blue."
    assert context.features == features
    assert context.complexity == ComplexityTier.SIMPLE
    assert context.candidates == [model]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_routing_context.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.routing.context'`

- [ ] **Step 3: Write the implementation**

```python
# backend/routing/context.py
from pydantic import BaseModel

from backend.analysis.prompt_analyzer import PromptFeatures
from backend.classifier.complexity_classifier import ComplexityTier
from backend.services.model_registry import ModelSpec


class RoutingContext(BaseModel):
    prompt: str
    features: PromptFeatures
    complexity: ComplexityTier
    candidates: list[ModelSpec]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_routing_context.py -v`
Expected: PASS (1 test)

### Task 24: Routing Strategies

**Files:**
- Create: `backend/routing/strategies.py`
- Test: `backend/tests/test_routing_strategies.py`

**Interfaces:**
- Consumes: `RoutingContext` (Task 23), `BalancedStrategyWeights` (Task 20), `ModelSpec` (Phase 1).
- Produces: `BaseRoutingStrategy` ABC (`select_model(context) -> ModelSpec`), `CostOptimizedStrategy`, `LatencyOptimizedStrategy`, `QualityOptimizedStrategy`, `BalancedStrategy(weights)`. Consumed by `RoutingEngine` (Task 26), `main.py` (Task 29).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_routing_strategies.py
import pytest

from backend.analysis.prompt_analyzer import PromptAnalyzer
from backend.classifier.complexity_classifier import ComplexityTier
from backend.routing.config import BalancedStrategyWeights
from backend.routing.context import RoutingContext
from backend.routing.strategies import (
    BalancedStrategy,
    BaseRoutingStrategy,
    CostOptimizedStrategy,
    LatencyOptimizedStrategy,
    QualityOptimizedStrategy,
)
from backend.services.model_registry import ModelSpec


def _model(id, input_cost, output_cost, latency, benchmark) -> ModelSpec:
    return ModelSpec(
        id=id, provider="openai", model=id, input_cost=input_cost, output_cost=output_cost,
        context_window=128000, max_output_tokens=16384, supports_streaming=True,
        supports_tools=True, supports_json=True, supports_vision=False,
        benchmark_score=benchmark, average_latency_ms=latency, available=True,
    )


def _context(candidates) -> RoutingContext:
    features = PromptAnalyzer().analyze("test prompt")
    return RoutingContext(
        prompt="test prompt", features=features, complexity=ComplexityTier.SIMPLE,
        candidates=candidates,
    )


def test_base_routing_strategy_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseRoutingStrategy()


def test_cost_optimized_picks_cheapest():
    cheap = _model("cheap", 0.1, 0.1, 500, 0.8)
    expensive = _model("expensive", 5.0, 5.0, 100, 0.99)
    selected = CostOptimizedStrategy().select_model(_context([cheap, expensive]))
    assert selected.id == "cheap"


def test_latency_optimized_picks_fastest():
    fast = _model("fast", 1.0, 1.0, 100, 0.8)
    slow = _model("slow", 0.1, 0.1, 900, 0.99)
    selected = LatencyOptimizedStrategy().select_model(_context([fast, slow]))
    assert selected.id == "fast"


def test_quality_optimized_picks_highest_benchmark():
    low = _model("low", 0.1, 0.1, 100, 0.7)
    high = _model("high", 5.0, 5.0, 900, 0.99)
    selected = QualityOptimizedStrategy().select_model(_context([low, high]))
    assert selected.id == "high"


def test_balanced_strategy_with_single_candidate_returns_it():
    only = _model("only", 1.0, 1.0, 500, 0.85)
    weights = BalancedStrategyWeights(cost_weight=1 / 3, latency_weight=1 / 3, quality_weight=1 / 3)
    selected = BalancedStrategy(weights).select_model(_context([only]))
    assert selected.id == "only"


def test_balanced_strategy_picks_best_combined_score():
    balanced = _model("balanced", 0.15, 0.60, 450, 0.82)
    premium = _model("premium", 2.50, 10.00, 900, 0.93)
    weights = BalancedStrategyWeights(cost_weight=1 / 3, latency_weight=1 / 3, quality_weight=1 / 3)
    selected = BalancedStrategy(weights).select_model(_context([balanced, premium]))
    assert selected.id == "balanced"


def test_balanced_strategy_respects_quality_weight_override():
    balanced = _model("balanced", 0.15, 0.60, 450, 0.82)
    premium = _model("premium", 2.50, 10.00, 900, 0.93)
    weights = BalancedStrategyWeights(cost_weight=0.05, latency_weight=0.05, quality_weight=0.90)
    selected = BalancedStrategy(weights).select_model(_context([balanced, premium]))
    assert selected.id == "premium"


def test_balanced_strategy_handles_tied_metric_without_division_by_zero():
    tied_a = _model("a", 1.0, 1.0, 500, 0.80)
    tied_b = _model("b", 1.0, 1.0, 500, 0.90)
    weights = BalancedStrategyWeights(cost_weight=1 / 3, latency_weight=1 / 3, quality_weight=1 / 3)
    selected = BalancedStrategy(weights).select_model(_context([tied_a, tied_b]))
    assert selected.id == "b"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_routing_strategies.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.routing.strategies'`

- [ ] **Step 3: Write the implementation**

```python
# backend/routing/strategies.py
from abc import ABC, abstractmethod

from backend.routing.config import BalancedStrategyWeights
from backend.routing.context import RoutingContext
from backend.services.model_registry import ModelSpec


class BaseRoutingStrategy(ABC):
    @abstractmethod
    def select_model(self, context: RoutingContext) -> ModelSpec: ...


class CostOptimizedStrategy(BaseRoutingStrategy):
    def select_model(self, context: RoutingContext) -> ModelSpec:
        return min(context.candidates, key=lambda c: c.input_cost + c.output_cost)


class LatencyOptimizedStrategy(BaseRoutingStrategy):
    def select_model(self, context: RoutingContext) -> ModelSpec:
        return min(context.candidates, key=lambda c: c.average_latency_ms)


class QualityOptimizedStrategy(BaseRoutingStrategy):
    def select_model(self, context: RoutingContext) -> ModelSpec:
        return max(context.candidates, key=lambda c: c.benchmark_score)


def _normalize(values: list[float], invert: bool) -> list[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5] * len(values)
    if invert:
        return [(hi - v) / (hi - lo) for v in values]
    return [(v - lo) / (hi - lo) for v in values]


class BalancedStrategy(BaseRoutingStrategy):
    def __init__(self, weights: BalancedStrategyWeights) -> None:
        self._weights = weights

    def select_model(self, context: RoutingContext) -> ModelSpec:
        candidates = context.candidates
        if len(candidates) == 1:
            return candidates[0]

        costs = [c.input_cost + c.output_cost for c in candidates]
        latencies = [c.average_latency_ms for c in candidates]
        qualities = [c.benchmark_score for c in candidates]

        cost_scores = _normalize(costs, invert=True)
        latency_scores = _normalize(latencies, invert=True)
        quality_scores = _normalize(qualities, invert=False)

        combined = [
            cost * self._weights.cost_weight
            + latency * self._weights.latency_weight
            + quality * self._weights.quality_weight
            for cost, latency, quality in zip(cost_scores, latency_scores, quality_scores)
        ]
        best_index = combined.index(max(combined))
        return candidates[best_index]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_routing_strategies.py -v`
Expected: PASS (8 tests)

- [ ] **Batch 3 verification & commit**

```bash
uv run pytest -v
```
Expected: all tests pass (142 + 4 + 1 + 8 = 155).

```bash
git add backend/routing/policy.py backend/routing/context.py backend/routing/strategies.py backend/tests/test_routing_policy.py backend/tests/test_routing_context.py backend/tests/test_routing_strategies.py
git commit -m "feat: add RoutingPolicy, RoutingContext, and routing strategies"
```

---

## Batch 4: Routing Engine

### Task 25: ExplanationGenerator

**Files:**
- Create: `backend/routing/explanation.py`
- Test: `backend/tests/test_explanation_generator.py`

**Interfaces:**
- Consumes: `RoutingContext` (Task 23), `ClassificationResult` (Task 21), `ModelSpec` (Phase 1).
- Produces: `ExplanationGenerator.generate(context, selected, strategy_name, classification) -> list[str]`. Consumed by `RoutingEngine` (Task 26).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_explanation_generator.py
from backend.analysis.prompt_analyzer import PromptAnalyzer
from backend.classifier.complexity_classifier import ClassificationResult, ComplexityTier
from backend.routing.context import RoutingContext
from backend.routing.explanation import ExplanationGenerator
from backend.services.model_registry import ModelSpec


def _model() -> ModelSpec:
    return ModelSpec(
        id="gpt-4o-mini", provider="openai", model="gpt-4o-mini", input_cost=0.15,
        output_cost=0.60, context_window=128000, max_output_tokens=16384,
        supports_streaming=True, supports_tools=True, supports_json=True,
        supports_vision=False, benchmark_score=0.82, average_latency_ms=450, available=True,
    )


def _context() -> RoutingContext:
    features = PromptAnalyzer().analyze("Explain why the sky is blue.")
    return RoutingContext(
        prompt="Explain why the sky is blue.", features=features,
        complexity=ComplexityTier.SIMPLE, candidates=[_model()],
    )


def test_generate_includes_signals_when_present():
    classification = ClassificationResult(
        tier=ComplexityTier.MEDIUM, score=2, confidence=0.66,
        signals=["reasoning keywords detected", "code content detected"],
    )
    reasoning = ExplanationGenerator().generate(_context(), _model(), "balanced", classification)

    assert "reasoning keywords detected" in reasoning[0]
    assert "code content detected" in reasoning[0]
    assert "medium" in reasoning[0]
    assert "0.66" in reasoning[0]
    assert "balanced" in reasoning[1]
    assert "1 eligible model" in reasoning[1]
    assert "gpt-4o-mini" in reasoning[2]


def test_generate_handles_no_signals():
    classification = ClassificationResult(
        tier=ComplexityTier.SIMPLE, score=0, confidence=0.66, signals=[]
    )
    reasoning = ExplanationGenerator().generate(_context(), _model(), "cost", classification)

    assert "no complexity signals detected" in reasoning[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_explanation_generator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.routing.explanation'`

- [ ] **Step 3: Write the implementation**

```python
# backend/routing/explanation.py
from backend.classifier.complexity_classifier import ClassificationResult
from backend.routing.context import RoutingContext
from backend.services.model_registry import ModelSpec


class ExplanationGenerator:
    def generate(
        self,
        context: RoutingContext,
        selected: ModelSpec,
        strategy_name: str,
        classification: ClassificationResult,
    ) -> list[str]:
        if classification.signals:
            signal_text = ", ".join(classification.signals)
            classification_line = (
                f"Classified as {classification.tier.value} "
                f"(confidence {classification.confidence}): {signal_text}."
            )
        else:
            classification_line = (
                f"Classified as {classification.tier.value} "
                f"(confidence {classification.confidence}): no complexity signals detected."
            )

        return [
            classification_line,
            f"Strategy '{strategy_name}' evaluated {len(context.candidates)} eligible model(s).",
            f"Selected '{selected.id}'.",
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_explanation_generator.py -v`
Expected: PASS (2 tests)

### Task 26: RoutingEngine

**Files:**
- Create: `backend/routing/engine.py`
- Test: `backend/tests/test_routing_engine.py`

**Interfaces:**
- Consumes: `ModelRegistry` (Phase 1), `PromptAnalyzer` (Task 19), `BaseComplexityClassifier` (Task 21), `RoutingPolicy` (Task 22), `RoutingContext` (Task 23), `BaseRoutingStrategy` (Task 24), `ExplanationGenerator` (Task 25).
- Produces: `RoutingDecision(selected_model, strategy, complexity, confidence, estimated_cost, estimated_latency_ms, reasoning)`, `NoEligibleModelError`, `RoutingEngine.route(prompt, strategy_name="balanced") -> RoutingDecision`. Consumed by `ChatService` (Task 28), `main.py` (Task 29).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_routing_engine.py
import textwrap

import pytest

from backend.analysis.prompt_analyzer import PromptAnalyzer
from backend.classifier.complexity_classifier import HeuristicComplexityClassifier
from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.events.bus import EventBus
from backend.providers.factory import ProviderFactory
from backend.providers.manager import ProviderManager
from backend.providers.mock_provider import MockProvider
from backend.providers.openai_provider import OpenAIProvider
from backend.routing.config import BalancedStrategyWeights, ClassifierPolicy, EligibilityPolicy
from backend.routing.engine import NoEligibleModelError, RoutingEngine
from backend.routing.explanation import ExplanationGenerator
from backend.routing.policy import RoutingPolicy
from backend.routing.strategies import BalancedStrategy, CostOptimizedStrategy
from backend.services.cost_estimator import DefaultCostEstimator
from backend.services.model_registry import ModelRegistry

TWO_MODEL_YAML = textwrap.dedent("""
    models:
      - id: gpt-4o-mini
        provider: openai
        model: gpt-4o-mini
        pricing:
          input_cost: 0.15
          output_cost: 0.60
        limits:
          context_window: 128000
          max_output_tokens: 16384
        capabilities:
          supports_streaming: true
          supports_tools: true
          supports_json: true
          supports_vision: false
        metadata:
          benchmark_score: 0.82
          average_latency_ms: 450
      - id: gpt-4o
        provider: openai
        model: gpt-4o
        pricing:
          input_cost: 2.50
          output_cost: 10.00
        limits:
          context_window: 128000
          max_output_tokens: 16384
        capabilities:
          supports_streaming: true
          supports_tools: true
          supports_json: true
          supports_vision: true
        metadata:
          benchmark_score: 0.93
          average_latency_ms: 900
""")


def _routing_policy() -> RoutingPolicy:
    return RoutingPolicy({
        "simple": EligibilityPolicy(min_benchmark_score=0.0),
        "medium": EligibilityPolicy(min_benchmark_score=0.75),
        "complex": EligibilityPolicy(min_benchmark_score=0.90),
    })


def _make_engine(tmp_path, openai_key="sk-test", strategies=None):
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(TWO_MODEL_YAML)

    settings = Settings(
        _env_file=None, database_url=f"sqlite:///{tmp_path}/test.db", openai_api_key=openai_key
    )
    engine_db = create_engine_from_settings(settings)
    init_db(engine_db)
    session_factory = create_session_factory(engine_db)

    factory = ProviderFactory()
    factory.register("mock", MockProvider)
    factory.register("openai", OpenAIProvider)
    provider_manager = ProviderManager(factory, settings)

    model_registry = ModelRegistry(
        provider_manager=provider_manager,
        event_bus=EventBus(),
        cost_estimator=DefaultCostEstimator(),
        session_factory=session_factory,
        yaml_path=str(yaml_path),
    )
    model_registry.reload()

    return RoutingEngine(
        model_registry=model_registry,
        analyzer=PromptAnalyzer(),
        classifier=HeuristicComplexityClassifier(ClassifierPolicy(simple_max=1, medium_max=3)),
        routing_policy=_routing_policy(),
        strategies=strategies
        or {
            "balanced": BalancedStrategy(BalancedStrategyWeights()),
            "cost": CostOptimizedStrategy(),
        },
        explanation_generator=ExplanationGenerator(),
    )


def test_route_returns_decision_for_simple_prompt(tmp_path):
    engine = _make_engine(tmp_path)
    decision = engine.route("List three fruits.", strategy_name="cost")

    assert decision.complexity.value == "simple"
    assert decision.strategy == "cost"
    assert decision.selected_model in {"gpt-4o-mini", "gpt-4o"}
    assert decision.estimated_cost > 0
    assert decision.estimated_latency_ms > 0
    assert len(decision.reasoning) == 3


def test_route_complex_prompt_only_selects_high_benchmark_model(tmp_path):
    engine = _make_engine(tmp_path)
    complex_prompt = (
        "Analyze and compare these two algorithms, explain the reasoning step by step, "
        "calculate their time complexity, and format the answer as bullet points. "
        "You must include examples and should ensure correctness."
    )
    decision = engine.route(complex_prompt, strategy_name="cost")

    assert decision.complexity.value == "complex"
    assert decision.selected_model == "gpt-4o"


def test_route_raises_when_no_provider_available(tmp_path):
    engine = _make_engine(tmp_path, openai_key=None, strategies={"cost": CostOptimizedStrategy()})

    with pytest.raises(NoEligibleModelError):
        engine.route("Hello.", strategy_name="cost")


def test_route_raises_key_error_for_unknown_strategy(tmp_path):
    engine = _make_engine(tmp_path)
    with pytest.raises(KeyError):
        engine.route("Hello.", strategy_name="does-not-exist")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_routing_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.routing.engine'`

- [ ] **Step 3: Write the implementation**

```python
# backend/routing/engine.py
from pydantic import BaseModel

from backend.analysis.prompt_analyzer import PromptAnalyzer
from backend.classifier.complexity_classifier import BaseComplexityClassifier, ComplexityTier
from backend.routing.context import RoutingContext
from backend.routing.explanation import ExplanationGenerator
from backend.routing.policy import RoutingPolicy
from backend.routing.strategies import BaseRoutingStrategy
from backend.services.model_registry import ModelRegistry


class RoutingDecision(BaseModel):
    selected_model: str
    strategy: str
    complexity: ComplexityTier
    confidence: float
    estimated_cost: float
    estimated_latency_ms: float
    reasoning: list[str]


class NoEligibleModelError(Exception):
    pass


class RoutingEngine:
    def __init__(
        self,
        model_registry: ModelRegistry,
        analyzer: PromptAnalyzer,
        classifier: BaseComplexityClassifier,
        routing_policy: RoutingPolicy,
        strategies: dict[str, BaseRoutingStrategy],
        explanation_generator: ExplanationGenerator,
    ) -> None:
        self._model_registry = model_registry
        self._analyzer = analyzer
        self._classifier = classifier
        self._routing_policy = routing_policy
        self._strategies = strategies
        self._explanation_generator = explanation_generator

    def route(self, prompt: str, strategy_name: str = "balanced") -> RoutingDecision:
        features = self._analyzer.analyze(prompt)
        classification = self._classifier.classify(features)

        available = self._model_registry.get_available_models()
        candidates = self._routing_policy.filter_candidates(classification.tier, available)
        if not candidates:
            raise NoEligibleModelError(
                f"No available model meets the '{classification.tier.value}' complexity policy"
            )

        context = RoutingContext(
            prompt=prompt,
            features=features,
            complexity=classification.tier,
            candidates=candidates,
        )
        selected = self._strategies[strategy_name].select_model(context)

        estimated_cost = self._model_registry.estimate_cost(
            selected.id, features.estimated_tokens, features.estimated_output_tokens
        )
        reasoning = self._explanation_generator.generate(
            context, selected, strategy_name, classification
        )

        return RoutingDecision(
            selected_model=selected.id,
            strategy=strategy_name,
            complexity=classification.tier,
            confidence=classification.confidence,
            estimated_cost=estimated_cost,
            estimated_latency_ms=selected.average_latency_ms,
            reasoning=reasoning,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_routing_engine.py -v`
Expected: PASS (4 tests)

- [ ] **Batch 4 verification & commit**

```bash
uv run pytest -v
```
Expected: all tests pass (155 + 2 + 4 = 161).

```bash
git add backend/routing/explanation.py backend/routing/engine.py backend/tests/test_explanation_generator.py backend/tests/test_routing_engine.py
git commit -m "feat: add ExplanationGenerator and RoutingEngine"
```

---

## Batch 5: Chat Service & API

### Task 27: Database additions

**Files:**
- Modify: `backend/database/models.py`
- Test: `backend/tests/test_chat_database.py`

**Interfaces:**
- Produces: `RequestRow` (table `requests`), `ResponseRow` (table `responses`), `RoutingEventRow` (table `routing_events`). Consumed by `ChatService` (Task 28).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_chat_database.py
from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.database.models import RequestRow, ResponseRow, RoutingEventRow


def test_create_and_query_request_response_routing_event(tmp_path):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    with session_factory() as session:
        session.add(RequestRow(request_id="req-1", prompt="Hello", strategy="balanced"))
        session.commit()

    with session_factory() as session:
        session.add(ResponseRow(
            request_id="req-1", response_text="Hi there", actual_input_tokens=5,
            actual_output_tokens=3, actual_cost=0.001,
        ))
        session.add(RoutingEventRow(
            request_id="req-1", complexity="simple", confidence=0.66,
            selected_model="gpt-4o-mini", selected_strategy="balanced",
            estimated_cost=0.001, estimated_latency_ms=450.0, reasoning="[]",
        ))
        session.commit()

    with session_factory() as session:
        request_row = session.query(RequestRow).filter_by(request_id="req-1").one()
        response_row = session.query(ResponseRow).filter_by(request_id="req-1").one()
        routing_event_row = session.query(RoutingEventRow).filter_by(request_id="req-1").one()

    assert request_row.prompt == "Hello"
    assert response_row.response_text == "Hi there"
    assert response_row.error is None
    assert routing_event_row.selected_strategy == "balanced"


def test_response_row_persists_error_without_response_text(tmp_path):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    with session_factory() as session:
        session.add(RequestRow(request_id="req-2", prompt="Hello", strategy="cost"))
        session.commit()

    with session_factory() as session:
        session.add(ResponseRow(request_id="req-2", error="ProviderError: boom"))
        session.commit()

    with session_factory() as session:
        response_row = session.query(ResponseRow).filter_by(request_id="req-2").one()

    assert response_row.response_text is None
    assert response_row.error == "ProviderError: boom"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_chat_database.py -v`
Expected: FAIL — `ImportError: cannot import name 'RequestRow' from 'backend.database.models'`

- [ ] **Step 3: Modify `backend/database/models.py`**

Change:
```python
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.database.base import Base
```

To:
```python
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.database.base import Base
```

Add at the end of the file (after `ModelRow`):

```python


class RequestRow(Base):
    __tablename__ = "requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    prompt: Mapped[str] = mapped_column(String, nullable=False)
    strategy: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ResponseRow(Base):
    __tablename__ = "responses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String, ForeignKey("requests.request_id"), nullable=False)
    response_text: Mapped[str | None] = mapped_column(String, nullable=True)
    actual_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    actual_output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    actual_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class RoutingEventRow(Base):
    __tablename__ = "routing_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String, ForeignKey("requests.request_id"), nullable=False)
    complexity: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    selected_model: Mapped[str] = mapped_column(String, nullable=False)
    selected_strategy: Mapped[str] = mapped_column(String, nullable=False)
    estimated_cost: Mapped[float] = mapped_column(Float, nullable=False)
    estimated_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    reasoning: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_chat_database.py -v`
Expected: PASS (2 tests)

### Task 28: ChatService

**Files:**
- Create: `backend/chat/__init__.py` (empty)
- Create: `backend/chat/service.py`
- Test: `backend/tests/test_chat_service.py`

**Interfaces:**
- Consumes: `RoutingEngine` (Task 26), `ProviderManager`/`ModelRegistry` (Phase 1), `ProviderError` (Phase 1), `RequestRow`/`ResponseRow`/`RoutingEventRow` (Task 27).
- Produces: `ChatResult(request_id, response, routing: RoutingDecision)`, `ChatService.chat(prompt, strategy="balanced") -> ChatResult` (async). Consumed by `POST /v1/chat` (Task 29).

- [ ] **Step 1: Create the package directory**

Run:
```bash
mkdir -p backend/chat
touch backend/chat/__init__.py
```

- [ ] **Step 2: Write the failing test**

```python
# backend/tests/test_chat_service.py
import json
import textwrap
from unittest.mock import AsyncMock

import pytest

from backend.analysis.prompt_analyzer import PromptAnalyzer
from backend.chat.service import ChatService
from backend.classifier.complexity_classifier import HeuristicComplexityClassifier
from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.database.models import RequestRow, ResponseRow, RoutingEventRow
from backend.events.bus import EventBus
from backend.providers.base import ProviderError
from backend.providers.factory import ProviderFactory
from backend.providers.manager import ProviderManager
from backend.providers.mock_provider import MockProvider
from backend.routing.config import BalancedStrategyWeights, ClassifierPolicy, EligibilityPolicy
from backend.routing.engine import RoutingEngine
from backend.routing.explanation import ExplanationGenerator
from backend.routing.policy import RoutingPolicy
from backend.routing.strategies import BalancedStrategy
from backend.services.cost_estimator import DefaultCostEstimator
from backend.services.model_registry import ModelRegistry

ONE_MODEL_YAML = textwrap.dedent("""
    models:
      - id: mock-model
        provider: mock
        model: mock-model
        pricing:
          input_cost: 0.15
          output_cost: 0.60
        limits:
          context_window: 128000
          max_output_tokens: 16384
        capabilities:
          supports_streaming: true
          supports_tools: true
          supports_json: true
          supports_vision: false
        metadata:
          benchmark_score: 0.82
          average_latency_ms: 450
""")


def _make_chat_service(tmp_path):
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(ONE_MODEL_YAML)

    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine_db = create_engine_from_settings(settings)
    init_db(engine_db)
    session_factory = create_session_factory(engine_db)

    factory = ProviderFactory()
    factory.register("mock", MockProvider)
    provider_manager = ProviderManager(factory, settings)

    model_registry = ModelRegistry(
        provider_manager=provider_manager,
        event_bus=EventBus(),
        cost_estimator=DefaultCostEstimator(),
        session_factory=session_factory,
        yaml_path=str(yaml_path),
    )
    model_registry.reload()

    routing_engine = RoutingEngine(
        model_registry=model_registry,
        analyzer=PromptAnalyzer(),
        classifier=HeuristicComplexityClassifier(ClassifierPolicy(simple_max=1, medium_max=3)),
        routing_policy=RoutingPolicy({
            "simple": EligibilityPolicy(min_benchmark_score=0.0),
            "medium": EligibilityPolicy(min_benchmark_score=0.75),
            "complex": EligibilityPolicy(min_benchmark_score=0.90),
        }),
        strategies={"balanced": BalancedStrategy(BalancedStrategyWeights())},
        explanation_generator=ExplanationGenerator(),
    )

    chat_service = ChatService(
        routing_engine=routing_engine,
        provider_manager=provider_manager,
        model_registry=model_registry,
        session_factory=session_factory,
    )
    return chat_service, session_factory


async def test_chat_returns_result_and_persists_rows(tmp_path):
    chat_service, session_factory = _make_chat_service(tmp_path)

    result = await chat_service.chat("List three fruits.", strategy="balanced")

    assert result.response
    assert result.routing.selected_model == "mock-model"

    with session_factory() as session:
        request_row = session.query(RequestRow).filter_by(request_id=result.request_id).one()
        response_row = session.query(ResponseRow).filter_by(request_id=result.request_id).one()
        routing_event_row = (
            session.query(RoutingEventRow).filter_by(request_id=result.request_id).one()
        )

    assert request_row.prompt == "List three fruits."
    assert response_row.response_text == result.response
    assert response_row.error is None
    assert routing_event_row.selected_model == "mock-model"
    assert json.loads(routing_event_row.reasoning) == result.routing.reasoning


async def test_chat_persists_error_and_reraises_on_provider_failure(tmp_path, mocker):
    chat_service, session_factory = _make_chat_service(tmp_path)

    mocker.patch(
        "backend.providers.mock_provider.MockProvider.generate",
        new_callable=AsyncMock,
        side_effect=ProviderError("simulated failure"),
    )

    with pytest.raises(ProviderError):
        await chat_service.chat("List three fruits.", strategy="balanced")

    with session_factory() as session:
        response_row = session.query(ResponseRow).one()

    assert response_row.response_text is None
    assert response_row.error == "simulated failure"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_chat_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.chat.service'`

- [ ] **Step 4: Write the implementation**

```python
# backend/chat/service.py
import json
import uuid

from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker

from backend.database.models import RequestRow, ResponseRow, RoutingEventRow
from backend.providers.base import ProviderError
from backend.providers.manager import ProviderManager
from backend.routing.engine import RoutingDecision, RoutingEngine
from backend.services.model_registry import ModelRegistry


class ChatResult(BaseModel):
    request_id: str
    response: str
    routing: RoutingDecision


class ChatService:
    def __init__(
        self,
        routing_engine: RoutingEngine,
        provider_manager: ProviderManager,
        model_registry: ModelRegistry,
        session_factory: sessionmaker,
    ) -> None:
        self._routing_engine = routing_engine
        self._provider_manager = provider_manager
        self._model_registry = model_registry
        self._session_factory = session_factory

    async def chat(self, prompt: str, strategy: str = "balanced") -> ChatResult:
        request_id = str(uuid.uuid4())
        decision = self._routing_engine.route(prompt, strategy_name=strategy)

        with self._session_factory() as session:
            session.add(RequestRow(request_id=request_id, prompt=prompt, strategy=strategy))
            session.add(RoutingEventRow(
                request_id=request_id,
                complexity=decision.complexity.value,
                confidence=decision.confidence,
                selected_model=decision.selected_model,
                selected_strategy=decision.strategy,
                estimated_cost=decision.estimated_cost,
                estimated_latency_ms=decision.estimated_latency_ms,
                reasoning=json.dumps(decision.reasoning),
            ))
            session.commit()

        model_spec = self._model_registry.get_model(decision.selected_model)
        provider = self._provider_manager.get_provider(model_spec.provider)

        try:
            response_text = await provider.generate(prompt, model=model_spec.model)
        except ProviderError as exc:
            with self._session_factory() as session:
                session.add(ResponseRow(request_id=request_id, error=str(exc)))
                session.commit()
            raise

        input_tokens = provider.count_tokens(prompt)
        output_tokens = provider.count_tokens(response_text)
        actual_cost = self._model_registry.estimate_cost(model_spec.id, input_tokens, output_tokens)

        with self._session_factory() as session:
            session.add(ResponseRow(
                request_id=request_id,
                response_text=response_text,
                actual_input_tokens=input_tokens,
                actual_output_tokens=output_tokens,
                actual_cost=actual_cost,
            ))
            session.commit()

        return ChatResult(request_id=request_id, response=response_text, routing=decision)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_chat_service.py -v`
Expected: PASS (2 tests)

### Task 29: POST /v1/chat & wiring

**Files:**
- Create: `backend/api/routers/chat.py`
- Modify: `backend/api/dependencies.py`
- Modify: `backend/api/main.py`
- Test: `backend/tests/test_chat_endpoint.py`

**Interfaces:**
- Consumes: `ChatService`/`ChatResult` (Task 28), `NoEligibleModelError` (Task 26), `ProviderError` (Phase 1).
- Produces: `GET/POST /v1/chat` route, `ChatServiceDep`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_chat_endpoint.py
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.dependencies import get_chat_service
from backend.api.routers.chat import router as chat_router
from backend.chat.service import ChatResult
from backend.classifier.complexity_classifier import ComplexityTier
from backend.providers.base import ProviderError
from backend.routing.engine import NoEligibleModelError, RoutingDecision


class _FakeChatService:
    def __init__(self, result=None, exception=None):
        self._result = result
        self._exception = exception

    async def chat(self, prompt, strategy="balanced"):
        if self._exception:
            raise self._exception
        return self._result


def _sample_result() -> ChatResult:
    return ChatResult(
        request_id="req-1",
        response="Here are three fruits: apple, banana, cherry.",
        routing=RoutingDecision(
            selected_model="mock-model", strategy="balanced", complexity=ComplexityTier.SIMPLE,
            confidence=0.66, estimated_cost=0.001, estimated_latency_ms=450.0,
            reasoning=[
                "Classified as simple.",
                "Strategy 'balanced' evaluated 1 eligible model(s).",
                "Selected 'mock-model'.",
            ],
        ),
    )


def test_chat_endpoint_returns_result():
    app = FastAPI()
    app.include_router(chat_router, prefix="/v1")
    app.dependency_overrides[get_chat_service] = lambda: _FakeChatService(result=_sample_result())

    client = TestClient(app)
    response = client.post("/v1/chat", json={"prompt": "List three fruits.", "strategy": "balanced"})

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "req-1"
    assert body["routing"]["selected_model"] == "mock-model"


def test_chat_endpoint_defaults_strategy_to_balanced():
    app = FastAPI()
    app.include_router(chat_router, prefix="/v1")
    app.dependency_overrides[get_chat_service] = lambda: _FakeChatService(result=_sample_result())

    client = TestClient(app)
    response = client.post("/v1/chat", json={"prompt": "List three fruits."})

    assert response.status_code == 200


def test_chat_endpoint_returns_503_for_no_eligible_model():
    app = FastAPI()
    app.include_router(chat_router, prefix="/v1")
    app.dependency_overrides[get_chat_service] = lambda: _FakeChatService(
        exception=NoEligibleModelError("no models available")
    )

    client = TestClient(app)
    response = client.post("/v1/chat", json={"prompt": "Hello."})

    assert response.status_code == 503


def test_chat_endpoint_returns_502_for_provider_error():
    app = FastAPI()
    app.include_router(chat_router, prefix="/v1")
    app.dependency_overrides[get_chat_service] = lambda: _FakeChatService(
        exception=ProviderError("upstream failure")
    )

    client = TestClient(app)
    response = client.post("/v1/chat", json={"prompt": "Hello."})

    assert response.status_code == 502
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_chat_endpoint.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.api.routers.chat'`

- [ ] **Step 3: Modify `backend/api/dependencies.py`**

Change:
```python
from typing import Annotated

from fastapi import Depends, Request

from backend.config.settings import Settings
from backend.events.bus import EventBus
from backend.providers.manager import ProviderManager
from backend.services.model_registry import ModelRegistry
```

To:
```python
from typing import Annotated

from fastapi import Depends, Request

from backend.chat.service import ChatService
from backend.config.settings import Settings
from backend.events.bus import EventBus
from backend.providers.manager import ProviderManager
from backend.services.model_registry import ModelRegistry
```

Change:
```python
def get_app_start_time(request: Request) -> float:
    return request.app.state.start_time


SettingsDep = Annotated[Settings, Depends(get_settings)]
```

To:
```python
def get_app_start_time(request: Request) -> float:
    return request.app.state.start_time


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


SettingsDep = Annotated[Settings, Depends(get_settings)]
```

Change:
```python
AppVersionDep = Annotated[str, Depends(get_app_version)]
AppStartTimeDep = Annotated[float, Depends(get_app_start_time)]
```

To:
```python
AppVersionDep = Annotated[str, Depends(get_app_version)]
AppStartTimeDep = Annotated[float, Depends(get_app_start_time)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
```

- [ ] **Step 4: Write `backend/api/routers/chat.py`**

```python
# backend/api/routers/chat.py
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.api.dependencies import ChatServiceDep
from backend.chat.service import ChatResult
from backend.providers.base import ProviderError
from backend.routing.engine import NoEligibleModelError

router = APIRouter()


class ChatRequest(BaseModel):
    prompt: str
    strategy: Literal["cost", "latency", "quality", "balanced"] = "balanced"


@router.post("/chat", response_model=ChatResult)
async def chat(request: ChatRequest, chat_service: ChatServiceDep) -> ChatResult:
    try:
        return await chat_service.chat(request.prompt, strategy=request.strategy)
    except NoEligibleModelError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
```

- [ ] **Step 5: Modify `backend/api/main.py`**

Change:
```python
from backend.api.routers.health import router as health_router
from backend.api.routers.models import router as models_router
from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.events.bus import EventBus
from backend.events.subscribers import register_logging_subscriber
from backend.providers.factory import ProviderFactory
from backend.providers.manager import ProviderManager
from backend.providers.mock_provider import MockProvider
from backend.providers.openai_provider import OpenAIProvider
from backend.services.cost_estimator import DefaultCostEstimator
from backend.services.model_registry import ModelRegistry
from backend.telemetry.logging import configure_logging
```

To:
```python
from backend.analysis.prompt_analyzer import PromptAnalyzer
from backend.api.routers.chat import router as chat_router
from backend.api.routers.health import router as health_router
from backend.api.routers.models import router as models_router
from backend.chat.service import ChatService
from backend.classifier.complexity_classifier import HeuristicComplexityClassifier
from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.events.bus import EventBus
from backend.events.subscribers import register_logging_subscriber
from backend.providers.factory import ProviderFactory
from backend.providers.manager import ProviderManager
from backend.providers.mock_provider import MockProvider
from backend.providers.openai_provider import OpenAIProvider
from backend.routing.config_loader import RoutingConfigLoader
from backend.routing.engine import RoutingEngine
from backend.routing.explanation import ExplanationGenerator
from backend.routing.policy import RoutingPolicy
from backend.routing.strategies import (
    BalancedStrategy,
    CostOptimizedStrategy,
    LatencyOptimizedStrategy,
    QualityOptimizedStrategy,
)
from backend.services.cost_estimator import DefaultCostEstimator
from backend.services.model_registry import ModelRegistry
from backend.telemetry.logging import configure_logging
```

Change:
```python
    model_registry.reload()
    await model_registry.refresh_provider_status()

    app.state.settings = settings
    app.state.event_bus = event_bus
    app.state.provider_manager = provider_manager
    app.state.model_registry = model_registry
    app.state.session_factory = session_factory
    app.state.version = APP_VERSION
    app.state.start_time = time.time()

    yield
```

To:
```python
    model_registry.reload()
    await model_registry.refresh_provider_status()

    routing_config = RoutingConfigLoader.load(settings.routing_config_path)
    classifier = HeuristicComplexityClassifier(routing_config.classifier)
    routing_policy = RoutingPolicy(routing_config.policy)
    strategies = {
        "cost": CostOptimizedStrategy(),
        "latency": LatencyOptimizedStrategy(),
        "quality": QualityOptimizedStrategy(),
        "balanced": BalancedStrategy(routing_config.balanced_strategy),
    }
    routing_engine = RoutingEngine(
        model_registry=model_registry,
        analyzer=PromptAnalyzer(),
        classifier=classifier,
        routing_policy=routing_policy,
        strategies=strategies,
        explanation_generator=ExplanationGenerator(),
    )
    chat_service = ChatService(
        routing_engine=routing_engine,
        provider_manager=provider_manager,
        model_registry=model_registry,
        session_factory=session_factory,
    )

    app.state.settings = settings
    app.state.event_bus = event_bus
    app.state.provider_manager = provider_manager
    app.state.model_registry = model_registry
    app.state.session_factory = session_factory
    app.state.chat_service = chat_service
    app.state.version = APP_VERSION
    app.state.start_time = time.time()

    yield
```

Change:
```python
    app = FastAPI(title="LLM Cost Autopilot", version=APP_VERSION, lifespan=lifespan)
    app.include_router(health_router, prefix="/v1")
    app.include_router(models_router, prefix="/v1")
    return app
```

To:
```python
    app = FastAPI(title="LLM Cost Autopilot", version=APP_VERSION, lifespan=lifespan)
    app.include_router(health_router, prefix="/v1")
    app.include_router(models_router, prefix="/v1")
    app.include_router(chat_router, prefix="/v1")
    return app
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_chat_endpoint.py backend/tests/test_main.py -v`
Expected: PASS (4 + 1 = 5 tests)

- [ ] **Batch 5 verification & commit**

```bash
uv run pytest -v
```
Expected: all tests pass (161 + 2 + 2 + 4 = 169).

```bash
git add backend/database/models.py backend/chat backend/api/routers/chat.py backend/api/dependencies.py backend/api/main.py backend/tests/test_chat_database.py backend/tests/test_chat_service.py backend/tests/test_chat_endpoint.py
git commit -m "feat: add ChatService, database persistence, and POST /v1/chat"
```

---

## Batch 6: Integration, Documentation & Tag

### Task 30: Integration tests, docs, manual verification, tag v0.2.0

**Files:**
- Create: `backend/tests/test_integration_chat_flow.py`
- Modify: `README.md`
- Modify: `docs/ARCHITECTURE.md`

**Interfaces:**
- None — integration testing, documentation, and verification only.

- [ ] **Step 1: Write the integration test**

```python
# backend/tests/test_integration_chat_flow.py
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.providers.base import ProviderError


def test_full_chat_flow_end_to_end(monkeypatch, tmp_path, mocker):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/integration.db")

    mocker.patch(
        "backend.providers.openai_provider.OpenAIProvider.generate",
        new_callable=AsyncMock,
        return_value="Here are three fruits: apple, banana, cherry.",
    )

    app = create_app()
    with TestClient(app) as client:
        health_response = client.get("/v1/health")
        assert health_response.status_code == 200
        assert health_response.json()["providers"]["openai"] == "available"

        chat_response = client.post(
            "/v1/chat", json={"prompt": "List three fruits.", "strategy": "cost"}
        )

    assert chat_response.status_code == 200
    body = chat_response.json()
    assert body["response"] == "Here are three fruits: apple, banana, cherry."
    assert body["routing"]["selected_model"] == "gpt-4o-mini"
    assert body["routing"]["strategy"] == "cost"
    assert "request_id" in body


def test_full_chat_flow_returns_502_on_real_provider_error(monkeypatch, tmp_path, mocker):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/integration2.db")

    mocker.patch(
        "backend.providers.openai_provider.OpenAIProvider.generate",
        new_callable=AsyncMock,
        side_effect=ProviderError("simulated upstream failure"),
    )

    app = create_app()
    with TestClient(app) as client:
        response = client.post("/v1/chat", json={"prompt": "Hello.", "strategy": "cost"})

    assert response.status_code == 502
```

- [ ] **Step 2: Run the integration test**

Run: `uv run pytest backend/tests/test_integration_chat_flow.py -v`
Expected: PASS (2 tests) — this boots the real `lifespan` (real `ModelRegistry`, real `RoutingConfigLoader`, real `ProviderManager`) with only the OpenAI SDK boundary mocked.

- [ ] **Step 3: Update `README.md`**

Add a new bullet list item under "What exists today" (after the `GET /v1/models` line):

```markdown
- `POST /v1/chat` — routes a prompt through prompt analysis, heuristic
  complexity classification, and a configurable strategy (`cost`,
  `latency`, `quality`, `balanced`) to select a model, then returns the
  response plus a full routing explanation (complexity, confidence,
  estimated cost/latency, human-readable reasoning)
```

Add a new "Example" subsection after the existing `/v1/models` example, before the closing note:

```markdown
`POST /v1/chat`:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain why the sky is blue.", "strategy": "balanced"}'
```

```json
{
  "request_id": "b3f1...",
  "response": "...",
  "routing": {
    "selected_model": "gpt-4o-mini",
    "strategy": "balanced",
    "complexity": "simple",
    "confidence": 0.66,
    "estimated_cost": 0.00013,
    "estimated_latency_ms": 450.0,
    "reasoning": [
      "Classified as simple (confidence 0.66): reasoning keywords detected.",
      "Strategy 'balanced' evaluated 2 eligible model(s).",
      "Selected 'gpt-4o-mini'."
    ]
  }
}
```
```

- [ ] **Step 4: Update `docs/ARCHITECTURE.md`**

Replace the `## Routing` and `## Classification` placeholder sections (currently `_Not built yet — Phase 2._` and `_Not built yet — Phase 3._`) with:

```markdown
## Classification

`PromptAnalyzer.analyze()` extracts a deterministic `PromptFeatures` from
a prompt (regex/keyword-based, no ML) — including `estimated_output_tokens`,
which uses explicit brevity/word-count/long-form signals rather than
assuming output length mirrors input length. `HeuristicComplexityClassifier`
(behind `BaseComplexityClassifier`) turns those features into a
`ClassificationResult` (tier, score, confidence, human-readable `signals`)
via an additive, YAML-configurable threshold score — designed so a future
ML classifier is a drop-in replacement with zero call-site changes.

## Routing

`RoutingPolicy` filters `ModelRegistry`'s available models by a
per-complexity-tier minimum `benchmark_score` (YAML-configurable) — the
engine never hardcodes eligibility. Four strategies
(`CostOptimizedStrategy`, `LatencyOptimizedStrategy`,
`QualityOptimizedStrategy`, `BalancedStrategy`) implement
`BaseRoutingStrategy.select_model(context: RoutingContext) -> ModelSpec`;
`BalancedStrategy`'s weights are also YAML-configurable, defaulting to
equal thirds. `ExplanationGenerator` builds the human-readable reasoning
from the classifier's own `signals` rather than rediscovering them, kept
entirely separate from `RoutingEngine` so the orchestrator never
accumulates conditional string-building. `RoutingEngine` is pure
orchestration — it never calls a provider or touches the database.
`RoutingConfigLoader` is the single owner of `routing.yaml` file I/O;
`ClassifierPolicy`, `RoutingPolicy`, and `BalancedStrategy` all receive
already-parsed, already-validated configuration.

`ChatService` is the one component that calls both `RoutingEngine` and a
provider: it routes, persists the request + routing event immediately,
calls `provider.generate()`, and persists the response (or the error, on
`ProviderError`, before re-raising). `POST /v1/chat` is a thin HTTP layer
mapping `NoEligibleModelError` → 503 and `ProviderError` → 502.

## Verification

_Not built yet — Phase 3._

## Learning

_Not built yet._

## Dashboard

_Not built yet._
```

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest -v`
Expected: PASS (all tests across Phase 1 + Phase 2)

- [ ] **Step 6: Manual verification — run the server and hit `/v1/chat`**

```bash
uv run uvicorn backend.api.main:app --port 8000 &
sleep 2
curl -s http://127.0.0.1:8000/v1/health | python3 -m json.tool
curl -s -X POST http://127.0.0.1:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Compare Python and JavaScript for backend development, and explain your reasoning.", "strategy": "quality"}' \
  | python3 -m json.tool
kill %1
```

Expected: `/v1/health` returns `status: healthy`; `/v1/chat` returns either a `200` with a full `routing` block (if `OPENAI_API_KEY` is set in `.env`) or a `502`/error (if not, since no provider would be available for the `openai`-only `models.yaml` catalog) — either outcome confirms the full pipeline wired correctly end-to-end.

- [ ] **Step 7: Commit and tag**

```bash
git add backend/tests/test_integration_chat_flow.py README.md docs/ARCHITECTURE.md
git commit -m "test: add end-to-end chat integration tests, update docs for Phase 2"
git tag -a v0.2.0 -m "Phase 2: intelligent routing engine (prompt analysis, heuristic classification, policy-filtered strategies, POST /v1/chat)"
```
