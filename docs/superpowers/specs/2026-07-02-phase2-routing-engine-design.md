# LLM Cost Autopilot — Phase 2 Design: Intelligent Routing Engine

Status: **Approved — frozen as implementation contract**
Date: 2026-07-02

## 1. Purpose & Scope

Phase 2 adds a heuristic (non-ML) routing pipeline that analyzes a prompt,
classifies its complexity, selects a model via a pluggable strategy, and
exposes the whole thing through `POST /v1/chat` — persisting each request,
its routing decision, and its response.

```
Client
  -> POST /v1/chat
  -> PromptAnalyzer            (prompt -> PromptFeatures)
  -> HeuristicComplexityClassifier  (PromptFeatures -> ClassificationResult)
  -> RoutingPolicy              (complexity -> eligible ModelSpec candidates)
  -> RoutingStrategy             (RoutingContext -> selected ModelSpec)
  -> ExplanationGenerator          (-> reasoning)
  -> RoutingEngine                  (assembles RoutingDecision)
  -> ChatService                     (calls provider, persists, returns ChatResult)
  -> HTTP response
```

**In scope:**
- `PromptAnalyzer` / `PromptFeatures` (deterministic feature extraction,
  including an `estimated_output_tokens` heuristic distinct from input
  length)
- `HeuristicComplexityClassifier` behind a `BaseComplexityClassifier`
  interface, with configurable thresholds and a `ClassificationResult`
  that includes human-readable `signals`
- `RoutingPolicy` (complexity -> eligibility filter, config-driven, not
  hardcoded)
- Four routing strategies (cost/latency/quality/balanced) behind
  `BaseRoutingStrategy`, operating on a single `RoutingContext` object
- `ExplanationGenerator` (routing decision -> human-readable reasoning,
  decoupled from `RoutingEngine`)
- `RoutingEngine` (pure orchestration + `RoutingDecision`,
  `NoEligibleModelError`) — **never calls a provider**
- `ChatService` (orchestrates `RoutingEngine` + `ProviderManager` +
  persistence) and `POST /v1/chat`
- New tables: `requests`, `responses`, `routing_events`
- A single `RoutingConfigLoader` owning all routing YAML file I/O

**Explicitly out of scope for Phase 2** (deferred to later phases):
- ML classifier (the heuristic classifier is designed as a drop-in
  replaceable interface, but no ML model is built now)
- LLM-as-judge / quality verification
- Auto-escalation on quality failure
- Background workers / async queues
- Retry policies, semantic caching, rate limiting
- Learning loop / classifier retraining
- Streaming responses (`provider.stream()`) — `/v1/chat` uses
  `provider.generate()` only
- `quality_profile` (per-category benchmark scores) — `ModelSpec.benchmark_score`
  stays as-is; noted as a future extension, not built now

## 2. Directory Structure

```
backend/
  analysis/
    __init__.py
    prompt_analyzer.py       # PromptFeatures, PromptAnalyzer
  classifier/
    __init__.py
    complexity_classifier.py  # ComplexityTier, ClassificationResult,
                                # BaseComplexityClassifier, HeuristicComplexityClassifier
  routing/
    __init__.py
    config.py                  # ClassifierPolicy, EligibilityPolicy,
                                 # BalancedStrategyWeights, RoutingConfig
    config_loader.py             # RoutingConfigLoader (the only YAML file I/O)
    policy.py                     # RoutingPolicy
    context.py                     # RoutingContext
    strategies.py                   # BaseRoutingStrategy + 4 implementations
    explanation.py                   # ExplanationGenerator
    engine.py                         # RoutingDecision, NoEligibleModelError, RoutingEngine
  chat/
    __init__.py
    service.py                        # ChatResult, ChatService
  api/
    routers/
      chat.py                          # POST /v1/chat
  config/
    routing.yaml                        # classifier + policy + balanced_strategy config
  database/
    models.py                           # + RequestRow, ResponseRow, RoutingEventRow (modify)
  config/
    settings.py                          # + routing_config_path field (modify)
```

## 3. Configuration

### 3.1 `backend/config/routing.yaml`

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

### 3.2 `Settings` addition

`Settings` (Phase 1, `backend/config/settings.py`) gains one field:
`routing_config_path: str = Field(default="backend/config/routing.yaml", min_length=1)`.
`Settings` still only carries the path as a string — it does not read or
parse the file, matching the same separation established for
`models_yaml_path` in Phase 1.

### 3.3 `RoutingConfig` / `RoutingConfigLoader` (`backend/routing/config.py`, `config_loader.py`)

A single loader owns all routing YAML file I/O — not three independent
parsers:

```python
class ClassifierPolicy(BaseModel):
    simple_max: int
    medium_max: int

class EligibilityPolicy(BaseModel):
    min_benchmark_score: float

class BalancedStrategyWeights(BaseModel):
    cost_weight: float = 1 / 3
    latency_weight: float = 1 / 3
    quality_weight: float = 1 / 3

class RoutingConfig(BaseModel):
    classifier: ClassifierPolicy
    policy: dict[str, EligibilityPolicy]  # keyed by "simple"/"medium"/"complex" -- see note below
    balanced_strategy: BalancedStrategyWeights
```

`policy` is keyed by plain strings (matching the YAML keys), not
`ComplexityTier`, deliberately: `backend/routing/config.py` must not
import `ComplexityTier` from `backend/classifier/complexity_classifier.py`,
because that module already imports `ClassifierPolicy` *from*
`routing/config.py` — keying by `ComplexityTier` here would create a
circular import. `RoutingPolicy` (§6.1, which already imports
`ComplexityTier` one-directionally from `classifier/`) is the one place
that converts `complexity.value` to look up this dict.

```python
class RoutingConfigLoader:
    @staticmethod
    def load(yaml_path: str) -> RoutingConfig:
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        return RoutingConfig.model_validate(raw)
```

`RoutingConfigLoader.load()` fails fast (`yaml.YAMLError` on malformed
YAML, `pydantic.ValidationError` on an invalid schema) — the same
fail-fast discipline as `ModelRegistry.reload()` in Phase 1. Each
downstream consumer (`HeuristicComplexityClassifier`, `RoutingPolicy`,
`BalancedStrategy`) receives its already-parsed, already-validated
sub-model (`config.classifier`, `config.policy`, `config.balanced_strategy`)
— none of them touch the filesystem or YAML themselves. Loaded once at
startup; no runtime reload in Phase 2 (a future phase could add one,
following the same `reload()` pattern as `ModelRegistry`, but it isn't
needed yet).

## 4. Prompt Analysis

### 4.1 `PromptFeatures` (`backend/analysis/prompt_analyzer.py`)

```python
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
```

### 4.2 `PromptAnalyzer`

```python
class PromptAnalyzer:
    def analyze(self, prompt: str) -> PromptFeatures: ...
```

Pure extraction — **never selects a model**. All detection is
deterministic (regex/keyword matching), no ML:

- `estimated_tokens`: `max(1, len(prompt) // 4)` (same heuristic as
  `count_tokens` elsewhere in the codebase, for consistency)
- `constraint_count`: count of regex matches for phrases like `must`,
  `should`, `need to`, `ensure`, `require`, `make sure`
- `has_code`: markdown code fence (`` ``` ``) or common code keywords
  (`def `, `function `, `class `, `import `)
- `has_json`: the word "json" (case-insensitive) or a `{...}`-with-colon
  structural pattern
- `has_reasoning_keywords`: `why`, `explain`, `reasoning`, `because`
- `has_comparison_keywords`: `compare`, `versus`, `vs`, `difference between`
- `has_analysis_keywords`: `analyze`, `analysis`, `evaluate`, `assess`, `review`
- `has_creative_keywords`: `story`, `poem`, `creative`, `imagine`, `brainstorm`
- `has_math_indicators`: digits adjacent to `+ - * / =`, or `calculate`,
  `solve`, `equation`
- `has_chain_of_thought_indicators`: `step by step`, `walk me through`,
  `first...then`
- `requires_output_formatting`: `format as`, `return as`, `bullet points`,
  `as a table`, `in json`
- `requested_language`: only set if `has_code`; regex match against a
  fixed list of language names (`python`, `javascript`, `typescript`,
  `go`, `rust`, `java`, `c++`, `sql`) mentioned in the prompt, else `None`

**`estimated_output_tokens` heuristic** (distinct from input length):
1. Explicit brevity phrases (`one sentence`, `briefly`, `short answer`,
   `one word`, `yes or no`) → `20`
2. Explicit word-count mention (regex `(\d+)\s*word`) → `round(word_count * 1.3)`,
   floor `20`
3. Long-form keywords (`essay`, `comprehensive`, `detailed`, `in-depth`,
   `thorough`, `elaborate`) → `max(800, estimated_tokens * 2)`
4. Otherwise → `max(50, min(estimated_tokens, 500))`

No `BasePromptAnalyzer` interface — unlike the classifier, there's no
near-term alternative implementation planned for this component, so an
ABC would be speculative. `HeuristicComplexityClassifier` gets one
because the ML classifier replacing it is explicitly planned.

## 5. Complexity Classification

### 5.1 `ComplexityTier` / `ClassificationResult` (`backend/classifier/complexity_classifier.py`)

```python
class ComplexityTier(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

class ClassificationResult(BaseModel):
    tier: ComplexityTier
    score: int
    confidence: float
    signals: list[str]
```

### 5.2 `BaseComplexityClassifier` / `HeuristicComplexityClassifier`

```python
class BaseComplexityClassifier(ABC):
    @abstractmethod
    def classify(self, features: PromptFeatures) -> ClassificationResult: ...
```

`HeuristicComplexityClassifier(policy: ClassifierPolicy)` computes an
additive score, one point per signal present, appending a human-readable
string to `signals` for each:

```
+1 estimated_tokens > 200        -> "prompt exceeds 200 estimated tokens"
+1 constraint_count >= 2         -> "multiple constraints detected"
+1 has_code                      -> "code content detected"
+1 has_reasoning_keywords        -> "reasoning keywords detected"
+1 has_comparison_keywords       -> "comparison keywords detected"
+1 has_analysis_keywords         -> "analysis keywords detected"
+1 has_math_indicators           -> "math indicators detected"
+1 has_chain_of_thought_indicators -> "chain-of-thought indicators detected"
+1 requires_output_formatting    -> "output formatting requested"
```

Tier: `score <= policy.simple_max` → `SIMPLE`;
`score <= policy.medium_max` → `MEDIUM`; else `COMPLEX`.

**Confidence**: distance from the score to the nearest configured tier
boundary (`policy.simple_max`, `policy.medium_max`), normalized:
`confidence = round(0.5 + min(nearest_boundary_distance / 3, 1.0) * 0.49, 2)`.
A score sitting exactly on a boundary (maximally ambiguous between two
tiers) yields `0.5`; a score 3+ away from any boundary yields `0.99`.

This design makes the classifier's thresholds and the eligibility
policy's thresholds independently configurable via the same YAML file
without either needing to know about the other.

## 6. Routing Policy & Context

### 6.1 `RoutingPolicy` (`backend/routing/policy.py`)

```python
class RoutingPolicy:
    def __init__(self, policies: dict[str, EligibilityPolicy]): ...
    def filter_candidates(
        self, complexity: ComplexityTier, candidates: list[ModelSpec]
    ) -> list[ModelSpec]:
        policy = self._policies[complexity.value]
        return [c for c in candidates if c.benchmark_score >= policy.min_benchmark_score]
```

`RoutingEngine` calls this instead of holding threshold logic itself —
the engine doesn't know *why* a model is eligible, only that the policy
says it is. Extending eligibility later (`requires_json`, `max_latency`,
etc.) means adding a field to `EligibilityPolicy`, not touching the
engine or any strategy.

### 6.2 `RoutingContext` (`backend/routing/context.py`)

```python
class RoutingContext(BaseModel):
    prompt: str
    features: PromptFeatures
    complexity: ComplexityTier
    candidates: list[ModelSpec]
```

One immutable object instead of long parameter lists — every strategy
takes exactly one argument.

## 7. Routing Strategies (`backend/routing/strategies.py`)

```python
class BaseRoutingStrategy(ABC):
    @abstractmethod
    def select_model(self, context: RoutingContext) -> ModelSpec: ...
```

- **`CostOptimizedStrategy`**: lowest `input_cost + output_cost`
- **`LatencyOptimizedStrategy`**: lowest `average_latency_ms`
- **`QualityOptimizedStrategy`**: highest `benchmark_score`
- **`BalancedStrategy(weights: BalancedStrategyWeights)`**: min-max
  normalize cost/latency (inverted — lower is better) and quality
  (higher is better) across `context.candidates`, combine with the
  configured weights, return the highest-scoring candidate. If all
  candidates tie on a metric (`max == min`), that metric contributes a
  neutral `0.5` to every candidate rather than dividing by zero.

All four raise nothing themselves for an empty candidate list —
`RoutingEngine` guarantees `context.candidates` is non-empty before any
strategy is invoked (see §8).

## 8. Explanation Generation (`backend/routing/explanation.py`)

```python
class ExplanationGenerator:
    def generate(
        self,
        context: RoutingContext,
        selected: ModelSpec,
        strategy_name: str,
        classification: ClassificationResult,
    ) -> list[str]: ...
```

Builds the `reasoning` list from `classification.signals` (already
computed by the classifier — never rediscovered) plus the strategy name,
candidate count, and selected model. Kept entirely separate from
`RoutingEngine` so the orchestration layer never grows a wall of
`if`/`elif` string-building as routing gets more sophisticated.

## 9. Routing Engine (`backend/routing/engine.py`)

```python
class RoutingDecision(BaseModel):
    selected_model: str
    strategy: str
    complexity: ComplexityTier
    confidence: float
    estimated_cost: float
    estimated_latency_ms: float
    reasoning: list[str]

class NoEligibleModelError(Exception): ...

class RoutingEngine:
    def __init__(
        self,
        model_registry: ModelRegistry,
        analyzer: PromptAnalyzer,
        classifier: BaseComplexityClassifier,
        routing_policy: RoutingPolicy,
        strategies: dict[str, BaseRoutingStrategy],
        explanation_generator: ExplanationGenerator,
    ): ...

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
            prompt=prompt, features=features, complexity=classification.tier, candidates=candidates
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

`RoutingEngine` never calls a provider and never touches the database —
pure orchestration + decision assembly, as specified. Unknown
`strategy_name` raises `KeyError` from the dict lookup (consistent with
`ProviderFactory.create()`'s "fail loudly" precedent from Phase 1 — no
silent fallback to a default strategy).

## 10. Chat Service & API

### 10.1 Database additions (`backend/database/models.py`)

```
requests:       id, request_id (unique), prompt, strategy, created_at
responses:      id, request_id (FK -> requests.request_id), response_text (nullable),
                actual_input_tokens (nullable), actual_output_tokens (nullable),
                actual_cost (nullable), error (nullable), created_at
routing_events: id, request_id (FK -> requests.request_id), complexity, confidence,
                selected_model, selected_strategy, estimated_cost,
                estimated_latency_ms, reasoning (JSON-encoded text), created_at
```

`responses.error`/nullable fields exist so a `ProviderError` during
generation still leaves a persisted record (with `error` set, response
fields `None`) rather than silently losing the request.

### 10.2 `ChatService` (`backend/chat/service.py`)

```python
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
    ): ...

    async def chat(self, prompt: str, strategy: str = "balanced") -> ChatResult:
        request_id = str(uuid.uuid4())
        decision = self._routing_engine.route(prompt, strategy_name=strategy)

        # Persist request + routing_event immediately -- the decision is
        # recorded even if generation subsequently fails.
        with self._session_factory() as session:
            session.add(RequestRow(request_id=request_id, prompt=prompt, strategy=strategy))
            session.add(RoutingEventRow(request_id=request_id, ...))  # from decision
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
                request_id=request_id, response_text=response_text,
                actual_input_tokens=input_tokens, actual_output_tokens=output_tokens,
                actual_cost=actual_cost,
            ))
            session.commit()

        return ChatResult(request_id=request_id, response=response_text, routing=decision)
```

`ChatService` is the only place that calls both `RoutingEngine` and a
provider — matching the Phase 1 discipline that `RoutingEngine` itself
stays provider-ignorant.

### 10.3 `POST /v1/chat` (`backend/api/routers/chat.py`)

```python
class ChatRequest(BaseModel):
    prompt: str
    strategy: Literal["cost", "latency", "quality", "balanced"] = "balanced"

@router.post("/chat", response_model=ChatResult)
async def chat(request: ChatRequest, chat_service: ChatServiceDep) -> ChatResult:
    try:
        return await chat_service.chat(request.prompt, strategy=request.strategy)
    except NoEligibleModelError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
```

Thin — HTTP concerns and status-code mapping only, no routing or
provider logic. A new `ChatServiceDep` (`backend/api/dependencies.py`,
modified) follows the exact same `Depends()`-reading-`app.state` pattern
as every other Phase 1 dependency; `ChatService` is constructed once in
`main.py`'s `lifespan`, alongside everything else.

## 11. Testing

Every new component gets the same fail-fast/interface-enforcement test
discipline established in Phase 1: `BaseComplexityClassifier`/
`BaseRoutingStrategy` cannot be instantiated without every abstract
method; `RoutingConfigLoader` rejects malformed YAML and invalid schema;
`RoutingPolicy.filter_candidates` returns `[]` (not an error) when
nothing qualifies, and `RoutingEngine` is what turns that into
`NoEligibleModelError`; each strategy is tested with a fixed
`RoutingContext` fixture so selection is deterministic and assertable;
`ChatService` tests use `MockProvider` (no network calls, per Phase 1's
provider-testing precedent) and assert both the happy path and the
`ProviderError`-persists-an-error-row path.

## 12. Tooling

No new dependencies — `pyyaml` and `pydantic` (already present) cover
`RoutingConfigLoader`; no new third-party packages required for Phase 2.
