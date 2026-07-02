# Phase 3 Implementation Plan: Quality Verification & Evaluation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an in-process, background LLM-as-judge verification pipeline that scores every chat response for quality after `ChatService` returns it, persists the verdict, and exposes it via `GET /v1/chat/{request_id}/verification` and `GET /v1/metrics/quality`.

**Architecture:** Two new packages (`verification`, plus two new API routers) layered on top of the existing Phase 1/2 stack. `BaseJudge`/`LLMJudge` are pure (prompt+response in, `JudgeVerdict` out — no I/O beyond the provider call). `JudgeEngine` times a judge call. `VerificationService` owns the DB lifecycle and event emission. `ChatService` schedules verification as a best-effort `BackgroundTasks` side effect that can never affect the chat response.

**Tech Stack:** Same as Phase 1/2 — Python 3.11+, `uv`, FastAPI (including `BackgroundTasks`, already part of FastAPI), Pydantic v2, SQLAlchemy 2.0, PyYAML. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-02-phase3-verification-design.md` (frozen — implement exactly).

## Global Constraints

- Same `uv`-managed Python 3.11+ project as Phases 1-2; no new dependencies.
- Batches of 2-3 tasks; one test run + one commit per batch (not per task). Only revert to single-task commits on a genuine architectural snag or repeated test failures.
- `BaseJudge`/`LLMJudge` never touch the database, never retry, never emit events, and never know about `request_id` — pure `(prompt, response) -> JudgeVerdict`.
- `JudgeEngine` is the only place that measures judge-call duration; `VerificationService` calls `JudgeEngine`, never `BaseJudge` directly.
- `VerificationService` opens a fresh, independent DB transaction for every state transition — no ORM instance is held or mutated across a `with session_factory()` boundary.
- Persist before emit: every `VerificationRow` write commits before the corresponding event is published.
- State transitions are always `PENDING -> RUNNING -> (COMPLETED | FAILED)` — never skipped, never combined.
- `VerificationService.verify()` swallows all judge-side exceptions itself; nothing propagates to `ChatService`.
- Verification scheduling from `ChatService` is best-effort: a failure scheduling or running verification must never cause `/v1/chat` to return an error or delay its response.
- `dimensions` is a typed `VerificationDimensions` model, never a bare `dict[str, float]`.
- Every score field (`correctness`, `completeness`, `instruction_following`, `format_adherence`, `confidence`) is constrained to `[0.0, 1.0]` via Pydantic `Field(ge=0.0, le=1.0)` — an out-of-range value from the judge model raises `pydantic.ValidationError`, which is treated as a parse failure (`FAILED` verification), not silently clamped.
- `LLMJudge` parses the judge's raw text via `_JudgeResponseSchema.model_validate_json(raw)`, never manual `json.loads()` + dict indexing.
- `raw_judge_response` and `error`/`error_type` are persisted for debugging/audit but `raw_judge_response` is never returned by any API response model.
- `judge_model` (which model served as judge) and `judge_prompt_version` (which prompt/schema version) are independent fields — both persisted on every completed verification.
- The existing `EventBus.emit(event_type: EventType, payload: dict)` signature (`backend/events/bus.py`) is unchanged. New verification events are typed Pydantic models whose `.model_dump()` is passed as `payload`.
- No auto-escalation, no classifier retraining, no feedback loop into routing, no prompt optimization, no retry policies for failed verifications, no separate worker process/queue — all explicitly out of scope per the spec.
- No placeholder code, no TODOs, no speculative abstractions.

---

## Batch 1: Verification Config & Judge

### Task 27: VerificationConfig, VerificationConfigLoader & Settings

**Files:**
- Create: `backend/verification/__init__.py` (empty)
- Create: `backend/verification/config.py`
- Create: `backend/verification/config_loader.py`
- Create: `backend/config/verification.yaml`
- Modify: `backend/config/settings.py`
- Test: `backend/tests/test_verification_config.py`
- Modify: `backend/tests/test_settings.py` (append)

**Interfaces:**
- Produces: `VerificationConfig(judge_model_id: str, pass_threshold: float, judge_prompt_version: str)`, `VerificationConfigLoader.load(yaml_path: str) -> VerificationConfig`. `Settings.verification_config_path: str` (new field, default `"backend/config/verification.yaml"`). Consumed by `LLMJudge`/`JudgeEngine`/`VerificationService` (Task 28, 30), `main.py` (Task 34, loads once at startup).

- [ ] **Step 1: Create the package directory**

Run:
```bash
mkdir -p backend/verification
touch backend/verification/__init__.py
```

- [ ] **Step 2: Write `backend/config/verification.yaml`**

```yaml
judge_model_id: gpt-4o
pass_threshold: 0.7
judge_prompt_version: v1
```

- [ ] **Step 3: Write the failing tests**

```python
# backend/tests/test_verification_config.py
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
```

Append to `backend/tests/test_settings.py`:

```python
def test_settings_verification_config_path_default():
    settings = Settings(_env_file=None)
    assert settings.verification_config_path == "backend/config/verification.yaml"


def test_settings_rejects_blank_verification_config_path():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, verification_config_path="")
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest backend/tests/test_verification_config.py backend/tests/test_settings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.verification.config'` (and the two new Settings tests fail with `AttributeError`)

- [ ] **Step 5: Write `backend/verification/config.py`**

```python
# backend/verification/config.py
from pydantic import BaseModel, Field


class VerificationConfig(BaseModel):
    judge_model_id: str
    pass_threshold: float = Field(ge=0.0, le=1.0)
    judge_prompt_version: str
```

- [ ] **Step 6: Write `backend/verification/config_loader.py`**

```python
# backend/verification/config_loader.py
import yaml

from backend.verification.config import VerificationConfig


class VerificationConfigLoader:
    @staticmethod
    def load(yaml_path: str) -> VerificationConfig:
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        return VerificationConfig.model_validate(raw)
```

- [ ] **Step 7: Modify `backend/config/settings.py`**

Change:
```python
    database_url: str = Field(default="sqlite:///./llm_cost_autopilot.db", min_length=1)
    models_yaml_path: str = Field(default="backend/config/models.yaml", min_length=1)
    routing_config_path: str = Field(default="backend/config/routing.yaml", min_length=1)

    openai_api_key: str | None = None
```

To:
```python
    database_url: str = Field(default="sqlite:///./llm_cost_autopilot.db", min_length=1)
    models_yaml_path: str = Field(default="backend/config/models.yaml", min_length=1)
    routing_config_path: str = Field(default="backend/config/routing.yaml", min_length=1)
    verification_config_path: str = Field(
        default="backend/config/verification.yaml", min_length=1
    )

    openai_api_key: str | None = None
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest backend/tests/test_verification_config.py backend/tests/test_settings.py -v`
Expected: PASS (4 + 13 = 17 tests)

### Task 28: JudgeVerdict, VerificationDimensions, BaseJudge & LLMJudge

**Files:**
- Create: `backend/verification/judge.py`
- Test: `backend/tests/test_llm_judge.py`

**Interfaces:**
- Consumes: `VerificationConfig` (Task 27), `BaseProvider` (Phase 1, `backend/providers/base.py`).
- Produces: `VerificationDimensions(correctness, completeness, instruction_following, format_adherence)`, `JudgeVerdict(score, passed, confidence, rationale, dimensions)`, `BaseJudge` ABC (`evaluate(prompt, response) -> JudgeVerdict`), `LLMJudge(provider, model, pass_threshold)`. Consumed by `JudgeEngine` (Task 29), `VerificationService` (Task 31), `main.py` (Task 34).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_llm_judge.py
import json

import pytest
from pydantic import ValidationError

from backend.providers.mock_provider import MockProvider
from backend.verification.judge import BaseJudge, LLMJudge


def _valid_judge_json(**overrides) -> str:
    payload = {
        "correctness": 0.9,
        "completeness": 0.8,
        "instruction_following": 0.85,
        "format_adherence": 0.95,
        "confidence": 0.9,
        "rationale": "The response correctly and completely answers the prompt.",
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_base_judge_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseJudge()


@pytest.mark.asyncio
async def test_evaluate_returns_verdict_with_mean_score():
    provider = MockProvider(response=_valid_judge_json())
    judge = LLMJudge(provider=provider, model="gpt-4o", pass_threshold=0.7)

    verdict = await judge.evaluate("What is 2+2?", "4")

    assert verdict.score == pytest.approx((0.9 + 0.8 + 0.85 + 0.95) / 4)
    assert verdict.passed is True
    assert verdict.confidence == 0.9
    assert verdict.rationale == "The response correctly and completely answers the prompt."
    assert verdict.dimensions.correctness == 0.9
    assert verdict.dimensions.completeness == 0.8
    assert verdict.dimensions.instruction_following == 0.85
    assert verdict.dimensions.format_adherence == 0.95


@pytest.mark.asyncio
async def test_evaluate_marks_failed_below_threshold():
    provider = MockProvider(
        response=_valid_judge_json(
            correctness=0.2, completeness=0.2, instruction_following=0.2, format_adherence=0.2
        )
    )
    judge = LLMJudge(provider=provider, model="gpt-4o", pass_threshold=0.7)

    verdict = await judge.evaluate("What is 2+2?", "purple")

    assert verdict.passed is False


@pytest.mark.asyncio
async def test_evaluate_raises_on_malformed_json():
    provider = MockProvider(response="not json at all")
    judge = LLMJudge(provider=provider, model="gpt-4o", pass_threshold=0.7)

    with pytest.raises(ValidationError):
        await judge.evaluate("prompt", "response")


@pytest.mark.asyncio
async def test_evaluate_raises_on_missing_field():
    incomplete = json.dumps({"correctness": 0.9, "completeness": 0.8, "confidence": 0.9})
    provider = MockProvider(response=incomplete)
    judge = LLMJudge(provider=provider, model="gpt-4o", pass_threshold=0.7)

    with pytest.raises(ValidationError):
        await judge.evaluate("prompt", "response")


@pytest.mark.asyncio
async def test_evaluate_raises_on_out_of_range_score():
    provider = MockProvider(response=_valid_judge_json(correctness=1.5))
    judge = LLMJudge(provider=provider, model="gpt-4o", pass_threshold=0.7)

    with pytest.raises(ValidationError):
        await judge.evaluate("prompt", "response")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_llm_judge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.verification.judge'`

- [ ] **Step 3: Write the implementation**

```python
# backend/verification/judge.py
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from backend.providers.base import BaseProvider


class VerificationDimensions(BaseModel):
    correctness: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    instruction_following: float = Field(ge=0.0, le=1.0)
    format_adherence: float = Field(ge=0.0, le=1.0)


class JudgeVerdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    dimensions: VerificationDimensions


class BaseJudge(ABC):
    @abstractmethod
    async def evaluate(self, prompt: str, response: str) -> JudgeVerdict: ...


class _JudgeResponseSchema(BaseModel):
    correctness: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    instruction_following: float = Field(ge=0.0, le=1.0)
    format_adherence: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


_JUDGE_PROMPT_TEMPLATE = """You are evaluating whether an AI response adequately answers a prompt.

Prompt:
{prompt}

Response:
{response}

Score each dimension from 0.0 to 1.0:
- correctness: is the response factually/logically correct?
- completeness: does it fully address the prompt?
- instruction_following: does it follow any explicit instructions/constraints in the prompt?
- format_adherence: does it match any requested format?

Respond with ONLY valid JSON matching this schema:
{{"correctness": float, "completeness": float, "instruction_following": float,
  "format_adherence": float, "confidence": float, "rationale": "one paragraph"}}
"""


class LLMJudge(BaseJudge):
    def __init__(self, provider: BaseProvider, model: str, pass_threshold: float) -> None:
        self._provider = provider
        self._model = model
        self._pass_threshold = pass_threshold

    async def evaluate(self, prompt: str, response: str) -> JudgeVerdict:
        raw = await self._provider.generate(
            _JUDGE_PROMPT_TEMPLATE.format(prompt=prompt, response=response), model=self._model
        )
        parsed = _JudgeResponseSchema.model_validate_json(raw)

        dimensions = VerificationDimensions(
            correctness=parsed.correctness,
            completeness=parsed.completeness,
            instruction_following=parsed.instruction_following,
            format_adherence=parsed.format_adherence,
        )
        score = (
            dimensions.correctness
            + dimensions.completeness
            + dimensions.instruction_following
            + dimensions.format_adherence
        ) / 4

        return JudgeVerdict(
            score=score,
            passed=score >= self._pass_threshold,
            confidence=parsed.confidence,
            rationale=parsed.rationale,
            dimensions=dimensions,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_llm_judge.py -v`
Expected: PASS (6 tests)

- [ ] **Batch 1 verification & commit**

Run the full suite to check for regressions:
```bash
uv run pytest -v
```
Expected: all tests pass (161 + 17 + 6 = 184).

Commit:
```bash
git add backend/verification/__init__.py backend/verification/config.py backend/verification/config_loader.py backend/verification/judge.py backend/config/verification.yaml backend/config/settings.py backend/tests/test_verification_config.py backend/tests/test_settings.py backend/tests/test_llm_judge.py
git commit -m "feat: add VerificationConfig/VerificationConfigLoader and LLMJudge"
```

---

## Batch 2: JudgeEngine, VerificationRow & Events

### Task 29: JudgeEngine

**Files:**
- Create: `backend/verification/engine.py`
- Test: `backend/tests/test_judge_engine.py`

**Interfaces:**
- Consumes: `BaseJudge`, `JudgeVerdict` (Task 28).
- Produces: `JudgeEngine(judge: BaseJudge, judge_model_id: str)`, `JudgeEngine.judge_model_id` (property), `JudgeEngine.run(prompt, response) -> tuple[JudgeVerdict, int]`. Consumed by `VerificationService` (Task 31), `main.py` (Task 34).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_judge_engine.py
import pytest

from backend.providers.mock_provider import MockProvider
from backend.verification.engine import JudgeEngine
from backend.verification.judge import LLMJudge


@pytest.mark.asyncio
async def test_run_returns_verdict_and_non_negative_duration():
    import json

    response_json = json.dumps({
        "correctness": 0.9, "completeness": 0.9, "instruction_following": 0.9,
        "format_adherence": 0.9, "confidence": 0.9, "rationale": "Good answer.",
    })
    provider = MockProvider(response=response_json)
    judge = LLMJudge(provider=provider, model="gpt-4o", pass_threshold=0.7)
    engine = JudgeEngine(judge=judge, judge_model_id="gpt-4o")

    verdict, duration_ms = await engine.run("prompt", "response")

    assert verdict.score == pytest.approx(0.9)
    assert duration_ms >= 0


def test_judge_model_id_property():
    provider = MockProvider(response="{}")
    judge = LLMJudge(provider=provider, model="gpt-4o", pass_threshold=0.7)
    engine = JudgeEngine(judge=judge, judge_model_id="gpt-4o")

    assert engine.judge_model_id == "gpt-4o"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_judge_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.verification.engine'`

- [ ] **Step 3: Write the implementation**

```python
# backend/verification/engine.py
import time

from backend.verification.judge import BaseJudge, JudgeVerdict


class JudgeEngine:
    def __init__(self, judge: BaseJudge, judge_model_id: str) -> None:
        self._judge = judge
        self._judge_model_id = judge_model_id

    @property
    def judge_model_id(self) -> str:
        return self._judge_model_id

    async def run(self, prompt: str, response: str) -> tuple[JudgeVerdict, int]:
        start = time.monotonic()
        verdict = await self._judge.evaluate(prompt, response)
        duration_ms = round((time.monotonic() - start) * 1000)
        return verdict, duration_ms
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_judge_engine.py -v`
Expected: PASS (2 tests)

### Task 30: VerificationRow, VerificationStatus & Typed Events

**Files:**
- Modify: `backend/database/models.py`
- Modify: `backend/events/types.py`
- Create: `backend/verification/status.py`
- Create: `backend/verification/events.py`
- Test: `backend/tests/test_verification_models.py`
- Test: `backend/tests/test_verification_events.py`

**Interfaces:**
- Produces: `VerificationStatus(str, Enum)` (`PENDING`, `RUNNING`, `COMPLETED`, `FAILED`), `VerificationRow` (SQLAlchemy model), `EventType.VERIFICATION_STARTED`/`VERIFICATION_COMPLETED`/`VERIFICATION_FAILED`, `VerificationStarted(request_id)`, `VerificationCompleted(request_id, score)`, `VerificationFailed(request_id, error_type, error)`. Consumed by `VerificationService` (Task 31), `main.py` (Task 34).

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_verification_models.py
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.config.settings import Settings
from backend.database.models import RequestRow, VerificationRow
from backend.verification.status import VerificationStatus


def test_verification_row_round_trip(tmp_path):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    with session_factory() as session:
        session.add(RequestRow(request_id="req-1", prompt="hi", strategy="balanced"))
        session.add(VerificationRow(
            request_id="req-1",
            status=VerificationStatus.PENDING.value,
            routing_model="gpt-4o-mini",
            routing_strategy="balanced",
            routing_complexity="simple",
        ))
        session.commit()

    with session_factory() as session:
        row = session.query(VerificationRow).filter_by(request_id="req-1").one()
        assert row.status == VerificationStatus.PENDING.value
        assert row.routing_model == "gpt-4o-mini"
        assert row.score is None
        assert row.dimensions is None
        assert row.raw_judge_response is None
        assert row.started_at is None
        assert row.completed_at is None
```

```python
# backend/tests/test_verification_events.py
from backend.events.types import EventType
from backend.verification.events import (
    VerificationCompleted,
    VerificationFailed,
    VerificationStarted,
)


def test_event_type_members_exist():
    assert EventType.VERIFICATION_STARTED == "verification_started"
    assert EventType.VERIFICATION_COMPLETED == "verification_completed"
    assert EventType.VERIFICATION_FAILED == "verification_failed"


def test_verification_started_payload():
    event = VerificationStarted(request_id="req-1")
    assert event.model_dump() == {"request_id": "req-1"}


def test_verification_completed_payload():
    event = VerificationCompleted(request_id="req-1", score=0.85)
    assert event.model_dump() == {"request_id": "req-1", "score": 0.85}


def test_verification_failed_payload():
    event = VerificationFailed(request_id="req-1", error_type="ValidationError", error="bad json")
    assert event.model_dump() == {
        "request_id": "req-1", "error_type": "ValidationError", "error": "bad json"
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest backend/tests/test_verification_models.py backend/tests/test_verification_events.py -v`
Expected: FAIL — `ImportError: cannot import name 'VerificationRow' from 'backend.database.models'` (and `ModuleNotFoundError` for `backend.verification.status`/`events`)

- [ ] **Step 3: Write `backend/verification/status.py`**

```python
# backend/verification/status.py
from enum import Enum


class VerificationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
```

- [ ] **Step 4: Modify `backend/events/types.py`**

Change:
```python
class EventType(str, Enum):
    PROVIDER_AVAILABLE = "provider_available"
    PROVIDER_DISABLED = "provider_disabled"
    PROVIDER_FAILED = "provider_failed"
    MODEL_REGISTERED = "model_registered"
```

To:
```python
class EventType(str, Enum):
    PROVIDER_AVAILABLE = "provider_available"
    PROVIDER_DISABLED = "provider_disabled"
    PROVIDER_FAILED = "provider_failed"
    MODEL_REGISTERED = "model_registered"
    VERIFICATION_STARTED = "verification_started"
    VERIFICATION_COMPLETED = "verification_completed"
    VERIFICATION_FAILED = "verification_failed"
```

- [ ] **Step 5: Write `backend/verification/events.py`**

```python
# backend/verification/events.py
from pydantic import BaseModel


class VerificationStarted(BaseModel):
    request_id: str


class VerificationCompleted(BaseModel):
    request_id: str
    score: float


class VerificationFailed(BaseModel):
    request_id: str
    error_type: str
    error: str
```

- [ ] **Step 6: Modify `backend/database/models.py`**

Add imports (change the top of the file):
```python
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.database.base import Base
```

Append at the end of the file:
```python
class VerificationRow(Base):
    __tablename__ = "verifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String, ForeignKey("requests.request_id"), nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)

    routing_model: Mapped[str] = mapped_column(String, nullable=False)
    routing_strategy: Mapped[str] = mapped_column(String, nullable=False)
    routing_complexity: Mapped[str] = mapped_column(String, nullable=False)

    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    rationale: Mapped[str | None] = mapped_column(String, nullable=True)
    dimensions: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    judge_model: Mapped[str | None] = mapped_column(String, nullable=True)
    judge_prompt_version: Mapped[str | None] = mapped_column(String, nullable=True)
    evaluation_duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    raw_judge_response: Mapped[str | None] = mapped_column(String, nullable=True)

    error_type: Mapped[str | None] = mapped_column(String, nullable=True)
    error: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest backend/tests/test_verification_models.py backend/tests/test_verification_events.py -v`
Expected: PASS (1 + 4 = 5 tests)

- [ ] **Batch 2 verification & commit**

```bash
uv run pytest -v
```
Expected: all tests pass (184 + 2 + 5 = 191).

```bash
git add backend/verification/engine.py backend/verification/status.py backend/verification/events.py backend/database/models.py backend/events/types.py backend/tests/test_judge_engine.py backend/tests/test_verification_models.py backend/tests/test_verification_events.py
git commit -m "feat: add JudgeEngine, VerificationRow, VerificationStatus, and typed verification events"
```

---

## Batch 3: VerificationService & ChatService Integration

### Task 31: VerificationService

**Files:**
- Create: `backend/verification/service.py`
- Test: `backend/tests/test_verification_service.py`

**Interfaces:**
- Consumes: `JudgeEngine` (Task 29), `VerificationRow`, `RoutingEventRow` (Phase 2, `backend/database/models.py`), `VerificationStatus` (Task 30), `VerificationStarted`/`VerificationCompleted`/`VerificationFailed` (Task 30), `EventBus` (Phase 1, `backend/events/bus.py`).
- Produces: `VerificationService(judge_engine, session_factory, event_bus, judge_prompt_version)`, `VerificationService.verify(request_id, prompt, response) -> None` (async). Consumed by `ChatService` (Task 32), `main.py` (Task 34).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_verification_service.py
import json

import pytest

from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.database.models import RequestRow, RoutingEventRow, VerificationRow
from backend.events.bus import EventBus
from backend.events.types import EventType
from backend.providers.mock_provider import MockProvider
from backend.verification.engine import JudgeEngine
from backend.verification.judge import LLMJudge
from backend.verification.service import VerificationService
from backend.verification.status import VerificationStatus


def _valid_judge_json() -> str:
    return json.dumps({
        "correctness": 0.9, "completeness": 0.9, "instruction_following": 0.9,
        "format_adherence": 0.9, "confidence": 0.9, "rationale": "Good answer.",
    })


def _seed_request_and_routing_event(session_factory, request_id: str) -> None:
    with session_factory() as session:
        session.add(RequestRow(request_id=request_id, prompt="What is 2+2?", strategy="balanced"))
        session.add(RoutingEventRow(
            request_id=request_id, complexity="simple", confidence=0.9,
            selected_model="gpt-4o-mini", selected_strategy="balanced",
            estimated_cost=0.001, estimated_latency_ms=450, reasoning="[]",
        ))
        session.commit()


def _make_service(tmp_path, provider_response: str, event_bus: EventBus | None = None):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    provider = MockProvider(response=provider_response)
    judge = LLMJudge(provider=provider, model="gpt-4o", pass_threshold=0.7)
    judge_engine = JudgeEngine(judge=judge, judge_model_id="gpt-4o")

    service = VerificationService(
        judge_engine=judge_engine,
        session_factory=session_factory,
        event_bus=event_bus or EventBus(),
        judge_prompt_version="v1",
    )
    return service, session_factory


@pytest.mark.asyncio
async def test_verify_completes_and_snapshots_routing(tmp_path):
    service, session_factory = _make_service(tmp_path, _valid_judge_json())
    _seed_request_and_routing_event(session_factory, "req-1")

    await service.verify("req-1", "What is 2+2?", "4")

    with session_factory() as session:
        row = session.query(VerificationRow).filter_by(request_id="req-1").one()
        assert row.status == VerificationStatus.COMPLETED.value
        assert row.score == pytest.approx(0.9)
        assert row.passed is True
        assert row.judge_model == "gpt-4o"
        assert row.judge_prompt_version == "v1"
        assert row.evaluation_duration_ms >= 0
        assert row.routing_model == "gpt-4o-mini"
        assert row.routing_strategy == "balanced"
        assert row.routing_complexity == "simple"
        assert row.started_at is not None
        assert row.completed_at is not None


@pytest.mark.asyncio
async def test_verify_persists_failed_row_on_malformed_judge_output(tmp_path):
    service, session_factory = _make_service(tmp_path, "not valid json")
    _seed_request_and_routing_event(session_factory, "req-2")

    await service.verify("req-2", "What is 2+2?", "4")

    with session_factory() as session:
        row = session.query(VerificationRow).filter_by(request_id="req-2").one()
        assert row.status == VerificationStatus.FAILED.value
        assert row.error_type == "ValidationError"
        assert row.error is not None
        assert row.score is None
        assert row.completed_at is not None


@pytest.mark.asyncio
async def test_verify_emits_events_in_order_after_persistence(tmp_path):
    events: list[tuple[str, dict]] = []
    bus = EventBus()
    bus.subscribe(EventType.VERIFICATION_STARTED, lambda p: events.append(("started", p)))
    bus.subscribe(EventType.VERIFICATION_COMPLETED, lambda p: events.append(("completed", p)))

    service, session_factory = _make_service(tmp_path, _valid_judge_json(), event_bus=bus)
    _seed_request_and_routing_event(session_factory, "req-3")

    await service.verify("req-3", "What is 2+2?", "4")

    assert [name for name, _ in events] == ["started", "completed"]
    assert events[1][1]["score"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_verify_never_raises_on_judge_failure(tmp_path):
    service, session_factory = _make_service(tmp_path, "not valid json")
    _seed_request_and_routing_event(session_factory, "req-4")

    await service.verify("req-4", "prompt", "response")  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_verification_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.verification.service'`

- [ ] **Step 3: Write the implementation**

```python
# backend/verification/service.py
from datetime import datetime, timezone

from sqlalchemy.orm import sessionmaker

from backend.database.models import RoutingEventRow, VerificationRow
from backend.events.bus import EventBus
from backend.events.types import EventType
from backend.verification.engine import JudgeEngine
from backend.verification.events import VerificationCompleted, VerificationFailed, VerificationStarted
from backend.verification.status import VerificationStatus


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class _RoutingSnapshot:
    def __init__(self, selected_model: str, strategy: str, complexity: str) -> None:
        self.selected_model = selected_model
        self.strategy = strategy
        self.complexity = complexity


class VerificationService:
    def __init__(
        self,
        judge_engine: JudgeEngine,
        session_factory: sessionmaker,
        event_bus: EventBus,
        judge_prompt_version: str,
    ) -> None:
        self._judge_engine = judge_engine
        self._session_factory = session_factory
        self._event_bus = event_bus
        self._judge_prompt_version = judge_prompt_version

    async def verify(self, request_id: str, prompt: str, response: str) -> None:
        routing = self._load_routing_snapshot(request_id)

        with self._session_factory() as session:
            session.add(VerificationRow(
                request_id=request_id,
                status=VerificationStatus.PENDING.value,
                routing_model=routing.selected_model,
                routing_strategy=routing.strategy,
                routing_complexity=routing.complexity,
            ))
            session.commit()

        with self._session_factory() as session:
            row = session.query(VerificationRow).filter_by(request_id=request_id).one()
            row.status = VerificationStatus.RUNNING.value
            row.started_at = _utcnow()
            session.commit()

        self._event_bus.emit(
            EventType.VERIFICATION_STARTED, VerificationStarted(request_id=request_id).model_dump()
        )

        try:
            verdict, duration_ms = await self._judge_engine.run(prompt, response)
        except Exception as exc:
            with self._session_factory() as session:
                row = session.query(VerificationRow).filter_by(request_id=request_id).one()
                row.status = VerificationStatus.FAILED.value
                row.error_type = type(exc).__name__
                row.error = str(exc)
                row.completed_at = _utcnow()
                session.commit()
            self._event_bus.emit(
                EventType.VERIFICATION_FAILED,
                VerificationFailed(
                    request_id=request_id, error_type=type(exc).__name__, error=str(exc)
                ).model_dump(),
            )
            return

        with self._session_factory() as session:
            row = session.query(VerificationRow).filter_by(request_id=request_id).one()
            row.status = VerificationStatus.COMPLETED.value
            row.score = verdict.score
            row.passed = verdict.passed
            row.confidence = verdict.confidence
            row.rationale = verdict.rationale
            row.dimensions = verdict.dimensions.model_dump()
            row.judge_model = self._judge_engine.judge_model_id
            row.judge_prompt_version = self._judge_prompt_version
            row.evaluation_duration_ms = duration_ms
            row.completed_at = _utcnow()
            session.commit()

        self._event_bus.emit(
            EventType.VERIFICATION_COMPLETED,
            VerificationCompleted(request_id=request_id, score=verdict.score).model_dump(),
        )

    def _load_routing_snapshot(self, request_id: str) -> _RoutingSnapshot:
        with self._session_factory() as session:
            event = session.query(RoutingEventRow).filter_by(request_id=request_id).one()
            return _RoutingSnapshot(
                selected_model=event.selected_model,
                strategy=event.selected_strategy,
                complexity=event.complexity,
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_verification_service.py -v`
Expected: PASS (4 tests)

### Task 32: ChatService Integration

**Files:**
- Modify: `backend/chat/service.py`
- Modify: `backend/api/routers/chat.py`
- Test: `backend/tests/test_chat_service.py` (create if it doesn't already exist)

**Interfaces:**
- Consumes: `VerificationService` (Task 31).
- Produces: `ChatService.__init__` gains `verification_service: VerificationService`; `ChatService.chat(prompt, strategy, background_tasks: BackgroundTasks) -> ChatResult`. Consumed by `api/routers/chat.py`, `main.py` (Task 34).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_chat_service.py
import textwrap

import pytest
from fastapi import BackgroundTasks

from backend.analysis.prompt_analyzer import PromptAnalyzer
from backend.chat.service import ChatService
from backend.classifier.complexity_classifier import HeuristicComplexityClassifier
from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.events.bus import EventBus
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
from backend.verification.engine import JudgeEngine
from backend.verification.judge import LLMJudge
from backend.verification.service import VerificationService

ONE_MODEL_YAML = textwrap.dedent("""
    models:
      - id: gpt-4o-mini
        provider: mock
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
""")


class _FailingVerificationService:
    async def verify(self, request_id, prompt, response):
        raise RuntimeError("verification exploded")


def _make_chat_service(tmp_path, verification_service=None):
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(ONE_MODEL_YAML)

    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

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
        routing_policy=RoutingPolicy({"simple": EligibilityPolicy(min_benchmark_score=0.0)}),
        strategies={"balanced": BalancedStrategy(BalancedStrategyWeights())},
        explanation_generator=ExplanationGenerator(),
    )

    if verification_service is None:
        judge = LLMJudge(provider=MockProvider(response="{}"), model="mock", pass_threshold=0.7)
        judge_engine = JudgeEngine(judge=judge, judge_model_id="mock")
        verification_service = VerificationService(
            judge_engine=judge_engine, session_factory=session_factory,
            event_bus=EventBus(), judge_prompt_version="v1",
        )

    return ChatService(
        routing_engine=routing_engine,
        provider_manager=provider_manager,
        model_registry=model_registry,
        session_factory=session_factory,
        verification_service=verification_service,
    )


@pytest.mark.asyncio
async def test_chat_schedules_verification_background_task(tmp_path):
    chat_service = _make_chat_service(tmp_path)
    background_tasks = BackgroundTasks()

    result = await chat_service.chat("Hello.", strategy="balanced", background_tasks=background_tasks)

    assert result.response
    assert len(background_tasks.tasks) == 1


@pytest.mark.asyncio
async def test_chat_succeeds_even_if_verification_service_would_fail(tmp_path):
    chat_service = _make_chat_service(tmp_path, verification_service=_FailingVerificationService())
    background_tasks = BackgroundTasks()

    result = await chat_service.chat("Hello.", strategy="balanced", background_tasks=background_tasks)

    assert result.response
    # The scheduled background task itself would raise if awaited directly,
    # but chat() must return successfully regardless -- scheduling never fails
    # here since add_task() only registers the call, it doesn't invoke it.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_chat_service.py -v`
Expected: FAIL — `TypeError: ChatService.__init__() got an unexpected keyword argument 'verification_service'`

- [ ] **Step 3: Modify `backend/chat/service.py`**

Change:
```python
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
```

To:
```python
import json
import uuid

from fastapi import BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker

from backend.database.models import RequestRow, ResponseRow, RoutingEventRow
from backend.providers.base import ProviderError
from backend.providers.manager import ProviderManager
from backend.routing.engine import RoutingDecision, RoutingEngine
from backend.services.model_registry import ModelRegistry
from backend.telemetry.logging import get_logger
from backend.verification.service import VerificationService


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
        verification_service: VerificationService,
    ) -> None:
        self._routing_engine = routing_engine
        self._provider_manager = provider_manager
        self._model_registry = model_registry
        self._session_factory = session_factory
        self._verification_service = verification_service
        self._logger = get_logger("chat")

    async def chat(
        self, prompt: str, strategy: str, background_tasks: BackgroundTasks
    ) -> ChatResult:
```

Then, immediately before the existing final `return ChatResult(...)` line, insert:
```python
        try:
            background_tasks.add_task(
                self._verification_service.verify, request_id, prompt, response_text
            )
        except Exception:
            self._logger.exception(
                "verification_scheduling_failed", extra={"request_id": request_id}
            )

```

(The rest of `chat()` — routing, persistence, provider call, response persistence — is unchanged from Phase 2.)

- [ ] **Step 4: Modify `backend/api/routers/chat.py`**

Change:
```python
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

To:
```python
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException
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
async def chat(
    request: ChatRequest, chat_service: ChatServiceDep, background_tasks: BackgroundTasks
) -> ChatResult:
    try:
        return await chat_service.chat(
            request.prompt, strategy=request.strategy, background_tasks=background_tasks
        )
    except NoEligibleModelError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_chat_service.py -v`
Expected: PASS (2 tests)

- [ ] **Batch 3 verification & commit**

```bash
uv run pytest -v
```
Expected: all tests pass (191 + 4 + 2 = 197).

```bash
git add backend/verification/service.py backend/chat/service.py backend/api/routers/chat.py backend/tests/test_verification_service.py backend/tests/test_chat_service.py
git commit -m "feat: add VerificationService and wire best-effort verification scheduling into ChatService"
```

---

## Batch 4: API Endpoints, Wiring & Manual Verification

### Task 33: Verification & Metrics Endpoints

**Files:**
- Create: `backend/api/routers/verification.py`
- Create: `backend/api/routers/metrics.py`
- Test: `backend/tests/test_verification_router.py`
- Test: `backend/tests/test_metrics_router.py`

**Interfaces:**
- Consumes: `VerificationRow` (Task 30), `VerificationDimensions` (Task 28), `SessionFactoryDep` (Phase 1, `backend/api/dependencies.py`, unchanged — no modification needed here).
- Produces: `VerificationResult` (response model), `GET /v1/chat/{request_id}/verification`, `QualityMetrics` (response model), `GET /v1/metrics/quality`. Consumed by `main.py` (Task 34).

Both routers are written now but not yet registered in `create_app()` — that happens in Task 34, alongside the rest of the wiring. Their tests will still fail with 404 until Task 34 lands; this is expected and called out below.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_verification_router.py
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.database.models import RequestRow, VerificationRow
from backend.verification.status import VerificationStatus


def test_get_verification_returns_completed_result(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()
    with TestClient(app) as client:
        session_factory = app.state.session_factory
        with session_factory() as session:
            session.add(RequestRow(request_id="req-1", prompt="hi", strategy="balanced"))
            session.add(VerificationRow(
                request_id="req-1", status=VerificationStatus.COMPLETED.value,
                routing_model="gpt-4o-mini", routing_strategy="balanced", routing_complexity="simple",
                score=0.9, passed=True, confidence=0.85, rationale="Good.",
                dimensions={
                    "correctness": 0.9, "completeness": 0.9,
                    "instruction_following": 0.9, "format_adherence": 0.9,
                },
                judge_model="gpt-4o", judge_prompt_version="v1", evaluation_duration_ms=120,
                started_at=datetime.now(timezone.utc), completed_at=datetime.now(timezone.utc),
            ))
            session.commit()

        response = client.get("/v1/chat/req-1/verification")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "completed"
        assert body["score"] == 0.9
        assert body["dimensions"]["correctness"] == 0.9
        assert "raw_judge_response" not in body


def test_get_verification_404_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/v1/chat/does-not-exist/verification")
        assert response.status_code == 404
```

```python
# backend/tests/test_metrics_router.py
import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.database.models import RequestRow, VerificationRow
from backend.verification.status import VerificationStatus


def _seed(session_factory):
    with session_factory() as session:
        for i, (model, strategy, complexity, score) in enumerate([
            ("gpt-4o-mini", "balanced", "simple", 0.9),
            ("gpt-4o-mini", "balanced", "simple", 0.5),
            ("gpt-4o", "quality", "complex", 0.95),
        ]):
            request_id = f"req-{i}"
            session.add(RequestRow(request_id=request_id, prompt="hi", strategy=strategy))
            session.add(VerificationRow(
                request_id=request_id, status=VerificationStatus.COMPLETED.value,
                routing_model=model, routing_strategy=strategy, routing_complexity=complexity,
                score=score, passed=score >= 0.7, confidence=0.8,
                evaluation_duration_ms=100,
            ))
        session.add(RequestRow(request_id="req-failed", prompt="hi", strategy="balanced"))
        session.add(VerificationRow(
            request_id="req-failed", status=VerificationStatus.FAILED.value,
            routing_model="gpt-4o-mini", routing_strategy="balanced", routing_complexity="simple",
            error_type="ValidationError", error="bad json",
        ))
        session.commit()


def test_quality_metrics_aggregates(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    app = create_app()
    with TestClient(app) as client:
        _seed(app.state.session_factory)

        response = client.get("/v1/metrics/quality")

        assert response.status_code == 200
        body = response.json()
        assert body["total_verified"] == 3
        assert body["verification_failure_count"] == 1
        assert body["pass_rate"] == pytest.approx(2 / 3)
        assert body["by_model"]["gpt-4o-mini"] == pytest.approx((0.9 + 0.5) / 2)
        assert body["by_model"]["gpt-4o"] == pytest.approx(0.95)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest backend/tests/test_verification_router.py backend/tests/test_metrics_router.py -v`
Expected: FAIL — `404 Not Found` for both routes (routers not yet registered in `create_app()`)

- [ ] **Step 3: Write `backend/api/routers/verification.py`**

```python
# backend/api/routers/verification.py
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.api.dependencies import SessionFactoryDep
from backend.database.models import VerificationRow
from backend.verification.judge import VerificationDimensions
from backend.verification.status import VerificationStatus

router = APIRouter()


class VerificationResult(BaseModel):
    request_id: str
    status: VerificationStatus
    score: float | None
    passed: bool | None
    confidence: float | None
    rationale: str | None
    dimensions: VerificationDimensions | None
    judge_model: str | None
    judge_prompt_version: str | None
    evaluation_duration_ms: int | None
    error_type: str | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


@router.get("/chat/{request_id}/verification", response_model=VerificationResult)
async def get_verification(request_id: str, session_factory: SessionFactoryDep) -> VerificationResult:
    with session_factory() as session:
        row = session.query(VerificationRow).filter_by(request_id=request_id).first()
        if row is None:
            raise HTTPException(status_code=404, detail=f"No verification found for '{request_id}'")

        return VerificationResult(
            request_id=row.request_id,
            status=VerificationStatus(row.status),
            score=row.score,
            passed=row.passed,
            confidence=row.confidence,
            rationale=row.rationale,
            dimensions=VerificationDimensions(**row.dimensions) if row.dimensions else None,
            judge_model=row.judge_model,
            judge_prompt_version=row.judge_prompt_version,
            evaluation_duration_ms=row.evaluation_duration_ms,
            error_type=row.error_type,
            error=row.error,
            created_at=row.created_at,
            started_at=row.started_at,
            completed_at=row.completed_at,
        )
```

- [ ] **Step 4: Write `backend/api/routers/metrics.py`**

```python
# backend/api/routers/metrics.py
from fastapi import APIRouter
from pydantic import BaseModel

from backend.api.dependencies import SessionFactoryDep
from backend.database.models import VerificationRow
from backend.verification.status import VerificationStatus

router = APIRouter()


class QualityMetrics(BaseModel):
    total_verified: int
    average_score: float
    average_confidence: float
    pass_rate: float
    average_queue_delay_ms: float
    average_evaluation_duration_ms: float
    average_total_verification_ms: float
    verification_failure_count: int
    by_model: dict[str, float]
    by_strategy: dict[str, float]
    by_complexity: dict[str, float]


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _group_avg(rows: list[VerificationRow], key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(getattr(row, key), []).append(row.score)
    return {name: _avg(scores) for name, scores in grouped.items()}


@router.get("/metrics/quality", response_model=QualityMetrics)
async def get_quality_metrics(session_factory: SessionFactoryDep) -> QualityMetrics:
    with session_factory() as session:
        completed = (
            session.query(VerificationRow)
            .filter_by(status=VerificationStatus.COMPLETED.value)
            .all()
        )
        failure_count = (
            session.query(VerificationRow)
            .filter_by(status=VerificationStatus.FAILED.value)
            .count()
        )

    queue_delays = [
        (row.started_at - row.created_at).total_seconds() * 1000
        for row in completed
        if row.started_at is not None
    ]
    total_durations = [
        (row.completed_at - row.started_at).total_seconds() * 1000
        for row in completed
        if row.started_at is not None and row.completed_at is not None
    ]
    eval_durations = [
        row.evaluation_duration_ms for row in completed if row.evaluation_duration_ms is not None
    ]

    return QualityMetrics(
        total_verified=len(completed),
        average_score=_avg([row.score for row in completed]),
        average_confidence=_avg([row.confidence for row in completed if row.confidence is not None]),
        pass_rate=_avg([1.0 if row.passed else 0.0 for row in completed]),
        average_queue_delay_ms=_avg(queue_delays),
        average_evaluation_duration_ms=_avg(eval_durations),
        average_total_verification_ms=_avg(total_durations),
        verification_failure_count=failure_count,
        by_model=_group_avg(completed, "routing_model"),
        by_strategy=_group_avg(completed, "routing_strategy"),
        by_complexity=_group_avg(completed, "routing_complexity"),
    )
```

- [ ] **Step 5: Confirm tests still fail with 404 (routers not yet registered)**

Run: `uv run pytest backend/tests/test_verification_router.py backend/tests/test_metrics_router.py -v`
Expected: FAIL — both return `404` because `create_app()` doesn't include these routers yet. This is expected; Task 34 wires them in and re-runs these same files to confirm PASS.

### Task 34: Wire Everything Into `main.py` and Tag `v0.3.0`

**Files:**
- Modify: `backend/api/main.py`
- Test: `backend/tests/test_verification_integration.py`

**Interfaces:**
- Consumes: everything from Tasks 27-33.
- Produces: a fully wired app where `/v1/chat` schedules verification, `/v1/chat/{request_id}/verification` and `/v1/metrics/quality` are live.

- [ ] **Step 1: Write the failing integration test**

```python
# backend/tests/test_verification_integration.py
import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app


def test_chat_then_verification_completes_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    judge_json = json.dumps({
        "correctness": 0.9, "completeness": 0.9, "instruction_following": 0.9,
        "format_adherence": 0.9, "confidence": 0.9, "rationale": "Good answer.",
    })

    app = create_app()
    with (
        patch(
            "backend.providers.openai_provider.OpenAIProvider.health_check",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "backend.providers.openai_provider.OpenAIProvider.generate",
            new=AsyncMock(side_effect=["The answer is 4.", judge_json]),
        ),
        TestClient(app) as client,
    ):
        chat_response = client.post(
            "/v1/chat", json={"prompt": "What is 2+2?", "strategy": "balanced"}
        )
        assert chat_response.status_code == 200
        request_id = chat_response.json()["request_id"]

        verification_response = client.get(f"/v1/chat/{request_id}/verification")
        assert verification_response.status_code == 200
        body = verification_response.json()
        assert body["status"] == "completed"
        assert body["score"] == pytest.approx(0.9)

        metrics_response = client.get("/v1/metrics/quality")
        assert metrics_response.status_code == 200
        assert metrics_response.json()["total_verified"] == 1
```

Note: since the judge model in `backend/config/verification.yaml` (`gpt-4o`) and the routing model registered under `provider: openai` both resolve through `OpenAIProvider`, and the mocked `generate` returns responses via `side_effect` in call order, the first `generate` call (the chat response) receives `"The answer is 4."` and the second (the judge call, triggered by the scheduled background task) receives `judge_json`. `TestClient`'s context manager executes `BackgroundTasks` synchronously as part of the request/response cycle, so the verification is guaranteed complete by the time the test issues its second request.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_verification_integration.py -v`
Expected: FAIL — `404 Not Found` for `/v1/chat/{request_id}/verification` (router not yet registered) and/or `TypeError` on `ChatService(...)` missing `verification_service`

- [ ] **Step 3: Modify `backend/api/main.py`**

Change the import block:
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

APP_VERSION = "0.1.0"
```

To:
```python
from backend.analysis.prompt_analyzer import PromptAnalyzer
from backend.api.routers.chat import router as chat_router
from backend.api.routers.health import router as health_router
from backend.api.routers.metrics import router as metrics_router
from backend.api.routers.models import router as models_router
from backend.api.routers.verification import router as verification_router
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
from backend.verification.config_loader import VerificationConfigLoader
from backend.verification.engine import JudgeEngine
from backend.verification.judge import LLMJudge
from backend.verification.service import VerificationService

APP_VERSION = "0.3.0"
```

Change the `lifespan` body from:
```python
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
```

To:
```python
    routing_engine = RoutingEngine(
        model_registry=model_registry,
        analyzer=PromptAnalyzer(),
        classifier=classifier,
        routing_policy=routing_policy,
        strategies=strategies,
        explanation_generator=ExplanationGenerator(),
    )

    verification_config = VerificationConfigLoader.load(settings.verification_config_path)
    judge_provider = provider_manager.get_provider(
        model_registry.get_model(verification_config.judge_model_id).provider
    )
    judge = LLMJudge(
        provider=judge_provider,
        model=verification_config.judge_model_id,
        pass_threshold=verification_config.pass_threshold,
    )
    judge_engine = JudgeEngine(judge=judge, judge_model_id=verification_config.judge_model_id)
    verification_service = VerificationService(
        judge_engine=judge_engine,
        session_factory=session_factory,
        event_bus=event_bus,
        judge_prompt_version=verification_config.judge_prompt_version,
    )

    chat_service = ChatService(
        routing_engine=routing_engine,
        provider_manager=provider_manager,
        model_registry=model_registry,
        session_factory=session_factory,
        verification_service=verification_service,
    )
```

Change `create_app()` from:
```python
def create_app() -> FastAPI:
    app = FastAPI(title="LLM Cost Autopilot", version=APP_VERSION, lifespan=lifespan)
    app.include_router(health_router, prefix="/v1")
    app.include_router(models_router, prefix="/v1")
    app.include_router(chat_router, prefix="/v1")
    return app
```

To:
```python
def create_app() -> FastAPI:
    app = FastAPI(title="LLM Cost Autopilot", version=APP_VERSION, lifespan=lifespan)
    app.include_router(health_router, prefix="/v1")
    app.include_router(models_router, prefix="/v1")
    app.include_router(chat_router, prefix="/v1")
    app.include_router(verification_router, prefix="/v1")
    app.include_router(metrics_router, prefix="/v1")
    return app
```

Note: `verification_config.judge_model_id` (`gpt-4o`) must exist in `backend/config/models.yaml` for `model_registry.get_model(...)` to resolve its provider. If `gpt-4o` is not already present in that file from Phase 2's test fixtures, add it to the real `backend/config/models.yaml` now (matching the `gpt-4o` entry already used in Phase 2's `test_routing_engine.py` fixture — `provider: openai`, `input_cost: 2.50`, `output_cost: 10.00`, `benchmark_score: 0.93`, `average_latency_ms: 900`, `context_window: 128000`, `max_output_tokens: 16384`, all capabilities `true` except none excluded).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_verification_integration.py backend/tests/test_verification_router.py backend/tests/test_metrics_router.py -v`
Expected: PASS (1 + 2 + 1 = 4 tests)

- [ ] **Step 5: Run full regression suite**

```bash
uv run pytest -v
```
Expected: all tests pass (197 + 4 = 201, plus the 2 router tests from Task 33 that were pending on this task's wiring — verify the exact collected count against actual output rather than assuming it matches precisely, since it depends on final file assembly).

- [ ] **Step 6: Manual end-to-end verification**

Run the real server:
```bash
uv run uvicorn backend.api.main:app --reload
```

In another terminal:
```bash
curl -s http://localhost:8000/v1/health | python3 -m json.tool
curl -s -X POST http://localhost:8000/v1/chat -H "Content-Type: application/json" \
  -d '{"prompt": "What is 2+2?", "strategy": "balanced"}' | python3 -m json.tool
```

Without a real `OPENAI_API_KEY` configured, expect a `503` from `/v1/chat` (`NoEligibleModelError`), consistent with Phase 2's manual verification — confirms the app still starts and routes correctly with Phase 3 wired in. If a real key is available, additionally poll:
```bash
curl -s http://localhost:8000/v1/chat/<request_id>/verification | python3 -m json.tool
curl -s http://localhost:8000/v1/metrics/quality | python3 -m json.tool
```
and confirm the verification transitions from absent/`pending` to `completed` with a populated `score` shortly after the chat call returns.

- [ ] **Step 7: Update docs and tag `v0.3.0`**

Update whichever doc Phase 2 updated (`README.md`/`ARCHITECTURE.md`) with a short Phase 3 section describing the verification pipeline and the two new endpoints, following the same doc-update pattern Phase 2 used in its closing batch.

```bash
git add backend/api/routers/verification.py backend/api/routers/metrics.py backend/api/main.py backend/config/models.yaml backend/tests/test_verification_router.py backend/tests/test_metrics_router.py backend/tests/test_verification_integration.py README.md
git commit -m "feat: wire verification pipeline into main.py, add verification/metrics endpoints"
git tag v0.3.0
```
