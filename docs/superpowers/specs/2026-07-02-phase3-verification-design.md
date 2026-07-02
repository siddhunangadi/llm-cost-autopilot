# LLM Cost Autopilot — Phase 3 Design: Quality Verification & Evaluation

Status: **Approved — frozen as implementation contract**
Date: 2026-07-02

## 1. Purpose & Scope

Phase 3 answers one question: **was the response actually good?** After
`ChatService` (Phase 2) persists a response, an in-process background task
scores it with an LLM-as-judge and persists the verdict, without ever
adding latency to `/v1/chat` or risking its availability.

```
ChatService.chat()
  -> persists request/routing_event/response  (Phase 2, unchanged)
  -> schedules background task: VerificationService.verify(request_id, prompt, response)
  -> returns ChatResult to client immediately

VerificationService.verify()
  -> loads routing snapshot (selected_model, strategy, complexity) from RoutingEventRow
  -> PENDING -> RUNNING -> JudgeEngine.run() -> persist -> COMPLETED | FAILED
  -> emits typed events via the existing EventBus

GET /v1/chat/{request_id}/verification -> single VerificationRow
GET /v1/metrics/quality                -> aggregate QualityMetrics
```

**In scope:**
- `BaseJudge` interface + `LLMJudge` implementation (pure: prompt+response
  in, `JudgeVerdict` out — no persistence, no retries, no events)
- `JudgeEngine` (orchestrates a `BaseJudge` call, measures duration)
- `VerificationService` (owns the DB lifecycle, event emission, routing
  snapshot capture)
- `VerificationRow` table, `VerificationStatus` enum
- Typed verification event payloads (`VerificationStarted`,
  `VerificationCompleted`, `VerificationFailed`) emitted through the
  existing `EventBus`
- `GET /v1/chat/{request_id}/verification` and `GET /v1/metrics/quality`
- In-process background execution via FastAPI `BackgroundTasks` — no new
  infra

**Explicitly out of scope for Phase 3** (deferred to a later
"Self-Improvement" phase):
- Auto-escalation on low quality scores
- Classifier retraining / online learning
- Feedback loop back into routing decisions
- Prompt optimization
- Retry policies for failed verifications
- A separate worker process, message queue, or scheduler (Celery, Redis,
  etc.)
- Judging routing appropriateness (model choice vs. cost) — the judge only
  scores response quality, not whether a cheaper/more expensive model
  would have been better

## 2. Directory Structure

```
backend/
  verification/
    __init__.py
    judge.py            # BaseJudge, LLMJudge, JudgeVerdict, VerificationDimensions
    engine.py             # JudgeEngine
    service.py              # VerificationService
    events.py                # VerificationStarted, VerificationCompleted, VerificationFailed
  api/
    routers/
      verification.py          # GET /v1/chat/{request_id}/verification
      metrics.py                 # GET /v1/metrics/quality
  config/
    verification.yaml            # judge_model, pass_threshold, judge_prompt_version
  database/
    models.py                    # + VerificationRow (modify)
  events/
    types.py                     # + VERIFICATION_STARTED/COMPLETED/FAILED (modify)
  chat/
    service.py                   # + schedules verification (modify)
```

## 3. Configuration

### 3.1 `backend/config/verification.yaml`

```yaml
judge_model_id: gpt-4o
pass_threshold: 0.7
judge_prompt_version: v1
```

### 3.2 `Settings` addition

`Settings` gains `verification_config_path: str = Field(default="backend/config/verification.yaml", min_length=1)`, matching the `routing_config_path` precedent from Phase 2 — `Settings` only carries the path, it does not parse the file.

### 3.3 `VerificationConfig` / `VerificationConfigLoader`

Same pattern as `RoutingConfigLoader` (Phase 2 §3.3) — one loader, one parse:

```python
class VerificationConfig(BaseModel):
    judge_model_id: str
    pass_threshold: float
    judge_prompt_version: str

class VerificationConfigLoader:
    @staticmethod
    def load(yaml_path: str) -> VerificationConfig:
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        return VerificationConfig.model_validate(raw)
```

Fails fast on malformed YAML (`yaml.YAMLError`) or invalid schema
(`pydantic.ValidationError`), loaded once at startup, no runtime reload —
same discipline as Phase 1/2.

## 4. Judge Verdict & Dimensions

```python
class VerificationDimensions(BaseModel):
    correctness: float
    completeness: float
    instruction_following: float
    format_adherence: float

class JudgeVerdict(BaseModel):
    score: float          # mean of the four dimensions, computed by LLMJudge, never asked of the model
    passed: bool           # score >= pass_threshold
    confidence: float
    rationale: str
    dimensions: VerificationDimensions
```

A typed model (not `dict[str, float]`) for `dimensions` guarantees
consistent keys across every verdict, which the metrics aggregation and
any future dashboard depend on. Adding a fifth dimension later means
adding a field here and bumping `judge_prompt_version` — not touching
consumers.

Every numeric field in `VerificationDimensions` and `JudgeVerdict.score`/
`confidence` **must be validated as being within `[0.0, 1.0]`** before a
`JudgeVerdict` is constructed (via a Pydantic `field_validator`). A value
outside that range means the judge model drifted from the requested
schema — `LLMJudge.evaluate()` raises `ValueError` in that case rather
than silently clamping, so `VerificationService` records it as a `FAILED`
verification (see §7) instead of persisting a suspicious score as if it
were trustworthy.

## 5. `BaseJudge` / `LLMJudge` (`backend/verification/judge.py`)

```python
class BaseJudge(ABC):
    @abstractmethod
    async def evaluate(self, prompt: str, response: str) -> JudgeVerdict: ...
```

`LLMJudge` is the only Phase 3 implementation; `BaseJudge` exists because
concrete alternatives are already anticipated (rule-based judge, human
review, benchmark replay), unlike Phase 2's `PromptAnalyzer`, which had no
planned alternative and so got no interface.

`LLMJudge` is called with the raw model text and immediately parses it
into an intermediate Pydantic model before touching `JudgeVerdict`:

```python
class _JudgeResponseSchema(BaseModel):
    correctness: float
    completeness: float
    instruction_following: float
    format_adherence: float
    confidence: float
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
    def __init__(self, provider: Provider, model: str, pass_threshold: float) -> None:
        self._provider = provider
        self._model = model
        self._pass_threshold = pass_threshold

    async def evaluate(self, prompt: str, response: str) -> JudgeVerdict:
        raw = await self._provider.generate(
            _JUDGE_PROMPT_TEMPLATE.format(prompt=prompt, response=response), model=self._model
        )
        parsed = _JudgeResponseSchema.model_validate_json(raw)  # raises pydantic.ValidationError on bad shape or out-of-range fields

        dimensions = VerificationDimensions(
            correctness=parsed.correctness, completeness=parsed.completeness,
            instruction_following=parsed.instruction_following,
            format_adherence=parsed.format_adherence,
        )
        score = (
            dimensions.correctness + dimensions.completeness
            + dimensions.instruction_following + dimensions.format_adherence
        ) / 4

        return JudgeVerdict(
            score=score, passed=score >= self._pass_threshold,
            confidence=parsed.confidence, rationale=parsed.rationale, dimensions=dimensions,
        )
```

Parsing through `_JudgeResponseSchema.model_validate_json()` instead of
`json.loads()` + manual dict access means malformed JSON, missing fields,
wrong types, *and* out-of-range floats (via a `field_validator` bounding
each score to `[0.0, 1.0]`) all raise the same `pydantic.ValidationError`
at one point, with a specific, inspectable error message — rather than a
bare `KeyError`/`ValueError` from manual dict indexing.

`LLMJudge` never writes to the database, never retries, never emits
events, and never knows about `request_id` — pure function of
`(prompt, response) -> JudgeVerdict`.

## 6. `JudgeEngine` (`backend/verification/engine.py`)

```python
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

`JudgeEngine` is the only place that measures timing — `BaseJudge`
implementations stay completely pure. `VerificationService` calls
`JudgeEngine`, never `BaseJudge` directly.

## 7. `VerificationService` (`backend/verification/service.py`)

### 7.1 `VerificationStatus` / `VerificationRow`

```python
class VerificationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
```

`VerificationRow` (`backend/database/models.py`):

```
id, request_id (FK -> requests.request_id),
status,
routing_model, routing_strategy, routing_complexity,   -- snapshot at verification start
score (nullable), passed (nullable), confidence (nullable),
rationale (nullable), dimensions (JSON, nullable),
judge_model (nullable), judge_prompt_version (nullable),
evaluation_duration_ms (nullable),
raw_judge_response (nullable, text),                    -- full raw judge output, debugging/audit only, never returned by any API
error_type (nullable), error (nullable),
created_at, started_at (nullable), completed_at (nullable)
```

`routing_model`/`routing_strategy`/`routing_complexity` are copied from
the corresponding `RoutingEventRow` at verification-creation time — a
verification result always reflects the routing decision that actually
produced the response, even if `routing.yaml` or strategy weights change
later.

`judge_prompt_version` (from `VerificationConfig.judge_prompt_version`,
e.g. `"v1"`) is distinct from `judge_model` (e.g. `"gpt-4o"`): the model
serving as judge and the prompt/schema used to instruct it evolve
independently, and both must be known to compare scores across time.

`raw_judge_response` stores the exact text returned by the judge model
before parsing (even on parse failure, where feasible) — never surfaced
through `GET /v1/chat/{request_id}/verification` or the metrics endpoint,
purely for manual inspection when a score looks anomalous.

### 7.2 Lifecycle

```python
class VerificationService:
    def __init__(
        self,
        judge_engine: JudgeEngine,
        session_factory: sessionmaker,
        event_bus: EventBus,
        judge_prompt_version: str,
    ) -> None: ...

    async def verify(self, request_id: str, prompt: str, response: str) -> None:
        routing = self._load_routing_snapshot(request_id)  # reads RoutingEventRow

        with self._session_factory() as session:
            session.add(VerificationRow(
                request_id=request_id, status=VerificationStatus.PENDING,
                routing_model=routing.selected_model, routing_strategy=routing.strategy,
                routing_complexity=routing.complexity,
            ))
            session.commit()

        with self._session_factory() as session:
            row = session.query(VerificationRow).filter_by(request_id=request_id).one()
            row.status = VerificationStatus.RUNNING
            row.started_at = datetime.utcnow()
            session.commit()

        self._event_bus.emit(
            EventType.VERIFICATION_STARTED, VerificationStarted(request_id=request_id).model_dump()
        )

        try:
            verdict, duration_ms = await self._judge_engine.run(prompt, response)
        except Exception as exc:
            with self._session_factory() as session:
                row = session.query(VerificationRow).filter_by(request_id=request_id).one()
                row.status = VerificationStatus.FAILED
                row.error_type = type(exc).__name__
                row.error = str(exc)
                row.completed_at = datetime.utcnow()
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
            row.status = VerificationStatus.COMPLETED
            row.score = verdict.score
            row.passed = verdict.passed
            row.confidence = verdict.confidence
            row.rationale = verdict.rationale
            row.dimensions = verdict.dimensions.model_dump()
            row.judge_model = self._judge_engine.judge_model_id
            row.judge_prompt_version = self._judge_prompt_version
            row.evaluation_duration_ms = duration_ms
            row.completed_at = datetime.utcnow()
            session.commit()

        self._event_bus.emit(
            EventType.VERIFICATION_COMPLETED,
            VerificationCompleted(request_id=request_id, score=verdict.score).model_dump(),
        )
```

Rules, all non-negotiable for Phase 3:
- **Every DB transaction is independent** — no ORM instance is held or
  mutated across a `with self._session_factory()` boundary; each block
  opens, queries fresh, commits, and closes.
- **Persist before emit** — both the `FAILED` and `COMPLETED` branches
  write the row, then publish the event. A subscriber never observes an
  event for a state that isn't durable.
- **No skipped states** — `PENDING -> RUNNING -> (COMPLETED | FAILED)`,
  always.
- **`verify()` swallows all judge-side exceptions itself** — nothing
  propagates to the caller (`ChatService`'s scheduled background task).

### 7.3 Typed event payloads (`backend/verification/events.py`)

```python
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

Three new `EventType` enum members
(`VERIFICATION_STARTED`/`COMPLETED`/`FAILED`, `backend/events/types.py`).
Payloads are constructed as typed Pydantic models and passed into the
existing `EventBus.emit(event_type: EventType, payload: dict)` via
`.model_dump()` — the Phase 1 `EventBus` interface itself is unchanged,
preserving compatibility with every existing Phase 1/2 subscriber, while
verification code gets full type safety when constructing its own
payloads.

## 8. `ChatService` Integration (`backend/chat/service.py`, modified)

Verification scheduling is a **best-effort side effect** — it must never
affect the chat response:

```python
    async def chat(self, prompt: str, strategy: str, background_tasks: BackgroundTasks) -> ChatResult:
        # ... existing Phase 2 logic: route, call provider, persist response ...
        try:
            background_tasks.add_task(
                self._verification_service.verify, request_id, prompt, response_text
            )
        except Exception:
            self._logger.exception("verification_scheduling_failed", extra={"request_id": request_id})
        return ChatResult(request_id=request_id, response=response_text, routing=decision)
```

`ChatService.chat()` gains a `background_tasks: BackgroundTasks`
parameter; `POST /v1/chat` (`backend/api/routers/chat.py`) accepts a
`BackgroundTasks` parameter via FastAPI's built-in injection and passes it
through. A failure at `add_task()` (essentially impossible in practice,
but the boundary is made explicit) is logged, never raised — the client
still receives their successful `ChatResult`. If `verify()` itself later
raises (it currently never does — see §7.2's blanket try/except), that
exception occurs after the HTTP response has already been sent and is
handled by FastAPI's background-task error logging, not by `ChatService`.

## 9. API Endpoints

### 9.1 `GET /v1/chat/{request_id}/verification` (`backend/api/routers/verification.py`)

```python
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
async def get_verification(request_id: str, ...) -> VerificationResult:
    ...  # 404 if no VerificationRow exists for request_id
```

`raw_judge_response` is deliberately excluded from `VerificationResult` —
debugging/audit data only, never returned over the API.

### 9.2 `GET /v1/metrics/quality` (`backend/api/routers/metrics.py`)

```python
class QualityMetrics(BaseModel):
    total_verified: int
    average_score: float
    average_confidence: float
    pass_rate: float
    average_queue_delay_ms: float           # started_at - created_at
    average_evaluation_duration_ms: float    # judge call time only
    average_total_verification_ms: float      # completed_at - started_at
    verification_failure_count: int
    by_model: dict[str, float]                # avg score
    by_strategy: dict[str, float]              # avg score
    by_complexity: dict[str, float]             # avg score
```

Computed via aggregate queries over `COMPLETED` rows for score/confidence/
pass-rate/timing fields, grouped by `routing_model`/`routing_strategy`/
`routing_complexity` for the breakdowns. `verification_failure_count` is a
separate `COUNT` over `FAILED` rows. `average_confidence` is included
specifically so a future phase can compare judge confidence against
actual score and Phase 2's classification confidence — no such comparison
is built in Phase 3, only the raw metric is exposed.

## 10. Testing

Same fail-fast/interface-enforcement discipline as Phases 1-2:
`BaseJudge` cannot be instantiated without `evaluate()`;
`VerificationConfigLoader` rejects malformed YAML and invalid schema;
`LLMJudge.evaluate()` is tested with a `MockProvider` returning fixed JSON
strings, covering valid output, malformed JSON, missing fields, and
out-of-range dimension values (each of the latter three asserting a
`pydantic.ValidationError`); `JudgeEngine.run()` is tested for both the
`(verdict, duration)` return shape and that duration is non-negative;
`VerificationService` is tested end-to-end against a real (SQLite) test
database for both the `COMPLETED` and `FAILED` paths, asserting exact
status transitions, that events fire in the correct order relative to
persistence, and that `routing_model`/`routing_strategy`/
`routing_complexity` are correctly snapshotted; `ChatService` tests assert
that a `VerificationService.verify` exception (simulated) never propagates
into the `ChatResult` response path; metrics endpoint tests seed known
`VerificationRow` fixtures and assert exact aggregate values.

## 11. Tooling

No new dependencies — `pydantic`, `pyyaml`, and FastAPI's built-in
`BackgroundTasks` (already a FastAPI dependency) cover everything in
Phase 3.
