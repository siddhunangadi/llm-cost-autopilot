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
