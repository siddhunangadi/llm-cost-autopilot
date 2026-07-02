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
