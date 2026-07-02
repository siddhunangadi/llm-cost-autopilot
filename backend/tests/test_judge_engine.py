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
