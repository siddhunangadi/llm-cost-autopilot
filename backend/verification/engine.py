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
