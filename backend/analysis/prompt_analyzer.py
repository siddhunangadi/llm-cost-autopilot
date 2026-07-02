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
