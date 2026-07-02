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
