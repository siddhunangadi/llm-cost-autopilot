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
