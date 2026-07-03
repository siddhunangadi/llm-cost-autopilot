import pytest

from backend.learning.generator import RecommendationGenerator, Severity
from backend.learning.rules import Finding, RuleType


def test_generate_model_complexity_signature_and_text():
    finding = Finding(
        rule_type=RuleType.MODEL_COMPLEXITY, subject="gpt-4o-mini:medium",
        sample_size=20, pass_rate=0.35, threshold=0.6,
    )
    [rec] = RecommendationGenerator().generate([finding])

    assert rec.signature == "model_complexity:gpt-4o-mini:medium"
    assert "gpt-4o-mini" in rec.text
    assert "medium" in rec.text
    assert "35%" in rec.text
    assert rec.severity == Severity.HIGH  # pass_rate 0.35 < 0.4
    assert rec.evidence.sample_size == 20
    assert rec.evidence.pass_rate == 0.35
    assert rec.evidence.threshold == 0.6


def test_generate_model_complexity_medium_severity_above_0_4():
    finding = Finding(
        rule_type=RuleType.MODEL_COMPLEXITY, subject="gpt-4o-mini:medium",
        sample_size=20, pass_rate=0.55, threshold=0.6,
    )
    [rec] = RecommendationGenerator().generate([finding])
    assert rec.severity == Severity.MEDIUM


def test_generate_complexity_tier_signature_and_text():
    finding = Finding(
        rule_type=RuleType.COMPLEXITY_TIER, subject="complex",
        sample_size=30, pass_rate=0.25, threshold=0.5,
    )
    [rec] = RecommendationGenerator().generate([finding])

    assert rec.signature == "complexity_tier:complex"
    assert "complex" in rec.text
    assert rec.severity == Severity.HIGH  # pass_rate 0.25 < 0.3


def test_generate_complexity_tier_medium_severity_above_0_3():
    finding = Finding(
        rule_type=RuleType.COMPLEXITY_TIER, subject="complex",
        sample_size=30, pass_rate=0.45, threshold=0.5,
    )
    [rec] = RecommendationGenerator().generate([finding])
    assert rec.severity == Severity.MEDIUM


def test_evidence_confidence_scales_with_sample_size():
    small = Finding(rule_type=RuleType.COMPLEXITY_TIER, subject="complex", sample_size=30, pass_rate=0.4, threshold=0.5)
    large = Finding(rule_type=RuleType.COMPLEXITY_TIER, subject="complex", sample_size=300, pass_rate=0.4, threshold=0.5)

    [rec_small] = RecommendationGenerator().generate([small])
    [rec_large] = RecommendationGenerator().generate([large])

    assert rec_small.evidence_confidence == pytest.approx(0.5 + 30 / 200)
    assert rec_large.evidence_confidence == pytest.approx(0.95)  # capped


def test_generate_default_source_is_verification():
    finding = Finding(rule_type=RuleType.COMPLEXITY_TIER, subject="complex", sample_size=30, pass_rate=0.4, threshold=0.5)
    [rec] = RecommendationGenerator().generate([finding])
    assert rec.source.value == "verification"


from backend.learning.cost_metrics import ModelCostMetrics
from backend.learning.generator import ModelComparison, RecommendationSource


def _metrics(model, complexity, avg_cost, requests_per_day=10.0, pass_rate=0.9, eligible=True):
    return ModelCostMetrics(
        model=model, complexity=complexity, input_cost=1.0, output_cost=1.0,
        avg_cost_per_request=avg_cost, requests_per_day=requests_per_day,
        pass_rate=pass_rate, eligible_for_optimization=eligible,
    )


def test_generate_cost_optimization_picks_cheapest_eligible_alternative():
    finding = Finding(
        rule_type=RuleType.COST_OPTIMIZATION, subject="gpt-4o:complex",
        sample_size=20, pass_rate=0.9, threshold=0.7,
    )
    cost_metrics = {
        ("gpt-4o", "complex"): _metrics("gpt-4o", "complex", avg_cost=0.10),
        ("gpt-4o-mini", "complex"): _metrics("gpt-4o-mini", "complex", avg_cost=0.02, pass_rate=0.75),
        ("claude-3-haiku", "complex"): _metrics("claude-3-haiku", "complex", avg_cost=0.05, pass_rate=0.8),
    }

    [rec] = RecommendationGenerator().generate([finding], cost_metrics)

    assert rec.signature == "cost_optimization:gpt-4o:complex"
    assert rec.source == RecommendationSource.COST_OPTIMIZATION
    assert rec.evidence.comparison.suggested_model == "gpt-4o-mini"  # cheapest eligible, not claude


def test_generate_cost_optimization_computes_rounded_monthly_savings():
    finding = Finding(
        rule_type=RuleType.COST_OPTIMIZATION, subject="gpt-4o:complex",
        sample_size=20, pass_rate=0.9, threshold=0.7,
    )
    cost_metrics = {
        ("gpt-4o", "complex"): _metrics("gpt-4o", "complex", avg_cost=0.10, requests_per_day=10.0),
        ("gpt-4o-mini", "complex"): _metrics("gpt-4o-mini", "complex", avg_cost=0.03),
    }

    [rec] = RecommendationGenerator().generate([finding], cost_metrics)

    # (0.10 - 0.03) * 10 requests/day * 30 = 21.00
    assert rec.evidence.comparison.estimated_monthly_savings == pytest.approx(21.00)


def test_generate_cost_optimization_skips_when_current_model_is_cheapest():
    finding = Finding(
        rule_type=RuleType.COST_OPTIMIZATION, subject="gpt-4o-mini:complex",
        sample_size=20, pass_rate=0.9, threshold=0.7,
    )
    cost_metrics = {
        ("gpt-4o-mini", "complex"): _metrics("gpt-4o-mini", "complex", avg_cost=0.02),
        ("gpt-4o", "complex"): _metrics("gpt-4o", "complex", avg_cost=0.10),
    }

    recs = RecommendationGenerator().generate([finding], cost_metrics)

    assert recs == []


def test_generate_cost_optimization_skips_when_current_not_in_cost_metrics():
    finding = Finding(
        rule_type=RuleType.COST_OPTIMIZATION, subject="gpt-4o:complex",
        sample_size=20, pass_rate=0.9, threshold=0.7,
    )
    cost_metrics = {("gpt-4o-mini", "complex"): _metrics("gpt-4o-mini", "complex", avg_cost=0.02)}

    recs = RecommendationGenerator().generate([finding], cost_metrics)

    assert recs == []


def test_generate_cost_optimization_ignores_ineligible_candidates():
    finding = Finding(
        rule_type=RuleType.COST_OPTIMIZATION, subject="gpt-4o:complex",
        sample_size=20, pass_rate=0.9, threshold=0.7,
    )
    cost_metrics = {
        ("gpt-4o", "complex"): _metrics("gpt-4o", "complex", avg_cost=0.10),
        # cheaper but not eligible (didn't clear the pass-rate/sample bar itself)
        ("gpt-4o-mini", "complex"): _metrics("gpt-4o-mini", "complex", avg_cost=0.01, eligible=False),
    }

    recs = RecommendationGenerator().generate([finding], cost_metrics)

    assert recs == []


def test_generate_cost_optimization_severity_bands():
    def _rec_for_savings(daily_delta):
        finding = Finding(
            rule_type=RuleType.COST_OPTIMIZATION, subject="gpt-4o:complex",
            sample_size=20, pass_rate=0.9, threshold=0.7,
        )
        cost_metrics = {
            ("gpt-4o", "complex"): _metrics("gpt-4o", "complex", avg_cost=daily_delta, requests_per_day=1.0 / 30),
            ("gpt-4o-mini", "complex"): _metrics("gpt-4o-mini", "complex", avg_cost=0.0),
        }
        [rec] = RecommendationGenerator().generate([finding], cost_metrics)
        return rec

    assert _rec_for_savings(9.99).severity == Severity.LOW
    assert _rec_for_savings(10.00).severity == Severity.MEDIUM
    assert _rec_for_savings(100.00).severity == Severity.MEDIUM
    assert _rec_for_savings(100.01).severity == Severity.HIGH


def test_generate_cost_optimization_text_mentions_both_models_and_savings():
    finding = Finding(
        rule_type=RuleType.COST_OPTIMIZATION, subject="gpt-4o:complex",
        sample_size=20, pass_rate=0.9, threshold=0.7,
    )
    cost_metrics = {
        ("gpt-4o", "complex"): _metrics("gpt-4o", "complex", avg_cost=0.10, requests_per_day=10.0),
        ("gpt-4o-mini", "complex"): _metrics("gpt-4o-mini", "complex", avg_cost=0.03),
    }

    [rec] = RecommendationGenerator().generate([finding], cost_metrics)

    assert "gpt-4o" in rec.text
    assert "gpt-4o-mini" in rec.text
    assert "21.00" in rec.text


def test_generate_mixed_findings_quality_and_cost():
    quality_finding = Finding(
        rule_type=RuleType.MODEL_COMPLEXITY, subject="gpt-4o-mini:medium",
        sample_size=20, pass_rate=0.35, threshold=0.6,
    )
    cost_finding = Finding(
        rule_type=RuleType.COST_OPTIMIZATION, subject="gpt-4o:complex",
        sample_size=20, pass_rate=0.9, threshold=0.7,
    )
    cost_metrics = {
        ("gpt-4o", "complex"): _metrics("gpt-4o", "complex", avg_cost=0.10, requests_per_day=10.0),
        ("gpt-4o-mini", "complex"): _metrics("gpt-4o-mini", "complex", avg_cost=0.03),
    }

    recs = RecommendationGenerator().generate([quality_finding, cost_finding], cost_metrics)

    assert len(recs) == 2
    assert {r.source for r in recs} == {RecommendationSource.VERIFICATION, RecommendationSource.COST_OPTIMIZATION}


def test_generate_without_cost_metrics_argument_still_works_for_quality_findings():
    finding = Finding(
        rule_type=RuleType.COMPLEXITY_TIER, subject="complex", sample_size=30, pass_rate=0.4, threshold=0.5,
    )
    [rec] = RecommendationGenerator().generate([finding])  # no cost_metrics arg -- backward compatible
    assert rec.source == RecommendationSource.VERIFICATION


def test_generate_cost_optimization_breaks_avg_cost_ties_by_model_name():
    finding = Finding(
        rule_type=RuleType.COST_OPTIMIZATION, subject="gpt-4o:complex",
        sample_size=20, pass_rate=0.9, threshold=0.7,
    )
    cost_metrics = {
        ("gpt-4o", "complex"): _metrics("gpt-4o", "complex", avg_cost=0.10),
        ("gpt-4o-mini", "complex"): _metrics("gpt-4o-mini", "complex", avg_cost=0.02),
        ("claude-3-haiku", "complex"): _metrics("claude-3-haiku", "complex", avg_cost=0.02),  # tied with gpt-4o-mini
    }

    [rec] = RecommendationGenerator().generate([finding], cost_metrics)

    assert rec.evidence.comparison.suggested_model == "claude-3-haiku"  # alphabetically first among tied-cheapest
