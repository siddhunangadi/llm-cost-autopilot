from datetime import datetime, timezone
from types import SimpleNamespace

from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.database.models import RecommendationRow, RequestRow, ResponseRow, VerificationRow
from backend.learning.detector import FailurePatternDetector
from backend.learning.generator import RecommendationGenerator
from backend.learning.rules import (
    ComplexityTierRule, DetectionRuleConfig, ModelComplexityRule, OverpoweredModelRule,
)
from backend.learning.service import LearningService
from backend.verification.status import VerificationStatus


class _FakeModelRegistry:
    def __init__(self, pricing: dict[str, tuple[float, float]]) -> None:
        self._pricing = pricing

    def get_model(self, model_id: str):
        input_cost, output_cost = self._pricing[model_id]
        return SimpleNamespace(input_cost=input_cost, output_cost=output_cost)


def _make_service(tmp_path, pricing=None):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    detector = FailurePatternDetector(rules=[
        ModelComplexityRule(DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.6)),
        ComplexityTierRule(DetectionRuleConfig(min_samples=30, pass_rate_threshold=0.5)),
        OverpoweredModelRule(DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.7)),
    ])
    service = LearningService(
        detector=detector,
        generator=RecommendationGenerator(),
        session_factory=session_factory,
        model_registry=_FakeModelRegistry(pricing or {}),
        cost_optimization_config=DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.7),
    )
    return service, session_factory


def _seed_failing_model(session_factory, count=20, passed_count=7, prefix="req"):
    with session_factory() as session:
        for i in range(count):
            request_id = f"{prefix}-{i}"
            session.add(RequestRow(request_id=request_id, prompt="hi", strategy="balanced"))
            session.add(VerificationRow(
                request_id=request_id, status=VerificationStatus.COMPLETED.value,
                routing_model="gpt-4o-mini", routing_strategy="balanced",
                routing_complexity="medium", passed=(i < passed_count),
            ))
        session.commit()


def test_refresh_inserts_new_recommendation(tmp_path):
    service, session_factory = _make_service(tmp_path)
    _seed_failing_model(session_factory)

    results = service.refresh_recommendations()

    assert len(results) == 1
    assert results[0].signature == "model_complexity:gpt-4o-mini:medium"
    assert results[0].status == "new"


def test_get_recommendations_returns_persisted_rows_without_recomputing(tmp_path):
    service, session_factory = _make_service(tmp_path)
    _seed_failing_model(session_factory)
    service.refresh_recommendations()

    with session_factory() as session:
        row = session.query(RecommendationRow).filter_by(
            signature="model_complexity:gpt-4o-mini:medium"
        ).one()
        row.recommendation_text = "manually edited, should not be overwritten"
        session.commit()

    # Seed more failing data that WOULD change the recommendation text if
    # get_recommendations() recomputed -- it must not.
    _seed_failing_model(session_factory, count=20, passed_count=2, prefix="req2")

    results = service.get_recommendations()

    assert len(results) == 1
    assert results[0].recommendation_text == "manually edited, should not be overwritten"


def test_get_recommendations_ordering_matches_refresh(tmp_path):
    service, session_factory = _make_service(tmp_path)
    _seed_failing_model(session_factory)
    refreshed = service.refresh_recommendations()

    results = service.get_recommendations()

    assert [r.signature for r in results] == [r.signature for r in refreshed]


def test_get_recommendations_returns_empty_list_when_none_persisted(tmp_path):
    service, _ = _make_service(tmp_path)

    results = service.get_recommendations()

    assert results == []


def test_refresh_is_idempotent_no_duplicates(tmp_path):
    service, session_factory = _make_service(tmp_path)
    _seed_failing_model(session_factory)

    service.refresh_recommendations()
    results = service.refresh_recommendations()

    with session_factory() as session:
        count = session.query(RecommendationRow).count()
    assert count == 1
    assert len(results) == 1


def test_refresh_updates_evidence_but_preserves_human_set_status(tmp_path):
    service, session_factory = _make_service(tmp_path)
    _seed_failing_model(session_factory, count=20, passed_count=7)
    service.refresh_recommendations()

    with session_factory() as session:
        row = session.query(RecommendationRow).filter_by(
            signature="model_complexity:gpt-4o-mini:medium"
        ).one()
        row.status = "acknowledged"
        session.commit()

    _seed_failing_model(session_factory, count=20, passed_count=5, prefix="req2")  # shifts pass_rate lower
    results = service.refresh_recommendations()

    model_complexity = next(r for r in results if r.signature == "model_complexity:gpt-4o-mini:medium")
    assert model_complexity.status == "acknowledged"  # untouched by refresh
    assert model_complexity.evidence["sample_size"] == 40  # evidence did update


def test_refresh_returns_empty_list_when_no_findings(tmp_path):
    service, session_factory = _make_service(tmp_path)
    results = service.refresh_recommendations()
    assert results == []


def _seed_passing_model_with_cost(session_factory, model, cost, count=20, prefix="req"):
    with session_factory() as session:
        base_day = 1
        for i in range(count):
            request_id = f"{prefix}-{model}-{i}"
            created = datetime(2026, 7, base_day + (i % 5), tzinfo=timezone.utc)
            session.add(RequestRow(request_id=request_id, prompt="hi", strategy="balanced"))
            session.add(VerificationRow(
                request_id=request_id, status=VerificationStatus.COMPLETED.value,
                routing_model=model, routing_strategy="balanced",
                routing_complexity="complex", passed=True, created_at=created,
            ))
            session.add(ResponseRow(request_id=request_id, response_text="ok", actual_cost=cost))
        session.commit()


def test_refresh_inserts_cost_optimization_recommendation(tmp_path):
    pricing = {"gpt-4o": (2.50, 10.00), "gpt-4o-mini": (0.15, 0.60)}
    service, session_factory = _make_service(tmp_path, pricing=pricing)
    _seed_passing_model_with_cost(session_factory, "gpt-4o", cost=0.10, prefix="expensive")
    _seed_passing_model_with_cost(session_factory, "gpt-4o-mini", cost=0.02, prefix="cheap")

    results = service.refresh_recommendations()

    cost_recs = [r for r in results if r.source == "cost_optimization"]
    assert len(cost_recs) == 1
    assert cost_recs[0].signature == "cost_optimization:gpt-4o:complex"
    assert cost_recs[0].evidence["comparison"]["suggested_model"] == "gpt-4o-mini"


def test_refresh_omits_cost_recommendation_when_no_cheaper_alternative(tmp_path):
    pricing = {"gpt-4o-mini": (0.15, 0.60)}
    service, session_factory = _make_service(tmp_path, pricing=pricing)
    _seed_passing_model_with_cost(session_factory, "gpt-4o-mini", cost=0.02, prefix="only")

    results = service.refresh_recommendations()

    assert [r for r in results if r.source == "cost_optimization"] == []
