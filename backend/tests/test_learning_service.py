from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.database.models import RecommendationRow, RequestRow, VerificationRow
from backend.learning.detector import FailurePatternDetector
from backend.learning.generator import RecommendationGenerator
from backend.learning.rules import ComplexityTierRule, DetectionRuleConfig, ModelComplexityRule
from backend.learning.service import LearningService
from backend.verification.status import VerificationStatus


def _make_service(tmp_path):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    detector = FailurePatternDetector(rules=[
        ModelComplexityRule(DetectionRuleConfig(min_samples=20, pass_rate_threshold=0.6)),
        ComplexityTierRule(DetectionRuleConfig(min_samples=30, pass_rate_threshold=0.5)),
    ])
    service = LearningService(
        detector=detector, generator=RecommendationGenerator(), session_factory=session_factory
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
