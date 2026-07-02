from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.database.models import RecommendationRow


def test_recommendation_row_round_trip(tmp_path):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    with session_factory() as session:
        session.add(RecommendationRow(
            signature="model_complexity:gpt-4o-mini:medium",
            rule_type="model_complexity",
            subject="gpt-4o-mini:medium",
            recommendation_text="Consider a higher-benchmark model.",
            evidence_confidence=0.6,
            severity="high",
            evidence={"sample_size": 20, "pass_rate": 0.35, "threshold": 0.6},
            source="verification",
        ))
        session.commit()

    with session_factory() as session:
        row = session.query(RecommendationRow).filter_by(
            signature="model_complexity:gpt-4o-mini:medium"
        ).one()
        assert row.status == "new"  # default
        assert row.severity == "high"
        assert row.evidence["pass_rate"] == 0.35
        assert row.updated_at is not None
