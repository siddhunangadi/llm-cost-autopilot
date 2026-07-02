from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.config.settings import Settings
from backend.database.models import RequestRow, VerificationRow
from backend.verification.status import VerificationStatus


def test_verification_row_round_trip(tmp_path):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    with session_factory() as session:
        session.add(RequestRow(request_id="req-1", prompt="hi", strategy="balanced"))
        session.add(VerificationRow(
            request_id="req-1",
            status=VerificationStatus.PENDING.value,
            routing_model="gpt-4o-mini",
            routing_strategy="balanced",
            routing_complexity="simple",
        ))
        session.commit()

    with session_factory() as session:
        row = session.query(VerificationRow).filter_by(request_id="req-1").one()
        assert row.status == VerificationStatus.PENDING.value
        assert row.routing_model == "gpt-4o-mini"
        assert row.score is None
        assert row.dimensions is None
        assert row.raw_judge_response is None
        assert row.started_at is None
        assert row.completed_at is None
