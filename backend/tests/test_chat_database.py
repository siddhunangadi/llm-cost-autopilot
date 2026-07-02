from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.database.models import RequestRow, ResponseRow, RoutingEventRow


def test_create_and_query_request_response_routing_event(tmp_path):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    with session_factory() as session:
        session.add(RequestRow(request_id="req-1", prompt="Hello", strategy="balanced"))
        session.commit()

    with session_factory() as session:
        session.add(ResponseRow(
            request_id="req-1", response_text="Hi there", actual_input_tokens=5,
            actual_output_tokens=3, actual_cost=0.001,
        ))
        session.add(RoutingEventRow(
            request_id="req-1", complexity="simple", confidence=0.66,
            selected_model="gpt-4o-mini", selected_strategy="balanced",
            estimated_cost=0.001, estimated_latency_ms=450.0, reasoning="[]",
        ))
        session.commit()

    with session_factory() as session:
        request_row = session.query(RequestRow).filter_by(request_id="req-1").one()
        response_row = session.query(ResponseRow).filter_by(request_id="req-1").one()
        routing_event_row = session.query(RoutingEventRow).filter_by(request_id="req-1").one()

    assert request_row.prompt == "Hello"
    assert response_row.response_text == "Hi there"
    assert response_row.error is None
    assert routing_event_row.selected_strategy == "balanced"


def test_response_row_persists_error_without_response_text(tmp_path):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    with session_factory() as session:
        session.add(RequestRow(request_id="req-2", prompt="Hello", strategy="cost"))
        session.commit()

    with session_factory() as session:
        session.add(ResponseRow(request_id="req-2", error="ProviderError: boom"))
        session.commit()

    with session_factory() as session:
        response_row = session.query(ResponseRow).filter_by(request_id="req-2").one()

    assert response_row.response_text is None
    assert response_row.error == "ProviderError: boom"
