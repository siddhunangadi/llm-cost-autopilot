import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import OperationalError

from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.database.models import ModelRow, ProviderRow


def _sample_provider_row() -> ProviderRow:
    return ProviderRow(name="openai", status="available")


def _sample_model_row() -> ModelRow:
    return ModelRow(
        model_id="gpt-4o-mini",
        provider="openai",
        model_name="gpt-4o-mini",
        input_cost=0.15,
        output_cost=0.60,
        context_window=128000,
        benchmark_score=0.82,
        supports_streaming=True,
        supports_tools=True,
        supports_json=True,
        average_latency_ms=450,
        available=True,
    )


def test_create_engine_from_settings_returns_a_bound_engine(tmp_path):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)

    assert str(engine.url) == settings.database_url


def test_init_db_creates_providers_and_models_tables(tmp_path):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)

    init_db(engine)

    table_names = set(inspect(engine).get_table_names())
    assert {"providers", "models"}.issubset(table_names)


def test_session_factory_returns_a_new_session_each_call(tmp_path):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    session_a = session_factory()
    session_b = session_factory()

    assert session_a is not session_b
    session_a.close()
    session_b.close()


def test_crud_insert_and_query_provider_and_model_rows(tmp_path):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    with session_factory() as session:
        session.add(_sample_provider_row())
        session.add(_sample_model_row())
        session.commit()

    with session_factory() as session:
        provider_row = session.query(ProviderRow).filter_by(name="openai").one()
        model_row = session.query(ModelRow).filter_by(model_id="gpt-4o-mini").one()

    assert provider_row.status == "available"
    assert model_row.benchmark_score == 0.82


def test_crud_update_and_delete_provider_row(tmp_path):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    with session_factory() as session:
        session.add(_sample_provider_row())
        session.commit()

    with session_factory() as session:
        row = session.query(ProviderRow).filter_by(name="openai").one()
        row.status = "disabled"
        session.commit()

    with session_factory() as session:
        row = session.query(ProviderRow).filter_by(name="openai").one()
        assert row.status == "disabled"
        session.delete(row)
        session.commit()

    with session_factory() as session:
        assert session.query(ProviderRow).filter_by(name="openai").one_or_none() is None


def test_rollback_discards_uncommitted_changes(tmp_path):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)

    with session_factory() as session:
        session.add(_sample_provider_row())
        session.flush()
        session.rollback()

    with session_factory() as session:
        assert session.query(ProviderRow).filter_by(name="openai").one_or_none() is None


def test_init_db_fails_fast_on_unwritable_database_path():
    settings = Settings(
        _env_file=None, database_url="sqlite:////nonexistent-directory-xyz/test.db"
    )
    engine = create_engine_from_settings(settings)

    with pytest.raises(OperationalError):
        init_db(engine)
