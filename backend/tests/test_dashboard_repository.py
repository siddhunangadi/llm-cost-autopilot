from datetime import datetime, timedelta, timezone

import pytest

from backend.config.settings import Settings
from backend.database.base import create_engine_from_settings, create_session_factory, init_db
from backend.database.models import RequestRow, ResponseRow, RoutingEventRow, VerificationRow
from backend.services.dashboard_repository import DashboardRepository, TimeWindow
from backend.verification.status import VerificationStatus


def _make_repository(tmp_path):
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/test.db")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    session_factory = create_session_factory(engine)
    return DashboardRepository(session_factory=session_factory), session_factory


def _routing_event(request_id: str, created_at: datetime, model: str = "gpt-4o-mini") -> RoutingEventRow:
    return RoutingEventRow(
        request_id=request_id, complexity="simple", confidence=0.9, selected_model=model,
        selected_strategy="balanced", estimated_cost=0.01, estimated_latency_ms=100,
        reasoning="[]", created_at=created_at,
    )


def test_time_window_cutoff_is_days_ago():
    window = TimeWindow(days=7)
    now = datetime.now(timezone.utc)

    delta = now - window.cutoff

    assert timedelta(days=6, hours=23) < delta < timedelta(days=7, hours=1)


def test_get_quality_aggregation_matches_verification_data(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-1", prompt="hi", strategy="balanced"))
        session.add(VerificationRow(
            request_id="req-1", status=VerificationStatus.COMPLETED.value,
            routing_model="gpt-4o-mini", routing_strategy="balanced", routing_complexity="simple",
            score=0.9, passed=True, confidence=0.8, evaluation_duration_ms=100,
        ))
        session.add(RequestRow(request_id="req-2", prompt="hi", strategy="balanced"))
        session.add(VerificationRow(
            request_id="req-2", status=VerificationStatus.FAILED.value,
            routing_model="gpt-4o-mini", routing_strategy="balanced", routing_complexity="simple",
            error_type="ValidationError", error="bad json",
        ))
        session.commit()

    result = repository.get_quality_aggregation()

    assert result.total_verified == 1
    assert result.verification_failure_count == 1
    assert result.pass_rate == pytest.approx(1.0)
    assert result.by_model["gpt-4o-mini"] == pytest.approx(0.9)


def test_get_cost_trend_buckets_by_day_and_omits_empty_days(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    now = datetime.now(timezone.utc)
    day0 = now
    day2 = now - timedelta(days=2)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-a", prompt="hi", strategy="balanced"))
        session.add(ResponseRow(
            request_id="req-a", response_text="ok", actual_cost=0.10, created_at=day0,
        ))
        session.add(RequestRow(request_id="req-b", prompt="hi", strategy="balanced"))
        session.add(ResponseRow(
            request_id="req-b", response_text="ok", actual_cost=0.20, created_at=day0,
        ))
        session.add(RequestRow(request_id="req-c", prompt="hi", strategy="balanced"))
        session.add(ResponseRow(
            request_id="req-c", response_text="ok", actual_cost=0.05, created_at=day2,
        ))
        session.commit()

    buckets = repository.get_cost_trend(TimeWindow(days=7))

    assert len(buckets) == 2
    assert buckets[0].date < buckets[1].date  # ascending order
    day0_bucket = buckets[1]
    assert day0_bucket.request_count == 2
    assert day0_bucket.total_cost == pytest.approx(0.30)


def test_get_cost_trend_excludes_data_outside_window(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    old = datetime.now(timezone.utc) - timedelta(days=30)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-old", prompt="hi", strategy="balanced"))
        session.add(ResponseRow(
            request_id="req-old", response_text="ok", actual_cost=0.10, created_at=old,
        ))
        session.commit()

    buckets = repository.get_cost_trend(TimeWindow(days=7))

    assert buckets == []


def test_get_cost_trend_excludes_error_responses_with_no_cost(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-err", prompt="hi", strategy="balanced"))
        session.add(ResponseRow(request_id="req-err", error_type="provider_error", error="boom"))
        session.commit()

    buckets = repository.get_cost_trend(TimeWindow(days=7))

    assert buckets == []


def test_get_failover_summary_only_counts_requests_with_two_routing_events(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    now = datetime.now(timezone.utc)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-single", prompt="hi", strategy="balanced"))
        session.add(_routing_event("req-single", now))

        session.add(RequestRow(request_id="req-failover", prompt="hi", strategy="balanced"))
        session.add(_routing_event("req-failover", now, model="gpt-4o-mini"))
        session.add(_routing_event("req-failover", now, model="gpt-4o"))
        session.commit()

    summary = repository.get_failover_summary(TimeWindow(days=7))

    assert summary.request_ids == ["req-failover"]


def test_get_failover_summary_excludes_events_outside_window(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    old = datetime.now(timezone.utc) - timedelta(days=30)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-old-failover", prompt="hi", strategy="balanced"))
        session.add(_routing_event("req-old-failover", old, model="gpt-4o-mini"))
        session.add(_routing_event("req-old-failover", old, model="gpt-4o"))
        session.commit()

    summary = repository.get_failover_summary(TimeWindow(days=7))

    assert summary.request_ids == []


def test_get_quality_trend_buckets_by_day(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    now = datetime.now(timezone.utc)
    day2 = now - timedelta(days=2)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-a", prompt="hi", strategy="balanced"))
        session.add(VerificationRow(
            request_id="req-a", status=VerificationStatus.COMPLETED.value,
            routing_model="gpt-4o-mini", routing_strategy="balanced", routing_complexity="simple",
            score=0.9, passed=True, confidence=0.8, created_at=now,
        ))
        session.add(RequestRow(request_id="req-b", prompt="hi", strategy="balanced"))
        session.add(VerificationRow(
            request_id="req-b", status=VerificationStatus.COMPLETED.value,
            routing_model="gpt-4o-mini", routing_strategy="balanced", routing_complexity="simple",
            score=0.5, passed=False, confidence=0.6, created_at=now,
        ))
        session.add(RequestRow(request_id="req-c", prompt="hi", strategy="balanced"))
        session.add(VerificationRow(
            request_id="req-c", status=VerificationStatus.COMPLETED.value,
            routing_model="gpt-4o-mini", routing_strategy="balanced", routing_complexity="simple",
            score=1.0, passed=True, confidence=0.9, created_at=day2,
        ))
        session.add(RequestRow(request_id="req-failed", prompt="hi", strategy="balanced"))
        session.add(VerificationRow(
            request_id="req-failed", status=VerificationStatus.FAILED.value,
            routing_model="gpt-4o-mini", routing_strategy="balanced", routing_complexity="simple",
            error_type="ValidationError", error="bad json", created_at=now,
        ))
        session.commit()

    buckets = repository.get_quality_trend(TimeWindow(days=7))

    assert len(buckets) == 2
    assert buckets[0].date < buckets[1].date
    today_bucket = buckets[1]
    assert today_bucket.average_score == pytest.approx(0.7)
    assert today_bucket.pass_rate == pytest.approx(0.5)


def test_get_quality_trend_excludes_data_outside_window(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    old = datetime.now(timezone.utc) - timedelta(days=30)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-old", prompt="hi", strategy="balanced"))
        session.add(VerificationRow(
            request_id="req-old", status=VerificationStatus.COMPLETED.value,
            routing_model="gpt-4o-mini", routing_strategy="balanced", routing_complexity="simple",
            score=0.9, passed=True, confidence=0.8, created_at=old,
        ))
        session.commit()

    buckets = repository.get_quality_trend(TimeWindow(days=7))

    assert buckets == []


def test_get_failover_events_returns_from_to_and_timestamp(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    now = datetime.now(timezone.utc)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-single", prompt="hi", strategy="balanced"))
        session.add(_routing_event("req-single", now))

        session.add(RequestRow(request_id="req-failover", prompt="hi", strategy="balanced"))
        session.add(_routing_event("req-failover", now, model="gpt-4o-mini"))
        session.add(_routing_event("req-failover", now + timedelta(seconds=1), model="gpt-4o"))
        session.commit()

    events = repository.get_failover_events(TimeWindow(days=7))

    assert len(events) == 1
    assert events[0].request_id == "req-failover"
    assert events[0].from_model == "gpt-4o-mini"
    assert events[0].to_model == "gpt-4o"


def test_get_failover_events_excludes_events_outside_window(tmp_path):
    repository, session_factory = _make_repository(tmp_path)
    old = datetime.now(timezone.utc) - timedelta(days=30)
    with session_factory() as session:
        session.add(RequestRow(request_id="req-old-failover", prompt="hi", strategy="balanced"))
        session.add(_routing_event("req-old-failover", old, model="gpt-4o-mini"))
        session.add(_routing_event("req-old-failover", old, model="gpt-4o"))
        session.commit()

    events = repository.get_failover_events(TimeWindow(days=7))

    assert events == []
