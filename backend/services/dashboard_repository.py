from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from sqlalchemy.orm import sessionmaker

from backend.database.models import RequestRow, ResponseRow, RoutingEventRow, VerificationRow
from backend.verification.status import VerificationStatus


@dataclass(frozen=True)
class TimeWindow:
    days: int

    @property
    def cutoff(self) -> datetime:
        return datetime.now(timezone.utc) - timedelta(days=self.days)


@dataclass(frozen=True)
class QualityAggregation:
    total_verified: int
    average_score: float
    average_confidence: float
    pass_rate: float
    average_queue_delay_ms: float
    average_evaluation_duration_ms: float
    average_total_verification_ms: float
    verification_failure_count: int
    by_model: dict[str, float]
    by_strategy: dict[str, float]
    by_complexity: dict[str, float]


@dataclass(frozen=True)
class CostBucketData:
    date: date
    request_count: int
    total_cost: float


@dataclass(frozen=True)
class RecentRequestRow:
    request_id: str
    model: str
    strategy: str
    complexity: str
    cost: float | None
    score: float | None
    passed: bool | None
    created_at: datetime


@dataclass(frozen=True)
class QualityTrendBucket:
    date: date
    average_score: float
    pass_rate: float


@dataclass(frozen=True)
class FailoverData:
    request_ids: list[str]


@dataclass(frozen=True)
class FailoverEvent:
    request_id: str
    from_model: str
    to_model: str
    occurred_at: datetime


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _group_avg(rows: list[VerificationRow], key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(getattr(row, key), []).append(row.score)
    return {name: _avg(scores) for name, scores in grouped.items()}


class DashboardRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def get_quality_aggregation(self) -> QualityAggregation:
        with self._session_factory() as session:
            completed = (
                session.query(VerificationRow)
                .filter_by(status=VerificationStatus.COMPLETED.value)
                .all()
            )
            failure_count = (
                session.query(VerificationRow)
                .filter_by(status=VerificationStatus.FAILED.value)
                .count()
            )

        queue_delays = [
            (row.started_at - row.created_at).total_seconds() * 1000
            for row in completed
            if row.started_at is not None
        ]
        total_durations = [
            (row.completed_at - row.started_at).total_seconds() * 1000
            for row in completed
            if row.started_at is not None and row.completed_at is not None
        ]
        eval_durations = [
            row.evaluation_duration_ms for row in completed if row.evaluation_duration_ms is not None
        ]

        return QualityAggregation(
            total_verified=len(completed),
            average_score=_avg([row.score for row in completed]),
            average_confidence=_avg(
                [row.confidence for row in completed if row.confidence is not None]
            ),
            pass_rate=_avg([1.0 if row.passed else 0.0 for row in completed]),
            average_queue_delay_ms=_avg(queue_delays),
            average_evaluation_duration_ms=_avg(eval_durations),
            average_total_verification_ms=_avg(total_durations),
            verification_failure_count=failure_count,
            by_model=_group_avg(completed, "routing_model"),
            by_strategy=_group_avg(completed, "routing_strategy"),
            by_complexity=_group_avg(completed, "routing_complexity"),
        )

    def get_cost_trend(self, window: TimeWindow) -> list[CostBucketData]:
        with self._session_factory() as session:
            rows = (
                session.query(ResponseRow)
                .filter(ResponseRow.created_at >= window.cutoff)
                .filter(ResponseRow.actual_cost.isnot(None))
                .all()
            )

        buckets: dict[date, list[float]] = {}
        for row in rows:
            day = row.created_at.date()
            buckets.setdefault(day, []).append(row.actual_cost)

        return [
            CostBucketData(date=day, request_count=len(costs), total_cost=sum(costs))
            for day, costs in sorted(buckets.items())
        ]

    def get_cost_by_model(self, window: TimeWindow) -> dict[str, float]:
        with self._session_factory() as session:
            responses = (
                session.query(ResponseRow)
                .filter(ResponseRow.created_at >= window.cutoff)
                .filter(ResponseRow.actual_cost.isnot(None))
                .all()
            )
            request_ids = [r.request_id for r in responses]
            routing_events = (
                session.query(RoutingEventRow)
                .filter(RoutingEventRow.request_id.in_(request_ids))
                .order_by(RoutingEventRow.created_at)
                .all()
            )

        latest_model: dict[str, str] = {}
        for row in routing_events:
            latest_model[row.request_id] = row.selected_model

        totals: dict[str, float] = {}
        for response in responses:
            model = latest_model.get(response.request_id, "unknown")
            totals[model] = totals.get(model, 0.0) + response.actual_cost
        return totals

    def get_quality_trend(self, window: TimeWindow) -> list[QualityTrendBucket]:
        with self._session_factory() as session:
            rows = (
                session.query(VerificationRow)
                .filter(VerificationRow.created_at >= window.cutoff)
                .filter(VerificationRow.status == VerificationStatus.COMPLETED.value)
                .all()
            )

        buckets: dict[date, list[VerificationRow]] = {}
        for row in rows:
            day = row.created_at.date()
            buckets.setdefault(day, []).append(row)

        return [
            QualityTrendBucket(
                date=day,
                average_score=_avg([r.score for r in group]),
                pass_rate=_avg([1.0 if r.passed else 0.0 for r in group]),
            )
            for day, group in sorted(buckets.items())
        ]

    def get_failover_summary(self, window: TimeWindow) -> FailoverData:
        with self._session_factory() as session:
            rows = (
                session.query(RoutingEventRow)
                .filter(RoutingEventRow.created_at >= window.cutoff)
                .all()
            )

        counts: dict[str, int] = {}
        for row in rows:
            counts[row.request_id] = counts.get(row.request_id, 0) + 1

        return FailoverData(
            request_ids=sorted(rid for rid, count in counts.items() if count == 2)
        )

    def get_failover_events(self, window: TimeWindow) -> list[FailoverEvent]:
        with self._session_factory() as session:
            rows = (
                session.query(RoutingEventRow)
                .filter(RoutingEventRow.created_at >= window.cutoff)
                .order_by(RoutingEventRow.request_id, RoutingEventRow.created_at)
                .all()
            )

        grouped: dict[str, list[RoutingEventRow]] = {}
        for row in rows:
            grouped.setdefault(row.request_id, []).append(row)

        events = [
            FailoverEvent(
                request_id=request_id,
                from_model=group[0].selected_model,
                to_model=group[1].selected_model,
                occurred_at=group[1].created_at,
            )
            for request_id, group in grouped.items()
            if len(group) == 2
        ]
        return sorted(events, key=lambda e: e.occurred_at)

    def get_recent_requests(self, limit: int = 50) -> list[RecentRequestRow]:
        with self._session_factory() as session:
            requests = (
                session.query(RequestRow)
                .order_by(RequestRow.created_at.desc())
                .limit(limit)
                .all()
            )
            request_ids = [r.request_id for r in requests]
            routing_events = (
                session.query(RoutingEventRow)
                .filter(RoutingEventRow.request_id.in_(request_ids))
                .order_by(RoutingEventRow.created_at)
                .all()
            )
            responses = (
                session.query(ResponseRow)
                .filter(ResponseRow.request_id.in_(request_ids))
                .all()
            )
            verifications = (
                session.query(VerificationRow)
                .filter(VerificationRow.request_id.in_(request_ids))
                .all()
            )

        latest_routing: dict[str, RoutingEventRow] = {}
        for row in routing_events:
            latest_routing[row.request_id] = row  # ascending order, last write wins
        response_by_request = {row.request_id: row for row in responses}
        verification_by_request = {row.request_id: row for row in verifications}

        result = []
        for req in requests:
            routing = latest_routing.get(req.request_id)
            response = response_by_request.get(req.request_id)
            verification = verification_by_request.get(req.request_id)
            result.append(RecentRequestRow(
                request_id=req.request_id,
                model=routing.selected_model if routing else "unknown",
                strategy=routing.selected_strategy if routing else req.strategy,
                complexity=routing.complexity if routing else "unknown",
                cost=response.actual_cost if response else None,
                score=verification.score if verification else None,
                passed=verification.passed if verification else None,
                created_at=req.created_at,
            ))
        return result
