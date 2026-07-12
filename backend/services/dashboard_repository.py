import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import Integer, func
from sqlalchemy.orm import sessionmaker

from backend.database.models import (
    RecommendationRow, RequestRow, ResponseRow, RoutingEventRow, VerificationRow,
)
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
class TokenTotals:
    input_tokens: int
    output_tokens: int


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
    reasoning: list[str]
    alternatives: list[dict]


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


@dataclass(frozen=True)
class FailoverTrendBucket:
    date: date
    failover_count: int


@dataclass(frozen=True)
class RoutingDistributionBucket:
    date: date
    model: str
    request_count: int


@dataclass(frozen=True)
class RecommendationTrendBucket:
    date: date
    generated_count: int
    open_count: int


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class DashboardRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def get_quality_aggregation(self, window: TimeWindow | None = None) -> QualityAggregation:
        with self._session_factory() as session:
            completed_query = session.query(VerificationRow).filter_by(
                status=VerificationStatus.COMPLETED.value
            )
            failed_query = session.query(VerificationRow).filter_by(
                status=VerificationStatus.FAILED.value
            )
            if window is not None:
                completed_query = completed_query.filter(VerificationRow.created_at >= window.cutoff)
                failed_query = failed_query.filter(VerificationRow.created_at >= window.cutoff)

            total_verified, avg_score, avg_confidence, passed_count, avg_eval_ms = (
                completed_query.with_entities(
                    func.count(VerificationRow.id),
                    func.avg(VerificationRow.score),
                    func.avg(VerificationRow.confidence),
                    func.sum(func.cast(VerificationRow.passed, Integer)),
                    func.avg(VerificationRow.evaluation_duration_ms),
                ).one()
            )
            failure_count = failed_query.count()

            by_model = dict(
                completed_query.with_entities(
                    VerificationRow.routing_model, func.avg(VerificationRow.score)
                ).group_by(VerificationRow.routing_model).all()
            )
            by_strategy = dict(
                completed_query.with_entities(
                    VerificationRow.routing_strategy, func.avg(VerificationRow.score)
                ).group_by(VerificationRow.routing_strategy).all()
            )
            by_complexity = dict(
                completed_query.with_entities(
                    VerificationRow.routing_complexity, func.avg(VerificationRow.score)
                ).group_by(VerificationRow.routing_complexity).all()
            )

            # Queue delay and total verification time are timestamp differences,
            # not portable across SQL dialects (sqlite vs postgres) -- computed
            # in Python, but only the 3 timestamp columns are fetched, not the
            # full row (skips dimensions/rationale/raw_judge_response, etc).
            timing_rows = completed_query.with_entities(
                VerificationRow.created_at, VerificationRow.started_at, VerificationRow.completed_at,
            ).all()

        queue_delays = [
            (started - created).total_seconds() * 1000
            for created, started, _ in timing_rows
            if started is not None
        ]
        total_durations = [
            (completed - started).total_seconds() * 1000
            for _, started, completed in timing_rows
            if started is not None and completed is not None
        ]

        return QualityAggregation(
            total_verified=total_verified,
            average_score=avg_score or 0.0,
            average_confidence=avg_confidence or 0.0,
            pass_rate=(passed_count or 0) / total_verified if total_verified else 0.0,
            average_queue_delay_ms=_avg(queue_delays),
            average_evaluation_duration_ms=avg_eval_ms or 0.0,
            average_total_verification_ms=_avg(total_durations),
            verification_failure_count=failure_count,
            by_model=by_model,
            by_strategy=by_strategy,
            by_complexity=by_complexity,
        )

    def get_token_totals(self, window: TimeWindow) -> list[TokenTotals]:
        """Per-response token counts for successful (costed) responses in
        the window -- used to compute a counterfactual baseline cost
        (what a single reference model would have cost for this traffic).
        Only 2 columns fetched, not full ResponseRow."""
        with self._session_factory() as session:
            rows = (
                session.query(ResponseRow)
                .filter(ResponseRow.created_at >= window.cutoff)
                .filter(ResponseRow.actual_cost.isnot(None))
                .with_entities(
                    ResponseRow.actual_input_tokens, ResponseRow.actual_output_tokens,
                )
                .all()
            )
        return [
            TokenTotals(input_tokens=i or 0, output_tokens=o or 0)
            for i, o in rows
        ]

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

    def get_failover_trend(self, window: TimeWindow) -> list[FailoverTrendBucket]:
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

        # Same failover predicate as get_failover_events: exactly two
        # routing events for a request_id. Bucketed by the day of the
        # second (failover) event and counted, not listed.
        buckets: dict[date, int] = {}
        for group in grouped.values():
            if len(group) == 2:
                day = group[1].created_at.date()
                buckets[day] = buckets.get(day, 0) + 1

        return [
            FailoverTrendBucket(date=day, failover_count=count)
            for day, count in sorted(buckets.items())
        ]

    def get_routing_distribution(self, window: TimeWindow) -> list[RoutingDistributionBucket]:
        with self._session_factory() as session:
            rows = (
                session.query(RoutingEventRow)
                .filter(RoutingEventRow.created_at >= window.cutoff)
                .all()
            )

        buckets: dict[tuple[date, str], int] = {}
        for row in rows:
            key = (row.created_at.date(), row.selected_model)
            buckets[key] = buckets.get(key, 0) + 1

        return [
            RoutingDistributionBucket(date=day, model=model, request_count=count)
            for (day, model), count in sorted(buckets.items())
        ]

    def get_recommendation_trend(self, window: TimeWindow) -> list[RecommendationTrendBucket]:
        with self._session_factory() as session:
            rows = (
                session.query(RecommendationRow)
                .filter(RecommendationRow.created_at >= window.cutoff)
                .all()
            )

        buckets: dict[date, list[RecommendationRow]] = {}
        for row in rows:
            day = row.created_at.date()
            buckets.setdefault(day, []).append(row)

        # open_count is a simplification: recommendations generated that
        # day whose *current* status is still "new" as of report
        # generation time, not a true point-in-time daily open count.
        return [
            RecommendationTrendBucket(
                date=day,
                generated_count=len(group),
                open_count=sum(1 for r in group if r.status == "new"),
            )
            for day, group in sorted(buckets.items())
        ]

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
                reasoning=json.loads(routing.reasoning) if routing else [],
                alternatives=routing.alternatives if routing and routing.alternatives else [],
            ))
        return result
