from datetime import datetime, timezone

from sqlalchemy.orm import sessionmaker

from backend.database.models import RecommendationRow, ResponseRow, VerificationRow
from backend.learning.cost_metrics import build_model_cost_metrics
from backend.learning.detector import FailurePatternDetector
from backend.learning.generator import RecommendationGenerator
from backend.learning.rules import DetectionRuleConfig


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LearningService:
    def __init__(
        self,
        detector: FailurePatternDetector,
        generator: RecommendationGenerator,
        session_factory: sessionmaker,
        model_registry,
        cost_optimization_config: DetectionRuleConfig,
    ) -> None:
        self._detector = detector
        self._generator = generator
        self._session_factory = session_factory
        self._model_registry = model_registry
        self._cost_optimization_config = cost_optimization_config

    def refresh_recommendations(self) -> list[RecommendationRow]:
        with self._session_factory() as session:
            rows = session.query(VerificationRow).order_by(VerificationRow.id).all()
            cost_by_request_id = {
                r.request_id: r.actual_cost
                for r in session.query(ResponseRow).all()
                if r.actual_cost is not None
            }

        findings = self._detector.detect(rows)
        cost_metrics = build_model_cost_metrics(
            rows, cost_by_request_id, self._model_registry, self._cost_optimization_config
        )
        recommendations = self._generator.generate(findings, cost_metrics)

        with self._session_factory() as session:
            for rec in recommendations:
                existing = (
                    session.query(RecommendationRow)
                    .filter_by(signature=rec.signature)
                    .first()
                )
                if existing is None:
                    session.add(RecommendationRow(
                        signature=rec.signature,
                        rule_type=rec.rule_type.value,
                        subject=rec.subject,
                        recommendation_text=rec.text,
                        evidence_confidence=rec.evidence_confidence,
                        severity=rec.severity.value,
                        evidence=rec.evidence.model_dump(),
                        status="new",
                        source=rec.source.value,
                    ))
                else:
                    existing.recommendation_text = rec.text
                    existing.evidence_confidence = rec.evidence_confidence
                    existing.severity = rec.severity.value
                    existing.evidence = rec.evidence.model_dump()
                    existing.updated_at = _utcnow()
                    # existing.status is intentionally never modified here --
                    # status is owned exclusively by humans.
            session.commit()

        with self._session_factory() as session:
            return (
                session.query(RecommendationRow)
                .order_by(
                    RecommendationRow.severity.desc(),
                    RecommendationRow.evidence_confidence.desc(),
                    RecommendationRow.updated_at.desc(),
                )
                .all()
            )

    def get_recommendations(self) -> list[RecommendationRow]:
        with self._session_factory() as session:
            return (
                session.query(RecommendationRow)
                .order_by(
                    RecommendationRow.severity.desc(),
                    RecommendationRow.evidence_confidence.desc(),
                    RecommendationRow.updated_at.desc(),
                )
                .all()
            )
