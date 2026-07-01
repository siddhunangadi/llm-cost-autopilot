from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.database.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ProviderRow(Base):
    __tablename__ = "providers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="unknown")
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(String, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class ModelRow(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    provider: Mapped[str] = mapped_column(String, nullable=False)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    input_cost: Mapped[float] = mapped_column(Float, nullable=False)
    output_cost: Mapped[float] = mapped_column(Float, nullable=False)
    context_window: Mapped[int] = mapped_column(Integer, nullable=False)
    benchmark_score: Mapped[float] = mapped_column(Float, nullable=False)
    supports_streaming: Mapped[bool] = mapped_column(Boolean, default=False)
    supports_tools: Mapped[bool] = mapped_column(Boolean, default=False)
    supports_json: Mapped[bool] = mapped_column(Boolean, default=False)
    average_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    available: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
