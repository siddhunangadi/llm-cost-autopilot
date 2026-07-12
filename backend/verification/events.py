from pydantic import BaseModel


class VerificationStarted(BaseModel):
    request_id: str


class VerificationCompleted(BaseModel):
    request_id: str
    score: float


class VerificationFailed(BaseModel):
    request_id: str
    error_type: str
    error: str


class EscalationTriggered(BaseModel):
    request_id: str
    routing_model: str
    score: float
    reason: str
    escalated_model: str | None = None
    cost_delta: float | None = None
    latency_ms: float | None = None
    quality_gap: float | None = None
