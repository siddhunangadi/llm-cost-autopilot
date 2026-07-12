from enum import Enum


class EventType(str, Enum):
    PROVIDER_AVAILABLE = "provider_available"
    PROVIDER_DISABLED = "provider_disabled"
    PROVIDER_FAILED = "provider_failed"
    MODEL_REGISTERED = "model_registered"
    VERIFICATION_STARTED = "verification_started"
    VERIFICATION_COMPLETED = "verification_completed"
    VERIFICATION_FAILED = "verification_failed"
    CIRCUIT_OPENED = "circuit_opened"
    CIRCUIT_HALF_OPEN = "circuit_half_open"
    CIRCUIT_CLOSED = "circuit_closed"
    PROVIDER_FAILOVER_TRIGGERED = "provider_failover_triggered"
    ESCALATION_TRIGGERED = "escalation_triggered"
