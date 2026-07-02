from backend.events.types import EventType
from backend.verification.events import (
    VerificationCompleted,
    VerificationFailed,
    VerificationStarted,
)


def test_event_type_members_exist():
    assert EventType.VERIFICATION_STARTED == "verification_started"
    assert EventType.VERIFICATION_COMPLETED == "verification_completed"
    assert EventType.VERIFICATION_FAILED == "verification_failed"


def test_verification_started_payload():
    event = VerificationStarted(request_id="req-1")
    assert event.model_dump() == {"request_id": "req-1"}


def test_verification_completed_payload():
    event = VerificationCompleted(request_id="req-1", score=0.85)
    assert event.model_dump() == {"request_id": "req-1", "score": 0.85}


def test_verification_failed_payload():
    event = VerificationFailed(request_id="req-1", error_type="ValidationError", error="bad json")
    assert event.model_dump() == {
        "request_id": "req-1", "error_type": "ValidationError", "error": "bad json"
    }
