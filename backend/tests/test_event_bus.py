import io
import json
import logging

from backend.events.bus import EventBus
from backend.events.types import EventType
from backend.telemetry.logging import JsonFormatter


def test_emit_calls_subscribed_handler_with_payload():
    bus = EventBus()
    received = []
    bus.subscribe(EventType.MODEL_REGISTERED, lambda payload: received.append(payload))

    bus.emit(EventType.MODEL_REGISTERED, {"model_id": "gpt-4o-mini"})

    assert received == [{"model_id": "gpt-4o-mini"}]


def test_emit_with_no_subscribers_does_not_raise():
    bus = EventBus()
    bus.emit(EventType.PROVIDER_FAILED, {"provider": "openai"})


def test_multiple_subscribers_all_called():
    bus = EventBus()
    calls = []
    bus.subscribe(EventType.PROVIDER_AVAILABLE, lambda p: calls.append("a"))
    bus.subscribe(EventType.PROVIDER_AVAILABLE, lambda p: calls.append("b"))

    bus.emit(EventType.PROVIDER_AVAILABLE, {})

    assert calls == ["a", "b"]


def test_subscriber_exception_does_not_prevent_other_subscribers():
    bus = EventBus()
    calls = []

    def failing_handler(payload):
        raise RuntimeError("boom")

    def working_handler(payload):
        calls.append("worked")

    bus.subscribe(EventType.MODEL_REGISTERED, failing_handler)
    bus.subscribe(EventType.MODEL_REGISTERED, working_handler)

    bus.emit(EventType.MODEL_REGISTERED, {})

    assert calls == ["worked"]


def test_subscriber_exception_is_logged():
    bus = EventBus()

    logger = logging.getLogger("events")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter(service="llm-cost-autopilot", environment="test"))
    logger.addHandler(handler)
    logger.setLevel(logging.ERROR)
    logger.propagate = False

    def failing_handler(payload):
        raise RuntimeError("boom")

    bus.subscribe(EventType.MODEL_REGISTERED, failing_handler)
    bus.emit(EventType.MODEL_REGISTERED, {})

    lines = [line for line in stream.getvalue().strip().splitlines() if line]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["message"] == "event_subscriber_failed"
    assert record["level"] == "ERROR"

    logger.removeHandler(handler)
