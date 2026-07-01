import io
import json
import logging

from backend.events.bus import EventBus
from backend.events.subscribers import register_logging_subscriber
from backend.events.types import EventType
from backend.telemetry.logging import JsonFormatter


def test_registered_subscriber_logs_every_event_type():
    bus = EventBus()
    register_logging_subscriber(bus)

    logger = logging.getLogger("events")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter(service="llm-cost-autopilot", environment="test"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    bus.emit(EventType.MODEL_REGISTERED, {"model_id": "gpt-4o-mini"})

    lines = [line for line in stream.getvalue().strip().splitlines() if line]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["message"] == "event_emitted"
    assert record["component"] == "events"

    logger.removeHandler(handler)
