from backend.events.bus import EventBus
from backend.events.types import EventType
from backend.telemetry.logging import get_logger


def register_logging_subscriber(event_bus: EventBus) -> None:
    logger = get_logger("events")

    def handler(payload: dict) -> None:
        logger.info("event_emitted", extra={"payload": payload})

    for event_type in EventType:
        event_bus.subscribe(event_type, handler)
