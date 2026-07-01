from collections import defaultdict
from typing import Callable

from backend.events.types import EventType
from backend.telemetry.logging import get_logger

EventHandler = Callable[[dict], None]


class EventBus:
    """In-process, synchronous event bus for Phase 1.

    No external broker (Redis/NATS/Kafka) is used. `emit` calls each
    subscribed handler synchronously in registration order. A future
    broker-backed implementation would preserve this subscribe/emit
    interface. A subscriber that raises is logged and skipped -- it never
    prevents other subscribers for the same event from running.
    """

    def __init__(self) -> None:
        self._subscribers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._logger = get_logger("events")

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        self._subscribers[event_type].append(handler)

    def emit(self, event_type: EventType, payload: dict) -> None:
        for handler in self._subscribers.get(event_type, []):
            try:
                handler(payload)
            except Exception:
                self._logger.exception(
                    "event_subscriber_failed", extra={"event_type": event_type.value}
                )
