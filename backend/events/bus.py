from collections import defaultdict
from typing import Callable

from backend.events.types import EventType

EventHandler = Callable[[dict], None]


class EventBus:
    """In-process, synchronous event bus for Phase 1.

    No external broker (Redis/NATS/Kafka) is used. `emit` calls each
    subscribed handler synchronously in registration order. A future
    broker-backed implementation would preserve this subscribe/emit
    interface.
    """

    def __init__(self) -> None:
        self._subscribers: dict[EventType, list[EventHandler]] = defaultdict(list)

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        self._subscribers[event_type].append(handler)

    def emit(self, event_type: EventType, payload: dict) -> None:
        for handler in self._subscribers.get(event_type, []):
            handler(payload)
