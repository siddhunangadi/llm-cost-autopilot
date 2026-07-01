from backend.events.bus import EventBus
from backend.events.types import EventType


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
