import asyncio
import io
import json
import logging

import pytest

from backend.config.settings import Settings
from backend.telemetry.logging import (
    JsonFormatter,
    clear_request_context,
    configure_logging,
    get_logger,
    get_request_context,
    request_context,
)


def _capture_logger(component: str) -> tuple[logging.LoggerAdapter, io.StringIO]:
    adapter = get_logger(component)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter(service="llm-cost-autopilot", environment="test"))
    adapter.logger.addHandler(handler)
    adapter.logger.setLevel(logging.INFO)
    adapter.logger.propagate = False
    return adapter, stream


def _read_last_record(stream: io.StringIO) -> dict:
    lines = [line for line in stream.getvalue().strip().splitlines() if line]
    return json.loads(lines[-1])


def test_log_record_contains_all_required_fields():
    logger, stream = _capture_logger("test_component")
    logger.info("hello")

    record = _read_last_record(stream)

    for field in (
        "timestamp", "level", "message", "service", "environment", "hostname",
        "component", "request_id", "trace_id", "provider", "model",
        "latency_ms", "cost_estimate",
    ):
        assert field in record

    assert record["message"] == "hello"
    assert record["level"] == "INFO"
    assert record["service"] == "llm-cost-autopilot"
    assert record["environment"] == "test"
    assert record["component"] == "test_component"
    assert record["hostname"]


def test_json_output_is_valid_json():
    logger, stream = _capture_logger("test_component")
    logger.info("hello")

    record = _read_last_record(stream)
    assert isinstance(record, dict)


def test_optional_request_fields_serialize_as_null_when_unset():
    clear_request_context()
    logger, stream = _capture_logger("test_component")
    logger.info("hello")

    record = _read_last_record(stream)
    assert record["request_id"] is None
    assert record["trace_id"] is None
    assert record["provider"] is None
    assert record["model"] is None
    assert record["latency_ms"] is None
    assert record["cost_estimate"] is None


def test_request_context_sets_fields_on_log_records():
    clear_request_context()
    logger, stream = _capture_logger("test_component")

    with request_context(request_id="req-1", trace_id="trace-1", provider="openai", model="gpt-4o-mini"):
        logger.info("inside context")

    record = _read_last_record(stream)
    assert record["request_id"] == "req-1"
    assert record["trace_id"] == "trace-1"
    assert record["provider"] == "openai"
    assert record["model"] == "gpt-4o-mini"


def test_request_context_resets_after_block_exits():
    clear_request_context()
    logger, stream = _capture_logger("test_component")

    with request_context(request_id="req-1"):
        pass
    logger.info("outside context")

    record = _read_last_record(stream)
    assert record["request_id"] is None


def test_request_context_rejects_unknown_fields():
    with pytest.raises(ValueError):
        with request_context(not_a_real_field="oops"):
            pass


def test_clear_request_context_resets_to_defaults():
    with request_context(request_id="req-1"):
        clear_request_context()
        assert get_request_context()["request_id"] is None


async def test_context_isolation_between_concurrent_tasks():
    results = {}

    async def run(name, request_id):
        with request_context(request_id=request_id):
            await asyncio.sleep(0.01)
            results[name] = get_request_context()["request_id"]

    await asyncio.gather(run("a", "req-a"), run("b", "req-b"))

    assert results == {"a": "req-a", "b": "req-b"}


def test_configure_logging_creates_log_dir_and_writes_json_lines(tmp_path):
    settings = Settings(_env_file=None, environment="test")
    log_dir = str(tmp_path / "logs")

    configure_logging(settings, log_dir=log_dir)
    logging.getLogger("smoke").info("boot")

    log_file = tmp_path / "logs" / "app.log"
    assert log_file.exists()
    last_line = [line for line in log_file.read_text().strip().splitlines() if line][-1]
    json.loads(last_line)


def test_get_logger_returns_logger_adapter_with_component_bound():
    logger = get_logger("my_component")
    assert isinstance(logger, logging.LoggerAdapter)
    assert logger.extra["component"] == "my_component"


def test_extra_fields_passed_to_logger_call_surface_in_output():
    """Regression test: JsonFormatter previously only rendered the fixed
    request-context whitelist, silently dropping any `extra={...}` passed
    directly to a log call -- e.g. subscribers.py's event payloads."""
    logger, stream = _capture_logger("test_component")
    logger.info("event_emitted", extra={"payload": {"score": 0.9, "escalated": False}})

    record = _read_last_record(stream)
    assert record["payload"] == {"score": 0.9, "escalated": False}


def test_extra_field_does_not_override_fixed_schema_fields():
    logger, stream = _capture_logger("test_component")
    with request_context(request_id="req-1"):
        logger.info("hello", extra={"request_id": "should-not-win"})

    record = _read_last_record(stream)
    assert record["request_id"] == "req-1"
