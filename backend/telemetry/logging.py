import contextvars
import json
import logging
import logging.handlers
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from backend.config.settings import Settings

_REQUEST_CONTEXT_FIELDS = (
    "request_id",
    "trace_id",
    "provider",
    "model",
    "latency_ms",
    "cost_estimate",
)

_DEFAULT_REQUEST_CONTEXT: dict = {field: None for field in _REQUEST_CONTEXT_FIELDS}

_request_context: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "request_context", default=_DEFAULT_REQUEST_CONTEXT
)


@contextmanager
def request_context(**fields) -> Iterator[None]:
    """Temporarily bind request-scoped fields onto every log record emitted
    within this block. Backed by contextvars, so concurrent asyncio tasks
    each see their own isolated copy -- one request's fields never leak
    into another's log lines."""
    unknown = set(fields) - set(_REQUEST_CONTEXT_FIELDS)
    if unknown:
        raise ValueError(f"Unknown request context field(s): {sorted(unknown)}")

    merged = {**_request_context.get(), **fields}
    token = _request_context.set(merged)
    try:
        yield
    finally:
        _request_context.reset(token)


def clear_request_context() -> None:
    _request_context.set(_DEFAULT_REQUEST_CONTEXT)


def get_request_context() -> dict:
    return dict(_request_context.get())


_RESERVED_LOG_RECORD_ATTRS = frozenset(logging.LogRecord(
    "", 0, "", 0, "", (), None
).__dict__) | {"message", "asctime", "component"}


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str, environment: str) -> None:
        super().__init__()
        self._service = service
        self._environment = environment
        self._hostname = socket.gethostname()

    def format(self, record: logging.LogRecord) -> str:
        context = _request_context.get()
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "message": record.getMessage(),
            "service": self._service,
            "environment": self._environment,
            "hostname": self._hostname,
            "component": getattr(record, "component", None),
        }
        for field in _REQUEST_CONTEXT_FIELDS:
            payload[field] = context.get(field)
        # Surface any caller-supplied `extra={...}` fields (e.g. event
        # payloads logged by subscribers.py) that aren't already part of
        # the fixed request-context schema above -- without this they are
        # silently dropped by the formatter.
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_RECORD_ATTRS and key not in payload:
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(settings: Settings, log_dir: str = "logs") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(settings.log_level)
    root.handlers.clear()

    formatter = JsonFormatter(service="llm-cost-autopilot", environment=settings.environment)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        f"{log_dir}/app.log", maxBytes=10_000_000, backupCount=5
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


class _MergingLoggerAdapter(logging.LoggerAdapter):
    """LoggerAdapter's default process() replaces kwargs['extra'] with
    self.extra entirely, silently dropping any extra= a caller passes to
    the log call. Merge them instead so both the adapter-bound `component`
    and caller-supplied fields (e.g. subscribers.py's event payloads)
    reach the formatter."""

    def process(self, msg, kwargs):
        kwargs["extra"] = {**self.extra, **kwargs.get("extra", {})}
        return msg, kwargs


def get_logger(component: str) -> logging.LoggerAdapter:
    """Centralized logger accessor. Application code must always go
    through this instead of calling logging.getLogger() directly, so
    every log line consistently carries `component` and merges the
    request-scoped context fields via JsonFormatter."""
    return _MergingLoggerAdapter(logging.getLogger(component), {"component": component})
