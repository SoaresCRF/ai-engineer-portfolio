"""Request-scoped tracing spans and one-line JSON logging (no content unless LOG_CONTENT=1)."""

import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# Message content, tool args/results and the canary are logged only when explicitly enabled.
LOG_CONTENT = os.getenv("LOG_CONTENT") == "1"

# Attributes present on every LogRecord; anything else came from `extra=`.
_STD_RECORD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class Trace:
    def __init__(self) -> None:
        self.spans: list[dict] = []

    @contextmanager
    def span(self, name: str, **attrs) -> Iterator[dict]:
        record: dict = {"name": name, "start": datetime.now(UTC).isoformat(), **attrs}
        began = time.perf_counter()
        try:
            yield record  # callers may add attributes known only after the work (e.g. tokens)
        finally:
            record["duration_ms"] = (time.perf_counter() - began) * 1000
            self.spans.append(record)

    def to_dict(self) -> dict:
        return {"request_id": request_id_var.get(), "spans": self.spans}


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        payload.update({k: v for k, v in vars(record).items() if k not in _STD_RECORD_ATTRS})
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
