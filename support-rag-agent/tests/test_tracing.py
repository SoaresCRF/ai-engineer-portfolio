import json
import logging
import time

from support_agent import tracing
from support_agent.llm import strip_think


def test_span_records_duration():
    trace = tracing.Trace()
    with trace.span("work", kind="x"):
        time.sleep(0.01)
    (span,) = trace.to_dict()["spans"]
    assert span["name"] == "work"
    assert span["kind"] == "x"
    assert span["duration_ms"] >= 10
    assert span["start"]


def test_log_has_request_id_and_no_content_by_default(capsys):
    tracing.setup_logging("INFO")
    token = tracing.request_id_var.set("req-123")
    try:
        logging.getLogger("t").info("done", extra={"steps": 2})
    finally:
        tracing.request_id_var.reset(token)
    line = capsys.readouterr().err.strip()
    entry = json.loads(line)
    assert entry["request_id"] == "req-123"
    assert entry["level"] == "INFO"
    assert entry["msg"] == "done"
    assert entry["steps"] == 2
    assert entry["ts"]
    assert tracing.LOG_CONTENT is False


def test_strip_think_removes_reasoning_block():
    assert strip_think("<think>\nplan\n</think>\nOlá!") == "Olá!"
