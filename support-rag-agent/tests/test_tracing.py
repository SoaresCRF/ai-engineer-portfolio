import json
import logging
import re
import sys

import pytest

from support_agent import tracing
from support_agent.agent import AgentConfig, run_agent
from support_agent.llm import AssistantTurn, ToolCall, strip_think
from support_agent.orders import REFERENCE_NOW, connect
from support_agent.tools import ToolContext


@pytest.fixture
def restore_logging():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def test_setup_logging_quiets_http_clients(restore_logging):
    tracing.setup_logging("INFO")
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING


def test_span_records_duration(monkeypatch):
    clock = iter([100.0, 100.25])
    monkeypatch.setattr(tracing.time, "perf_counter", lambda: next(clock))
    trace = tracing.Trace()
    with trace.span("work", kind="x"):
        pass
    (span,) = trace.to_dict()["spans"]
    assert span["name"] == "work"
    assert span["kind"] == "x"
    assert span["duration_ms"] == 250.0
    assert span["start"]


def test_log_has_request_id_and_no_content_by_default(capsys, monkeypatch, restore_logging):
    monkeypatch.setattr(tracing, "LOG_CONTENT", False)
    tracing.setup_logging("INFO")
    token = tracing.request_id_var.set("req-123")
    secret_user_text = "meu cpf e 123.456.789-00"
    turns = [AssistantTurn("", [ToolCall("t", "consultar_pedido", '{"pedido_id": "P1"}')])]
    try:
        run_agent(
            [{"role": "user", "content": secret_user_text}],
            ToolContext(conn=connect(":memory:"), customer_id="C001", now=REFERENCE_NOW),
            AgentConfig(max_steps=1),
            lambda q, k: [],
            _OneShotLLM(turns),
            "CANARY-xyz",
            "nonce",
        )
    finally:
        tracing.request_id_var.reset(token)
    out = capsys.readouterr().err
    entry = json.loads(out.strip().splitlines()[-1])
    assert entry["request_id"] == "req-123"
    assert entry["level"] == "WARNING"
    assert entry["ts"]
    assert "CANARY-xyz" not in out
    assert secret_user_text not in out
    assert "P1" not in out


class _OneShotLLM:
    def __init__(self, turns):
        self.turns = turns

    def chat(self, messages, tools):
        return self.turns[0]


def test_strip_think_removes_reasoning_block():
    assert strip_think("<think>\nplan\n</think>\nOlá!") == "Olá!"


def test_json_formatter_includes_exception_and_utc_z_timestamp():
    try:
        raise RuntimeError("boom-detail")
    except RuntimeError:
        record = logging.getLogger("x").makeRecord(
            "x", logging.ERROR, __file__, 1, "unhandled", (), sys.exc_info()
        )
    payload = json.loads(tracing._JsonFormatter().format(record))
    assert "RuntimeError: boom-detail" in payload["exc"]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z", payload["ts"])
