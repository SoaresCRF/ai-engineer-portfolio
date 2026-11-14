import asyncio
import json
import sqlite3
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from support_agent import tracing
from support_agent.api import _SecurityMiddleware, create_app
from support_agent.corpus import Chunk
from support_agent.llm import AssistantTurn, ToolCall
from support_agent.orders import REFERENCE_NOW, connect, generate, to_iso

CHUNKS = [Chunk("c1", "cdc:art49", "Arrependimento em 7 dias.")]
TOKENS = {"tok-c001": "C001", "tok-c002": "C002"}
AUTH = {"Authorization": "Bearer tok-c001"}
HSTS = "max-age=31536000; includeSubDomains"
CSP = "default-src 'none'; frame-ancestors 'none'"


class ScriptedLLM:
    def __init__(self, turns):
        self.turns = list(turns)

    def chat(self, messages, tools):
        return self.turns.pop(0)


class BrokenLLM:
    def chat(self, messages, tools):
        raise RuntimeError("secret-internal-detail")


def _order(oid, customer):
    now = REFERENCE_NOW
    return {
        "id": oid,
        "customer_id": customer,
        "store_type": "varejo",
        "delivery_mode": "propria",
        "status": "entregue",
        "created_at": to_iso(now - timedelta(days=2)),
        "promised_at": to_iso(now - timedelta(days=2)),
        "delivered_at": to_iso(now - timedelta(hours=1)),
        "delivery_fee_cents": 0,
        "service_fee_cents": 0,
        "discount_cents": 0,
        "total_cents": 1000,
        "items": [{"nome": "Camiseta", "qty": 1, "unit_cents": 1000}],
    }


def _tool_turn(name, args):
    return AssistantTurn("", [ToolCall("t1", name, json.dumps(args))], 1, 1)


def _chat(client, content="Oi", headers=AUTH, **body):
    payload = {"messages": [{"role": "user", "content": content}], **body}
    return client.post("/chat", json=payload, headers=headers)


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "orders.sqlite"
    conn = connect(path)
    generate(conn, n_orders=5, n_customers=3, extra_orders=[_order("P90001", "C002")])
    conn.close()
    return path


@pytest.fixture
def trace_path(tmp_path, monkeypatch):
    path = tmp_path / "traces.jsonl"
    monkeypatch.setenv("TRACE_PATH", str(path))
    return path


@pytest.fixture
def make_client(db_path, trace_path):
    def make(llm=None, **kwargs):
        llm = llm or ScriptedLLM([AssistantTurn("Olá [cdc:art49]")] * 10)
        app = create_app(
            llm=llm,
            retrieve=lambda q, k: CHUNKS,
            conn_factory=lambda: connect(db_path),
            tokens=TOKENS,
            now=lambda: REFERENCE_NOW,
            **kwargs,
        )
        return TestClient(app, raise_server_exceptions=False)

    return make


def test_health(make_client):
    res = make_client().get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_bearer_scheme_is_case_insensitive(make_client):
    assert _chat(make_client(), headers={"Authorization": "bearer tok-c001"}).status_code == 200


def test_401_advertises_bearer_scheme(make_client):
    assert _chat(make_client(), headers={}).headers["WWW-Authenticate"] == "Bearer"


def test_middleware_resets_request_id_var():
    seen = []

    async def app(scope, receive, send):
        seen.append(tracing.request_id_var.get())
        await send({"type": "http.response.start", "status": 200, "headers": []})

    async def send(message):
        pass

    asyncio.run(_SecurityMiddleware(app)({"type": "http", "path": "/x"}, None, send))
    assert seen[0] != "-"
    assert tracing.request_id_var.get() == "-"


def test_chat_happy_path(make_client):
    body = _chat(make_client()).json()
    assert body["answer"] == "Olá [cdc:art49]"
    assert body["sources"] == ["cdc:art49"]
    assert body["tool_calls"] == []
    assert body["request_id"]


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer nope"}, {"Authorization": "x"}])
def test_chat_requires_token(make_client, headers):
    res = _chat(make_client(), headers=headers)
    assert res.status_code == 401
    assert res.json()["erro"] == "nao_autorizado"


@pytest.mark.parametrize("role", ["system", "tool"])
def test_chat_rejects_non_chat_roles(make_client, role):
    res = make_client().post(
        "/chat", json={"messages": [{"role": role, "content": "x"}]}, headers=AUTH
    )
    assert res.status_code == 422


def test_chat_rejects_oversized_message(make_client):
    assert _chat(make_client(), content="x" * 2001).status_code == 422


def test_last_message_must_be_user(make_client):
    messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    res = make_client().post("/chat", json={"messages": messages}, headers=AUTH)
    assert res.status_code == 422


def test_body_cannot_disable_guard(make_client):
    assert _chat(make_client(), defense="none").status_code == 422


def test_validation_error_does_not_echo_input(make_client):
    res = _chat(make_client(), content="x" * 2001 + "SECRET-CUSTOMER-TEXT")
    assert res.status_code == 422
    assert "SECRET-CUSTOMER-TEXT" not in res.text
    assert res.json()["erro"] == "requisicao_invalida"
    assert res.json()["request_id"] == res.headers["X-Request-ID"]


def test_request_id_and_security_headers(make_client):
    res = make_client().get("/health")
    assert len(res.headers["X-Request-ID"]) == 32
    assert res.headers["X-Content-Type-Options"] == "nosniff"
    assert res.headers["X-Frame-Options"] == "DENY"
    assert res.headers["Strict-Transport-Security"] == HSTS
    assert res.headers["Content-Security-Policy"] == CSP


def test_security_headers_on_errors(make_client):
    res = _chat(make_client(), headers={})
    assert res.headers["X-Content-Type-Options"] == "nosniff"
    assert res.headers["Strict-Transport-Security"] == HSTS


def test_csp_absent_on_docs(make_client):
    res = make_client().get("/docs")
    assert res.status_code == 200
    assert "Content-Security-Policy" not in res.headers
    assert res.headers["X-Content-Type-Options"] == "nosniff"


def test_rate_limit_returns_429(make_client):
    client = make_client(rate_limit_per_min=2)
    assert _chat(client).status_code == 200
    assert _chat(client).status_code == 200
    res = _chat(client)
    assert res.status_code == 429
    assert res.json()["erro"] == "limite_excedido"
    assert _chat(client, headers={"Authorization": "Bearer tok-c002"}).status_code == 200


def test_unauthorized_requests_do_not_consume_rate_limit(make_client):
    client = make_client(rate_limit_per_min=1)
    assert _chat(client, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert _chat(client).status_code == 200


def test_cannot_refund_other_customers_order_via_api(make_client, db_path):
    llm = ScriptedLLM(
        [
            _tool_turn("abrir_reembolso", {"pedido_id": "P90001", "motivo": "qualidade"}),
            AssistantTurn("Não consegui."),
        ]
    )
    res = _chat(make_client(llm))  # authenticated as C001; P90001 belongs to C002
    assert res.status_code == 200
    assert res.json()["tool_calls"] == [{"name": "abrir_reembolso", "status": "recusado"}]
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM refunds").fetchone()[0] == 0
    conn.close()


def test_tool_call_status_mapping(make_client):
    llm = ScriptedLLM(
        [
            _tool_turn("consultar_pedido", {"pedido_id": "P90001"}),
            _tool_turn("inexistente", {}),
            AssistantTurn("fim"),
        ]
    )
    res = _chat(make_client(llm), headers={"Authorization": "Bearer tok-c002"})
    assert res.json()["tool_calls"] == [
        {"name": "consultar_pedido", "status": "ok"},
        {"name": "inexistente", "status": "ferramenta_desconhecida"},
    ]


def test_trace_appended_with_request_id(make_client, trace_path):
    res = _chat(make_client())
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    trace = json.loads(lines[0])
    assert trace["request_id"] == res.headers["X-Request-ID"] == res.json()["request_id"]
    assert [s["name"] for s in trace["spans"]] == ["retrieve", "llm_call"]


def test_chat_completed_log_has_no_content(make_client, caplog):
    caplog.set_level("INFO")
    _chat(make_client(), content="TEXTO-SECRETO")
    record = next(r for r in caplog.records if r.getMessage() == "chat_completed")
    assert record.sources == ["cdc:art49"]
    assert record.tool_names == []
    assert "TEXTO-SECRETO" not in json.dumps(vars(record), default=str)


def test_internal_error_is_generic(make_client, caplog):
    res = _chat(make_client(BrokenLLM()))
    assert res.status_code == 500
    body = res.json()
    assert body["erro"] == "erro_interno"
    assert body["request_id"] == res.headers["X-Request-ID"]
    assert "secret-internal-detail" not in res.text
    assert "Traceback" not in res.text
    assert res.headers["X-Content-Type-Options"] == "nosniff"
    assert any(r.exc_info for r in caplog.records)  # logged with the stack trace


def test_build_app_from_env_requires_variables(monkeypatch):
    from support_agent.api import build_app_from_env

    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="LLM_BASE_URL"):
        build_app_from_env()
