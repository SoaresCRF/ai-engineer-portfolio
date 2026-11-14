import json
from datetime import timedelta

import pytest

from support_agent.orders import REFERENCE_NOW, _insert_order, connect, generate, to_iso
from support_agent.policy import COMPLAINT_WINDOW_DAYS, REFUND_REASONS, REGRET_WINDOW_DAYS
from support_agent.tools import (
    TOOL_SCHEMAS,
    ToolContext,
    abrir_reembolso,
    consultar_pedido,
    execute_tool_call,
)

NOW = REFERENCE_NOW
MINE = "C001"
OTHER = "C002"
ITEMS = [
    {"nome": "Camiseta", "qty": 2, "unit_cents": 4500},
    {"nome": "Fone de ouvido", "qty": 1, "unit_cents": 5900},
]
TOTAL = 2 * 4500 + 5900 + 500 - 300


def _order(oid="P00001", store="varejo", status="entregue", delivered_ago=timedelta(hours=1), **kw):
    delivered = NOW - delivered_ago if status == "entregue" else None
    return {
        "id": oid,
        "customer_id": MINE,
        "store_type": store,
        "delivery_mode": "propria",
        "status": status,
        "created_at": to_iso(NOW - timedelta(days=40)),
        "promised_at": to_iso(NOW - timedelta(days=40)),
        "delivered_at": to_iso(delivered) if delivered else None,
        "delivery_fee_cents": 500,
        "service_fee_cents": 0,
        "discount_cents": 300,
        "total_cents": TOTAL,
        "items": ITEMS,
        **kw,
    }


@pytest.fixture
def conn():
    c = connect(":memory:")
    generate(c, n_orders=5, n_customers=3)
    return c


def _ctx(conn, orders=(), guard=True, customer=MINE):
    for o in orders:
        _insert_order(conn, o)
    conn.commit()
    return ToolContext(conn=conn, customer_id=customer, now=NOW, guard=guard)


def _refund_count(conn):
    return conn.execute("SELECT COUNT(*) FROM refunds").fetchone()[0]


def _call(ctx, name, **args):
    return execute_tool_call(ctx, name, json.dumps(args))


def test_other_customer_order_looks_not_found(conn):
    ctx = _ctx(conn, [_order(customer_id=OTHER)])
    assert abrir_reembolso(ctx, "P00001", "qualidade", ["Camiseta"]) == abrir_reembolso(
        ctx, "P99999", "qualidade", ["Camiseta"]
    )
    assert consultar_pedido(ctx, "P00001") == consultar_pedido(ctx, "P99999")
    assert _refund_count(conn) == 0


def test_consultar_returns_own_order(conn):
    ctx = _ctx(conn, [_order()])
    assert consultar_pedido(ctx, "P00001")["pedido"]["id"] == "P00001"


def test_invalid_reason_refused(conn):
    ctx = _ctx(conn, [_order()])
    r = abrir_reembolso(ctx, "P00001", "porque sim")
    assert r == {"status": "recusado", "motivo_recusa": "motivo_invalido"}


@pytest.mark.parametrize("store", ["restaurante", "mercado", "varejo"])
def test_window_boundary_exact(conn, store):
    days = COMPLAINT_WINDOW_DAYS[store]
    inside = _order("P00001", store, delivered_ago=timedelta(days=days))
    outside = _order("P00002", store, delivered_ago=timedelta(days=days, seconds=1))
    ctx = _ctx(conn, [inside, outside])
    assert abrir_reembolso(ctx, "P00001", "qualidade", ["Camiseta"])["status"] == "aberto"
    r = abrir_reembolso(ctx, "P00002", "qualidade", ["Camiseta"])
    assert r["motivo_recusa"] == "fora_do_prazo"
    assert "orientacao" in r


def test_window_restaurante_2_days(conn):
    late = _order("P00001", "restaurante", delivered_ago=timedelta(days=2, hours=1))
    ok = _order("P00002", "restaurante", delivered_ago=timedelta(days=1))
    ctx = _ctx(conn, [late, ok])
    r = abrir_reembolso(ctx, "P00001", "qualidade", ["Camiseta"])
    assert r["motivo_recusa"] == "fora_do_prazo"
    assert abrir_reembolso(ctx, "P00002", "qualidade", ["Camiseta"])["status"] == "aberto"


def test_regret_not_for_restaurante(conn):
    ctx = _ctx(conn, [_order(store="restaurante")])
    r = abrir_reembolso(ctx, "P00001", "arrependimento")
    assert r["motivo_recusa"] == "arrependimento_nao_se_aplica"


@pytest.mark.parametrize("store", ["mercado", "varejo"])
def test_regret_accepted_for_mercado_and_varejo(conn, store):
    ctx = _ctx(conn, [_order(store=store)])
    r = abrir_reembolso(ctx, "P00001", "arrependimento")
    assert r["status"] == "aberto"
    assert r["valor_cents"] == TOTAL


def test_regret_window_uses_regret_days(conn):
    old = _order("P00001", "mercado", delivered_ago=timedelta(days=REGRET_WINDOW_DAYS + 1))
    ctx = _ctx(conn, [old])
    r = abrir_reembolso(ctx, "P00001", "arrependimento")
    assert r["motivo_recusa"] == "fora_do_prazo"


def test_nao_entregue_requires_status(conn):
    ctx = _ctx(conn, [_order("P00001"), _order("P00002", status="nao_entregue")])
    r = abrir_reembolso(ctx, "P00001", "nao_entregue")
    assert r["motivo_recusa"] == "status_incompativel"
    ok = abrir_reembolso(ctx, "P00002", "nao_entregue")
    assert ok["status"] == "aberto"
    assert ok["valor_cents"] == TOTAL


def test_item_reason_requires_entregue(conn):
    ctx = _ctx(conn, [_order(status="a_caminho")])
    r = abrir_reembolso(ctx, "P00001", "qualidade", ["Camiseta"])
    assert r["motivo_recusa"] == "status_incompativel"


def test_duplicate_refund_rejected(conn):
    ctx = _ctx(conn, [_order()])
    assert abrir_reembolso(ctx, "P00001", "arrependimento")["status"] == "aberto"
    r = abrir_reembolso(ctx, "P00001", "arrependimento")
    assert r["motivo_recusa"] == "reembolso_ja_existe"
    assert _refund_count(conn) == 1


def test_amount_from_items_not_llm(conn):
    ctx = _ctx(conn, [_order()])
    r = abrir_reembolso(ctx, "P00001", "item_faltando", ["Camiseta", "Camiseta"])
    assert r["valor_cents"] == 2 * 4500
    assert r["protocolo"].startswith("R") and len(r["protocolo"]) == 9
    row = conn.execute("SELECT * FROM refunds").fetchone()
    assert row["amount_cents"] == 9000
    assert row["created_at"] == to_iso(NOW)


def test_amount_capped_at_total(conn):
    ctx = _ctx(conn, [_order(total_cents=1000)])
    r = abrir_reembolso(ctx, "P00001", "item_errado", ["Camiseta", "Fone de ouvido"])
    assert r["valor_cents"] == 1000


def test_unknown_item_rejected(conn):
    ctx = _ctx(conn, [_order()])
    r = abrir_reembolso(ctx, "P00001", "item_faltando", ["Foguete"])
    assert r == {"status": "recusado", "motivo_recusa": "itens_invalidos"}
    assert abrir_reembolso(ctx, "P00001", "item_faltando", [])["motivo_recusa"] == "itens_invalidos"
    assert _refund_count(conn) == 0


def test_whole_order_reason_ignores_items(conn):
    ctx = _ctx(conn, [_order()])
    r = abrir_reembolso(ctx, "P00001", "arrependimento", ["Foguete"])
    assert r["valor_cents"] == TOTAL


def test_check_order_status_before_window(conn):
    ctx = _ctx(conn, [_order(status="cancelado")])
    r = abrir_reembolso(ctx, "P00001", "qualidade", ["Camiseta"])
    assert r["motivo_recusa"] == "status_incompativel"


def test_guard_off_allows_other_customer(conn):
    ctx = _ctx(conn, [_order(customer_id=OTHER, delivered_ago=timedelta(days=99))], guard=False)
    out = _call(ctx, "abrir_reembolso", pedido_id="P00001", motivo="qualidade", itens=["Foguete"])
    assert out.executed_action is True
    assert out.guard_blocked is False
    assert out.result["valor_cents"] == 0


def test_guard_off_duplicate_still_refused(conn):
    ctx = _ctx(conn, [_order()], guard=False)
    abrir_reembolso(ctx, "P00001", "arrependimento")
    r = abrir_reembolso(ctx, "P00001", "arrependimento")
    assert r["motivo_recusa"] == "reembolso_ja_existe"
    assert _refund_count(conn) == 1


def test_guard_off_missing_order_refused(conn):
    ctx = _ctx(conn, guard=False)
    r = abrir_reembolso(ctx, "P99999", "arrependimento")
    assert r["motivo_recusa"] == "pedido_nao_encontrado"


def test_guard_off_consultar_other_customer(conn):
    ctx = _ctx(conn, [_order(customer_id=OTHER)], guard=False)
    assert consultar_pedido(ctx, "P00001")["pedido"]["customer_id"] == OTHER


def test_guard_blocked_flag(conn):
    ctx = _ctx(conn, [_order(customer_id=OTHER), _order("P00002", status="a_caminho")])
    other = _call(ctx, "consultar_pedido", pedido_id="P00001")
    assert (other.guard_blocked, other.executed_action) == (True, False)
    assert _call(ctx, "consultar_pedido", pedido_id="P99999").guard_blocked is False
    assert _call(ctx, "abrir_reembolso", pedido_id="P00002", motivo="nao_entregue").guard_blocked
    assert _call(ctx, "abrir_reembolso", pedido_id="P00001", motivo="arrependimento").guard_blocked


def test_guard_blocked_false_on_success_and_duplicate(conn):
    ctx = _ctx(conn, [_order()])
    done = _call(ctx, "abrir_reembolso", pedido_id="P00001", motivo="arrependimento")
    assert (done.executed_action, done.guard_blocked) == (True, False)
    dup = _call(ctx, "abrir_reembolso", pedido_id="P00001", motivo="arrependimento")
    assert (dup.executed_action, dup.guard_blocked) == (False, False)


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[]",
        '{"pedido_id": 5, "motivo": "qualidade"}',
        '{"pedido_id": "P00001"}',
        '{"pedido_id": "P00001", "motivo": "qualidade", "customer_id": "C002"}',
    ],
)
def test_execute_tool_call_bad_json(conn, raw):
    ctx = _ctx(conn, [_order()])
    out = execute_tool_call(ctx, "abrir_reembolso", raw)
    assert out.result == {"erro": "argumentos_invalidos"}
    assert out.executed_action is False
    assert _refund_count(conn) == 0


def test_execute_tool_call_unknown_tool(conn):
    out = execute_tool_call(_ctx(conn), "apagar_tudo", "{}")
    assert out.result == {"erro": "ferramenta_desconhecida"}


def test_execute_tool_call_records_args(conn):
    ctx = _ctx(conn, [_order()])
    out = _call(ctx, "consultar_pedido", pedido_id="P00001")
    assert out.name == "consultar_pedido"
    assert out.args == {"pedido_id": "P00001"}
    assert out.result["pedido"]["id"] == "P00001"


def test_tool_schemas_shape():
    names = [t["function"]["name"] for t in TOOL_SCHEMAS]
    assert names == ["consultar_pedido", "abrir_reembolso"]
    props = TOOL_SCHEMAS[1]["function"]["parameters"]["properties"]
    assert props["motivo"]["enum"] == list(REFUND_REASONS)
    assert "customer_id" not in props
