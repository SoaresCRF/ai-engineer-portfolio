import json
from datetime import timedelta

import pytest

from support_agent import run_eval
from support_agent.agent import AgentConfig
from support_agent.corpus import Chunk
from support_agent.eval_set import Message, ToolScenario
from support_agent.llm import AssistantTurn, ToolCall
from support_agent.orders import REFERENCE_NOW, to_iso

ORDER_ID = "P01001"
CHUNKS = [Chunk("c1", "cdc:art49", "Arrependimento em 7 dias.")]


class ScriptedLLM:
    def __init__(self, turns):
        self.turns = list(turns)

    def chat(self, messages, tools):
        return self.turns.pop(0)


def _refund_turn():
    args = {"pedido_id": ORDER_ID, "motivo": "arrependimento"}
    return AssistantTurn("", [ToolCall("t1", "abrir_reembolso", json.dumps(args))], 10, 2)


def _injected(delivered_days_before_ref):
    delivered = to_iso(REFERENCE_NOW - timedelta(days=delivered_days_before_ref))
    return {
        "id": ORDER_ID,
        "customer_id": "C001",
        "store_type": "varejo",
        "delivery_mode": "propria",
        "status": "entregue",
        "created_at": delivered,
        "promised_at": delivered,
        "delivered_at": delivered,
        "delivery_fee_cents": 0,
        "service_fee_cents": 0,
        "discount_cents": 0,
        "total_cents": 1000,
        "items": [{"nome": "Camiseta", "qty": 1, "unit_cents": 1000}],
    }


def _scenario(sid):
    return ToolScenario(
        id=sid,
        split="dev",
        author="claude",
        customer_id="C001",
        messages=[Message(role="user", content="Quero reembolso")],
        expected_tool="abrir_reembolso",
        expected_args={"pedido_id": ORDER_ID},
        expected_status="aberto",
    )


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(run_eval, "RUNS_DIR", tmp_path)

    def patch(days_before_ref, n_items):
        order = _injected(days_before_ref)
        monkeypatch.setattr(run_eval.eval_set, "load_injected_orders", lambda: [order])
        items = [_scenario(f"tool-{i}") for i in range(n_items)]
        monkeypatch.setattr(run_eval.eval_set, "load", lambda kind: items)

    return patch


def _rows(run_dir):
    lines = (run_dir / "outputs.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def _run(llm, name="r1"):
    return run_eval.run(name, AgentConfig(), "dev", ["tool_scenarios"], llm, lambda q, k: CHUNKS)


def test_run_isolates_db_between_items(setup):
    setup(days_before_ref=0, n_items=2)
    llm = ScriptedLLM([_refund_turn(), AssistantTurn("ok"), _refund_turn(), AssistantTurn("ok")])
    rows = _rows(_run(llm))
    statuses = [r["tool_outcomes"][0]["result"]["status"] for r in rows]
    assert statuses == ["aberto", "aberto"]


def test_run_uses_reference_clock_not_wall_clock(setup):
    # Delivered 3 days before REFERENCE_NOW: inside the 7-day window only under that clock.
    setup(days_before_ref=3, n_items=1)
    rows = _rows(_run(ScriptedLLM([_refund_turn(), AssistantTurn("ok")])))
    assert rows[0]["tool_outcomes"][0]["result"]["status"] == "aberto"


def test_run_writes_outputs_and_meta(setup):
    setup(days_before_ref=0, n_items=1)
    run_dir = _run(ScriptedLLM([_refund_turn(), AssistantTurn("ok")]))
    row = _rows(run_dir)[0]
    assert row["retrieved_chunk_ids"] == ["c1"]
    assert row["tokens"] == {"prompt": 10, "completion": 2}
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["date"].endswith("Z") and meta["canary"]["tool-0"].startswith("CANARY-")
    assert len(meta["manifest_sha256"]) == 64 and meta["config"]["top_k"] == 5
    with pytest.raises(FileExistsError):
        _run(ScriptedLLM([]), name=run_dir.name)
