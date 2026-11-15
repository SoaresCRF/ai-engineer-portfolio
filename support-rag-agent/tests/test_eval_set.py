import json
import unicodedata
from pathlib import Path

import pytest
from pydantic import ValidationError

from support_agent import eval_set
from support_agent.eval_set import (
    Attack,
    PolicyQuestion,
    assign_splits,
    freeze,
    load,
    load_injected_orders,
    validate_all,
    verify_frozen,
)
from support_agent.orders import connect, generate

REAL_EVAL_DIR = Path(__file__).resolve().parents[1] / "data" / "eval"
MSG = [{"role": "user", "content": "Oi"}]


def _normalize(text: str) -> str:  # same as tests/test_policy.py
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def _write(eval_dir: Path, kind: str, rows: list[dict]) -> None:
    for other in eval_set.KINDS:  # validate_all needs every file to exist
        (eval_dir / f"{other}.jsonl").touch()
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (eval_dir / f"{kind}.jsonl").write_text(text, encoding="utf-8", newline="\n")


def _policy(i: str, gold=("cdc:art49",), **kw) -> dict:
    base = {
        "id": i, "split": "dev", "author": "claude", "question": "q", "gold_passages": list(gold),
        "reference_answer": "a", "answerable": True,
    }  # fmt: skip
    return base | kw


@pytest.fixture
def conn():
    c = connect(":memory:")
    generate(c)
    return c


@pytest.fixture
def customer(conn):
    return conn.execute("SELECT id FROM customers LIMIT 1").fetchone()[0]


def test_unknown_gold_passage_rejected(tmp_path, conn):
    _write(tmp_path, "policy_questions", [_policy("p1", gold=("cdc:art999",))])
    with pytest.raises(ValueError, match="cdc:art999"):
        validate_all({"cdc:art49"}, conn, eval_dir=tmp_path)


def test_valid_set_passes_and_ids_unique_across_kinds(tmp_path, conn, customer):
    _write(tmp_path, "policy_questions", [_policy("x1")])
    benign = {
        "id": "b1",
        "split": "test",
        "author": "user",
        "customer_id": customer,
        "messages": MSG,
    }
    _write(tmp_path, "benign", [benign])
    validate_all({"cdc:art49"}, conn, eval_dir=tmp_path)
    _write(tmp_path, "benign", [benign | {"id": "x1"}])
    with pytest.raises(ValueError, match="x1"):
        validate_all({"cdc:art49"}, conn, eval_dir=tmp_path)


def test_unsplit_item_rejected(tmp_path, conn):
    _write(tmp_path, "policy_questions", [_policy("p1", split=None)])
    with pytest.raises(ValueError, match="split"):
        validate_all({"cdc:art49"}, conn, eval_dir=tmp_path)


def test_unknown_customer_and_order_rejected(tmp_path, conn, customer):
    row = {"id": "b1", "split": "dev", "author": "claude", "customer_id": "nope", "messages": MSG}
    _write(tmp_path, "benign", [row])
    with pytest.raises(ValueError, match="nope"):
        validate_all(set(), conn, eval_dir=tmp_path)
    ghost = [{"role": "user", "content": "Cadê o pedido P00007?"}]
    _write(tmp_path, "benign", [row | {"customer_id": customer, "messages": ghost}])
    with pytest.raises(ValueError, match="P00007"):
        validate_all(set(), conn, eval_dir=tmp_path)


def test_injected_orders_id_range_enforced(tmp_path, conn):
    _write(tmp_path, "benign", [])
    (tmp_path / "injected_orders.json").write_text(json.dumps([{"id": "P12345"}]), encoding="utf-8")
    assert load_injected_orders(tmp_path) == [{"id": "P12345"}]
    with pytest.raises(ValueError, match="P12345"):
        validate_all(set(), conn, eval_dir=tmp_path)


def test_message_contract():
    for bad in ([], [{"role": "system", "content": "x"}], [{"role": "assistant", "content": "x"}]):
        with pytest.raises(ValidationError):
            eval_set.Benign(id="b", author="claude", customer_id="c", messages=bad)


def test_attack_forbidden_validated_by_target():
    base = {"id": "a", "author": "claude", "customer_id": "c", "vector": "direto", "messages": MSG}
    with pytest.raises(ValidationError):
        Attack(**base, target="acao", forbidden={"tool": "abrir_reembolso"})
    with pytest.raises(ValidationError):
        Attack(**base, target="frase", forbidden={"phrase": ""})
    with pytest.raises(ValidationError):
        Attack(**base, target="vazamento", forbidden={"phrase": "x"})
    Attack(**base, target="vazamento", forbidden={})


def test_answerable_false_requires_no_gold(tmp_path, conn):
    _write(tmp_path, "policy_questions", [_policy("p1", answerable=False)])
    with pytest.raises(ValueError, match="p1"):
        validate_all({"cdc:art49"}, conn, eval_dir=tmp_path)


def _items(strata: dict[str, int]) -> list[PolicyQuestion]:
    return [
        PolicyQuestion(**_policy(f"{s}{n:03d}", split=None, question=s))
        for s, count in strata.items()
        for n in range(count)
    ]


def _splits(items) -> dict[str, str | None]:
    return {i.id: i.split for i in items}


def test_split_is_stratified_and_deterministic():
    runs = []
    for _ in range(2):
        items = _items({"a": 10, "b": 10, "c": 10})
        assign_splits(items, lambda i: i.question, dev_frac=0.4)
        runs.append(_splits(items))
        for s in "abc":
            assert sum(v == "dev" for k, v in runs[-1].items() if k[0] == s) == 4
    assert runs[0] == runs[1]


def test_split_total_dev_is_exact_with_fractional_shares():
    items = _items({"a": 7, "b": 7, "c": 6})
    assign_splits(items, lambda i: i.question, dev_frac=0.6)
    dev = [i for i in items if i.split == "dev"]
    assert len(dev) == 12
    for s, n in {"a": 7, "b": 7, "c": 6}.items():
        got = sum(i.question == s for i in dev)
        assert int(n * 0.6) <= got <= int(n * 0.6) + 1


def test_verify_frozen_detects_edit(tmp_path):
    (tmp_path / "a.jsonl").write_bytes(b"one\n")
    freeze(tmp_path)
    verify_frozen(tmp_path)
    (tmp_path / "a.jsonl").write_bytes(b"two\n")
    with pytest.raises(ValueError, match="a.jsonl"):
        verify_frozen(tmp_path)


def test_verify_frozen_detects_added_and_removed_files(tmp_path):
    (tmp_path / "a.jsonl").write_bytes(b"one\n")
    freeze(tmp_path)
    (tmp_path / "b.jsonl").write_bytes(b"x")
    with pytest.raises(ValueError, match="b.jsonl"):
        verify_frozen(tmp_path)
    (tmp_path / "b.jsonl").unlink()
    (tmp_path / "a.jsonl").unlink()
    with pytest.raises(ValueError, match="a.jsonl"):
        verify_frozen(tmp_path)


def test_load_missing_kind_is_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load("benign", eval_dir=tmp_path)


def test_eval_set_is_frozen():
    if not (REAL_EVAL_DIR / "manifest.json").exists():
        pytest.skip("eval set not frozen yet")
    verify_frozen(REAL_EVAL_DIR)


def test_eval_set_has_no_brand():
    if not (REAL_EVAL_DIR / "manifest.json").exists():
        pytest.skip("eval set not frozen yet")
    files = [p for p in REAL_EVAL_DIR.iterdir() if p.is_file()]
    assert files
    for path in files:
        assert "ifood" not in _normalize(path.read_text(encoding="utf-8")), path
