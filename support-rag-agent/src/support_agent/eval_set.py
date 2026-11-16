"""Eval set: schemas, loader, validator, stratified split and SHA-256 freeze.

Files live in data/eval/*.jsonl; they are written as UTF-8 with LF newlines so raw-byte
hashes are identical on every checkout (see .gitattributes).
"""

import hashlib
import json
import math
import random
import re
import sqlite3
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from support_agent import ROOT
from support_agent.tools import _TOOLS

EVAL_DIR = ROOT / "data" / "eval"
MANIFEST = "manifest.json"
INJECTED_ORDERS = "injected_orders.json"
INJECTED_ID = re.compile(r"^P0\d{4}$")
ORDER_REF = re.compile(r"\bP\d{5}\b")
DEFAULT_SEED = 13
NOT_FOUND_STATUS = "recusado:pedido_nao_encontrado"
TOOL_NAMES = frozenset(_TOOLS)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Message(_Strict):
    """Same contract as the API: no system turns."""

    role: Literal["user", "assistant"]
    content: str


class _Item(_Strict):
    id: str
    split: Literal["dev", "test"] | None = None  # None while drafting; validate_all rejects it
    author: Literal["claude", "user"]


class _Conversation(_Item):
    customer_id: str
    messages: list[Message]

    @model_validator(mode="after")
    def _ends_with_user(self):
        if not self.messages or self.messages[-1].role != "user":
            raise ValueError("messages must be non-empty and end with a user message")
        return self


class PolicyQuestion(_Item):
    question: str
    gold_passages: list[str]
    reference_answer: str
    answerable: bool


class ToolScenario(_Conversation):
    expected_tool: Literal["consultar_pedido", "abrir_reembolso"] | None
    expected_args: dict  # subset of the tool arguments
    expected_status: str | None = None  # "aberto" or "recusado:<motivo_recusa>"


class Attack(_Conversation):
    vector: Literal["direto", "observacao", "nome_item"]
    target: Literal["vazamento", "acao", "frase"]
    forbidden: dict

    @model_validator(mode="after")
    def _forbidden_matches_target(self):
        f = self.forbidden
        if self.target == "acao":
            ok = f.get("tool") in TOOL_NAMES and bool(f.get("pedido_id"))
        elif self.target == "frase":
            ok = bool(f.get("phrase"))
        else:
            ok = f == {}
        if not ok:
            raise ValueError(f"invalid forbidden {f!r} for target {self.target!r}")
        return self


class Benign(_Conversation):
    must_not_refuse: bool = True


KINDS: dict[str, type[_Item]] = {
    "policy_questions": PolicyQuestion,
    "tool_scenarios": ToolScenario,
    "attacks": Attack,
    "benign": Benign,
}


def dump_jsonl(path: Path, items: list[_Item]) -> None:
    lines = "".join(item.model_dump_json() + "\n" for item in items)
    path.write_text(lines, encoding="utf-8", newline="\n")


def load(kind: str, eval_dir: Path = EVAL_DIR) -> list:
    model = KINDS[kind]
    text = (eval_dir / f"{kind}.jsonl").read_text(encoding="utf-8")
    items = []
    for n, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            items.append(model.model_validate_json(line))
        except ValidationError as exc:
            raise ValueError(f"{kind}.jsonl line {n}: {exc}") from exc
    return items


def load_injected_orders(eval_dir: Path = EVAL_DIR) -> list[dict]:
    path = eval_dir / INJECTED_ORDERS
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _order_refs(item: _Item) -> set[str]:
    refs: set[str] = set()
    for source in (getattr(item, "expected_args", {}), getattr(item, "forbidden", {})):
        if source.get("pedido_id"):
            refs.add(source["pedido_id"])
    for m in getattr(item, "messages", []):
        refs.update(ORDER_REF.findall(m.content))
    return refs


def _ids(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[0] for r in conn.execute(f"SELECT id FROM {table}")}  # fixed table names


def _injected_errors(eval_dir: Path) -> list[str]:
    orders = load_injected_orders(eval_dir)
    errors = [f"injected order #{n} is not an object" for n, o in enumerate(orders, 1)
              if not isinstance(o, dict)]  # fmt: skip
    ids = [o.get("id") for o in orders if isinstance(o, dict)]
    bad = [i for i in ids if not isinstance(i, str) or not INJECTED_ID.match(i)]
    errors += [f"injected order id {i!r} must match {INJECTED_ID.pattern}" for i in bad]
    return errors + [f"duplicate injected order id {i!r}" for i in set(ids) if ids.count(i) > 1]


def _item_errors(
    kind: str, item: _Item, passages: set[str], customers: set[str], orders: set[str]
) -> list[str]:
    where = f"{kind}:{item.id}"
    errors = [f"{where}: split is not assigned"] if item.split is None else []
    if isinstance(item, PolicyQuestion):
        errors += [
            f"{where}: unknown gold passage {g!r}" for g in item.gold_passages if g not in passages
        ]
        if item.answerable == (not item.gold_passages):
            errors.append(
                f"{where}: answerable={item.answerable} but gold_passages is {item.gold_passages!r}"
            )
        return errors
    if item.customer_id not in customers:
        errors.append(f"{where}: unknown customer {item.customer_id!r}")
    refs = _order_refs(item)
    # A not-found scenario targets an order id that does not exist; only that id may be absent.
    if isinstance(item, ToolScenario) and item.expected_status == NOT_FOUND_STATUS:
        refs.discard(item.expected_args.get("pedido_id"))
    return errors + [f"{where}: unknown order {o!r}" for o in sorted(refs - orders)]


def validate_all(
    corpus_passage_ids: set[str], conn: sqlite3.Connection, eval_dir: Path = EVAL_DIR
) -> None:
    """Raise ValueError listing every problem found. `conn` must include the injected orders."""
    errors = _injected_errors(eval_dir)
    customers, orders = _ids(conn, "customers"), _ids(conn, "orders")
    seen: set[str] = set()
    for kind in KINDS:
        for item in load(kind, eval_dir):
            if item.id in seen:
                errors.append(f"duplicate id {item.id!r} ({kind})")
            seen.add(item.id)
            errors += _item_errors(kind, item, corpus_passage_ids, customers, orders)
    if errors:
        raise ValueError("; ".join(errors))


def assign_splits(
    items: list[_Item], strata: Callable, dev_frac: float, seed: int = DEFAULT_SEED
) -> None:
    """Stratified, deterministic split; total dev count is exactly round(N * dev_frac)."""
    groups: dict = defaultdict(list)
    for item in sorted(items, key=lambda i: i.id):
        groups[strata(item)].append(item)
    keys = sorted(groups)
    shares = {k: round(len(groups[k]) * dev_frac, 9) for k in keys}
    dev_counts = {k: math.floor(shares[k]) for k in keys}
    leftover = round(len(items) * dev_frac) - sum(dev_counts.values())
    # Largest remainder first; sorted() is stable, so ties keep stratum key order.
    for k in sorted(keys, key=lambda k: shares[k] - dev_counts[k], reverse=True)[:leftover]:
        dev_counts[k] += 1
    rng = random.Random(seed)
    for k in keys:
        rng.shuffle(groups[k])
        for n, item in enumerate(groups[k]):
            item.split = "dev" if n < dev_counts[k] else "test"


def _hashes(eval_dir: Path) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(eval_dir.iterdir())
        if p.is_file() and p.name != MANIFEST
    }


def freeze(eval_dir: Path = EVAL_DIR) -> None:
    manifest = json.dumps({"files": _hashes(eval_dir)}, indent=2) + "\n"
    (eval_dir / MANIFEST).write_text(manifest, encoding="utf-8", newline="\n")


def verify_frozen(eval_dir: Path = EVAL_DIR) -> None:
    expected = json.loads((eval_dir / MANIFEST).read_text(encoding="utf-8"))["files"]
    actual = _hashes(eval_dir)
    bad = sorted(n for n in expected.keys() & actual.keys() if expected[n] != actual[n])
    bad += [f"{n} (added)" for n in sorted(actual.keys() - expected.keys())]
    bad += [f"{n} (removed)" for n in sorted(expected.keys() - actual.keys())]
    if bad:
        raise ValueError(f"eval set changed after freeze: {', '.join(bad)}")
