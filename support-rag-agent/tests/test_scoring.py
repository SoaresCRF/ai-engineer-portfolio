from dataclasses import dataclass, field

import pytest

from support_agent.corpus import Chunk
from support_agent.scoring import mrr, passages_from_chunks, recall_at_k, retrieval_report

RANKED = ["cdc:art49#0", "cdc:art49#1", "x:1#0"]


@dataclass
class Q:
    id: str
    question: str
    gold_passages: list[str] = field(default_factory=list)


def test_passages_strip_only_final_suffix_and_dedup():
    assert passages_from_chunks(["reembolso_restaurante:2.1#0", "a:1#1", "a:1#2"]) == [
        "reembolso_restaurante:2.1",
        "a:1",
    ]


def test_duplicate_chunks_count_once():
    assert recall_at_k(RANKED, {"x:1"}, k=3) == 1.0
    assert recall_at_k(RANKED, {"x:1"}, k=2) == 0.0


def test_recall_partial():
    assert recall_at_k(RANKED, {"x:1", "y:2"}, k=3) == 0.5


def test_mrr():
    assert mrr(["a:1#0", "b:1#0", "g:1#0"], {"g:1"}) == pytest.approx(1 / 3)
    assert mrr(["a:1#0"], {"g:1"}) == 0.0


def test_retrieval_report():
    table = {
        "q1": ["cdc:art49#0", "cdc:art49#1", "x:1#0"],
        "q2": ["z:9#0"],
    }
    calls = []

    def retrieve(question, k):
        calls.append((question, k))
        return [Chunk(c, c.rsplit("#", 1)[0], "t") for c in table[question]]

    qs = [Q("a", "q1", ["x:1"]), Q("b", "q2", ["y:2"]), Q("c", "q3", [])]
    rep = retrieval_report(retrieve, qs, ks=(1, 3))

    assert calls == [("q1", 3), ("q2", 3)]
    assert (rep["n"], rep["n_skipped"]) == (2, 1)
    assert rep["recall@1"] == 0.0
    assert rep["recall@3"] == 0.5
    assert rep["mrr"] == pytest.approx((1 / 2 + 0) / 2)
    assert rep["per_question"][0] == {
        "id": "a",
        "recall@1": 0.0,
        "recall@3": 1.0,
        "rank_first_gold": 2,
    }
    assert rep["per_question"][1]["rank_first_gold"] is None


def test_report_empty_returns_none():
    rep = retrieval_report(lambda q, k: [], [Q("c", "q", [])])
    assert rep["n"] == 0 and rep["mrr"] is None and rep["recall@10"] is None
