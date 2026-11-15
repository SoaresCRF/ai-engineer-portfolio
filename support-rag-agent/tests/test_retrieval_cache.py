import json
import zlib

import numpy as np
import pytest

from support_agent import eval_set, report, retrieval_cache, run_eval
from support_agent.agent import AgentConfig
from support_agent.corpus import Chunk
from support_agent.eval_set import Message, PolicyQuestion, ToolScenario
from support_agent.llm import AssistantTurn
from support_agent.retrieval import tokenize_pt

DIMS = 64
CHUNKS = [
    Chunk("a#0", "a", "reembolso do valor pago em sete dias"),
    Chunk("b#0", "b", "prazo de entrega do frete expresso"),
    Chunk("c#0", "c", "troca de produto com defeito"),
]


def fake_embed(texts):
    out = np.zeros((len(texts), DIMS))
    for row, text in enumerate(texts):
        for token in tokenize_pt(text):
            out[row, zlib.crc32(token.encode()) % DIMS] += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)


def fake_rerank(query, texts):
    return [float(len(set(tokenize_pt(query)) & set(tokenize_pt(t)))) for t in texts]


def _question(qid, split, author, text, gold, answerable=True):
    return PolicyQuestion(
        id=qid,
        split=split,
        author=author,
        question=text,
        gold_passages=gold,
        reference_answer="ref",
        answerable=answerable,
    )


def _scenario(sid, split, text):
    return ToolScenario(
        id=sid,
        split=split,
        author="claude",
        customer_id="C001",
        messages=[
            Message(role="user", content="oi"),
            Message(role="assistant", content="ola"),
            Message(role="user", content=text),
        ],
        expected_tool=None,
        expected_args={},
    )


QUESTIONS = [
    _question("q1", "dev", "user", "prazo do frete expresso", ["b"]),
    _question("q2", "dev", "claude", "reembolso em sete dias", ["a"]),
    _question("q3", "dev", "claude", "assunto sem resposta", [], answerable=False),
    _question("q4", "test", "user", "troca com defeito", ["c"]),
]
SCENARIOS = [_scenario("t1", "dev", "qual o status?"), _scenario("t2", "test", "outro pedido")]


@pytest.fixture(autouse=True)
def fake_eval_set(monkeypatch):
    data = {"policy_questions": QUESTIONS, "tool_scenarios": SCENARIOS}
    monkeypatch.setattr(eval_set, "load", lambda kind, *a: data.get(kind, []))


def test_retrieval_report_shape_and_user_subset(tmp_path):
    path = report.retrieval("dev", CHUNKS, fake_embed, fake_rerank, out_dir=tmp_path)
    assert path == tmp_path / "retrieval_dev.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert list(data["configs"]) == ["dense", "bm25", "hybrid", "hybrid_rerank"]
    for per_config in data["configs"].values():
        assert per_config["all"]["n"] == 2  # unanswerable and test items excluded
        assert per_config["user"]["n"] == 1
        assert {"recall@1", "recall@3", "recall@5", "recall@10", "mrr"} <= set(per_config["all"])
    assert [r["id"] for r in data["configs"]["bm25"]["user"]["per_question"]] == ["q1"]
    assert data["configs"]["bm25"]["user"]["mrr"] == 1.0


def _precompute(tmp_path):
    return retrieval_cache.precompute("dev", CHUNKS, fake_embed, fake_rerank, cache_dir=tmp_path)


def test_precompute_covers_every_runner_query(tmp_path, monkeypatch):
    cache = json.loads(_precompute(tmp_path).read_text(encoding="utf-8"))
    monkeypatch.setattr(run_eval, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(run_eval.eval_set, "verify_frozen", lambda: None)
    monkeypatch.setattr(run_eval.eval_set, "load_injected_orders", lambda: [])
    monkeypatch.setattr(run_eval, "_initial_meta", lambda *a: {"canary": {}})
    queried = []
    cached = retrieval_cache.cached_retriever(cache, "hybrid_rerank", CHUNKS)

    def recording(query, k):
        queried.append(query)
        return cached(query, k)

    class LLM:
        def chat(self, messages, tools):
            return AssistantTurn("ok")

    for kinds in (["policy_questions"], ["tool_scenarios"]):
        run_eval.run(f"r_{kinds[0]}", AgentConfig(), "dev", kinds, LLM(), recording)
    assert set(queried) == set(cache["configs"]["dense"]) == set(cache["configs"]["hybrid_rerank"])
    assert "qual o status?" in queried  # last user message of a multi-turn item


def test_cached_retriever_slices_and_matches_live(tmp_path):
    cache = json.loads(_precompute(tmp_path).read_text(encoding="utf-8"))
    cached = retrieval_cache.cached_retriever(cache, "dense", CHUNKS)
    full = cached("prazo do frete expresso", 3)
    assert [c.chunk_id for c in cached("prazo do frete expresso", 1)] == [full[0].chunk_id]
    assert full[0].chunk_id == "b#0"


def test_cached_retriever_miss_raises(tmp_path):
    cache = json.loads(_precompute(tmp_path).read_text(encoding="utf-8"))
    with pytest.raises(KeyError):
        retrieval_cache.cached_retriever(cache, "dense", CHUNKS)("nunca vista", 3)


def test_cached_retriever_k_above_cache_raises(tmp_path):
    cache = json.loads(_precompute(tmp_path).read_text(encoding="utf-8"))
    retrieve = retrieval_cache.cached_retriever(cache, "dense", CHUNKS)
    with pytest.raises(ValueError, match="top 10"):
        retrieve("prazo do frete expresso", 11)


def test_cached_retriever_stale_chunk_hash_raises(tmp_path):
    cache = json.loads(_precompute(tmp_path).read_text(encoding="utf-8"))
    changed = [*CHUNKS[:2], Chunk("c#0", "c", "texto novo")]
    with pytest.raises(ValueError, match="stale"):
        retrieval_cache.cached_retriever(cache, "dense", changed)
