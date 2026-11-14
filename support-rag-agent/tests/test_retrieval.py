import subprocess
import sys
import zlib

import numpy as np
import pytest

from support_agent.corpus import Chunk
from support_agent.retrieval import (
    BM25Index,
    DenseIndex,
    build_retriever,
    cached_embeddings,
    rrf,
    tokenize_pt,
)

DIMS = 256


def fake_embed(texts: list[str]) -> np.ndarray:
    """Hashed bag-of-words, L2-normalized (deterministic across processes)."""
    out = np.zeros((len(texts), DIMS))
    for row, text in enumerate(texts):
        for token in tokenize_pt(text):
            out[row, zlib.crc32(token.encode()) % DIMS] += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)


CHUNKS = [
    Chunk("a#0", "a", "Reembolsó do valor pago em até sete dias"),
    Chunk("b#0", "b", "Prazo de entrega para o frete expresso"),
    Chunk("c#0", "c", "Troca de produto com defeito"),
]


def test_dense_finds_lexical_match():
    assert DenseIndex(CHUNKS, fake_embed).search("frete expresso", 1) == ["b#0"]


def test_bm25_ignores_accents_and_stopwords():
    index = BM25Index(CHUNKS)
    assert index.search("reembolso", 1) == ["a#0"]
    assert index.search("de", 3) == []


def test_bm25_clamps_k_and_handles_unknown_tokens():
    index = BM25Index(CHUNKS)
    assert len(index.search("produto", 50)) == 1
    assert index.search("zzzz", 3) == []


def test_rrf_merges():
    assert rrf([["a", "b"], ["b", "c"]])[0] == "b"


def test_rerank_reorders_candidates():
    def prefers_c(query: str, texts: list[str]) -> list[float]:
        return [1.0 if "Troca" in t else 0.0 for t in texts]

    retrieve = build_retriever("hybrid_rerank", CHUNKS, fake_embed, prefers_c)
    assert retrieve("produto frete reembolso", 3)[0].chunk_id == "c#0"


def test_all_configs_return_chunks_without_rerank_model():
    for config in ("dense", "bm25", "hybrid"):
        retrieve = build_retriever(config, CHUNKS, fake_embed, None)
        assert retrieve("frete", 1)[0].chunk_id == "b#0"


def test_bm25_config_needs_no_embedder():
    assert build_retriever("bm25", CHUNKS, None, None)("frete", 1)[0].chunk_id == "b#0"


@pytest.mark.parametrize(
    ("config", "embed", "rerank"),
    [("dense", None, None), ("hybrid", None, None), ("hybrid_rerank", fake_embed, None)],
)
def test_missing_dependency_raises(config, embed, rerank):
    with pytest.raises(ValueError):
        build_retriever(config, CHUNKS, embed, rerank)


def test_empty_query_returns_empty():
    for config in ("dense", "bm25", "hybrid"):
        assert build_retriever(config, CHUNKS, fake_embed, None)("", 3) == []


def test_cache_invalidates_on_text_change(tmp_path):
    calls = []

    def counting_embed(texts: list[str]) -> np.ndarray:
        calls.append(len(texts))
        return fake_embed(texts)

    cache = tmp_path / "cache"
    first = cached_embeddings(CHUNKS, counting_embed, "m", cache)
    again = cached_embeddings(CHUNKS, counting_embed, "m", cache)
    assert calls == [3] and np.array_equal(first, again)
    changed = [CHUNKS[0], CHUNKS[1], Chunk("c#0", "c", "outro texto")]
    cached_embeddings(changed, counting_embed, "m", cache)
    assert calls == [3, 3] and len(list(cache.glob("*.npy"))) == 2


def test_importing_retrieval_does_not_load_sentence_transformers():
    code = "import sys, support_agent.retrieval; assert 'sentence_transformers' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_build_retriever_uses_precomputed_embeddings():
    matrix = fake_embed([c.text for c in CHUNKS])
    embedded: list[list[str]] = []

    def spying_embed(texts):
        embedded.append(texts)
        return fake_embed(texts)

    retrieve = build_retriever("dense", CHUNKS, spying_embed, None, embeddings=matrix)
    assert [c.chunk_id for c in retrieve("frete expresso", 1)] == ["b#0"]
    assert embedded == [["frete expresso"]]  # only the query was embedded, never the documents
