"""Precomputed retrieval for Kaggle: the T4 only generates and judges, ranking ran locally.

The candidate pool in build_retriever does not depend on k, so the top-10 ids of a query
contain the top-k for every k <= 10.
"""

import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from support_agent import ROOT, eval_set
from support_agent.corpus import Chunk, chunk_passages, load_corpus
from support_agent.retrieval import (
    DENSE_MODEL,
    Embedder,
    RetrievalConfig,
    Scorer,
    build_retriever,
    cached_embeddings,
    load_models,
)
from support_agent.run_eval import _conversation

CORPUS_DIR = ROOT / "corpus"
EMBEDDING_CACHE_DIR = ROOT / "data" / "cache"  # same default as api.py CACHE_DIR
RETRIEVAL_CACHE_DIR = ROOT / "data" / "retrieval_cache"
CACHED_CONFIGS: tuple[RetrievalConfig, ...] = ("dense", "hybrid_rerank")
CACHED_K = 10


def chunk_set_hash(chunks: list[Chunk]) -> str:
    digest = hashlib.sha256()
    for c in chunks:
        digest.update(f"{c.chunk_id}\0{c.text}\0".encode())
    return digest.hexdigest()


def default_setup() -> tuple[list[Chunk], Embedder, Scorer, np.ndarray]:
    """Project corpus + real models + document embeddings (downloads models on first use)."""
    chunks = chunk_passages(load_corpus(CORPUS_DIR))
    embed, rerank = load_models(dense_name=DENSE_MODEL)
    return chunks, embed, rerank, cached_embeddings(chunks, embed, DENSE_MODEL, EMBEDDING_CACHE_DIR)


def runner_queries(split: str) -> list[str]:
    """Every query the runner issues on the split: the last user message of each item."""
    queries = {
        _conversation(item)[1][-1]["content"]
        for kind in eval_set.KINDS
        for item in eval_set.load(kind)
        if item.split == split
    }
    return sorted(queries)


def precompute(
    split: str,
    chunks: list[Chunk],
    embed: Embedder,
    rerank: Scorer,
    embeddings: np.ndarray | None = None,
    cache_dir: Path = RETRIEVAL_CACHE_DIR,
) -> Path:
    queries = runner_queries(split)
    configs = {}
    for config in CACHED_CONFIGS:
        retrieve = build_retriever(config, chunks, embed, rerank, embeddings=embeddings)
        configs[config] = {q: [c.chunk_id for c in retrieve(q, CACHED_K)] for q in queries}
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{split}.json"
    payload = {"chunk_hash": chunk_set_hash(chunks), "k": CACHED_K, "configs": configs}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def cached_retriever(
    cache: dict, config: RetrievalConfig, chunks: list[Chunk]
) -> Callable[[str, int], list[Chunk]]:
    if cache["chunk_hash"] != chunk_set_hash(chunks):
        raise ValueError("retrieval cache is stale: the corpus chunks changed, re-run precompute")
    by_id = {c.chunk_id: c for c in chunks}
    ranked = cache["configs"][config]

    def retrieve(query: str, k: int) -> list[Chunk]:
        if k > cache["k"]:
            raise ValueError(f"cache holds the top {cache['k']} only, got k={k}")
        if query not in ranked:  # never fall back to live ranking: it would hide a stale cache
            raise KeyError(f"query not in retrieval cache for {config!r}: {query!r}")
        return [by_id[i] for i in ranked[query][:k]]

    return retrieve


def load_cache(split: str, cache_dir: Path = RETRIEVAL_CACHE_DIR) -> dict:
    return json.loads((cache_dir / f"{split}.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    # usage: python -m support_agent.retrieval_cache precompute dev
    if len(sys.argv) != 3 or sys.argv[1] != "precompute":
        raise SystemExit("usage: python -m support_agent.retrieval_cache precompute <split>")
    print(precompute(sys.argv[2], *default_setup()))
