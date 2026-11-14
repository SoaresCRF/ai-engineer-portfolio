"""Retrieval configs R1-R4: dense, BM25, hybrid (RRF) and hybrid + cross-encoder rerank."""

import hashlib
import re
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import bm25s
import numpy as np

from support_agent.corpus import Chunk

Embedder = Callable[[list[str]], np.ndarray]  # L2-normalized vectors, shape (n, d)
Scorer = Callable[[str, list[str]], list[float]]  # (query, texts) -> one score per text
RetrievalConfig = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]

RRF_K = 60
DEFAULT_CANDIDATES = 20
CACHE_KEY_LENGTH = 16

# Written without accents: compared after accent stripping.
PT_STOPWORDS = frozenset(
    "a o as os um uma uns umas de do da dos das em no na nos nas por para com sem sob sobre "
    "e ou mas que se ao aos ou ja me te lhe nao ser foi sao estao eh ter tem como mais "
    "seu sua seus suas este esta esse essa isso isto".split()
)

_WORD_RE = re.compile(r"\w+")


def tokenize_pt(text: str) -> list[str]:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return [t for t in _WORD_RE.findall(plain) if t not in PT_STOPWORDS]


def _top_ids(scores: np.ndarray, ids: list[str], k: int, positive_only: bool) -> list[str]:
    # Stable sort: ties keep corpus order, so results are deterministic across runs.
    order = np.argsort(-scores, kind="stable")[: max(k, 0)]
    return [ids[i] for i in order if not positive_only or scores[i] > 0]


class DenseIndex:
    def __init__(self, chunks: list[Chunk], embed: Embedder, embeddings: np.ndarray | None = None):
        self._ids = [c.chunk_id for c in chunks]
        self._embed = embed
        self._matrix = embeddings if embeddings is not None else embed([c.text for c in chunks])

    def search(self, query: str, k: int) -> list[str]:
        if not query.strip():
            return []
        scores = self._matrix @ self._embed([query])[0]
        return _top_ids(scores, self._ids, k, positive_only=False)


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self._ids = [c.chunk_id for c in chunks]
        self._bm25 = bm25s.BM25()
        self._bm25.index([tokenize_pt(c.text) for c in chunks], show_progress=False)

    def search(self, query: str, k: int) -> list[str]:
        tokens = tokenize_pt(query)
        if not tokens:  # bm25s raises on an empty token list
            return []
        # get_scores skips out-of-vocabulary tokens; k is clamped by slicing the full ranking.
        return _top_ids(self._bm25.get_scores(tokens), self._ids, k, positive_only=True)


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    """Reciprocal rank fusion; ties keep first-seen order."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda item: -scores[item])  # sorted() is stable


def build_retriever(
    config: RetrievalConfig,
    chunks: list[Chunk],
    embed: Embedder | None,
    rerank: Scorer | None,
    candidates: int = DEFAULT_CANDIDATES,
    embeddings: np.ndarray | None = None,  # precomputed document vectors (see cached_embeddings)
) -> Callable[[str, int], list[Chunk]]:
    if config != "bm25" and embed is None:
        raise ValueError(f"config {config!r} needs an embedder")
    if config == "hybrid_rerank" and rerank is None:
        raise ValueError("config 'hybrid_rerank' needs a reranker")

    by_id = {c.chunk_id: c for c in chunks}
    dense = DenseIndex(chunks, embed, embeddings) if config != "bm25" else None
    bm25 = BM25Index(chunks) if config != "dense" else None

    def ids_for(query: str, k: int) -> list[str]:
        if config == "dense":
            return dense.search(query, k)
        if config == "bm25":
            return bm25.search(query, k)
        fused = rrf([dense.search(query, candidates), bm25.search(query, candidates)])
        if config == "hybrid":
            return fused[:k]
        pool = fused[:candidates]
        scores = rerank(query, [by_id[i].text for i in pool])
        order = sorted(range(len(pool)), key=lambda i: -scores[i])  # stable on ties
        return [pool[i] for i in order][:k]

    return lambda query, k: [by_id[i] for i in ids_for(query, k)]


def cached_embeddings(
    chunks: list[Chunk], embed: Embedder, model_name: str, cache_dir: Path
) -> np.ndarray:
    digest = hashlib.sha256((model_name + "".join(c.text for c in chunks)).encode("utf-8"))
    path = cache_dir / f"{digest.hexdigest()[:CACHE_KEY_LENGTH]}.npy"
    if path.exists():
        return np.load(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix = embed([c.text for c in chunks])
    np.save(path, matrix)
    return matrix


def load_models(
    dense_name: str = "BAAI/bge-m3", rerank_name: str = "BAAI/bge-reranker-v2-m3"
) -> tuple[Embedder, Scorer]:
    # Lazy import: sentence_transformers pulls in torch, which slows every import of this module.
    from sentence_transformers import CrossEncoder, SentenceTransformer

    encoder = SentenceTransformer(dense_name)
    cross = CrossEncoder(rerank_name)

    def embed(texts: list[str]) -> np.ndarray:
        return encoder.encode(texts, normalize_embeddings=True)

    def score(query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in cross.predict([(query, t) for t in texts])]

    return embed, score
