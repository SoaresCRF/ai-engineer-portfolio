"""Passage-level retrieval metrics.

Two units are mixed on purpose:
- recall@k: k counts CHUNKS (the context the LLM sees); gold passages are
  matched among the passages of those k chunks.
- MRR / rank_first_gold: counted in PASSAGE positions after dedup, independent
  of how many chunks precede the first gold passage.
"""

from collections.abc import Callable, Iterable
from statistics import fmean
from typing import Any

from support_agent.corpus import Chunk

DEFAULT_KS = (1, 3, 5, 10)


def passages_from_chunks(chunk_ids: list[str]) -> list[str]:
    """Strip the final `#<i>` suffix and dedupe, keeping first-seen order."""
    return list(dict.fromkeys(c.rsplit("#", 1)[0] for c in chunk_ids))


def _require_gold(gold_passages: set[str]) -> None:
    if not gold_passages:
        raise ValueError("gold_passages must not be empty")


def _rank_first_gold(ranked_chunk_ids: list[str], gold: set[str]) -> int | None:
    for rank, passage in enumerate(passages_from_chunks(ranked_chunk_ids), start=1):
        if passage in gold:
            return rank
    return None


def recall_at_k(ranked_chunk_ids: list[str], gold_passages: set[str], k: int) -> float:
    """Fraction of gold passages found among the passages of the first k CHUNKS.

    k counts chunks, i.e. the context the LLM sees, not passages.
    """
    _require_gold(gold_passages)
    found = set(passages_from_chunks(ranked_chunk_ids[:k]))
    return len(gold_passages & found) / len(gold_passages)


def mrr(ranked_chunk_ids: list[str], gold_passages: set[str]) -> float:
    """1 / rank of the first gold passage after dedup, in PASSAGE positions.

    Independent of how many chunks come before it (A#0, A#1, B#0 -> B is rank 2).
    """
    _require_gold(gold_passages)
    rank = _rank_first_gold(ranked_chunk_ids, gold_passages)
    return 0.0 if rank is None else 1 / rank


def _score_question(q: Any, ranked: list[str], ks: tuple[int, ...]) -> dict[str, Any]:
    gold = set(q.gold_passages)
    row: dict[str, Any] = {"id": q.id}
    for k in ks:
        row[f"recall@{k}"] = recall_at_k(ranked, gold, k)
    row["rank_first_gold"] = _rank_first_gold(ranked, gold)
    return row


def _mean_or_none(values: list[float]) -> float | None:
    return fmean(values) if values else None


def retrieval_report(
    retrieve: Callable[[str, int], list[Chunk]],
    questions: Iterable[Any],
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict[str, Any]:
    """Averages per k, MRR and per-question rows; unanswerable questions are skipped."""
    rows: list[dict[str, Any]] = []
    skipped = 0
    for q in questions:
        if not q.gold_passages:
            skipped += 1
            continue
        ranked = [c.chunk_id for c in retrieve(q.question, max(ks))]
        rows.append(_score_question(q, ranked, ks))
    report: dict[str, Any] = {"n": len(rows), "n_skipped": skipped}
    for k in ks:
        report[f"recall@{k}"] = _mean_or_none([r[f"recall@{k}"] for r in rows])
    ranks = [r["rank_first_gold"] for r in rows]
    report["mrr"] = _mean_or_none([0.0 if r is None else 1 / r for r in ranks])
    report["per_question"] = rows
    return report
