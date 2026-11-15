"""Result tables. Only the retrieval report lives here so far (final tables come in Task 18)."""

import json
import sys
from pathlib import Path

import numpy as np

from support_agent import eval_set
from support_agent.corpus import Chunk
from support_agent.retrieval import Embedder, RetrievalConfig, Scorer, build_retriever
from support_agent.retrieval_cache import ROOT, default_setup
from support_agent.scoring import retrieval_report

RESULTS_DIR = ROOT / "results"
RETRIEVAL_CONFIGS: tuple[RetrievalConfig, ...] = ("dense", "bm25", "hybrid", "hybrid_rerank")


def retrieval(
    split: str,
    chunks: list[Chunk],
    embed: Embedder,
    rerank: Scorer,
    embeddings: np.ndarray | None = None,
    out_dir: Path = RESULTS_DIR,
) -> Path:
    """R1-R4 over the answerable policy questions of the split, and over the user-written subset."""
    questions = [q for q in eval_set.load("policy_questions") if q.split == split and q.answerable]
    subsets = {
        "all": questions,
        "user": [q for q in questions if q.author == "user"],
    }
    report = {"split": split, "configs": {}}
    for config in RETRIEVAL_CONFIGS:
        retrieve = build_retriever(config, chunks, embed, rerank, embeddings=embeddings)
        report["configs"][config] = {
            name: retrieval_report(retrieve, subset) for name, subset in subsets.items()
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"retrieval_{split}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


if __name__ == "__main__":
    # usage: python -m support_agent.report retrieval dev
    if len(sys.argv) != 3 or sys.argv[1] != "retrieval":
        raise SystemExit("usage: python -m support_agent.report retrieval <split>")
    print(retrieval(sys.argv[2], *default_setup()))
