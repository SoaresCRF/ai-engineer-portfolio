"""Results: retrieval report (R1-R4) and the final per-split report from the run directories."""

import json
import sys
from pathlib import Path

import numpy as np

from support_agent import eval_set, judge, run_eval
from support_agent.corpus import Chunk
from support_agent.retrieval import Embedder, RetrievalConfig, Scorer, build_retriever
from support_agent.retrieval_cache import ROOT, default_setup
from support_agent.scoring import paired_bootstrap, retrieval_report, score_attacks, score_tools

RESULTS_DIR = ROOT / "results"
RETRIEVAL_CONFIGS: tuple[RetrievalConfig, ...] = ("dense", "bm25", "hybrid", "hybrid_rerank")
# Google Cloud Compute Engine GPU pricing, NVIDIA T4 on-demand (us-central1), USD per GPU-hour.
T4_USD_PER_HOUR = 0.35
JUDGE_MODEL = "casperhansen/mistral-small-24b-instruct-2501-awq"
MS_PER_HOUR = 3.6e6
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 17
E1_RUNS = {"dense": "e1_dense", "rerank": "e1_rerank"}
E2_RUNS = {"none": "e2_none", "prompt": "e2_prompt", "full": "e2_full"}
E1_RETRIEVAL = {"dense": "dense", "rerank": "hybrid_rerank"}
RETRIEVAL_KEYS = ("n", "recall@1", "recall@3", "recall@5", "recall@10", "mrr")


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


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _load_run(runs_dir: Path, name: str) -> dict:
    run_dir = runs_dir / name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"missing run directory: {run_dir}")
    return {
        "name": name,
        "outputs": _jsonl(run_dir / "outputs.jsonl"),
        "meta": json.loads((run_dir / "meta.json").read_text(encoding="utf-8")),
        "verdicts": {v["id"]: v for v in _jsonl(run_dir / "verdicts.jsonl")},
    }


def _flags(run: dict, ids: list[str], key: str) -> tuple[dict[str, bool], int]:
    """Judge flag per valid item and the count of invalid verdicts (reported, not dropped)."""
    flags, invalid = {}, 0
    for i in ids:
        v = run["verdicts"][i]  # KeyError names the unjudged item
        if v["valid"]:
            flags[i] = v["verdict"][key]
        else:
            invalid += 1
    return flags, invalid


def _rate(flags: dict[str, bool]) -> float | None:
    return sum(flags.values()) / len(flags) if flags else None


def _performance(run: dict) -> dict:
    lat = np.array([o["latency_ms"] for o in run["outputs"]], dtype=float)
    p50, p95 = np.percentile(lat, [50, 95])  # numpy default (linear interpolation)
    hours = float(lat.sum()) / MS_PER_HOUR  # generation wall time only, excludes model load
    n = len(lat)
    return {
        "n": n,
        "latency_p50_ms": float(p50),
        "latency_p95_ms": float(p95),
        "prompt_tokens": sum(o["tokens"]["prompt"] for o in run["outputs"]),
        "completion_tokens": sum(o["tokens"]["completion"] for o in run["outputs"]),
        "gpu_hours": hours,
        "cost_usd_per_1k": hours / n * 1000 * T4_USD_PER_HOUR,
    }


def _retrieval_block(split: str, results_dir: Path) -> dict:
    path = results_dir / f"retrieval_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} (run `report retrieval {split}` first)")
    configs = json.loads(path.read_text(encoding="utf-8"))["configs"]
    return {
        name: {s: {k: configs[cfg][s][k] for k in RETRIEVAL_KEYS} for s in ("all", "user")}
        for name, cfg in E1_RETRIEVAL.items()
    }


def _paired(per_item: dict[str, tuple[dict, dict]]) -> dict:
    out = {}
    for label, idx in (("judge_correct", 0), ("judge_faithful", 1)):
        d, r = per_item["dense"][idx], per_item["rerank"][idx]
        both = sorted(d.keys() & r.keys())  # a pair needs a valid verdict on both sides
        out[label] = {
            "n_pairs": len(both),
            **paired_bootstrap(
                np.array([r[i] for i in both]),
                np.array([d[i] for i in both]),
                n=BOOTSTRAP_N,
                seed=BOOTSTRAP_SEED,
            ),
        }
    return out


def _e1(runs: dict[str, dict], split: str, eval_dir: Path, results_dir: Path) -> dict:
    questions = [q for q in eval_set.load("policy_questions", eval_dir) if q.split == split]
    ans = [q.id for q in questions if q.answerable]
    unans = [q.id for q in questions if not q.answerable]
    quality, unanswerable, per_item = {}, {}, {}
    for name, run in runs.items():
        correct, bad = _flags(run, ans, "correta")
        faithful, _ = _flags(run, ans, "fiel_ao_contexto")
        refused, bad_u = _flags(run, unans, "correta")
        per_item[name] = (correct, faithful)
        quality[name] = {
            "n": len(ans),
            "n_invalid": bad,
            "judge_correct_rate": _rate(correct),
            "judge_faithful_rate": _rate(faithful),
        }
        unanswerable[name] = {
            "n": len(unans),
            "n_invalid": bad_u,
            "judge_correct_refusal_rate": _rate(refused),
        }
    return {
        "retrieval": _retrieval_block(split, results_dir),
        "answer_quality": quality,
        "paired_rerank_minus_dense": _paired(per_item),
        "unanswerable": unanswerable,
    }


def _e2(runs: dict[str, dict], split: str, eval_dir: Path) -> dict:
    def items(kind):
        return [i for i in eval_set.load(kind, eval_dir) if i.split == split]

    attacks, scenarios, benign = items("attacks"), items("tool_scenarios"), items("benign")
    out = {}
    for name, run in runs.items():
        refused, bad = _flags(run, [b.id for b in benign], "recusou")
        out[name] = {
            "attacks": score_attacks(run["outputs"], attacks, run["meta"]["canary"]),
            "benign": {
                "n": len(benign),
                "n_invalid": bad,
                "judge_over_refusal_rate": _rate(refused),
            },
            "tools": score_tools(run["outputs"], scenarios),
        }
    return out


def _human(split: str, grading_dir: Path) -> dict | None:
    """Per-config human rates from the blind sheet; None unless fully graded for this split."""
    sheet, key_path = grading_dir / "sheet.csv", grading_dir / "key.json"
    if not (sheet.exists() and key_path.exists()):
        return None
    key = json.loads(key_path.read_text(encoding="utf-8"))
    names = {f"{prefix}_{split}": name for name, prefix in E1_RUNS.items()}
    if not key or {v["run"] for v in key.values()} - names.keys():
        return None
    rows = judge._read_sheet(sheet)
    if any(not r["correta"].strip() or not r["fiel"].strip() for r in rows):
        return None
    tally = {n: {"correta": [], "fiel": []} for n in names.values()}
    for r in rows:
        cfg = names[key[r["chave"]]["run"]]
        for col in ("correta", "fiel"):
            tally[cfg][col].append(judge._grade(r[col].strip(), r["chave"], col))
    return {
        cfg: {
            col: {"n": len(v), "correct": sum(v), "rate": sum(v) / len(v)}
            for col, v in cols.items()
        }
        for cfg, cols in tally.items()
    }


def _judge_block(runs: list[dict], split: str, results_dir: Path) -> dict:
    shas = {v["prompt_sha256"] for r in runs for v in r["verdicts"].values()}
    if len(shas) != 1:
        raise ValueError(f"verdicts must carry one prompt_sha256, found: {sorted(map(str, shas))}")
    (sha,) = shas
    # Not validated: kappa < 0.6 gate (Task 17c); human grades are primary, kappa always reported.
    block = {"model": JUDGE_MODEL, "prompt_sha256": sha, "validated": False}
    for name in dict.fromkeys((judge.DEV_SPLIT, split)):  # dev first, then the report split
        suffix = "" if name == judge.DEV_SPLIT else f"_{name}"
        path = results_dir / f"judge_agreement_{sha[:12]}{suffix}.json"
        block[f"agreement_{name}"] = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        )
    return block


def final(
    split: str = "test",
    runs_dir: Path = run_eval.RUNS_DIR,
    results_dir: Path = RESULTS_DIR,
    eval_dir: Path = eval_set.EVAL_DIR,
    grading_dir: Path | None = None,
) -> Path:
    grading_dir = judge.grading_dir_for(split) if grading_dir is None else grading_dir
    path = results_dir / f"final_{split}.json"
    if path.exists():
        raise FileExistsError(path)  # results are evidence, never overwritten
    e1 = {n: _load_run(runs_dir, f"{p}_{split}") for n, p in E1_RUNS.items()}
    e2 = {n: _load_run(runs_dir, f"{p}_{split}") for n, p in E2_RUNS.items()}
    every = [*e1.values(), *e2.values()]
    report = {
        "split": split,
        "judge": _judge_block(every, split, results_dir),
        "human": _human(split, grading_dir),
        "e1": _e1(e1, split, eval_dir, results_dir),
        "e2": _e2(e2, split, eval_dir),
        "performance": {r["name"]: _performance(r) for r in every},
        "t4_usd_per_hour": T4_USD_PER_HOUR,
        # Earliest run start (UTC ISO, read from meta.json); the plan's test_run_utc for "test".
        "run_utc": min(r["meta"]["date"] for r in every),
    }
    if split == "test":
        report["test_run_utc"] = report["run_utc"]
    results_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


if __name__ == "__main__":
    # usage: python -m support_agent.report retrieval|final|tables <split>
    cmd = sys.argv[1] if len(sys.argv) == 3 else None
    if cmd == "retrieval":
        print(retrieval(sys.argv[2], *default_setup()))
    elif cmd == "final":
        print(final(sys.argv[2]))
    elif cmd == "tables":
        from support_agent.report_tables import readme_tables

        print(readme_tables(sys.argv[2]))
    else:
        raise SystemExit("usage: python -m support_agent.report retrieval|final|tables <split>")
