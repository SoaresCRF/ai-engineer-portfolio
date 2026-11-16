"""README tables in Markdown, rendered from results/final_{split}.json only."""

import json
from pathlib import Path

from support_agent.report import RESULTS_DIR

NA = "n/a"


def fmt(x: float | None, digits: int = 2) -> str:
    """Single rounding point: format the raw float, never a value that was already rounded."""
    return NA if x is None else format(x, f".{digits}f")


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _retrieval(e1: dict) -> str:
    keys = ("recall@1", "recall@3", "recall@5", "recall@10", "mrr")
    rows = [
        [cfg, subset, str(m["n"]), *(fmt(m[k]) for k in keys)]
        for cfg, subsets in e1["retrieval"].items()
        for subset, m in subsets.items()
    ]
    return _table(["Retrieval", "Questions", "n", "R@1", "R@3", "R@5", "R@10", "MRR"], rows)


def _quality(data: dict) -> str:
    parts = []
    if data["human"]:
        rows = [
            [cfg, str(m["correta"]["n"]), fmt(m["correta"]["rate"]), fmt(m["fiel"]["rate"])]
            for cfg, m in data["human"].items()
        ]
        parts.append(
            "Human grades (primary):\n\n" + _table(["Config", "n", "Correct", "Faithful"], rows)
        )
    judge = data["judge"]
    agreement = judge["agreement"]
    kappa = (
        f"kappa correct {fmt(agreement['correta']['kappa'])}, "
        f"faithful {fmt(agreement['fiel_ao_contexto']['kappa'])}"
        if agreement
        else "kappa not available"
    )
    e1 = data["e1"]
    rows = [
        [
            cfg,
            str(m["n"]),
            str(m["n_invalid"]),
            fmt(m["judge_correct_rate"]),
            fmt(m["judge_faithful_rate"]),
            fmt(e1["unanswerable"][cfg]["judge_correct_refusal_rate"]),
        ]
        for cfg, m in e1["answer_quality"].items()
    ]
    parts.append(
        f"LLM judge (secondary, not validated; {kappa}):\n\n"
        + _table(
            ["Config", "n", "Invalid", "Correct", "Faithful", "Correct refusal (unanswerable)"],
            rows,
        )
    )
    paired = e1["paired_rerank_minus_dense"]
    rows = [
        [label, fmt(p["diff"]), f"[{fmt(p['lo'])}, {fmt(p['hi'])}]", str(p["n_pairs"])]
        for label, p in paired.items()
    ]
    parts.append(
        "Rerank minus dense (paired bootstrap, 95% CI):\n\n"
        + _table(["Metric", "Diff", "95% CI", "Pairs"], rows)
    )
    return "\n\n".join(parts)


def _asr(e2: dict) -> str:
    rows = []
    for cfg, m in e2.items():
        groups = {"total": m["attacks"]["total"]} | {
            f"vector {v}": r for v, r in m["attacks"]["by_vector"].items()
        }
        rows += [
            [cfg, scope, str(r["n"]), fmt(r["attempted"]), fmt(r["executed"])]
            for scope, r in groups.items()
        ]
    return _table(["Defense", "Scope", "n", "ASR attempted", "ASR executed"], rows)


def _benign(e2: dict) -> str:
    rows = [
        [
            cfg,
            str(m["benign"]["n"]),
            str(m["benign"]["n_invalid"]),
            fmt(m["benign"]["judge_over_refusal_rate"]),
        ]
        for cfg, m in e2.items()
    ]
    return _table(["Defense", "n", "Invalid", "Judge over-refusal"], rows)


def _tools(e2: dict) -> str:
    rows = [
        [cfg, str(m["tools"]["n"])]
        + [fmt(m["tools"][k]) for k in ("tool_certa", "args_certos", "status_certo")]
        for cfg, m in e2.items()
    ]
    return _table(["Defense", "n", "Right tool", "Right args", "Right status"], rows)


def _performance(data: dict) -> str:
    rows = [
        [
            run,
            str(p["n"]),
            fmt(p["latency_p50_ms"] / 1000),
            fmt(p["latency_p95_ms"] / 1000),
            fmt(p["gpu_hours"], 3),
            fmt(p["cost_usd_per_1k"], 3),
        ]
        for run, p in data["performance"].items()
    ]
    note = f"T4 on-demand at {fmt(data['t4_usd_per_hour'])} USD/h; generation time only."
    table = _table(["Run", "n", "p50 (s)", "p95 (s)", "GPU hours", "USD per 1k requests"], rows)
    return f"{note}\n\n{table}"


def readme_tables(split: str = "test", results_dir: Path = RESULTS_DIR) -> Path:
    data = json.loads((results_dir / f"final_{split}.json").read_text(encoding="utf-8"))
    sections = [
        ("E1 - Retrieval", _retrieval(data["e1"])),
        ("E1 - Answer quality", _quality(data)),
        ("E2 - Attack success rate by defense", _asr(data["e2"])),
        ("E2 - Benign over-refusal", _benign(data["e2"])),
        ("E2 - Tool accuracy", _tools(data["e2"])),
        ("Performance", _performance(data)),
    ]
    text = "\n\n".join(f"### {title}\n\n{body}" for title, body in sections) + "\n"
    path = results_dir / f"readme_tables_{split}.md"
    path.write_text(text, encoding="utf-8")
    return path
