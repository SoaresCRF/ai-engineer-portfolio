import csv
import json
import sys

import pytest

from support_agent import judge, report
from support_agent.report_tables import fmt, readme_tables

SHA = "ab12cd34ef56" + "0" * 52
SPLIT = "dev"
DEFAULT_DATE = "2026-01-02T00:00:00Z"
EARLIEST = {"name": "e2_none", "date": "2026-01-01T08:30:00Z"}
RETRIEVAL_ROW = {
    "n": 2,
    "recall@1": 0.5,
    "recall@3": 0.6,
    "recall@5": 0.7,
    "recall@10": 0.8,
    "mrr": 0.55,
}


def _msgs():
    return {"customer_id": "C001", "messages": [{"role": "user", "content": "oi"}]}


def _write_eval(eval_dir):
    eval_dir.mkdir()
    common = {"split": SPLIT, "author": "claude"}
    policy = [
        {
            "id": "p1",
            "question": "q",
            "gold_passages": ["a"],
            "reference_answer": "r",
            "answerable": True,
        }
        | common,
        {
            "id": "p2",
            "question": "q",
            "gold_passages": ["a"],
            "reference_answer": "r",
            "answerable": True,
        }
        | common,
        {
            "id": "p3",
            "question": "q",
            "gold_passages": [],
            "reference_answer": "r",
            "answerable": False,
        }
        | common,
    ]
    attacks = [
        {"id": "a1", "vector": "direto", "target": "frase", "forbidden": {"phrase": "BAD"}}
        | _msgs()
        | common,
        {"id": "a2", "vector": "observacao", "target": "vazamento", "forbidden": {}}
        | _msgs()
        | common,
    ]
    benign = [{"id": "b1"} | _msgs() | common]
    tools = [
        {"id": "t1", "expected_tool": None, "expected_args": {}} | _msgs() | common,
    ]
    for kind, rows in (
        ("policy_questions", policy),
        ("attacks", attacks),
        ("benign", benign),
        ("tool_scenarios", tools),
    ):
        text = "".join(json.dumps(r) + "\n" for r in rows)
        (eval_dir / f"{kind}.jsonl").write_text(text, encoding="utf-8")


def _verdict(i, kind, valid=True, correta=True, sha=SHA):
    v = {"justificativa": "j", "correta": correta, "fiel_ao_contexto": correta, "recusou": False}
    return {
        "id": i,
        "kind": kind,
        "valid": valid,
        "verdict": v if valid else None,
        "raw": "",
        "prompt_sha256": sha,
    }


def _out(i, kind, answer="ok"):
    return {
        "id": i,
        "kind": kind,
        "answer": answer,
        "sources": [],
        "tool_outcomes": [],
        "output_blocked": False,
        "retrieved_chunk_ids": [],
        "latency_ms": 1000.0,
        "tokens": {"prompt": 10, "completion": 5},
    }


def _write_run(runs_dir, name, kinds, verdicts):
    run = runs_dir / name
    run.mkdir(parents=True)
    ids = {
        "policy_questions": ["p1", "p2", "p3"],
        "attacks": ["a1", "a2"],
        "benign": ["b1"],
        "tool_scenarios": ["t1"],
    }
    outs = [_out(i, k) for k in kinds for i in ids[k]]
    (run / "outputs.jsonl").write_text(
        "".join(json.dumps(o) + "\n" for o in outs), encoding="utf-8"
    )
    date = EARLIEST["date"] if name.startswith(EARLIEST["name"]) else DEFAULT_DATE
    meta = {"canary": {"a2": "CANARY-x"}, "date": date}
    (run / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (run / "verdicts.jsonl").write_text(
        "".join(json.dumps(v) + "\n" for v in verdicts), encoding="utf-8"
    )


@pytest.fixture
def env(tmp_path):
    return _make_env(tmp_path)


def _make_env(tmp_path):
    runs, results, grading, eval_dir = (
        tmp_path / "runs",
        tmp_path / "results",
        tmp_path / "grading",
        tmp_path / "eval",
    )
    results.mkdir()
    _write_eval(eval_dir)
    retrieval = {
        "configs": {
            c: {"all": RETRIEVAL_ROW, "user": RETRIEVAL_ROW} for c in report.RETRIEVAL_CONFIGS
        }
    }
    (results / f"retrieval_{SPLIT}.json").write_text(json.dumps(retrieval), encoding="utf-8")
    e1_verdicts = [_verdict(i, "policy_questions") for i in ("p1", "p2", "p3")]
    for p in report.E1_RUNS.values():
        _write_run(runs, f"{p}_{SPLIT}", ["policy_questions"], e1_verdicts)
    for p in report.E2_RUNS.values():
        _write_run(
            runs,
            f"{p}_{SPLIT}",
            ["attacks", "benign", "tool_scenarios"],
            [_verdict("b1", "benign")],
        )
    return {"runs_dir": runs, "results_dir": results, "eval_dir": eval_dir, "grading_dir": grading}


def _final(env):
    return report.final(SPLIT, **env)


def test_final_content(env):
    data = json.loads(_final(env).read_text(encoding="utf-8"))
    assert data["judge"]["validated"] is False and data["judge"]["agreement_dev"] is None
    assert data["human"] is None
    assert data["e1"]["answer_quality"]["dense"]["judge_correct_rate"] == 1.0
    assert data["e1"]["paired_rerank_minus_dense"]["judge_correct"]["diff"] == 0.0
    assert data["e2"]["none"]["attacks"]["total"]["n"] == 2
    perf = data["performance"]["e1_dense_dev"]
    assert perf["gpu_hours"] == pytest.approx(3 / 3600)
    assert perf["cost_usd_per_1k"] == pytest.approx(1 / 3600 * 1000 * report.T4_USD_PER_HOUR)


def test_missing_run_raises(env):
    (env["runs_dir"] / "e2_full_dev" / "verdicts.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="verdicts"):
        _final(env)
    import shutil

    shutil.rmtree(env["runs_dir"] / "e2_full_dev")
    with pytest.raises(FileNotFoundError, match="e2_full_dev"):
        _final(env)


def test_mixed_prompt_sha_raises(env):
    path = env["runs_dir"] / "e1_dense_dev" / "verdicts.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["prompt_sha256"] = "f" * 64
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="prompt_sha256"):
        _final(env)


def test_invalid_verdicts_counted_not_dropped(env):
    rows = [_verdict("p1", "policy_questions", valid=False)] + [
        _verdict(i, "policy_questions", correta=False) for i in ("p2", "p3")
    ]
    path = env["runs_dir"] / "e1_dense_dev" / "verdicts.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    quality = json.loads(_final(env).read_text(encoding="utf-8"))["e1"]["answer_quality"]["dense"]
    assert quality["n_invalid"] == 1 and quality["judge_correct_rate"] == 0.0


def test_final_refuses_overwrite(env):
    _final(env)
    with pytest.raises(FileExistsError):
        _final(env)


def _write_sheet(grading, key_runs, graded=True):
    grading.mkdir(parents=True)
    key = {f"g{i}": {"run": r, "id": "p1"} for i, r in enumerate(key_runs, 1)}
    (grading / "key.json").write_text(json.dumps(key), encoding="utf-8")
    with (grading / "sheet.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["chave", "correta", "fiel"])
        w.writeheader()
        for k in key:
            w.writerow(
                {"chave": k, "correta": "1" if graded else "", "fiel": "0" if graded else ""}
            )


def test_human_block_present(env):
    _write_sheet(env["grading_dir"], ["e1_dense_dev", "e1_rerank_dev", "e1_rerank_dev"])
    human = json.loads(_final(env).read_text(encoding="utf-8"))["human"]
    assert human["rerank"]["correta"] == {"n": 2, "correct": 2, "rate": 1.0}
    assert human["dense"]["fiel"]["rate"] == 0.0


def test_human_block_absent_for_other_split_or_ungraded(env):
    _write_sheet(env["grading_dir"], ["e1_dense_test"])
    assert json.loads(_final(env).read_text(encoding="utf-8"))["human"] is None


def test_human_block_absent_when_ungraded(env):
    _write_sheet(env["grading_dir"], ["e1_dense_dev"], graded=False)
    assert json.loads(_final(env).read_text(encoding="utf-8"))["human"] is None


def test_rounding_once():
    assert fmt(0.8449) == "0.84"
    assert fmt(None) == "n/a"


def test_tables_render_from_final(env):
    _final(env)
    path = readme_tables(SPLIT, results_dir=env["results_dir"])
    text = path.read_text(encoding="utf-8")
    assert path.name == f"readme_tables_{SPLIT}.md"
    for heading in (
        "Retrieval",
        "Answer quality",
        "Attack success",
        "over-refusal",
        "Tool",
        "Performance",
    ):
        assert heading in text
    assert "kappa correta not available" in text and "never human-graded" in text
    assert "e1_dense_dev" in text and "not validated" in text
    assert text.count("not validated") == 3  # quality, paired and benign tables
    assert (
        readme_tables(SPLIT, results_dir=env["results_dir"]) == path
    )  # derived files may be overwritten


def test_tables_with_agreement_human_and_vectors(env):
    agreement = {
        "correta": {"kappa": 0.4750000000000001},
        "fiel_ao_contexto": {"kappa": 0.2047},
    }
    (env["results_dir"] / f"judge_agreement_{SHA[:12]}.json").write_text(
        json.dumps(agreement), encoding="utf-8"
    )
    _write_sheet(env["grading_dir"], ["e1_dense_dev", "e1_rerank_dev", "e1_rerank_dev"])
    _final(env)
    text = readme_tables(SPLIT, results_dir=env["results_dir"]).read_text(encoding="utf-8")
    assert "kappa correta 0.48" in text and "kappa fiel_ao_contexto 0.20" in text
    assert "Human grades (primary)" in text
    assert "| rerank | 2 | 1.00 | 0.00 |" in text
    assert "| dense | 1 | 1.00 | 0.00 |" in text
    assert "| none | vector direto | 1 |" in text
    assert "| none | vector observacao | 1 |" in text


def test_run_utc_is_earliest_meta_date(env):
    data = json.loads(_final(env).read_text(encoding="utf-8"))
    assert data["run_utc"] == EARLIEST["date"]
    assert "test_run_utc" not in data


def test_final_test_uses_test_grading_dir_and_both_agreements(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "SPLIT", "test")
    env = _make_env(tmp_path)
    monkeypatch.setattr(judge, "GRADING_DIR", tmp_path / "grading_root")
    _write_sheet(judge.grading_dir_for("test"), ["e1_dense_test", "e1_rerank_test"])
    for suffix, kappa in (("", 0.4), ("_test", 0.7)):
        (env["results_dir"] / f"judge_agreement_{SHA[:12]}{suffix}.json").write_text(
            json.dumps({k: {"kappa": kappa} for k in ("correta", "fiel_ao_contexto")}),
            encoding="utf-8",
        )
    env.pop("grading_dir")  # the default must resolve to <GRADING_DIR>/test
    data = json.loads(report.final("test", **env).read_text(encoding="utf-8"))
    assert data["human"]["dense"]["correta"]["n"] == 1
    assert data["judge"]["agreement_dev"]["correta"]["kappa"] == 0.4
    assert data["judge"]["agreement_test"]["correta"]["kappa"] == 0.7
    assert data["test_run_utc"] == data["run_utc"] == EARLIEST["date"]
    text = readme_tables("test", results_dir=env["results_dir"]).read_text(encoding="utf-8")
    assert "kappa correta 0.40 (dev); kappa correta 0.70 (test)" in text
