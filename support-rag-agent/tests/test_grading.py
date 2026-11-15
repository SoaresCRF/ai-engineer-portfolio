import csv
import json
from types import SimpleNamespace

import pytest

from support_agent import eval_set, judge, run_eval
from support_agent.corpus import chunk_passages, load_corpus
from support_agent.eval_set import PolicyQuestion

N_ITEMS = 8
SHA = "ab12cd34ef56" + "0" * 52


def _write_run(path, name, chunk_id):
    run = path / name
    run.mkdir()
    rows = [
        {
            "id": f"pq{i}",
            "kind": "policy_questions",
            "answer": f"resp-{name[-1]}-{i}",
            "retrieved_chunk_ids": [chunk_id],
        }
        for i in range(N_ITEMS)
    ]
    rows.append({"id": "bn1", "kind": "benign", "answer": "x", "retrieved_chunk_ids": []})
    (run / "outputs.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    return run


@pytest.fixture
def env(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    (corpus / "policies").mkdir(parents=True)
    (corpus / "policies" / "doc.md").write_text(
        "# Doc\n\n## Reembolso\n\nO prazo de reembolso é de 7 dias.\n", encoding="utf-8"
    )
    chunk = chunk_passages(load_corpus(corpus))[0]
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    pqs = [
        PolicyQuestion(
            id=f"pq{i}",
            split="dev",
            author="claude",
            question=f"pergunta {i}?",
            gold_passages=[],
            reference_answer=f"ref {i}",
            answerable=True,
        )
        for i in range(N_ITEMS)
    ]
    eval_set.dump_jsonl(eval_dir / "policy_questions.jsonl", pqs)
    grading = tmp_path / "grading"
    results = tmp_path / "results"
    monkeypatch.setattr(judge, "CORPUS_DIR", corpus)
    monkeypatch.setattr(judge, "EVAL_DIR", eval_dir)
    monkeypatch.setattr(judge, "GRADING_DIR", grading)
    monkeypatch.setattr(judge, "RESULTS_DIR", results)
    monkeypatch.setattr(run_eval, "RUNS_DIR", tmp_path)
    runs = [_write_run(tmp_path, n, chunk.chunk_id) for n in ("run_a", "run_b")]
    return SimpleNamespace(runs=runs, chunk=chunk, grading=grading, results=results)


def _read_sheet(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_sheet_is_blind_balanced_and_utf8_sig(env):
    path = judge.export_grading_sheet(env.runs, n=8, seed=17)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    rows = _read_sheet(path)
    assert list(rows[0]) == [
        "chave",
        "pergunta",
        "referencia",
        "contexto",
        "resposta",
        "correta",
        "fiel",
    ]
    assert [r["chave"] for r in rows] == [f"g{i:02d}" for i in range(1, 9)]
    assert all(r["correta"] == "" and r["fiel"] == "" for r in rows)
    assert env.chunk.text in rows[0]["contexto"]
    key = json.loads((env.grading / "key.json").read_text(encoding="utf-8"))
    per_run = [v["run"] for v in key.values()]
    assert per_run.count("run_a") == 4 and per_run.count("run_b") == 4
    text = path.read_text(encoding="utf-8-sig")
    for leak in ("run_a", "run_b", "verdict", "policy_questions"):
        assert leak not in text
    # opaque keys only: the item id never appears as a cell of its own
    assert not {f"pq{i}" for i in range(N_ITEMS)} & {c for r in rows for c in r.values()}
    for r in rows:  # the answer shown matches the mapped run/item
        k = key[r["chave"]]
        assert r["resposta"] == f"resp-{k['run'][-1]}-{k['id'][2:]}"


def test_sheet_deterministic_with_seed(env, monkeypatch):
    first = _read_sheet(judge.export_grading_sheet(env.runs, n=8, seed=3))
    monkeypatch.setattr(judge, "GRADING_DIR", env.grading.parent / "g2")
    second = _read_sheet(judge.export_grading_sheet(env.runs, n=8, seed=3))
    assert first == second


def test_sheet_refuses_overwrite_and_bad_args(env):
    with pytest.raises(ValueError):
        judge.export_grading_sheet(env.runs, n=7)
    with pytest.raises(ValueError):
        judge.export_grading_sheet(env.runs[:1], n=8)
    judge.export_grading_sheet(env.runs, n=8)
    with pytest.raises(FileExistsError):
        judge.export_grading_sheet(env.runs, n=8)
    (env.grading / "sheet.csv").unlink()  # key.json alone must also block
    with pytest.raises(FileExistsError):
        judge.export_grading_sheet(env.runs, n=8)


def _verdict(correta, fiel):
    return {"correta": correta, "fiel_ao_contexto": fiel, "recusou": False, "justificativa": "x"}


def _grade(env, tmp_path, human, invalid=(), shas=(SHA, SHA), delimiter=","):
    """Export, write all-true verdicts per run, return a graded sheet copy. human(chave)->(c, f)."""
    sheet = judge.export_grading_sheet(env.runs, n=8)
    for run, sha in zip(env.runs, shas, strict=True):
        lines = []
        for i in range(N_ITEMS):
            bad = f"{run.name}/pq{i}" in invalid
            lines.append(
                json.dumps(
                    {
                        "id": f"pq{i}",
                        "kind": "policy_questions",
                        "valid": not bad,
                        "verdict": None if bad else _verdict(True, True),
                        "raw": "",
                        "prompt_sha256": sha,
                    }
                )
            )
        (run / "verdicts.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = _read_sheet(sheet)
    for r in rows:
        r["correta"], r["fiel"] = human(r["chave"])
    graded = tmp_path / "graded.csv"
    with graded.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=delimiter)
        w.writeheader()
        w.writerows(rows)
    return graded, env.grading / "key.json"


def test_agreement_accuracy_kappa_and_prompt_sha(env, tmp_path):
    flips = {"g01", "g02"}  # human says 0 where the all-true judge says 1
    graded, key_path = _grade(env, tmp_path, lambda k: ("0" if k in flips else "1", "1"))
    out = judge.judge_agreement(graded, key_path)
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert out.parent == env.results
    assert out.name == f"judge_agreement_{SHA[:12]}.json"
    assert rep["judge_prompt_sha256"] == SHA
    assert rep["correta"] == {"n": 8, "accuracy": 0.75, "kappa": 0.0}  # constant judge: pe == po
    assert rep["fiel_ao_contexto"] == {"n": 8, "accuracy": 1.0, "kappa": 1.0}
    assert rep["skipped_invalid"] == []


def test_agreement_lists_invalid_verdict_keys(env, tmp_path):
    graded, key_path = _grade(
        env, tmp_path, lambda k: ("1", "1"), invalid={f"run_a/pq{i}" for i in range(N_ITEMS)}
    )
    key = json.loads(key_path.read_text(encoding="utf-8"))
    rep = json.loads(judge.judge_agreement(graded, key_path).read_text(encoding="utf-8"))
    expected = [k for k, v in key.items() if v["run"] == "run_a"]
    assert len(expected) == 4 and rep["skipped_invalid"] == expected
    assert rep["correta"]["n"] == 4


def test_agreement_blank_grades_raise_listing_keys(env, tmp_path):
    graded, key_path = _grade(env, tmp_path, lambda k: ("" if k == "g03" else "1", "1"))
    with pytest.raises(ValueError, match="g03"):
        judge.judge_agreement(graded, key_path)


@pytest.mark.parametrize("bad", ["2", "sim", "true"])
def test_agreement_bad_value_raises_naming_key(env, tmp_path, bad):
    graded, key_path = _grade(env, tmp_path, lambda k: (bad if k == "g05" else "1", "1"))
    with pytest.raises(ValueError, match="g05"):
        judge.judge_agreement(graded, key_path)


def test_key_has_no_paths(env):
    judge.export_grading_sheet(env.runs, n=8)
    key = json.loads((env.grading / "key.json").read_text(encoding="utf-8"))
    assert all(set(v) == {"run", "id"} for v in key.values())


def test_agreement_mixed_prompt_hashes_raise(env, tmp_path):
    graded, key_path = _grade(env, tmp_path, lambda k: ("1", "1"), shas=(SHA, "f" * 64))
    with pytest.raises(ValueError, match="prompt_sha256"):
        judge.judge_agreement(graded, key_path)


def test_agreement_missing_hash_raises(env, tmp_path):
    graded, key_path = _grade(env, tmp_path, lambda k: ("1", "1"), shas=(None, None))
    with pytest.raises(ValueError, match="prompt_sha256"):
        judge.judge_agreement(graded, key_path)


def test_agreement_refuses_overwrite_per_prompt_version(env, tmp_path):
    graded, key_path = _grade(env, tmp_path, lambda k: ("1", "1"))
    judge.judge_agreement(graded, key_path)
    with pytest.raises(FileExistsError):
        judge.judge_agreement(graded, key_path)


def test_agreement_reads_semicolon_sheet(env, tmp_path):
    graded, key_path = _grade(env, tmp_path, lambda k: ("1", "1"), delimiter=";")
    rep = json.loads(judge.judge_agreement(graded, key_path).read_text(encoding="utf-8"))
    assert rep["correta"]["n"] == 8


def test_agreement_missing_columns_raise(env, tmp_path):
    graded, key_path = _grade(env, tmp_path, lambda k: ("1", "1"))
    text = graded.read_text(encoding="utf-8-sig").replace("fiel", "outra", 1)
    graded.write_text(text, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="fiel"):
        judge.judge_agreement(graded, key_path)


def test_agreement_unknown_chave_and_missing_id_raise(env, tmp_path):
    graded, key_path = _grade(env, tmp_path, lambda k: ("1", "1"))
    key = json.loads(key_path.read_text(encoding="utf-8"))
    del key["g02"]
    key_path.write_text(json.dumps(key), encoding="utf-8")
    with pytest.raises(ValueError, match="g02"):
        judge.judge_agreement(graded, key_path)
    key["g02"] = {"run": "run_a", "id": "pq999"}
    key_path.write_text(json.dumps(key), encoding="utf-8")
    with pytest.raises(ValueError, match="pq999"):
        judge.judge_agreement(graded, key_path)
