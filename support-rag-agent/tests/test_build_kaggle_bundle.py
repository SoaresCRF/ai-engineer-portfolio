import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_kaggle_bundle.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_kaggle_bundle", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundle_copies_needed_files_and_skips_raw_and_pycache(tmp_path):
    root = tmp_path / "proj"
    files = {
        "src/pkg/a.py": "a",
        "src/pkg/__pycache__/a.pyc": "x",
        "corpus/policy.md": "p",
        "corpus/raw/big.html": "r",
        "corpus/sub/raw/keep.txt": "k",
        "data/eval/manifest.json": "{}",
        "data/retrieval_cache/dev.json": "{}",
        "data/runs/e1/outputs.jsonl": "{}",
        "config/demo_tokens.json": "{}",
        "pyproject.toml": "[project]",
        "uv.lock": "lock",
    }
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    stale = root / "dist" / "kaggle" / "stale.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("old")

    out = _load().build(root)

    copied = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert copied == {
        "src/pkg/a.py",
        "corpus/policy.md",
        "corpus/sub/raw/keep.txt",
        "data/eval/manifest.json",
        "data/retrieval_cache/dev.json",
        "data/runs/e1/outputs.jsonl",
        "config/demo_tokens.json",
        "pyproject.toml",
        "uv.lock",
    }


def test_bundle_skips_absent_runs_dir(tmp_path):
    root = tmp_path / "proj"
    for rel in ("src/a.py", "corpus/p.md", "data/eval/m.json", "data/retrieval_cache/d.json"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("x")
    (root / "config").mkdir()
    (root / "pyproject.toml").write_text("x")
    (root / "uv.lock").write_text("x")

    out = _load().build(root)

    assert not (out / "data" / "runs").exists()
