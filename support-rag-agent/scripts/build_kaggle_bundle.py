"""Copy what a Kaggle notebook needs into dist/kaggle/ (wiped first)."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = ("src", "corpus", "data/eval", "config")
FILES = ("pyproject.toml", "uv.lock")  # the notebook finds the project through pyproject.toml
IGNORE = ("__pycache__", "*.pyc")
CORPUS_IGNORE = (*IGNORE, "raw")


def build(root: Path = ROOT, out: Path | None = None) -> Path:
    out = out or root / "dist" / "kaggle"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for d in DIRS:
        patterns = CORPUS_IGNORE if d == "corpus" else IGNORE
        shutil.copytree(root / d, out / d, ignore=shutil.ignore_patterns(*patterns))
    for f in FILES:
        shutil.copy2(root / f, out / f)
    return out


if __name__ == "__main__":
    print(build())
