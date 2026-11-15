"""Copy what a Kaggle notebook needs into dist/kaggle/ (wiped first)."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = ("src", "corpus", "data/eval", "config")
FILES = ("pyproject.toml", "uv.lock")  # the notebook finds the project through pyproject.toml
IGNORE = ("__pycache__", "*.pyc")


def _ignore(src: Path, skip_top_raw: bool):
    base = shutil.ignore_patterns(*IGNORE)

    def ignore(directory: str, names: list[str]) -> set[str]:
        skipped = set(base(directory, names))
        # Only corpus/raw/ is excluded; a nested directory named raw is kept.
        if skip_top_raw and Path(directory) == src:
            skipped |= {n for n in names if n == "raw"}
        return skipped

    return ignore


def build(root: Path = ROOT, out: Path | None = None) -> Path:
    out = out or root / "dist" / "kaggle"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for d in DIRS:
        shutil.copytree(root / d, out / d, ignore=_ignore(root / d, d == "corpus"))
    for f in FILES:
        shutil.copy2(root / f, out / f)
    return out


if __name__ == "__main__":
    print(build())
