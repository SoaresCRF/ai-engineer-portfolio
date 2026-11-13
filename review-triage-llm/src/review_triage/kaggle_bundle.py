"""Build the folder uploaded once as a private Kaggle Dataset for the GPU notebooks.

It carries review texts only (no gold labels, so the test answers never leave this repo)
plus the package code the notebooks import.

Usage:
    uv run python -m review_triage.kaggle_bundle   # -> dist/kaggle/review-triage-data.zip
"""

import shutil
from pathlib import Path

import pandas as pd

from review_triage.data import load_reviews, sample_silver
from review_triage.labeling import LABELS_DIR, PROJECT_DIR, RAW_DIR

BUNDLE_DIR = PROJECT_DIR / "dist" / "kaggle" / "review-triage-data"
PACKAGE_DIR = Path(__file__).resolve().parent
# Only what GPU notebooks import; data/labeling need local files and extra deps.
PACKAGE_FILES = ("__init__.py", "schema.py", "prompts.py", "teacher.py")
TEXT_COLUMNS = ["review_id", "text"]


def build(bundle_dir: Path = BUNDLE_DIR) -> Path:
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    (bundle_dir / "review_triage").mkdir(parents=True)

    gold = {split: pd.read_csv(LABELS_DIR / f"gold_{split}.csv") for split in ("dev", "test")}
    gold_ids = set(pd.concat(gold.values())["review_id"])
    for split, df in gold.items():
        df[TEXT_COLUMNS].to_csv(bundle_dir / f"gold_{split}.csv", index=False)

    silver = sample_silver(load_reviews(RAW_DIR), exclude_ids=gold_ids)
    silver[[*TEXT_COLUMNS, "review_score"]].to_csv(bundle_dir / "silver_pool.csv", index=False)

    for name in PACKAGE_FILES:
        shutil.copy2(PACKAGE_DIR / name, bundle_dir / "review_triage" / name)

    archive = shutil.make_archive(str(bundle_dir), "zip", root_dir=bundle_dir)
    print(f"gold dev/test: {len(gold['dev'])}/{len(gold['test'])}, silver: {len(silver)}")
    return Path(archive)


if __name__ == "__main__":
    print(build())
