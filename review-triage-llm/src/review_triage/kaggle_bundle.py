"""Build the folder uploaded once as a private Kaggle Dataset for the GPU notebooks.

It carries review texts only (no gold labels, so the test answers never leave this repo),
the teacher's silver labels the student trains on, and the package code the notebooks import.

Usage:
    uv run python -m review_triage.kaggle_bundle   # -> dist/kaggle/review-triage-data.zip
"""

import shutil
from pathlib import Path

import pandas as pd

from review_triage.data import load_reviews, sample_silver
from review_triage.labeling import LABELS_DIR, PROJECT_DIR, RAW_DIR
from review_triage.schema import LABEL_FIELDS

BUNDLE_DIR = PROJECT_DIR / "dist" / "kaggle" / "review-triage-data"
PACKAGE_DIR = Path(__file__).resolve().parent
# Only what GPU notebooks import; data/labeling need local files and extra deps.
PACKAGE_FILES = ("__init__.py", "schema.py", "prompts.py", "teacher.py", "student.py")
TEXT_COLUMNS = ["review_id", "text"]
# Teacher run whose silver labels train the student (prompt 0f83769b).
SILVER_LABELS = PROJECT_DIR / "data" / "predictions" / "teacher_full_v2" / "teacher_silver.csv"
SILVER_LABEL_COLUMNS = [*TEXT_COLUMNS, "valid_json", *LABEL_FIELDS]


def build() -> Path:
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    (BUNDLE_DIR / "review_triage").mkdir(parents=True)

    gold = {split: pd.read_csv(LABELS_DIR / f"gold_{split}.csv") for split in ("dev", "test")}
    gold_ids = set(pd.concat(gold.values())["review_id"])
    for split, df in gold.items():
        df[TEXT_COLUMNS].to_csv(BUNDLE_DIR / f"gold_{split}.csv", index=False)

    silver = sample_silver(load_reviews(RAW_DIR), exclude_ids=gold_ids)
    silver[[*TEXT_COLUMNS, "review_score"]].to_csv(BUNDLE_DIR / "silver_pool.csv", index=False)
    # The teacher run produces the silver labels, so its bundle is built without them;
    # rebuild after downloading teacher outputs to get the student's training data.
    n_labels = 0
    if SILVER_LABELS.exists():
        labels = pd.read_csv(SILVER_LABELS)
        if not gold_ids.isdisjoint(labels["review_id"]):
            raise ValueError(f"{SILVER_LABELS} contains gold reviews")
        labels[SILVER_LABEL_COLUMNS].to_csv(BUNDLE_DIR / "teacher_silver.csv", index=False)
        n_labels = len(labels)

    for name in PACKAGE_FILES:
        shutil.copy2(PACKAGE_DIR / name, BUNDLE_DIR / "review_triage" / name)

    archive = shutil.make_archive(str(BUNDLE_DIR), "zip", root_dir=BUNDLE_DIR)
    print(
        f"gold dev/test: {len(gold['dev'])}/{len(gold['test'])}, "
        f"silver pool: {len(silver)}, silver labels: {n_labels or 'none yet (teacher run)'}"
    )
    return Path(archive)


if __name__ == "__main__":
    print(build())
