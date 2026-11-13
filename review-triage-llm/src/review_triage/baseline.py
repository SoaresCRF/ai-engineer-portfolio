"""Approach A: TF-IDF + logistic regression, one classifier per label field.

Trained on the teacher's silver labels, like the fine-tuned student, so A and C learn
from the same data.

Usage:
    uv run python -m review_triage.baseline <teacher_silver.csv> <split: dev|test>
"""

import json
import sys
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

from review_triage.evaluate import score
from review_triage.labeling import LABELS_DIR, PROJECT_DIR
from review_triage.schema import LABEL_FIELDS

PREDICTIONS_DIR = PROJECT_DIR / "data" / "predictions"
# Character n-grams within word boundaries: Olist reviews are full of typos
# ("prpduto", "prqzo"), which word features would treat as unseen tokens.
NGRAM_RANGE = (2, 5)
MIN_DF = 2
MAX_ITER = 2_000
SEED = 42


def make_classifier() -> Pipeline:
    return make_pipeline(
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=NGRAM_RANGE,
            min_df=MIN_DF,
            sublinear_tf=True,
            strip_accents="unicode",
        ),
        # Balanced weights: silver labels are skewed toward praise.
        LogisticRegression(max_iter=MAX_ITER, class_weight="balanced", random_state=SEED),
    )


def train(silver: pd.DataFrame) -> dict[str, Pipeline]:
    """Fit one classifier per field on rows the teacher answered validly."""
    usable = silver[silver["valid_json"].astype(bool)]
    return {field: make_classifier().fit(usable["text"], usable[field]) for field in LABEL_FIELDS}


def predict(models: dict[str, Pipeline], df: pd.DataFrame) -> pd.DataFrame:
    out = df[["review_id", "text"]].copy()
    for field, model in models.items():
        out[field] = model.predict(df["text"])
    return out


def single_item_latency_ms(models: dict[str, Pipeline], texts: list[str]) -> float:
    """Median milliseconds to classify one review on CPU (all three fields)."""
    timings = []
    for text in texts:
        start = time.perf_counter()
        for model in models.values():
            model.predict([text])
        timings.append((time.perf_counter() - start) * 1_000)
    return float(pd.Series(timings).median())


def main(silver_path: str, split: str) -> None:
    gold = pd.read_csv(LABELS_DIR / f"gold_{split}.csv")
    models = train(pd.read_csv(silver_path))
    preds = predict(models, gold)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_csv(PREDICTIONS_DIR / f"baseline_{split}.csv", index=False)
    report = score(preds, gold) | {
        "latency_ms_p50_cpu": round(single_item_latency_ms(models, gold["text"].tolist()), 2)
    }
    (PREDICTIONS_DIR / f"baseline_{split}_run.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[2] not in ("dev", "test"):
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
