"""Label sanity checks against facts the labels never saw: star rating and delivery dates.

Neither the human labeler nor the models saw `review_score` or the order dates, so they
are a free, independent check on every label set, including the 5,000 silver labels that
have no human reference.

Usage:
    uv run python -m review_triage.sanity   # needs data/raw (see README)
"""

import json

import pandas as pd

from review_triage.baseline import PREDICTIONS_DIR
from review_triage.data import load_reviews
from review_triage.labeling import LABELS_DIR, RAW_DIR, load_gold
from review_triage.report import RESULTS_DIR

LOW_SCORES = (1, 2)
HIGH_SCORES = (4, 5)
META = ["review_id", "review_score", "delivered", "late"]
LABELS = ["review_id", "categoria", "sentimento"]


def checks(df: pd.DataFrame) -> dict:
    """df: one row per review with review_score, delivered, late, categoria, sentimento."""
    positive, negative = df["sentimento"] == "positivo", df["sentimento"] == "negativo"
    low, high = df["review_score"].isin(LOW_SCORES), df["review_score"].isin(HIGH_SCORES)
    late_label = df["categoria"] == "atraso_entrega"
    missing_label = df["categoria"] == "nao_recebido"

    def share(mask: pd.Series) -> float | None:
        return round(float(mask.mean()), 3) if len(mask) else None

    return {
        "n": len(df),
        "mean_score_by_sentiment": {
            s: round(float(g.mean()), 2) for s, g in df.groupby("sentimento")["review_score"]
        },
        # A star rating is not the tone of the text, but strong contradictions should be rare.
        "low_star_reviews_labeled_positivo": share(positive[low]),
        "high_star_reviews_labeled_negativo": share(negative[high]),
        "atraso_entrega_actually_late": share(df.loc[late_label & df["delivered"], "late"]),
        "late_orders_labeled_atraso_entrega": share(late_label[df["late"]]),
        "nao_recebido_actually_undelivered": share(~df.loc[missing_label, "delivered"]),
    }


def build() -> dict:
    gold = load_gold()
    gold_test = pd.read_csv(LABELS_DIR / "gold_test.csv")[META]
    silver = pd.read_csv(PREDICTIONS_DIR / "teacher_full_v2" / "teacher_silver.csv")
    silver = silver[silver["valid_json"].astype(bool)][LABELS]
    silver_meta = load_reviews(RAW_DIR)[META]

    def with_meta(preds_file: str) -> pd.DataFrame:
        preds = pd.read_csv(PREDICTIONS_DIR / preds_file)[LABELS]
        return gold_test.merge(preds, on="review_id", validate="one_to_one")

    return {
        "human_gold_dev_test": checks(gold),
        "teacher_silver": checks(silver.merge(silver_meta, on="review_id", validate="one_to_one")),
        "teacher_test": checks(with_meta("teacher_full_v2/teacher_test.csv")),
        "student_test": checks(with_meta("student_v1/student_test.csv")),
    }


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    report = build()
    (RESULTS_DIR / "sanity_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
