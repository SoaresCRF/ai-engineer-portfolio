"""Score predictions against gold labels with one metric definition for every approach.

Invalid or missing predictions count as wrong, so an approach can't look better by
refusing to answer.

Usage:
    uv run python -m review_triage.evaluate <predictions.csv> <gold.csv>
"""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from review_triage.schema import LABEL_FIELDS

INVALID = "__invalid__"


def align(preds: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Gold rows with the matching prediction columns (suffix _pred); missing -> INVALID."""
    pred_cols = preds[["review_id", *LABEL_FIELDS]].rename(
        columns={f: f"{f}_pred" for f in LABEL_FIELDS}
    )
    merged = gold.merge(pred_cols, on="review_id", how="left", validate="one_to_one")
    for field in LABEL_FIELDS:
        merged[f"{field}_pred"] = merged[f"{field}_pred"].fillna(INVALID)
    return merged


def score(preds: pd.DataFrame, gold: pd.DataFrame) -> dict:
    merged = align(preds, gold)
    report = {"n": len(merged)}
    for field, enum in LABEL_FIELDS.items():
        y_true, y_pred = merged[field], merged[f"{field}_pred"]
        report[field] = {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            # Macro over the schema's classes only: every class weighs the same.
            "macro_f1": round(
                f1_score(
                    y_true, y_pred, labels=[m.value for m in enum], average="macro", zero_division=0
                ),
                4,
            ),
        }
    all_valid = (merged[[f"{f}_pred" for f in LABEL_FIELDS]] != INVALID).all(axis=1)
    report["valid_rate"] = round(float(all_valid.mean()), 4)
    return report


def confusion(preds: pd.DataFrame, gold: pd.DataFrame, field: str) -> pd.DataFrame:
    merged = align(preds, gold)
    labels = [m.value for m in LABEL_FIELDS[field]]
    matrix = confusion_matrix(merged[field], merged[f"{field}_pred"], labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    predictions, gold_labels = (pd.read_csv(Path(p)) for p in sys.argv[1:])
    print(json.dumps(score(predictions, gold_labels), indent=2))
