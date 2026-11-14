"""Score predictions against gold labels with one metric definition for every approach.

Invalid or missing predictions count as wrong, so an approach can't look better by
refusing to answer.

Usage:
    uv run python -m review_triage.evaluate <predictions.csv> <gold.csv>
    uv run python -m review_triage.evaluate bootstrap <gold.csv> <name>=<predictions.csv> ...
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from review_triage.schema import LABEL_FIELDS

INVALID = "__invalid__"
BOOTSTRAP_RESAMPLES = 2000
CI_LEVEL = 0.95
BOOTSTRAP_SEED = 42


def align(preds: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Gold rows with the matching prediction columns (suffix _pred); missing -> INVALID."""
    pred_cols = preds[["review_id", *LABEL_FIELDS]].rename(
        columns={f: f"{f}_pred" for f in LABEL_FIELDS}
    )
    merged = gold.merge(pred_cols, on="review_id", how="left", validate="one_to_one")
    for field in LABEL_FIELDS:
        merged[f"{field}_pred"] = merged[f"{field}_pred"].fillna(INVALID)
    return merged


def _metrics(y_true, y_pred, labels: list[str]) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        # Macro over the schema's classes only: every class weighs the same.
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
    }


def score(preds: pd.DataFrame, gold: pd.DataFrame) -> dict:
    merged = align(preds, gold)
    report = {"n": len(merged)}
    for field, enum in LABEL_FIELDS.items():
        metrics = _metrics(merged[field], merged[f"{field}_pred"], [m.value for m in enum])
        report[field] = {name: round(value, 4) for name, value in metrics.items()}
    all_valid = (merged[[f"{f}_pred" for f in LABEL_FIELDS]] != INVALID).all(axis=1)
    report["valid_rate"] = round(float(all_valid.mean()), 4)
    return report


def confusion(merged: pd.DataFrame, field: str) -> pd.DataFrame:
    """Confusion matrix (rows = gold, columns = predicted) on an align() frame."""
    labels = [m.value for m in LABEL_FIELDS[field]]
    matrix = confusion_matrix(merged[field], merged[f"{field}_pred"], labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def _interval(point: float, samples: np.ndarray) -> dict[str, float]:
    tail = (1 - CI_LEVEL) / 2 * 100
    low, high = np.percentile(samples, [tail, 100 - tail])
    return {"value": round(point, 4), "low": round(float(low), 4), "high": round(float(high), 4)}


def bootstrap(
    runs: dict[str, pd.DataFrame],
    gold: pd.DataFrame,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict:
    """Paired percentile bootstrap CIs for every approach and every pairwise difference.

    Each resample draws the same reviews for all approaches, so a difference's CI reflects
    disagreement on shared reviews instead of two independent noise sources.
    """
    aligned = {name: align(preds, gold) for name, preds in runs.items()}
    n = len(gold)
    resamples = np.random.default_rng(BOOTSTRAP_SEED).integers(0, n, size=(n_resamples, n))
    report = {"n": n, "n_resamples": n_resamples, "ci_level": CI_LEVEL}
    for field, enum in LABEL_FIELDS.items():
        labels = [m.value for m in enum]
        y_true = gold[field].to_numpy()
        preds = {name: m[f"{field}_pred"].to_numpy() for name, m in aligned.items()}
        point = {name: _metrics(y_true, p, labels) for name, p in preds.items()}
        draws = {
            name: pd.DataFrame([_metrics(y_true[i], p[i], labels) for i in resamples])
            for name, p in preds.items()
        }
        report[field] = {
            metric: {
                "approaches": {
                    name: _interval(point[name][metric], draws[name][metric].to_numpy())
                    for name in runs
                },
                "differences": {
                    f"{a} - {b}": _interval(
                        point[a][metric] - point[b][metric],
                        (draws[a][metric] - draws[b][metric]).to_numpy(),
                    )
                    for a, b in itertools.combinations(runs, 2)
                },
            }
            for metric in ("accuracy", "macro_f1")
        }
    return report


def _bootstrap_cli(gold_path: str, run_args: list[str]) -> dict:
    runs = {}
    for arg in run_args:
        name, sep, path = arg.partition("=")
        if not sep:
            raise SystemExit(__doc__)
        runs[name] = pd.read_csv(Path(path))
    return bootstrap(runs, pd.read_csv(Path(gold_path)))


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "bootstrap":
        result = _bootstrap_cli(sys.argv[2], sys.argv[3:])
    elif len(sys.argv) == 3:
        predictions, gold_labels = (pd.read_csv(Path(p)) for p in sys.argv[1:])
        result = score(predictions, gold_labels)
    else:
        raise SystemExit(__doc__)
    print(json.dumps(result, indent=2))
