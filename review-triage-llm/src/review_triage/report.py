"""Assemble results/final.json: quality, speed and cost of A, B and C on the gold test set.

Reads the committed predictions and run reports; runs no model.

Usage:
    uv run python -m review_triage.report
"""

import json

import pandas as pd
from sklearn.metrics import f1_score

from review_triage.baseline import PREDICTIONS_DIR
from review_triage.evaluate import align, confusion, score
from review_triage.labeling import LABELS_DIR, PROJECT_DIR
from review_triage.schema import LABEL_FIELDS

RESULTS_DIR = PROJECT_DIR / "results"
TEST_EVALUATED_ON = "2026-11-13"  # the single test evaluation
# Google Cloud on-demand price of one T4, us-west8, checked 2026-11-13:
# https://cloud.google.com/products/compute/gpus-pricing
T4_USD_PER_HOUR = 0.35
SECONDS_PER_HOUR = 3_600
COST_UNIT_REVIEWS = 1_000

# approach -> (predictions, run report), relative to data/predictions
APPROACHES = {
    "A_baseline_tfidf": ("baseline_test.csv", "baseline_test_run.json"),
    "B_teacher_qwen3_8b": ("teacher_full_v2/teacher_test.csv", "teacher_full_v2/teacher_run.json"),
    "C_student_qwen3_1.7b_qlora": ("student_v1/student_test.csv", "student_v1/student_run.json"),
}


def gpu_usd(seconds: float) -> float:
    return round(seconds / SECONDS_PER_HOUR * T4_USD_PER_HOUR, 4)


def _speed_and_cost(run: dict) -> dict:
    if "latency_ms_p50_cpu" in run:  # baseline: CPU only, no GPU cost
        return {
            "hardware": "laptop CPU",
            "latency_batch1_p50_s": round(run["latency_ms_p50_cpu"] / 1_000, 4),
            "latency_batch1_p95_s": None,
            "reviews_per_s_batched": None,
            "gpu_usd_per_1k_reviews": 0.0,
        }
    reviews_per_s = run["splits"]["test"]["reviews_per_s"]
    return {
        "hardware": f"1x {run['gpu']}, 4-bit",
        "latency_batch1_p50_s": round(run["latency_batch1"]["p50_s"], 3),
        "latency_batch1_p95_s": round(run["latency_batch1"]["p95_s"], 3),
        "reviews_per_s_batched": reviews_per_s,
        "gpu_usd_per_1k_reviews": gpu_usd(COST_UNIT_REVIEWS / reviews_per_s),
    }


def _per_class(preds: pd.DataFrame, gold: pd.DataFrame) -> dict:
    """F1 per class and confusion matrix (rows = gold, columns = predicted) for each field."""
    merged = align(preds, gold)
    out = {}
    for field, enum in LABEL_FIELDS.items():
        labels = [m.value for m in enum]
        f1 = f1_score(
            merged[field], merged[f"{field}_pred"], labels=labels, average=None, zero_division=0
        )
        out[field] = {
            "support": merged[field].value_counts().reindex(labels, fill_value=0).to_dict(),
            "f1": {label: round(float(v), 3) for label, v in zip(labels, f1, strict=True)},
            "confusion": confusion(preds, gold, field).to_dict(orient="index"),
        }
    return out


def build() -> dict:
    gold = pd.read_csv(LABELS_DIR / "gold_test.csv")
    runs = {
        name: json.loads((PREDICTIONS_DIR / run_file).read_text())
        for name, (_, run_file) in APPROACHES.items()
    }
    preds = {name: pd.read_csv(PREDICTIONS_DIR / f) for name, (f, _) in APPROACHES.items()}
    teacher, student = runs["B_teacher_qwen3_8b"], runs["C_student_qwen3_1.7b_qlora"]
    return {
        "split": "test",
        "evaluated_on": TEST_EVALUATED_ON,
        "cost_assumption": f"GPU time only, T4 at ${T4_USD_PER_HOUR}/h (Google Cloud on-demand)",
        "approaches": {
            name: {
                **score(preds[name], gold),
                **_speed_and_cost(runs[name]),
                "per_class": _per_class(preds[name], gold),
            }
            for name in APPROACHES
        },
        "one_off_gpu_usd": {
            "teacher_silver_labeling": gpu_usd(teacher["splits"]["silver"]["seconds"]),
            "student_training": gpu_usd(student["training"]["seconds"]),
        },
        "bootstrap_ci": "bootstrap_test.json",
    }


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    report = build()
    (RESULTS_DIR / "final.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
