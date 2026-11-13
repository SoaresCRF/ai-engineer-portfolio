import pandas as pd
import pytest

from review_triage.evaluate import bootstrap, confusion, score

GOLD = pd.DataFrame(
    {
        "review_id": ["a", "b", "c", "d"],
        "categoria": ["nao_recebido", "nao_recebido", "outro", "outro"],
        "sentimento": ["negativo", "negativo", "neutro", "positivo"],
        "urgencia": ["alta", "alta", "baixa", "baixa"],
    }
)


def test_perfect_predictions_score_one():
    report = score(GOLD, GOLD)
    assert report["valid_rate"] == 1.0
    assert report["categoria"] == {"accuracy": 1.0, "macro_f1": pytest.approx(2 / 7, abs=1e-3)}


def test_missing_and_invalid_predictions_count_as_wrong():
    preds = GOLD.iloc[:3].copy()  # "d" missing entirely
    preds.loc[2, "categoria"] = None  # "c" unparseable

    report = score(preds, GOLD)

    assert report["categoria"]["accuracy"] == 0.5
    assert report["valid_rate"] == 0.5
    assert report["urgencia"]["accuracy"] == 0.75


def test_confusion_uses_schema_order():
    preds = GOLD.assign(categoria=["outro", "nao_recebido", "outro", "outro"])
    matrix = confusion(preds, GOLD, "categoria")
    assert matrix.loc["nao_recebido", "outro"] == 1
    assert matrix.loc["outro", "outro"] == 2


def test_bootstrap_is_paired_and_brackets_the_point_estimate():
    gold = pd.concat([GOLD] * 10, ignore_index=True).assign(review_id=lambda d: d.index)
    noisy = gold.copy()
    noisy.loc[::3, "sentimento"] = "positivo"

    report = bootstrap({"same": noisy, "copy": noisy, "perfect": gold}, gold, n_resamples=200)

    acc = report["sentimento"]["accuracy"]
    assert acc["approaches"]["perfect"] == {"value": 1.0, "low": 1.0, "high": 1.0}
    noisy_ci = acc["approaches"]["same"]
    assert noisy_ci["low"] < noisy_ci["value"] < noisy_ci["high"]
    # Identical predictions on the same resampled reviews differ by exactly zero.
    assert acc["differences"]["same - copy"] == {"value": 0.0, "low": 0.0, "high": 0.0}
    assert acc["differences"]["same - perfect"]["high"] < 0
