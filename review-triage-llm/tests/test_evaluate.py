import pandas as pd
import pytest

from review_triage.evaluate import confusion, score

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
