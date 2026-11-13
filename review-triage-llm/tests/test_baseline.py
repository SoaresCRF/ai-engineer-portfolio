import pandas as pd

from review_triage.baseline import predict, train

COMPLAINT = ("nao_recebido", "negativo", "alta")
PRAISE = ("elogio_sem_problema", "positivo", "baixa")


def make_silver() -> pd.DataFrame:
    rows = []
    for _ in range(20):
        rows.append(("não recebi o produto ainda", *COMPLAINT, True))
        rows.append(("ótimo produto chegou rápido", *PRAISE, True))
    # An invalid teacher answer must be ignored, not learned as a class.
    rows.append(("texto qualquer", None, None, None, False))
    return pd.DataFrame(
        [(f"r{i}", *r) for i, r in enumerate(rows)],
        columns=["review_id", "text", "categoria", "sentimento", "urgencia", "valid_json"],
    )


def test_baseline_learns_separable_classes_and_skips_invalid_rows():
    models = train(make_silver())
    test = pd.DataFrame({"review_id": ["x", "y"], "text": ["não recebi", "produto ótimo"]})

    preds = predict(models, test)

    assert tuple(preds.loc[0, ["categoria", "sentimento", "urgencia"]]) == COMPLAINT
    assert tuple(preds.loc[1, ["categoria", "sentimento", "urgencia"]]) == PRAISE
    assert set(models["categoria"].classes_) == {COMPLAINT[0], PRAISE[0]}
