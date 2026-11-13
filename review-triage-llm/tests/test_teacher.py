import pandas as pd

from review_triage.teacher import to_prediction_frame


def test_prediction_frame_keeps_order_and_flags_invalid_output():
    df = pd.DataFrame({"review_id": ["a", "b"], "text": ["não chegou", "???"]})
    raw = [
        '{"categoria": "nao_recebido", "sentimento": "negativo", "urgencia": "alta"}',
        "não sei",
    ]

    preds = to_prediction_frame(df, raw)

    assert preds["valid_json"].tolist() == [True, False]
    assert preds.loc[0, "categoria"] == "nao_recebido"
    assert pd.isna(preds.loc[1, "urgencia"])
