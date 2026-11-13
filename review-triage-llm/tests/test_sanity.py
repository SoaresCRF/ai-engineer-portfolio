import pandas as pd

from review_triage.sanity import checks


def test_checks_compare_labels_with_stars_and_delivery_facts():
    df = pd.DataFrame(
        {
            "review_score": [1, 1, 5, 5],
            "delivered": [True, False, True, True],
            "late": [True, False, False, True],
            "categoria": ["atraso_entrega", "nao_recebido", "elogio_sem_problema", "outro"],
            "sentimento": ["negativo", "positivo", "negativo", "positivo"],
        }
    )

    report = checks(df)

    assert report["low_star_reviews_labeled_positivo"] == 0.5
    assert report["high_star_reviews_labeled_negativo"] == 0.5
    assert report["atraso_entrega_actually_late"] == 1.0
    assert report["late_orders_labeled_atraso_entrega"] == 0.5
    assert report["nao_recebido_actually_undelivered"] == 1.0
    assert report["mean_score_by_sentiment"] == {"negativo": 3.0, "positivo": 3.0}
