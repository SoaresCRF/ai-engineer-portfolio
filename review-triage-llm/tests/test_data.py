import pandas as pd
import pytest

from review_triage.data import (
    GOLD_DEV_PER_SCORE,
    GOLD_TEST_PER_SCORE,
    SCORES,
    build_text,
    load_reviews,
    mask_pii,
    sample_gold,
    validate_labels,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("meu email é joao.silva@gmail.com ok", "meu email é [EMAIL] ok"),
        ("cpf 123.456.789-09", "cpf [CPF]"),
        ("cpf 12345678909", "cpf [CPF]"),
        ("liga (11) 98765-4321", "liga [TELEFONE]"),
        ("liga 11 987654321", "liga [TELEFONE]"),
        ("comprei 2 unidades em 07/05", "comprei 2 unidades em 07/05"),
    ],
)
def test_mask_pii(text, expected):
    assert mask_pii(text) == expected


def test_build_text_joins_title_and_message_and_skips_missing():
    assert build_text("Ótimo", "  chegou\n rápido ") == "Ótimo. chegou rápido"
    assert build_text(float("nan"), "só mensagem") == "só mensagem"


def test_load_reviews_dedups_and_flags_late(tmp_path):
    pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3", "r4"],
            "order_id": ["o1", "o2", "o3", "o4"],
            "review_score": [1, 1, 5, 3],
            "review_comment_title": [None, None, None, None],
            "review_comment_message": ["não chegou", "não  chegou", "ótimo", None],
            "review_creation_date": ["2018-01-01"] * 4,
            "review_answer_timestamp": ["2018-01-01"] * 4,
        }
    ).to_csv(tmp_path / "olist_order_reviews_dataset.csv", index=False)
    pd.DataFrame(
        {
            "order_id": ["o1", "o3"],
            "order_delivered_customer_date": [None, "2018-01-10 15:00:00"],
            "order_estimated_delivery_date": ["2018-01-05", "2018-01-09"],
        }
    ).to_csv(tmp_path / "olist_orders_dataset.csv", index=False)

    df = load_reviews(tmp_path).set_index("review_id")

    assert list(df.index) == ["r1", "r3"]  # r2 duplicate after whitespace normalization, r4 no text
    assert not df.loc["r1", "delivered"]
    assert df.loc["r3", "late"]


def test_sample_gold_is_stratified_split_and_deterministic():
    df = pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(500)],
            "review_score": [SCORES[i % len(SCORES)] for i in range(500)],
            "text": ["x"] * 500,
        }
    )
    gold = sample_gold(df)

    assert gold["review_id"].is_unique
    counts = gold.groupby(["review_score", "split"]).size().unstack()
    assert (counts["dev"] == GOLD_DEV_PER_SCORE).all()
    assert (counts["test"] == GOLD_TEST_PER_SCORE).all()
    assert gold.equals(sample_gold(df))


def test_validate_labels_normalizes_case_and_rejects_unknown_values():
    ok = pd.DataFrame(
        {
            "review_id": ["a"],
            "categoria": [" Atraso_Entrega "],
            "sentimento": ["negativo"],
            "urgencia": ["media"],
        }
    )
    assert validate_labels(ok).loc[0, "categoria"] == "atraso_entrega"

    bad = ok.assign(urgencia=["urgentissimo"])
    with pytest.raises(ValueError, match="review_id=a"):
        validate_labels(bad)
