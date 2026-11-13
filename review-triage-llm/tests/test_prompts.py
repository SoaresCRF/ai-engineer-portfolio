import pandas as pd
import pytest

from review_triage.labeling import LABELS_DIR
from review_triage.prompts import FEW_SHOT, build_messages, parse_output, to_json
from review_triage.schema import Triage

EXPECTED = Triage(categoria="nao_recebido", sentimento="negativo", urgencia="alta")


@pytest.mark.parametrize(
    "raw",
    [
        '{"categoria": "nao_recebido", "sentimento": "negativo", "urgencia": "alta"}',
        'Resposta: {"categoria": "nao_recebido", "sentimento": "negativo", "urgencia": "alta"} ok',
        '<think>hmm</think>\n{"categoria": "Nao_Recebido", "sentimento": "negativo", '
        '"urgencia": "alta"}',
        '{"categoria": "não_recebido", "sentimento": "negativo", "urgencia": "alta"}',
    ],
)
def test_parse_output_accepts_valid_json_with_noise(raw):
    assert parse_output(raw) == EXPECTED


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "nao_recebido",
        '{"categoria": "nao_recebido", "sentimento": "negativo"}',  # missing field
        '{"categoria": "inventada", "sentimento": "negativo", "urgencia": "alta"}',
        '{"categoria": "nao_recebido", "sentimento": "negativo", "urgencia": "alta", "x": 1}',
        '{"categoria": "nao_recebido", "sentimento": ',  # truncated generation
    ],
)
def test_parse_output_rejects_invalid(raw):
    assert parse_output(raw) is None


def test_to_json_round_trips():
    assert parse_output(to_json(EXPECTED)) == EXPECTED


def test_build_messages_shapes():
    with_shots = build_messages("chegou quebrado")
    assert with_shots[0]["role"] == "system"
    assert with_shots[-1] == {"role": "user", "content": "chegou quebrado"}
    assert len(with_shots) == 2 + 2 * len(FEW_SHOT)
    assert len(build_messages("chegou quebrado", few_shot=False)) == 2


def test_few_shot_examples_are_not_in_gold():
    gold = pd.concat(pd.read_csv(LABELS_DIR / f"gold_{s}.csv") for s in ("dev", "test"))
    gold_texts = {t.casefold().strip() for t in gold["text"]}
    for text, _ in FEW_SHOT:
        assert text.casefold().strip() not in gold_texts
