import pandas as pd

from review_triage.prompts import parse_output
from review_triage.schema import Triage
from review_triage.student import training_records


class FakeTokenizer:
    """Renders the chat as plain text so the test can see what the prompt contains."""

    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        return "|".join(m["content"] for m in messages) + "|assistant:"


def test_training_records_skip_invalid_rows_and_end_with_eos():
    silver = pd.DataFrame(
        {
            "review_id": ["a", "b"],
            "text": ["não chegou", "???"],
            "valid_json": [True, False],
            "categoria": ["nao_recebido", None],
            "sentimento": ["negativo", None],
            "urgencia": ["alta", None],
        }
    )

    records = training_records(silver, FakeTokenizer())

    assert len(records) == 1
    prompt, completion = records[0]["prompt"], records[0]["completion"]
    # No few-shot: system prompt, then the review, then the generation prompt.
    assert prompt.count("|") == 2 and prompt.endswith("não chegou|assistant:")
    assert completion.endswith("<eos>")
    assert parse_output(completion) == Triage(
        categoria="nao_recebido", sentimento="negativo", urgencia="alta"
    )
