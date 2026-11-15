import json
import logging
from types import SimpleNamespace

import pytest

from support_agent import eval_set, judge
from support_agent.corpus import chunk_passages, load_corpus
from support_agent.eval_set import Benign, Message, PolicyQuestion
from support_agent.judge import Verdict, build_judge_messages, parse_verdict
from support_agent.llm import AssistantTurn, OpenAIChatClient

VALID = {"correta": True, "fiel_ao_contexto": False, "recusou": False, "justificativa": "ok"}


def test_parse_plain_json():
    assert parse_verdict(json.dumps(VALID)) == Verdict(True, False, False, "ok")


def test_parse_fenced_json():
    text = f"\n```json\n{json.dumps(VALID)}\n```\n"
    assert parse_verdict(text) == Verdict(True, False, False, "ok")


@pytest.mark.parametrize(
    "text",
    [
        "sem json aqui",
        "",
        "[1, 2]",
        json.dumps({**VALID, "correta": 1}),
        json.dumps({**VALID, "recusou": "true"}),
        json.dumps({**VALID, "justificativa": 3}),
        json.dumps({k: v for k, v in VALID.items() if k != "recusou"}),
        json.dumps({**VALID, "extra": True}),
        "Veredicto: " + json.dumps(VALID),
    ],
)
def test_parse_garbage_returns_none(text):
    assert parse_verdict(text) is None


def test_judge_prompt_contains_reference_and_contexts():
    msgs = build_judge_messages(
        "Posso desistir?", "Sim, em 7 dias.", ["ctx-um", "ctx-dois"], "Sim."
    )
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    text = msgs[0]["content"]
    for part in ("Posso desistir?", "Sim, em 7 dias.", "ctx-um", "ctx-dois", "Sim."):
        assert part in text


def test_judge_prompt_without_reference_says_so():
    text = build_judge_messages("q", None, [], "a")[0]["content"]
    assert "sem referência" in text


class ScriptedJudge:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    def chat(self, messages, tools):
        self.calls.append((messages, tools))
        return AssistantTurn(self.texts.pop(0))


@pytest.fixture
def env(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    (corpus / "policies").mkdir(parents=True)
    (corpus / "policies" / "doc.md").write_text(
        "# Doc\n\n## Reembolso\n\nO prazo de reembolso é de 7 dias.\n", encoding="utf-8"
    )
    chunk = chunk_passages(load_corpus(corpus))[0]
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    pq = PolicyQuestion(
        id="pq1",
        split="dev",
        author="claude",
        question="Qual o prazo?",
        gold_passages=[],
        reference_answer="7 dias",
        answerable=True,
    )
    pq2 = pq.model_copy(update={"id": "pq2"})
    bn = Benign(
        id="bn1",
        split="dev",
        author="claude",
        customer_id="C001",
        messages=[
            Message(role="user", content="Oi"),
            Message(role="assistant", content="Olá"),
            Message(role="user", content="Onde está meu pedido?"),
        ],
    )
    eval_set.dump_jsonl(eval_dir / "policy_questions.jsonl", [pq, pq2])
    eval_set.dump_jsonl(eval_dir / "benign.jsonl", [bn])
    monkeypatch.setattr(judge, "CORPUS_DIR", corpus)
    monkeypatch.setattr(judge, "EVAL_DIR", eval_dir)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        {
            "id": "pq1",
            "kind": "policy_questions",
            "answer": "7 dias",
            "retrieved_chunk_ids": [chunk.chunk_id],
        },
        {"id": "pq2", "kind": "policy_questions", "answer": "talvez", "retrieved_chunk_ids": []},
        {"id": "bn1", "kind": "benign", "answer": "Já vou ver", "retrieved_chunk_ids": []},
        {"id": "ts1", "kind": "tool_scenarios", "answer": "x", "retrieved_chunk_ids": []},
        {"id": "at1", "kind": "attacks", "answer": "x", "retrieved_chunk_ids": []},
    ]
    (run_dir / "outputs.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    return SimpleNamespace(run_dir=run_dir, chunk=chunk)


def test_judge_run_writes_rows_skips_kinds_and_keeps_invalid(env, caplog):
    llm = ScriptedJudge([json.dumps(VALID), "lixo", json.dumps(VALID)])
    with caplog.at_level(logging.WARNING):
        path = judge.judge_run(env.run_dir, llm)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == ["pq1", "pq2", "bn1"]
    assert [r["valid"] for r in rows] == [True, False, True]
    assert rows[0]["verdict"] == VALID and rows[1]["verdict"] is None
    assert rows[1]["raw"] == "lixo"
    assert "pq2" in caplog.text
    assert all(tools == [] for _, tools in llm.calls)
    first, _, third = (c[0][0]["content"] for c in llm.calls)
    assert "7 dias" in first and env.chunk.text in first
    assert "Cliente: Oi" in third and "Atendente: Olá" in third and "sem referência" in third


def test_judge_run_refuses_to_overwrite(env):
    (env.run_dir / "verdicts.jsonl").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        judge.judge_run(env.run_dir, ScriptedJudge([]))
    assert (env.run_dir / "verdicts.jsonl").read_text(encoding="utf-8") == "keep"


def test_judge_run_unknown_chunk_id_raises(env):
    lines = (env.run_dir / "outputs.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["retrieved_chunk_ids"] = ["nao-existe"]
    (env.run_dir / "outputs.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(KeyError):
        judge.judge_run(env.run_dir, ScriptedJudge([json.dumps(VALID)]))


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        msg = SimpleNamespace(content="oi", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


@pytest.mark.parametrize(("tools", "sent"), [([], False), ([{"type": "function"}], True)])
def test_chat_omits_empty_tools(tools, sent):
    completions = FakeCompletions()
    fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    OpenAIChatClient("http://x", "m", client=fake).chat([{"role": "user", "content": "q"}], tools)
    assert ("tools" in completions.kwargs) is sent
