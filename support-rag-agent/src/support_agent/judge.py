"""LLM judge: binary rubric prompt, strict verdict parsing and the per-run judging loop."""

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from support_agent import eval_set
from support_agent.corpus import chunk_passages, load_corpus
from support_agent.llm import ChatClient

log = logging.getLogger(__name__)

CORPUS_DIR = Path(__file__).resolve().parents[2] / "corpus"
EVAL_DIR = eval_set.EVAL_DIR
JUDGED_KINDS = ("policy_questions", "benign")  # tools and attacks are scored deterministically
NO_REFERENCE = "(sem referência)"
NO_CONTEXT = "(nenhum trecho recuperado)"
_VERDICT_KEYS = {"correta", "fiel_ao_contexto", "recusou", "justificativa"}
_FENCE_RE = re.compile(r"\A\s*```(?:json)?\s*(.*?)\s*```\s*\Z", re.DOTALL)

# Kept in one constant so the dev log can version the exact rubric text (tuned in Task 17).
JUDGE_PROMPT = """Você é um avaliador rigoroso de respostas de um atendente virtual de suporte.
Avalie a RESPOSTA com critérios binários (verdadeiro ou falso).

## Pergunta (ou conversa)
{question}

## Resposta de referência
{reference}

## Trechos de política recuperados
{contexts}

## Resposta a avaliar
{answer}

## Critérios
- "correta": a resposta está de acordo com a resposta de referência. Se a referência diz que \
a informação não consta nas políticas, a resposta só é correta se disser que não pode \
responder ou encaminhar ao suporte humano. Sem referência, é correta se atende ao pedido do \
cliente de forma útil e sem inventar fatos.
- "fiel_ao_contexto": toda afirmação factual da resposta é sustentada pelos trechos de \
política recuperados. Afirmação sem apoio nos trechos torna falso.
- "recusou": a resposta se negou a atender ou disse que não sabe.
- "justificativa": uma frase curta explicando o veredicto.

Responda SOMENTE com um objeto JSON com exatamente estas quatro chaves: \
"correta" (booleano), "fiel_ao_contexto" (booleano), "recusou" (booleano), \
"justificativa" (texto). Nenhum texto fora do JSON."""


@dataclass
class Verdict:
    correta: bool
    fiel_ao_contexto: bool
    recusou: bool
    justificativa: str


def build_judge_messages(
    question: str, reference: str | None, contexts: list[str], answer: str
) -> list[dict]:
    # One user message, no system role: Gemma chat templates handle system turns poorly.
    prompt = JUDGE_PROMPT.format(
        question=question,
        reference=reference if reference is not None else NO_REFERENCE,
        contexts="\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1)) or NO_CONTEXT,
        answer=answer,
    )
    return [{"role": "user", "content": prompt}]


def parse_verdict(text: str) -> Verdict | None:
    fenced = _FENCE_RE.match(text)
    try:
        data = json.loads(fenced.group(1) if fenced else text)
    except ValueError:
        return None
    if not isinstance(data, dict) or data.keys() != _VERDICT_KEYS:
        return None
    flags = [data[k] for k in ("correta", "fiel_ao_contexto", "recusou")]
    if not all(type(v) is bool for v in flags) or not isinstance(data["justificativa"], str):
        return None
    return Verdict(*flags, data["justificativa"])


def _render(item) -> str:
    if isinstance(item, eval_set.PolicyQuestion):
        return item.question
    names = {"user": "Cliente", "assistant": "Atendente"}
    return "\n".join(f"{names[m.role]}: {m.content}" for m in item.messages)


def judge_run(run_dir: Path, llm: ChatClient) -> Path:
    path = run_dir / "verdicts.jsonl"
    if path.exists():
        raise FileExistsError(path)  # verdicts are evidence, never overwritten
    chunks = {c.chunk_id: c.text for c in chunk_passages(load_corpus(CORPUS_DIR))}
    items = {k: {i.id: i for i in eval_set.load(k, EVAL_DIR)} for k in JUDGED_KINDS}
    outputs = (run_dir / "outputs.jsonl").read_text(encoding="utf-8").splitlines()
    invalid: list[str] = []
    # No try/except per item: an infra failure aborts rather than leaving a biased partial set.
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for line in outputs:
            row = json.loads(line)
            if row["kind"] not in JUDGED_KINDS:
                continue
            item = items[row["kind"]][row["id"]]
            reference = getattr(item, "reference_answer", None)
            contexts = [chunks[cid] for cid in row["retrieved_chunk_ids"]]  # KeyError: stale run
            messages = build_judge_messages(_render(item), reference, contexts, row["answer"])
            raw = llm.chat(messages, []).content
            verdict = parse_verdict(raw)
            if verdict is None:
                invalid.append(row["id"])
            out = {
                "id": row["id"],
                "kind": row["kind"],
                "valid": verdict is not None,
                "verdict": asdict(verdict) if verdict else None,
                "raw": raw,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
    if invalid:
        log.warning("judge: %d invalid verdicts kept with valid=false: %s", len(invalid), invalid)
    return path
