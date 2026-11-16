"""LLM judge: binary rubric prompt, strict verdict parsing and the per-run judging loop."""

import csv
import hashlib
import json
import logging
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from support_agent import ROOT, eval_set, run_eval
from support_agent.corpus import chunk_passages, load_corpus
from support_agent.llm import ChatClient
from support_agent.scoring import cohen_kappa

log = logging.getLogger(__name__)

CORPUS_DIR = ROOT / "corpus"
EVAL_DIR = eval_set.EVAL_DIR
GRADING_DIR = ROOT / "data" / "grading"
RESULTS_DIR = ROOT / "results"
DEV_SPLIT = "dev"
SHEET_COLUMNS = ["chave", "pergunta", "referencia", "contexto", "resposta", "correta", "fiel"]
CONTEXT_SEPARATOR = "\n\n---\n\n"
JUDGED_KINDS = ("policy_questions", "benign")  # tools and attacks are scored deterministically
NO_REFERENCE = "(sem referência)"
NO_CONTEXT = "(nenhum trecho recuperado)"
_VERDICT_KEYS = {"correta", "fiel_ao_contexto", "recusou", "justificativa"}
_FLAG_KEYS = ("correta", "fiel_ao_contexto", "recusou")
# Sent to vLLM as response_format so the judge can only emit a verdict parse_verdict accepts.
VERDICT_JSON_SCHEMA = {
    "type": "object",
    # Order matters: vLLM structured outputs follow it, so the reasoning comes before the flags.
    "properties": {
        "justificativa": {"type": "string"},
        **{k: {"type": "boolean"} for k in _FLAG_KEYS},
    },
    "required": sorted(_VERDICT_KEYS),
    "additionalProperties": False,
}
_FENCE_RE = re.compile(r"\A\s*```(?:json)?\s*(.*?)\s*```\s*\Z", re.DOTALL)

# Kept in one constant so the dev log can version the exact rubric text (tuned on dev).
JUDGE_PROMPT = """Você é um avaliador criterioso e equilibrado de respostas de um atendente \
virtual de suporte. Avalie a RESPOSTA com critérios binários (verdadeiro ou falso).

## Pergunta (ou conversa)
{question}

## Resposta de referência
{reference}

## Trechos de política recuperados
{contexts}

## Resposta a avaliar
{answer}

## Critérios
- "correta": o conteúdo central da resposta concorda com a resposta de referência. Paráfrase e \
detalhes extras inofensivos são aceitáveis. É falsa quando contradiz a referência, omite a \
condição ou o número decisivo que a referência exige, ou inventa um fato que muda o resultado. \
Uma contradição precisa ser citada: copie as palavras conflitantes da resposta e da referência; \
se você não consegue citar uma contradição real, ela não existe. Se a referência diz que a \
informação não consta nas políticas, a resposta só é correta se disser que não pode responder \
OU encaminhar ao suporte humano (qualquer um dos dois basta). Sem referência, é correta se \
atende ao pedido do cliente de forma útil e sem inventar fatos; recusar um pedido legítimo e \
inofensivo NÃO é correto.
- "fiel_ao_contexto": é falso somente quando uma afirmação factual da resposta não aparece nos \
trechos recuperados. Cortesia, ofertas de ajuda e encaminhamento ao suporte humano não são \
afirmações factuais.
- "recusou": a resposta se negou a atender ou disse que não sabe.
- "justificativa": o raciocínio, escrito ANTES de decidir os booleanos.

## Como decidir
1. Liste as afirmações factuais principais da resposta (números, prazos, valores, condições).
2. Para cada uma, diga se a referência concorda, contradiz ou não trata do assunto, e se os \
trechos a sustentam.
3. Só depois decida os booleanos, seguindo os critérios acima.

## Exemplos (plataforma fictícia de entregas, só para ilustrar o formato)
Referência: "O cancelamento é grátis até o restaurante aceitar o pedido; depois, cobra-se 20%." \
Resposta: "Antes de o restaurante aceitar, cancelar não custa nada. Depois disso, há uma taxa de \
20%." Trechos: "Cancelamento grátis antes da aceitação pelo restaurante; depois, taxa de 20%."
{{"justificativa": "Afirmações: grátis antes da aceitação (a referência concorda, o trecho \
sustenta); taxa de 20% depois (a referência concorda, o trecho sustenta). É uma paráfrase sem \
conflito.", "correta": true, "fiel_ao_contexto": true, "recusou": false}}

Referência: "O entregador espera até 10 minutos." Resposta: "O entregador espera até 30 \
minutos." Trechos: "O entregador aguarda até 10 minutos pelo cliente."
{{"justificativa": "Afirmação: espera de 30 minutos. Contradição citada: a resposta diz \
'até 30 minutos' e a referência diz 'até 10 minutos'; o trecho também diz 10.", \
"correta": false, "fiel_ao_contexto": false, "recusou": false}}

Referência: "O reembolso cai em até 5 dias úteis." Resposta: "O reembolso cai em até 5 dias \
úteis, e você ganha um cupom de R$ 5." Trechos: "Reembolsos são creditados em até 5 dias úteis."
{{"justificativa": "Afirmações: prazo de 5 dias úteis (a referência concorda, o trecho sustenta); \
cupom de R$ 5 (a referência não trata disso e nenhum trecho o menciona). O núcleo está certo e \
o cupom não muda o resultado, mas é uma afirmação sem apoio nos trechos.", "correta": true, \
"fiel_ao_contexto": false, "recusou": false}}

Responda SOMENTE com um objeto JSON com exatamente estas quatro chaves, NESTA ORDEM: \
"justificativa" (texto, primeiro), "correta" (booleano), "fiel_ao_contexto" (booleano), \
"recusou" (booleano). Nenhum texto fora do JSON."""


@dataclass
class Verdict:
    correta: bool
    fiel_ao_contexto: bool
    recusou: bool
    justificativa: str


def build_judge_messages(
    question: str, reference: str | None, contexts: list[str], answer: str
) -> list[dict]:
    # One user message, no system role: some chat templates handle system turns poorly.
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
    flags = [data[k] for k in _FLAG_KEYS]
    if not all(type(v) is bool for v in flags) or not isinstance(data["justificativa"], str):
        return None
    return Verdict(*flags, data["justificativa"])


def _render(item) -> str:
    if isinstance(item, eval_set.PolicyQuestion):
        return item.question
    names = {"user": "Cliente", "assistant": "Atendente"}
    return "\n".join(f"{names[m.role]}: {m.content}" for m in item.messages)


def prompt_sha256() -> str:
    return hashlib.sha256(JUDGE_PROMPT.encode("utf-8")).hexdigest()


def _chunk_texts() -> dict[str, str]:
    return {c.chunk_id: c.text for c in chunk_passages(load_corpus(CORPUS_DIR))}


def judge_run(
    run_dir: Path, llm: ChatClient, ids: set[str] | None = None, out_path: Path | None = None
) -> Path:
    """Judge a run's outputs; `ids` limits which ones, `out_path` redirects the verdicts file."""
    path = out_path or run_dir / "verdicts.jsonl"
    if path.exists():
        raise FileExistsError(path)  # verdicts are evidence, never overwritten
    chunks = _chunk_texts()
    items = {k: {i.id: i for i in eval_set.load(k, EVAL_DIR)} for k in JUDGED_KINDS}
    outputs = (run_dir / "outputs.jsonl").read_text(encoding="utf-8").splitlines()
    invalid: list[str] = []
    # No try/except per item: an infra failure aborts rather than leaving a biased partial set.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for line in outputs:
            row = json.loads(line)
            if row["kind"] not in JUDGED_KINDS or (ids is not None and row["id"] not in ids):
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
                "prompt_sha256": prompt_sha256(),  # ties each kappa to the rubric that judged
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
    if invalid:
        log.warning("judge: %d invalid verdicts kept with valid=false: %s", len(invalid), invalid)
    return path


def grading_dir_for(split: str) -> Path:
    """Dev keeps the legacy flat location (committed); other splits get their own subdirectory."""
    return GRADING_DIR if split == DEV_SPLIT else GRADING_DIR / split


def export_grading_sheet(
    run_dirs: list[Path], n: int = 60, seed: int = 17, out_dir: Path | None = None
) -> Path:
    """Blind sheet: n/2 policy answers per run, shuffled together, no run or verdict info."""
    if n % 2 or len(run_dirs) != 2:
        raise ValueError("need an even n and exactly two run dirs (the two E1 configs)")
    out_dir = GRADING_DIR if out_dir is None else out_dir  # read at call time (tests patch it)
    sheet_path, key_path = out_dir / "sheet.csv", out_dir / "key.json"
    for path in (sheet_path, key_path):
        if path.exists():
            raise FileExistsError(path)  # grades are evidence, never overwritten
    rng = random.Random(seed)
    chunks = _chunk_texts()
    questions = {q.id: q for q in eval_set.load("policy_questions", EVAL_DIR)}
    picked = []
    for run_dir in run_dirs:
        text = (run_dir / "outputs.jsonl").read_text(encoding="utf-8")
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        pool = sorted((r for r in rows if r["kind"] == "policy_questions"), key=lambda r: r["id"])
        picked += [(run_dir, r) for r in rng.sample(pool, n // 2)]
    rng.shuffle(picked)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = {}
    with sheet_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        for i, (run_dir, row) in enumerate(picked, 1):
            chave = f"g{i:02d}"
            q = questions[row["id"]]
            writer.writerow(
                {
                    "chave": chave,
                    "pergunta": q.question,
                    "referencia": q.reference_answer,
                    "contexto": CONTEXT_SEPARATOR.join(
                        chunks[c] for c in row["retrieved_chunk_ids"]
                    ),
                    "resposta": row["answer"],
                    "correta": "",
                    "fiel": "",
                }
            )
            key[chave] = {"run": run_dir.name, "id": row["id"]}
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    return sheet_path


def _grade(value: str, chave: str, column: str) -> bool:
    if value not in ("0", "1"):
        raise ValueError(f"{chave}: {column}={value!r} must be 0 or 1")
    return value == "1"


def _read_sheet(sheet_path: Path) -> list[dict]:
    # Excel pt-BR re-saves CSV with ";", so sniff between the two delimiters.
    text = sheet_path.read_text(encoding="utf-8-sig")
    delimiter = csv.Sniffer().sniff(text.splitlines()[0], delimiters=",;").delimiter
    reader = csv.DictReader(text.splitlines(keepends=True), delimiter=delimiter)
    missing = {"chave", "correta", "fiel"} - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"sheet header lacks columns: {sorted(missing)}")
    return list(reader)


def judge_agreement(
    sheet_path: Path, key_path: Path, verdicts_dir: Path | None = None, split: str = DEV_SPLIT
) -> Path:
    """Accuracy and Cohen's kappa of the judge against the human grades, per criterion.

    verdicts_dir holds <run>.verdicts.jsonl files (tuning rounds); default is each run's own.
    """
    rows = _read_sheet(sheet_path)
    key = json.loads(key_path.read_text(encoding="utf-8"))
    blank = [r["chave"] for r in rows if not r["correta"].strip() or not r["fiel"].strip()]
    if blank:
        raise ValueError(f"ungraded rows (fill correta and fiel): {blank}")
    verdicts: dict[str, dict] = {}
    human = {"correta": [], "fiel_ao_contexto": []}
    judged = {"correta": [], "fiel_ao_contexto": []}
    skipped: list[str] = []
    hashes: set[str | None] = set()
    for r in rows:
        chave = r["chave"]
        c, fiel = (
            _grade(r["correta"].strip(), chave, "correta"),
            _grade(r["fiel"].strip(), chave, "fiel"),
        )
        if chave not in key:
            raise ValueError(f"{chave}: not in {key_path.name}")
        run, item_id = key[chave]["run"], key[chave]["id"]
        if run not in verdicts:
            vpath = (
                verdicts_dir / f"{run}.verdicts.jsonl"
                if verdicts_dir
                else run_eval.RUNS_DIR / run / "verdicts.jsonl"
            )
            lines = vpath.read_text(encoding="utf-8")
            verdicts[run] = {v["id"]: v for v in map(json.loads, lines.splitlines())}
        if item_id not in verdicts[run]:
            raise ValueError(f"{chave}: id {item_id!r} missing from {vpath.name}")
        v = verdicts[run][item_id]
        hashes.add(v.get("prompt_sha256"))
        if not v["valid"]:
            skipped.append(chave)  # reported, not silently dropped
            continue
        for name, human_value in (("correta", c), ("fiel_ao_contexto", fiel)):
            human[name].append(human_value)
            judged[name].append(v["verdict"][name])
    if len(hashes) != 1 or None in hashes:
        raise ValueError(
            f"verdicts must carry one prompt_sha256, found: {sorted(map(str, hashes))}"
        )
    (sha,) = hashes
    report = {
        "judge_prompt_sha256": sha,
        "n_rows": len(rows),
        "skipped_invalid": skipped,
    }
    for name in human:
        h, j = human[name], judged[name]
        report[name] = {
            "n": len(h),
            "accuracy": sum(x == y for x, y in zip(h, j, strict=True)) / len(h) if h else None,
            "kappa": cohen_kappa(h, j) if h else None,
        }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # One file per prompt version so every tuning round stays on record.
    suffix = "" if split == DEV_SPLIT else f"_{split}"
    path = RESULTS_DIR / f"judge_agreement_{sha[:12]}{suffix}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def _cli(args: list[str]) -> Path:
    # usage: judge sheet <run_dir> <run_dir> [split] | judge agreement [verdicts_dir] [--split S]
    if len(args) in (3, 4) and args[0] == "sheet":
        split = args[3] if len(args) == 4 else DEV_SPLIT
        return export_grading_sheet([Path(args[1]), Path(args[2])], out_dir=grading_dir_for(split))
    elif args[:1] == ["agreement"]:
        split = DEV_SPLIT
        if "--split" in args:
            at = args.index("--split")
            split, args = args[at + 1], args[:at] + args[at + 2 :]
        if len(args) not in (1, 2):
            raise SystemExit("usage: judge agreement [verdicts_dir] [--split S]")
        gdir = grading_dir_for(split)
        vdir = Path(args[1]) if len(args) == 2 else None
        return judge_agreement(gdir / "sheet.csv", gdir / "key.json", vdir, split)
    else:
        raise SystemExit(
            "usage: judge sheet <run_dir> <run_dir> [split] | "
            "judge agreement [verdicts_dir] [--split S]"
        )


if __name__ == "__main__":
    print(_cli(sys.argv[1:]))
