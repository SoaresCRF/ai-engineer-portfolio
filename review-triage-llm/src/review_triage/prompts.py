"""Prompt and output parsing shared by every LLM approach (teacher zero-shot and student).

The same parser is used for all approaches so JSON-validity numbers are comparable.
"""

import hashlib
import json
import re
import unicodedata

from pydantic import ValidationError

from review_triage.schema import Triage

SYSTEM_PROMPT = """\
Você classifica avaliações de clientes de um e-commerce brasileiro.
Responda APENAS com um JSON de uma linha, sem texto extra, no formato:
{"categoria": "...", "sentimento": "...", "urgencia": "..."}

categoria (o problema principal):
- atraso_entrega: chegou, mas depois do prazo, ou demorou demais
- nao_recebido: ainda não chegou ou nada chegou (inclui perguntar quando vai chegar)
- produto_errado_ou_incompleto: enviaram OUTRO item, modelo, cor, tamanho ou voltagem; faltou item ou peça; recebeu só parte do pedido
- defeito_ou_qualidade: o item certo, mas quebrado ou com parte quebrada, não funciona, não encaixa, qualidade ruim, pior ou diferente do anúncio/foto, falsificado ou recondicionado
- atendimento_vendedor: loja não responde, cancelamento, estorno, nota fiscal, troca difícil
- elogio_sem_problema: elogio ou comentário sem nenhum problema
- outro: arrependimento, gosto pessoal, frete caro, retirada nos Correios, texto vago

Desempates de categoria:
1. Vários problemas: o mais grave, nesta ordem: nao_recebido > produto_errado_ou_incompleto > defeito_ou_qualidade > atendimento_vendedor > atraso_entrega.
2. Elogio com ressalva ("ótimo, mas demorou"): use a categoria da ressalva.

sentimento (tom geral, não o problema):
- positivo: satisfeito no geral, mesmo com ressalva leve
- neutro: informativo ou calmo, sem emoção clara; relata espera ou problema sem reclamar ("estou no aguardo", "poderia chegar antes")
- negativo: insatisfação explícita (reclama, critica, "péssimo", "não recomendo")

urgencia (precisa de ação da empresa?):
- alta: não recebeu; pede reembolso, estorno, troca ou devolução; cobra resposta; menciona Procon ou justiça
- media: tem problema, mas não pede ação explícita
- baixa: nada a resolver (elogio, comentário neutro, gosto pessoal)"""

# Lightly edited Olist reviews that are NOT in the gold set (enforced by tests/test_prompts.py).
FEW_SHOT: tuple[tuple[str, Triage], ...] = tuple(
    (text, Triage(categoria=c, sentimento=s, urgencia=u))
    for text, c, s, u in [
        (
            "Já passou 3 dias e o produto não foi entregue.",
            "nao_recebido",
            "negativo",
            "alta",
        ),
        (
            "Gostei da mercadoria, chegou rápido, mas não veio na voltagem certa",
            "produto_errado_ou_incompleto",
            "neutro",
            "media",
        ),
        (
            "O produto é maravilhoso, recebi tudo direitinho. Só achei demorar um pouco a entrega",
            "atraso_entrega",
            "positivo",
            "baixa",
        ),
        (
            "Nao funcionou, parece queimado",
            "defeito_ou_qualidade",
            "negativo",
            "media",
        ),
        (
            "Excelente, recebi um dia antes do prazo e muito fácil de rastrear a entrega.",
            "elogio_sem_problema",
            "positivo",
            "baixa",
        ),
        (
            "Só não gostei das fragrâncias, bem enjoativas e doces.",
            "outro",
            "negativo",
            "baixa",
        ),
    ]
)

_JSON_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def to_json(triage: Triage) -> str:
    return json.dumps(triage.model_dump(mode="json"), ensure_ascii=False)


def build_messages(text: str, few_shot: bool = True) -> list[dict[str, str]]:
    """Chat messages for one review. The fine-tuned student is trained without few-shot."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if few_shot:
        for example_text, label in FEW_SHOT:
            messages.append({"role": "user", "content": example_text})
            messages.append({"role": "assistant", "content": to_json(label)})
    messages.append({"role": "user", "content": text})
    return messages


def prompt_version() -> str:
    """Short hash identifying the exact prompt, stored with every prediction run."""
    payload = SYSTEM_PROMPT + "".join(t + to_json(lbl) for t, lbl in FEW_SHOT)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def _strip_accents(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def parse_output(raw: str) -> Triage | None:
    """First JSON object in the model output, validated; None when unusable."""
    match = _JSON_OBJECT.search(_THINK_BLOCK.sub("", raw))
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    cleaned = {
        k: _strip_accents(v).strip().lower() if isinstance(v, str) else v for k, v in data.items()
    }
    try:
        return Triage(**cleaned)
    except (ValidationError, TypeError):
        return None
