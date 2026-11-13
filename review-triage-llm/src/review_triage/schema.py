"""Output contract shared by every approach (baseline, zero-shot LLM, fine-tuned LLM).

Keys and label values are Portuguese on purpose: they are the product's data contract
for PT-BR reviews, not code identifiers. See docs/labeling_guide.md for definitions.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Category(StrEnum):
    LATE_DELIVERY = "atraso_entrega"
    NOT_RECEIVED = "nao_recebido"
    WRONG_OR_INCOMPLETE = "produto_errado_ou_incompleto"
    DEFECT_OR_QUALITY = "defeito_ou_qualidade"
    SELLER_SERVICE = "atendimento_vendedor"
    PRAISE_NO_ISSUE = "elogio_sem_problema"
    OTHER = "outro"


class Sentiment(StrEnum):
    POSITIVE = "positivo"
    NEUTRAL = "neutro"
    NEGATIVE = "negativo"


class Urgency(StrEnum):
    LOW = "baixa"
    MEDIUM = "media"
    HIGH = "alta"


class Triage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    categoria: Category
    sentimento: Sentiment
    urgencia: Urgency


LABEL_FIELDS: dict[str, type[StrEnum]] = {
    "categoria": Category,
    "sentimento": Sentiment,
    "urgencia": Urgency,
}
