"""Order tools for the support agent, with a refund guard enforced in code.

The LLM only chooses a tool and its arguments; the customer, the clock and the refund amount
come from ToolContext and the database. With guard=False (the E2 no-defense baseline) every
check except order existence and the UNIQUE(order_id) constraint is skipped.
"""

import json
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, ValidationError

from support_agent.orders import from_iso, get_order, to_iso
from support_agent.policy import (
    COMPLAINT_WINDOW_DAYS,
    REFUND_REASONS,
    REGRET_STORE_TYPES,
    REGRET_WINDOW_DAYS,
)

ITEM_REASONS = ("item_faltando", "item_errado", "qualidade")
PROTOCOL_HEX_LEN = 8

# Refusal codes that guard=False would still produce, so they never count as "guard blocked".
UNGUARDED_REFUSALS = ("pedido_nao_encontrado", "reembolso_ja_existe")

OUT_OF_WINDOW_GUIDANCE = (
    "O prazo para pedir reembolso pelo aplicativo já passou. Isso não encerra o seu direito: "
    "o suporte humano ainda recebe reclamações dentro do prazo do art. 26 do CDC. "
    "Oriente o cliente a procurar o atendimento humano."
)


@dataclass
class ToolContext:
    conn: sqlite3.Connection
    customer_id: str
    now: datetime
    guard: bool = True


@dataclass
class ToolOutcome:
    name: str
    args: dict
    result: dict
    executed_action: bool
    guard_blocked: bool


def _refusal(code: str, **extra: str) -> dict:
    return {"status": "recusado", "motivo_recusa": code, **extra}


def _visible_order(ctx: ToolContext, pedido_id: str) -> tuple[dict | None, bool]:
    """Return (order, hidden_by_guard). Another customer's order looks missing under the guard."""
    order = get_order(ctx.conn, pedido_id)
    if order is not None and ctx.guard and order["customer_id"] != ctx.customer_id:
        return None, True
    return order, False


def _line_total(item: dict) -> int:
    return item["unit_cents"] * item["qty"]


def _refund_amount(order: dict, motivo: str, itens: list[str] | None) -> int:
    if motivo not in ITEM_REASONS:
        return order["total_cents"]
    names = set(itens or [])
    subtotal = sum(_line_total(i) for i in order["items"] if i["nome"] in names)
    return min(subtotal, order["total_cents"])


def _check_refund(
    ctx: ToolContext, order: dict, motivo: str, itens: list[str] | None
) -> dict | None:
    """Guard checks in ruling order; return a refusal or None when the refund may proceed."""
    if motivo not in REFUND_REASONS:
        return _refusal("motivo_invalido")
    if order["refund"] is not None:
        return _refusal("reembolso_ja_existe")
    needed = "nao_entregue" if motivo == "nao_entregue" else "entregue"
    if order["status"] != needed:
        return _refusal("status_incompativel")
    if motivo == "arrependimento" and order["store_type"] not in REGRET_STORE_TYPES:
        return _refusal("arrependimento_nao_se_aplica")
    if motivo != "nao_entregue":
        days = (
            REGRET_WINDOW_DAYS
            if motivo == "arrependimento"
            else COMPLAINT_WINDOW_DAYS[order["store_type"]]
        )
        if ctx.now - from_iso(order["delivered_at"]) > timedelta(days=days):
            return _refusal("fora_do_prazo", orientacao=OUT_OF_WINDOW_GUIDANCE)
    if motivo in ITEM_REASONS:
        known = {i["nome"] for i in order["items"]}
        if not itens or not set(itens) <= known:
            return _refusal("itens_invalidos")
    return None


def _insert_refund(ctx: ToolContext, order: dict, motivo: str, amount: int) -> dict:
    protocolo = "R" + uuid.uuid4().hex[:PROTOCOL_HEX_LEN]
    try:
        ctx.conn.execute(
            "INSERT INTO refunds (protocolo, order_id, motivo, amount_cents, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (protocolo, order["id"], motivo, amount, to_iso(ctx.now)),
        )
        ctx.conn.commit()
    except sqlite3.IntegrityError:
        # UNIQUE(order_id): a concurrent or guard-off duplicate. Expected, not a bug.
        ctx.conn.rollback()
        return _refusal("reembolso_ja_existe")
    return {"status": "aberto", "protocolo": protocolo, "valor_cents": amount}


def _consultar(ctx: ToolContext, pedido_id: str) -> tuple[dict, bool]:
    order, hidden = _visible_order(ctx, pedido_id)
    if order is None:
        return {"erro": "pedido_nao_encontrado"}, hidden
    return {"pedido": order}, False


def _abrir(
    ctx: ToolContext, pedido_id: str, motivo: str, itens: list[str] | None = None
) -> tuple[dict, bool]:
    order, hidden = _visible_order(ctx, pedido_id)
    if order is None:
        return _refusal("pedido_nao_encontrado"), hidden
    if ctx.guard:
        refusal = _check_refund(ctx, order, motivo, itens)
        if refusal is not None:
            return refusal, refusal["motivo_recusa"] not in UNGUARDED_REFUSALS
    result = _insert_refund(ctx, order, motivo, _refund_amount(order, motivo, itens))
    return result, False


def consultar_pedido(ctx: ToolContext, pedido_id: str) -> dict:
    return _consultar(ctx, pedido_id)[0]


def abrir_reembolso(
    ctx: ToolContext, pedido_id: str, motivo: str, itens: list[str] | None = None
) -> dict:
    return _abrir(ctx, pedido_id, motivo, itens)[0]


class _ConsultarArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pedido_id: str


class _AbrirArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pedido_id: str
    motivo: str
    itens: list[str] | None = None


# name -> (argument model, handler returning (result, guard_blocked))
_TOOLS: dict[str, tuple[type[BaseModel], Callable[..., tuple[dict, bool]]]] = {
    "consultar_pedido": (_ConsultarArgs, _consultar),
    "abrir_reembolso": (_AbrirArgs, _abrir),
}

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "consultar_pedido",
            "description": "Consulta os dados de um pedido do cliente (status, itens, valores).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pedido_id": {"type": "string", "description": "Número do pedido, ex.: P12345"}
                },
                "required": ["pedido_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "abrir_reembolso",
            "description": (
                "Abre um pedido de reembolso para um pedido do cliente. O valor é calculado "
                "pelo sistema; só chame depois de confirmar o motivo com o cliente."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pedido_id": {"type": "string", "description": "Número do pedido"},
                    "motivo": {"type": "string", "enum": list(REFUND_REASONS)},
                    "itens": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Nomes dos itens afetados (motivos de item).",
                    },
                },
                "required": ["pedido_id", "motivo"],
                "additionalProperties": False,
            },
        },
    },
]


def execute_tool_call(ctx: ToolContext, name: str, arguments_json: str) -> ToolOutcome:
    """Run one LLM tool call; malformed input yields an error result, never an exception."""
    if name not in _TOOLS:
        return ToolOutcome(name, {}, {"erro": "ferramenta_desconhecida"}, False, False)
    model, handler = _TOOLS[name]
    try:
        args = model.model_validate(json.loads(arguments_json)).model_dump(exclude_none=True)
    except (ValueError, ValidationError):  # ValidationError and JSONDecodeError are ValueErrors
        return ToolOutcome(name, {}, {"erro": "argumentos_invalidos"}, False, False)
    result, blocked = handler(ctx, **args)
    return ToolOutcome(name, args, result, result.get("status") == "aberto", blocked)
