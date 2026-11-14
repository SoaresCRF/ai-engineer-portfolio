"""Tool-calling agent loop with layered prompt-injection defenses (D1-D4)."""

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, replace

from support_agent import tracing
from support_agent.corpus import Chunk
from support_agent.llm import AssistantTurn, ChatClient
from support_agent.prompts import (
    BLOCKED_ANSWER,
    FALLBACK_ANSWER,
    Defense,
    build_system_prompt,
    format_context,
    wrap_untrusted,
)
from support_agent.retrieval import RetrievalConfig
from support_agent.tools import TOOL_SCHEMAS, ToolContext, ToolOutcome, execute_tool_call

__all__ = ["AgentConfig", "AgentResult", "Defense", "run_agent"]

log = logging.getLogger(__name__)
_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")


@dataclass
class AgentConfig:
    retrieval: RetrievalConfig = "hybrid_rerank"
    top_k: int = 5
    defense: Defense = "full"
    max_steps: int = 4  # max number of LLM calls


@dataclass
class AgentResult:
    answer: str
    sources: list[str]
    tool_outcomes: list[ToolOutcome]
    output_blocked: bool
    trace: dict


def _assistant_message(turn: AssistantTurn) -> dict:
    return {
        "role": "assistant",
        "content": turn.content,
        "tool_calls": [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": c.arguments},
            }
            for c in turn.tool_calls
        ],
    }


def _execute_turn_tools(
    turn: AssistantTurn, ctx: ToolContext, config: AgentConfig, nonce: str, trace: tracing.Trace
) -> tuple[list[dict], list[ToolOutcome]]:
    messages, outcomes = [], []
    for call in turn.tool_calls:
        with trace.span("tool_call", tool=call.name) as span:
            outcome = execute_tool_call(ctx, call.name, call.arguments)
            span["guard_blocked"] = outcome.guard_blocked
        content = json.dumps(outcome.result, ensure_ascii=False)
        if config.defense != "none":
            content = wrap_untrusted(content, nonce, "ferramenta")
        messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
        outcomes.append(outcome)
    return messages, outcomes


def _finalize(
    answer: str, chunks: list[Chunk], defense: Defense, canary: str
) -> tuple[str, list[str], bool]:
    if defense != "none" and canary in answer:
        return BLOCKED_ANSWER, [], True
    known = {c.passage_id for c in chunks}
    cited = _CITATION_RE.findall(answer)
    return answer, list(dict.fromkeys(p for p in cited if p in known)), False


def run_agent(
    messages: list[dict],
    ctx: ToolContext,
    config: AgentConfig,
    retrieve: Callable[[str, int], list[Chunk]],
    llm: ChatClient,
    canary: str,
    nonce: str,
) -> AgentResult:
    ctx = replace(ctx, guard=config.defense == "full")  # never mutate the caller's context
    trace = tracing.Trace()
    query = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    with trace.span("retrieve", top_k=config.top_k):
        chunks = retrieve(query, config.top_k)
    system = build_system_prompt(config.defense, canary)
    context = format_context(chunks, config.defense, nonce)
    convo = [{"role": "system", "content": f"{system}\n\n{context}"}, *messages]
    outcomes: list[ToolOutcome] = []
    for step in range(config.max_steps):
        with trace.span("llm_call") as span:
            turn = llm.chat(convo, TOOL_SCHEMAS)
            span["prompt_tokens"] = turn.prompt_tokens
            span["completion_tokens"] = turn.completion_tokens
        if not turn.tool_calls:
            answer, sources, blocked = _finalize(turn.content, chunks, config.defense, canary)
            return AgentResult(answer, sources, outcomes, blocked, trace.to_dict())
        if step == config.max_steps - 1:
            break  # out of LLM calls: this turn's tool calls are not executed
        convo.append(_assistant_message(turn))
        tool_messages, new = _execute_turn_tools(turn, ctx, config, nonce, trace)
        convo.extend(tool_messages)
        outcomes.extend(new)
    log.warning("max_steps reached", extra={"max_steps": config.max_steps})
    return AgentResult(FALLBACK_ANSWER, [], outcomes, False, trace.to_dict())
