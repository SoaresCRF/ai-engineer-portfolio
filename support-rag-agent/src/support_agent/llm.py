"""LLM client boundary: a Protocol for injection and an OpenAI-compatible implementation."""

import re
from dataclasses import dataclass, field
from typing import Protocol

from openai import OpenAI

QWEN3_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class AssistantTurn:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ChatClient(Protocol):
    def chat(self, messages: list[dict], tools: list[dict]) -> AssistantTurn: ...


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


class OpenAIChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "none",
        extra_body: dict | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self._client = client or OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._extra_body = QWEN3_EXTRA_BODY if extra_body is None else extra_body

    def chat(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
            temperature=0,
            extra_body=self._extra_body,
        )
        msg = resp.choices[0].message
        calls = [
            ToolCall(c.id, c.function.name, c.function.arguments) for c in (msg.tool_calls or [])
        ]
        usage = resp.usage
        return AssistantTurn(
            content=strip_think(msg.content or ""),
            tool_calls=calls,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
