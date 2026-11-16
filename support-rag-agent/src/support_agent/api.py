"""FastAPI layer over the support agent: token auth, validation, rate limit, tracing.

Run (the app is built from the environment, so it is a factory, not a module-level `app`):

    uvicorn --factory support_agent.api:build_app_from_env

Environment: LLM_BASE_URL, LLM_MODEL, DB_PATH, TOKENS_PATH (required); RETRIEVAL
(default hybrid_rerank), CACHE_DIR (data/cache), CORPUS_DIR (corpus), TRACE_PATH
(data/traces.jsonl), LOG_CONTENT=1 to log answer text.

config/demo_tokens.json holds obviously-fake demo tokens mapped to generated customer ids.
They are demo fixtures for the local demo, not secrets.
"""

import hmac
import json
import logging
import os
import secrets
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, model_validator

from support_agent import tracing
from support_agent.agent import AgentConfig, run_agent
from support_agent.corpus import chunk_passages, load_corpus
from support_agent.llm import ChatClient, OpenAIChatClient
from support_agent.orders import REFERENCE_NOW, connect, utc_now
from support_agent.retrieval import (
    RetrievalConfig,
    build_retriever,
    cached_embeddings,
    load_models,
)
from support_agent.tools import ToolContext, ToolOutcome

log = logging.getLogger(__name__)

MAX_MESSAGES = 20
MAX_CONTENT_CHARS = 2000
RATE_WINDOW_SECONDS = 60.0
DEFAULT_RATE_LIMIT_PER_MIN = 30
NONCE_BYTES = 8
DENSE_MODEL = "BAAI/bge-m3"
DOC_PATHS = frozenset({"/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"})
BASE_SECURITY_HEADERS = {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "strict-transport-security": "max-age=31536000; includeSubDomains",
}
CSP = "default-src 'none'; frame-ancestors 'none'"


class _Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_CONTENT_CHARS)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: list[_Message] = Field(min_length=1, max_length=MAX_MESSAGES)

    @model_validator(mode="after")
    def _last_is_user(self) -> "ChatRequest":
        if self.messages[-1].role != "user":
            raise ValueError("last message must come from the user")
        return self


class ToolCallInfo(BaseModel):
    name: str
    status: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    tool_calls: list[ToolCallInfo]
    request_id: str


class ApiError(Exception):
    def __init__(self, status: int, code: str, headers: dict[str, str] | None = None) -> None:
        self.status, self.code, self.headers = status, code, headers


def _error(status: int, code: str, headers: dict[str, str] | None = None) -> JSONResponse:
    body = {"erro": code, "request_id": tracing.request_id_var.get()}
    return JSONResponse(body, status, headers)


class RateLimiter:
    """Sliding-window limiter per token."""

    # ponytail: in-memory per-process limiter; move to Redis if more than one instance
    def __init__(self, limit: int, window: float = RATE_WINDOW_SECONDS) -> None:
        self._limit, self._window = limit, window
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, token: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(token, deque())
            while hits and now - hits[0] >= self._window:
                hits.popleft()
            if len(hits) >= self._limit:
                return False
            hits.append(now)
            return True


class _SecurityMiddleware:
    """Pure ASGI: sets request_id for the whole request (incl. the sync endpoint's thread),
    adds security headers and turns unexpected errors into a generic 500."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid4().hex
        reset = tracing.request_id_var.set(request_id)
        started = False

        async def send_with_headers(message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                headers = {**BASE_SECURITY_HEADERS, "x-request-id": request_id}
                if scope["path"] not in DOC_PATHS:
                    headers["content-security-policy"] = CSP
                message["headers"] = [
                    *message.get("headers", []),
                    *((k.encode(), v.encode()) for k, v in headers.items()),
                ]
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        except Exception:
            log.exception("unhandled error")  # full detail only in logs
            if started:
                raise
            await _error(500, "erro_interno")(scope, receive, send_with_headers)
        finally:
            tracing.request_id_var.reset(reset)


def _tool_status(outcome: ToolOutcome) -> str:
    result = outcome.result
    return str(result.get("status") or result.get("erro") or "ok")


def _append_trace(path: Path, trace: dict, lock: threading.Lock) -> None:
    line = json.dumps(trace, ensure_ascii=False)
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


_DEFAULT_CONFIG = AgentConfig()  # never mutated: create_app works on a copy


def create_app(
    *,
    llm: ChatClient,
    retrieve: Callable,
    conn_factory: Callable[[], sqlite3.Connection],
    tokens: dict[str, str],
    config: AgentConfig = _DEFAULT_CONFIG,
    now: Callable[[], datetime] = utc_now,
    rate_limit_per_min: int = DEFAULT_RATE_LIMIT_PER_MIN,
) -> FastAPI:
    config = replace(config, defense="full")  # no request field can weaken the defenses
    limiter = RateLimiter(rate_limit_per_min)
    trace_path = Path(os.getenv("TRACE_PATH", "data/traces.jsonl"))
    trace_lock = threading.Lock()
    app = FastAPI(title="Support agent")
    app.add_middleware(_SecurityMiddleware)

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return _error(exc.status, exc.code, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, __: RequestValidationError) -> JSONResponse:
        return _error(422, "requisicao_invalida")  # the default body would echo customer text

    # Declared as a security scheme so Swagger's Authorize button can send the token
    # (Swagger drops plain header parameters named Authorization). auto_error=False
    # keeps our generic 401 body; non-Bearer schemes arrive here as None.
    bearer = HTTPBearer(auto_error=False)

    def authenticate(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)] = None,
    ) -> tuple[str, str]:
        supplied = credentials.credentials if credentials else ""
        match = next(
            (t for t in tokens if hmac.compare_digest(t.encode(), supplied.encode())), None
        )
        if match is None:
            raise ApiError(401, "nao_autorizado", {"WWW-Authenticate": "Bearer"})
        if not limiter.allow(match):
            raise ApiError(429, "limite_excedido")
        return match, tokens[match]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/chat")
    def chat(body: ChatRequest, auth: Annotated[tuple[str, str], Depends(authenticate)]):
        began = time.perf_counter()
        messages = [m.model_dump() for m in body.messages]
        conn = conn_factory()
        try:
            ctx = ToolContext(conn=conn, customer_id=auth[1], now=now(), guard=True)
            result = run_agent(
                messages,
                ctx,
                config,
                retrieve,
                llm,
                canary=secrets.token_hex(NONCE_BYTES),
                nonce=secrets.token_hex(NONCE_BYTES),
            )
        finally:
            conn.close()
        _append_trace(trace_path, result.trace, trace_lock)
        extra = {
            "latency_ms": round((time.perf_counter() - began) * 1000),
            "sources": result.sources,
            "tool_names": [o.name for o in result.tool_outcomes],
        }
        if tracing.LOG_CONTENT:
            extra["answer"] = result.answer
        log.info("chat_completed", extra=extra)
        return ChatResponse(
            answer=result.answer,
            sources=result.sources,
            tool_calls=[
                ToolCallInfo(name=o.name, status=_tool_status(o)) for o in result.tool_outcomes
            ],
            request_id=tracing.request_id_var.get(),
        )

    return app


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def build_app_from_env() -> FastAPI:
    llm = OpenAIChatClient(_require_env("LLM_BASE_URL"), _require_env("LLM_MODEL"))
    db_path = _require_env("DB_PATH")
    tokens = json.loads(Path(_require_env("TOKENS_PATH")).read_text(encoding="utf-8"))
    tracing.setup_logging()
    retrieval: RetrievalConfig = os.getenv("RETRIEVAL", "hybrid_rerank")  # type: ignore[assignment]
    chunks = chunk_passages(load_corpus(Path(os.getenv("CORPUS_DIR", "corpus"))))
    embed = rerank = embeddings = None
    if retrieval != "bm25":
        embed, rerank = load_models(dense_name=DENSE_MODEL)
        cache_dir = Path(os.getenv("CACHE_DIR", "data/cache"))
        embeddings = cached_embeddings(chunks, embed, DENSE_MODEL, cache_dir)
    retrieve = build_retriever(retrieval, chunks, embed, rerank, embeddings=embeddings)
    return create_app(
        llm=llm,
        retrieve=retrieve,
        conn_factory=lambda: connect(db_path),
        tokens=tokens,
        config=AgentConfig(retrieval=retrieval),
        now=lambda: REFERENCE_NOW,  # synthetic orders are anchored to REFERENCE_NOW, not the clock
    )
