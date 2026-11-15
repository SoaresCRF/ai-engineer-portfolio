"""Run the agent over the frozen eval set; one fresh in-memory DB per item."""

import hashlib
import importlib.metadata
import json
import platform
import secrets
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from support_agent import eval_set
from support_agent.agent import AgentConfig, run_agent
from support_agent.corpus import Chunk
from support_agent.llm import ChatClient
from support_agent.orders import REFERENCE_NOW, connect, generate
from support_agent.tools import ToolContext

RUNS_DIR = Path(__file__).resolve().parents[2] / "data" / "runs"
POLICY_CUSTOMER = "C001"  # policy questions have no customer; any existing one works
TRACKED_PACKAGES = ("numpy", "openai", "vllm")
TOKEN_BYTES = 8


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _conversation(item) -> tuple[str, list[dict]]:
    if isinstance(item, eval_set.PolicyQuestion):
        return POLICY_CUSTOMER, [{"role": "user", "content": item.question}]
    return item.customer_id, [m.model_dump() for m in item.messages]


def _tokens(trace: dict) -> dict[str, int]:
    spans = [s for s in trace["spans"] if s["name"] == "llm_call"]
    return {
        "prompt": sum(s["prompt_tokens"] for s in spans),
        "completion": sum(s["completion_tokens"] for s in spans),
    }


def _run_item(item, kind, config, llm, retrieve, injected, canary) -> dict:
    customer, messages = _conversation(item)
    conn = connect(":memory:")
    # Fixed clock: results must not depend on the day the run happens.
    generate(conn, now=REFERENCE_NOW, extra_orders=injected)
    ctx = ToolContext(conn=conn, customer_id=customer, now=REFERENCE_NOW, guard=False)
    chunk_ids: list[str] = []

    def recording_retrieve(query: str, k: int) -> list[Chunk]:
        chunks = retrieve(query, k)
        chunk_ids.extend(c.chunk_id for c in chunks)
        return chunks

    nonce = secrets.token_hex(TOKEN_BYTES)
    start = time.perf_counter()
    try:
        res = run_agent(messages, ctx, config, recording_retrieve, llm, canary, nonce)
    finally:
        conn.close()
    return {
        "id": item.id,
        "kind": kind,
        "answer": res.answer,
        "sources": res.sources,
        "tool_outcomes": [asdict(o) for o in res.tool_outcomes],
        "output_blocked": res.output_blocked,
        "retrieved_chunk_ids": chunk_ids,
        "latency_ms": (time.perf_counter() - start) * 1000,
        "tokens": _tokens(res.trace),
        "trace": res.trace,
    }


def _initial_meta(config: AgentConfig, split: str, llm: ChatClient) -> dict:
    manifest = (eval_set.EVAL_DIR / eval_set.MANIFEST).read_bytes()
    return {
        "config": asdict(config),
        "model": getattr(llm, "_model", type(llm).__name__),
        "split": split,
        "versions": {"python": platform.python_version()}
        | {p: _version(p) for p in TRACKED_PACKAGES},
        "date": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "canary": {},
    }


def run(
    run_name: str,
    config: AgentConfig,
    split: Literal["dev", "test"],
    kinds: list[str],
    llm: ChatClient,
    retrieve: Callable[[str, int], list[Chunk]],
) -> Path:
    eval_set.verify_frozen()
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True)  # FileExistsError: a run is evidence, never overwritten
    meta = _initial_meta(config, split, llm)
    meta_path = run_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    injected = eval_set.load_injected_orders()
    # No try/except per item: a failure aborts the run, partial scores would bias results.
    with (run_dir / "outputs.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        for kind in kinds:
            for item in eval_set.load(kind):
                if item.split != split:
                    continue
                canary = f"CANARY-{secrets.token_hex(TOKEN_BYTES)}"
                meta["canary"][item.id] = canary
                row = _run_item(item, kind, config, llm, retrieve, injected, canary)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return run_dir
