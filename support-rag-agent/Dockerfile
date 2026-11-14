# Single stage: uv's venv and the CPU torch wheels are needed at runtime anyway.
FROM python:3.11.18-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    DB_PATH=/app/data/orders.db \
    TOKENS_PATH=/app/config/demo_tokens.json \
    CORPUS_DIR=/app/corpus \
    CACHE_DIR=/app/data/cache \
    TRACE_PATH=/app/data/traces.jsonl \
    HF_HOME=/home/app/.cache/huggingface

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/data /home/app/.cache/huggingface \
    && chown -R app:app /app /home/app/.cache
WORKDIR /app
USER app

# Dependencies first so this layer is cached until the lockfile changes.
COPY --chown=app:app pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY --chown=app:app src ./src
COPY --chown=app:app corpus ./corpus
COPY --chown=app:app config ./config
RUN uv sync --locked --no-dev \
    && python -m support_agent.orders generate "$DB_PATH"

# Model weights are not baked in: mount a volume here to persist the download.
VOLUME /home/app/.cache/huggingface
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "--factory", "support_agent.api:build_app_from_env", "--host", "0.0.0.0", "--port", "8000"]
