# Dev log

## vLLM smoke test (Task 10)

Notebook: `notebooks/run_kaggle.ipynb` (commit 1542188), `defense="full"`, `retrieval="bm25"`.
Environment: Kaggle, Python 3.13 image, GPU T4 x2 with vLLM pinned to GPU 0
(`CUDA_VISIBLE_DEVICES=0`), internet on. Install + model download + load took about 9 minutes.

- vLLM version: 0.32.0
- Model ID: `Qwen/Qwen3-8B-AWQ`, served with `--dtype half --enable-auto-tool-choice
  --tool-call-parser hermes --max-model-len 8192`.
- Latency per question (5 questions, 2 with tools): mean 4.7 s, max 5.9 s.
- Tokens: 9042 prompt / 513 completion in total.
- Tool-call parsing: 2/2. `consultar_pedido` and `abrir_reembolso` arrived parsed in `tool_calls`
  and executed. The refund call was `motivo=item_faltando`, `itens=['Hambúrguer']`, result
  `aberto` (9000 cents).
- Citations: 3/3 policy answers cited a known `[passage_id]` (for example
  `[reembolso_mercado_varejo:3.1]`, `[atrasos:5]`, `[cancelamento:2.2]`) with correct content.

### Problems

- Run 1 failed, fixed in 1542188:
  - Kaggle nests the dataset (not `/kaggle/input/<slug>`); the notebook now searches for
    `pyproject.toml`.
  - `bm25s` imports `jax` at import time; JAX took GPU memory (XLA CUDA out of memory) and vLLM
    died. Fixed with `JAX_PLATFORMS=cpu` and `XLA_PYTHON_CLIENT_PREALLOCATE=false` before any
    import.
- `hermes` parsed Qwen3 tool calls correctly, although the vLLM docs list it only for Qwen2.5/QwQ.
- Model copies the literal placeholder: the order-status answer ends with `[passage_id:5]` even
  though the data came from a tool result. `sources` was correctly filtered to `[]`. Carry to
  prompt tuning: citations only for context passages.
- Refund answer says "valor integral do reembolso" for an item-only refund (misleading) and adds an
  emoji. Carry to prompt tuning (wording and tone).
- The summary cell was not executed under Run All on the first pass; running it manually worked.
  Cause unknown, not blocking.
