# Dev log

## vLLM smoke test

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

## Judge validation

Judge: `stelterlab/phi-4-AWQ` (vLLM 0.32.0, T4), structured output through `response_format`.
Gate: Cohen's kappa >= 0.6 per criterion against 60 blind human grades (30 rows per E1 run).

### Prompt v1 (sha256 prefix `40010de1e040`)

- `results/judge_agreement_40010de1e040.json`: `correta` accuracy 0.833, kappa 0.0; `fiel`
  accuracy 0.783, kappa 0.0.
- The judge answered true/true on all 60 rows; the human gave 0 to 14 rows (`correta` 0 on 10,
  `fiel` 0 on 13). Accuracy is just the human base rate, kappa is zero.
- A free-form (unconstrained) run of the same prompt never answered false either, so the cause is
  a too lenient rubric, not only the output format. The schema also put the booleans before
  `justificativa`, so the model committed before any reasoning.

### Prompt v2 (sha256 `9d90a091203ada2e3ac80b335eda290ecc922877331805966a2737a8cbc065e0`)

- Reason first: `justificativa` is the first schema property (vLLM follows property order) and the
  prompt demands a claim-by-claim check before the booleans.
- Strictness: one wrong or unsupported fact makes the flag false; a missing key condition or number
  makes `correta` false; a claim absent from the excerpts makes `fiel_ao_contexto` false even if
  true in real life; when in doubt, false.
- Three short synthetic examples (fictional delivery platform, not from `data/eval` or
  `data/grading`) showing a false verdict per criterion.
- Rubric notes: "cannot answer OR refer to human support" (either counts); a refusal of
  a legitimate benign request is not `correta`.
- Tuning loop: `judge_run(..., ids, out_path)` judges only the graded rows into
  `judge_tuning/<sha12>/<run>.verdicts.jsonl` (notebook section "Judge tuning"), then
  `python -m support_agent.judge agreement <dir>` scores them locally. The bundle ships
  `data/grading/key.json` only, never the human grades.
- Metrics (graded subset, 60 rows): `correta` accuracy 0.283, kappa 0.051; `fiel` accuracy 0.383,
  kappa 0.105. The judge said true on only 7/60 (`correta`) and 10/60 (`fiel`): strictness
  overshot into false negatives.
- Finding: the justifications invent contradictions (they claim the answer contradicts the
  reference when it agrees). Phi-4 AWQ cannot verify pt-BR claims reliably; decision: switch the
  judge model rather than tune the prompt further.

### Prompt v3 (sha256 `aa3d2d0e79694ed9a89c94b0122b49401a24bb56f9b5f046019b76313c9a7cd9`)

- Judge model: `casperhansen/mistral-small-24b-instruct-2501-awq` (Mistral Small 24B, 4-bit AWQ,
  public, non-Qwen family). It does not fit one T4 with KV cache, so the judge server runs with
  `--tensor-parallel-size 2` and `CUDA_VISIBLE_DEVICES=0,1` (accelerator must be GPU T4 x2);
  `--dtype half` and `--max-model-len 8192` unchanged.
- Prompt: balanced instead of strict. Still reason-first (`justificativa` first). Claim-by-claim
  procedure (agrees / contradicts / not covered, supported by excerpts), then the flags.
  `correta` is false only for a real contradiction (which must be quoted from both texts), a
  missing decisive condition/number, or an invented fact that changes the outcome; paraphrase and
  harmless extra detail are fine. `fiel_ao_contexto` is false only for a factual claim absent from
  the excerpts (courtesy, offers of help and routing to support are not claims). "Na dúvida,
  falso" removed. v2 rules for "not in the policies" and refused benign requests kept.
- Examples: three synthetic ones (fictional delivery platform): all-true paraphrase, quoted
  contradiction (`correta` false), unsupported claim with `correta` true and `fiel` false.
- Metrics (graded subset, 60 rows, 0 invalid): `correta` accuracy 0.883, kappa 0.475; `fiel`
  accuracy 0.700, kappa 0.205. Confusion (h = human, j = judge, 1 = true):
  - `correta`: h1j1 49, h1j0 1, h0j1 6, h0j0 4.
  - `fiel`: h1j1 36, h1j0 11, h0j1 7, h0j0 6.

### Decision

Tuning stopped after three prompt versions (user decision 2026-11-15). Neither criterion reaches
the kappa >= 0.6 gate, so the judge is not validated. The 60 blind human grades are the primary
measure of answer quality; the LLM judge is secondary and always reported together with its kappa.
Only 10 of 60 rows have `correta` = 0, so the `correta` kappa is unstable.

The Phi-4 v1 full-run verdicts were moved to `results/judge_attempts/phi4_v1_full/`; they are not
a valid measure (the judge answered true on every row).

## E1 dev results - human grades

Blind human grades on the dev answers, 30 rows per config.

| Retrieval     | n  | `correta`     | `fiel_ao_contexto` |
|---------------|----|---------------|--------------------|
| dense         | 30 | 25/30 (0.833) | 22/30 (0.733)      |
| hybrid_rerank | 30 | 25/30 (0.833) | 25/30 (0.833)      |

n is small (30 per config): the `fiel` difference is 3 rows and is not statistically meaningful.

## Final config

`src/support_agent/final_config.py`: `FINAL_CONFIG = AgentConfig(retrieval="hybrid_rerank",
top_k=5, defense="full")`; `max_steps` keeps the `AgentConfig` default.

- Answer quality is measured by the human grades. The judge is not validated, so `top_k` and the
  prompt wording are not tuned on answer quality, and new human grades per variant are out of
  scope. Prompts are unchanged.
- `retrieval = "hybrid_rerank"`: best on every retrieval metric (dev, n=60 answerable,
  `results/retrieval_dev.json`): recall@3 0.718, recall@5 0.843, recall@10 0.882, MRR 0.741.
  For comparison dense has recall@5 0.711 and MRR 0.662, bm25 recall@5 0.435, hybrid recall@5
  0.678. Human `fiel`: 22/30 dense vs 25/30 rerank.
- `top_k = 5` (plan default): k=3 loses 12.5 points of recall vs k=5; k=8 sits between @5 and @10
  (at most +3.9 points) at the cost of longer context in an 8192-token window.
- `defense = "full"`: the layered defense is the system under test in E2; its dev results are
  reported in the final results, not tuned here.
