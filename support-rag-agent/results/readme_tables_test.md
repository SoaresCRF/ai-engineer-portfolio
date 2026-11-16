### E1 - Retrieval

| Retrieval | Questions | n | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|---|---|
| dense | all | 100 | 0.38 | 0.67 | 0.74 | 0.84 | 0.69 |
| dense | user | 19 | 0.52 | 0.64 | 0.71 | 0.80 | 0.87 |
| rerank | all | 100 | 0.46 | 0.69 | 0.78 | 0.85 | 0.75 |
| rerank | user | 19 | 0.51 | 0.68 | 0.75 | 0.79 | 0.87 |

### E1 - Answer quality

Human grades (primary):

| Config | n | Correct | Faithful |
|---|---|---|---|
| dense | 30 | 0.53 | 0.50 |
| rerank | 30 | 0.77 | 0.63 |

LLM judge casperhansen/mistral-small-24b-instruct-2501-awq (secondary, not validated; kappa correta 0.48 (dev); kappa correta 0.36 (test); kappa fiel_ao_contexto 0.20 (dev); kappa fiel_ao_contexto 0.44 (test)):

| Config | n | Invalid | Correct | Faithful | Correct refusal (unanswerable) |
|---|---|---|---|---|---|
| dense | 100 | 0 | 0.89 | 0.78 | 0.42 |
| rerank | 100 | 0 | 0.87 | 0.78 | 0.50 |

Rerank minus dense (paired bootstrap, 95% CI); LLM judge casperhansen/mistral-small-24b-instruct-2501-awq (secondary, not validated; kappa correta 0.48 (dev); kappa correta 0.36 (test); kappa fiel_ao_contexto 0.20 (dev); kappa fiel_ao_contexto 0.44 (test)):

| Metric | Diff | 95% CI | Pairs |
|---|---|---|---|
| judge_correct | -0.02 | [-0.09, 0.04] | 100 |
| judge_faithful | 0.00 | [-0.10, 0.10] | 100 |

### E2 - Attack success rate by defense

| Defense | Scope | n | ASR attempted | ASR executed |
|---|---|---|---|---|
| none | total | 40 | 0.30 | 0.30 |
| none | vector direto | 12 | 0.67 | 0.67 |
| none | vector nome_item | 14 | 0.21 | 0.21 |
| none | vector observacao | 14 | 0.07 | 0.07 |
| prompt | total | 40 | 0.30 | 0.30 |
| prompt | vector direto | 12 | 0.58 | 0.58 |
| prompt | vector nome_item | 14 | 0.21 | 0.21 |
| prompt | vector observacao | 14 | 0.14 | 0.14 |
| full | total | 40 | 0.30 | 0.23 |
| full | vector direto | 12 | 0.58 | 0.33 |
| full | vector nome_item | 14 | 0.21 | 0.21 |
| full | vector observacao | 14 | 0.14 | 0.14 |

### E2 - Benign over-refusal

LLM judge casperhansen/mistral-small-24b-instruct-2501-awq (secondary, not validated; recusou: judge only; recusou was never human-graded):

| Defense | n | Invalid | Judge over-refusal |
|---|---|---|---|
| none | 12 | 0 | 0.00 |
| prompt | 12 | 0 | 0.00 |
| full | 12 | 0 | 0.00 |

### E2 - Tool accuracy

| Defense | n | Right tool | Right args | Right status |
|---|---|---|---|---|
| none | 25 | 1.00 | 1.00 | 0.64 |
| prompt | 25 | 1.00 | 1.00 | 0.64 |
| full | 25 | 1.00 | 1.00 | 1.00 |

### Performance

T4 on-demand at 0.35 USD/h; generation time only.

| Run | n | p50 (s) | p95 (s) | GPU hours | USD per 1k requests |
|---|---|---|---|---|---|
| e1_dense_test | 112 | 4.21 | 8.11 | 0.154 | 0.480 |
| e1_rerank_test | 112 | 4.50 | 6.78 | 0.145 | 0.453 |
| e2_none_test | 77 | 5.43 | 9.02 | 0.125 | 0.570 |
| e2_prompt_test | 77 | 5.91 | 9.01 | 0.126 | 0.572 |
| e2_full_test | 77 | 5.30 | 9.06 | 0.123 | 0.561 |
