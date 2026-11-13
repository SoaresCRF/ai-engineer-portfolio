# review-triage-llm

**Can a small LLM fine-tuned with QLoRA match a 4.7× larger open LLM at triaging real Brazilian Portuguese customer reviews into structured JSON, at a fraction of the cost?**

Short answer: on two of three fields, yes. A Qwen3-1.7B student trained on 5,000 labels produced by a few-shot Qwen3-8B teacher ties the teacher on `categoria` and `urgencia`, loses 4 accuracy points on `sentimento` (the only statistically clear gap), and runs 3.7× faster in batch. Both LLMs beat a TF-IDF baseline by 7–10 points on `categoria` and `urgencia`; on `sentimento`, neither LLM is clearly better than the baseline. Everything ran on free Kaggle T4s; the whole experiment would cost about $1.30 in rented GPU time.

Model: [SoaresCRF/qwen3-1.7b-review-triage-ptbr-lora](https://huggingface.co/SoaresCRF/qwen3-1.7b-review-triage-ptbr-lora)

## The task

Each review becomes one line of JSON:

```json
{"categoria": "nao_recebido", "sentimento": "negativo", "urgencia": "alta"}
```

| Field | Values |
|---|---|
| `categoria` (main problem) | `atraso_entrega`, `nao_recebido`, `produto_errado_ou_incompleto`, `defeito_ou_qualidade`, `atendimento_vendedor`, `elogio_sem_problema`, `outro` |
| `sentimento` (overall tone) | `positivo`, `neutro`, `negativo` |
| `urgencia` (does the company need to act?) | `alta`, `media`, `baixa` |

Definitions and tie-break rules (several problems: the most severe wins) are in [docs/labeling_guide.md](docs/labeling_guide.md). Keys and labels are Portuguese on purpose: they are the product's data contract.

## Setup

| Approach | What it is | Trained on |
|---|---|---|
| A. Baseline | TF-IDF + logistic regression, one classifier per field | 5,000 silver labels |
| B. Teacher | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), 4-bit, system prompt + 3 few-shot examples, no thinking | nothing |
| C. Student | [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), QLoRA, system prompt only | 5,000 silver labels |

- **Data:** reviews with text from the public [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
- **Gold:** 300 reviews labeled by hand, blind (text only, no star rating), split 100 dev / 200 test. Gold reviews never appear in the silver set or the few-shot examples (checked by tests and by the Kaggle bundle builder).
- **Silver:** 5,000 other reviews labeled by the teacher (100% valid JSON). A and C both learn from them, so the comparison isolates the model, not the data.
- **Discipline:** the prompt was revised once, on dev. The test set was scored once, on 2026-11-13, after every decision was frozen ([results/final.json](results/final.json): quality, latency, throughput and cost of A, B and C in one file).
- All three approaches share one parser (`review_triage.prompts.parse_output`), so valid-JSON rates are comparable.

## Results

Gold test set, n = 200. Accuracy / macro-F1:

| Approach | categoria | sentimento | urgencia | Valid JSON |
|---|---|---|---|---|
| A. TF-IDF baseline | 0.775 / 0.70 | 0.795 / 0.73 | 0.745 / 0.74 | 100% |
| B. Qwen3-8B teacher | **0.870** / **0.78** | **0.840** / **0.77** | 0.820 / 0.81 | 100% |
| C. Qwen3-1.7B student | 0.855 / 0.77 | 0.800 / 0.69 | **0.825** / **0.82** | 100% |

With 200 reviews, one review is 0.5 points. To separate real gaps from noise, a paired bootstrap (2,000 resamples of the test reviews, the same reviews for every approach) gives 95% CIs for each difference in accuracy ([results/bootstrap_test.json](results/bootstrap_test.json), which also has macro-F1):

| Difference in accuracy (points) | categoria | sentimento | urgencia |
|---|---|---|---|
| Teacher − student | +1.5 [−1.5, +4.5] | **+4.0 [+1.0, +7.5]** | −0.5 [−4.0, +3.0] |
| Teacher − baseline | **+9.5 [+4.0, +15.0]** | +4.5 [−1.5, +10.5] | **+7.5 [+2.0, +13.0]** |
| Student − baseline | **+8.0 [+3.0, +13.5]** | +0.5 [−4.5, +5.5] | **+8.0 [+3.0, +13.5]** |

Bold: the CI excludes zero. The student's only real loss is sentiment (macro-F1 −7.6 [−13.4, −1.9]). And on sentiment, a TF-IDF model trained on the same silver labels is statistically as good as either LLM: extra model capacity buys little until `neutro` is better defined and labeled. The student agrees with the teacher on 93–94% of test reviews in each field.

### Speed and cost

| | A. Baseline | B. Teacher 8B | C. Student 1.7B |
|---|---|---|---|
| Hardware | laptop CPU | 1× T4, 4-bit | 1× T4, 4-bit |
| Batched throughput | not measured | 0.71 reviews/s | 2.6 reviews/s |
| Latency, 1 review (p50 / p95) | 2 ms | 3.82 s / 4.05 s | 3.09 s / 3.32 s |
| GPU cost per 1,000 reviews* | ~$0 | $0.137 | $0.037 |

\* GPU time only, at Google Cloud's $0.35 per T4-hour on-demand ([GPU pricing](https://cloud.google.com/products/compute/gpus-pricing), us-west8, Nov 2026). The runs themselves used Kaggle's free quota. One-off costs at that rate: teacher labeling of the silver set $0.69, student training (96 min) $0.56.

Part of the student's throughput gain comes from the shorter prompt: the fine-tune replaces the teacher's few-shot examples. Batch-1 latency barely moves (3.8 s → 3.1 s): a 30-token answer on a T4 is dominated by per-token overhead, not model size. The student pays off in batch jobs, not single requests.

## Error analysis

Patterns shared by both LLMs (test set):

- **`neutro` sentiment is the hardest class.** Of 40 neutral reviews, the teacher gets 16 right and the student 10. Calm questions about a problem ("Gostaria de saber quando vou receber meu pedido [...] Aguardo uma posicao") get read as `negativo`; plain confirmations ("Recebi corretamente.") as `positivo`. Most of the student's 4-point sentiment gap is here (6 of 8 reviews): it learned the teacher's bias and amplified it.
- **`outro` is under-used.** Recall is 0.28 (teacher) and 0.22 (student): vague or off-topic reviews get forced into a concrete category, mostly `atendimento_vendedor` or `produto_errado_ou_incompleto`. That is why `atendimento_vendedor` precision is only 0.43–0.47.
- **Urgency over-escalates.** About 20 of 68 `media` reviews are predicted `alta` by each model. The rule is "`alta` only if the customer asks for an action"; the models escalate on tone. In a support queue this errs on the safe side, but the predicted `alta` queue is ~45% larger than the true one (65 vs 45).
- **Nothing breaks.** Zero invalid JSON across 300 gold and 5,000 silver reviews, without constrained decoding.

### Annotated examples

16 of the 84 test reviews where at least one model missed at least one field. T = teacher, S = student; only the fields with an error are shown. Texts are verbatim, typos included.

**Keyword shortcuts.** The models react to words instead of meaning:

| Review | Gold | T | S | Comment |
|---|---|---|---|---|
| "Recebi antes do prqzo" | `elogio_sem_problema` | `atraso_entrega` | `atraso_entrega` | "prazo" (here misspelled) pulls toward delivery delay, even when it arrived early. |
| "Entrega realizada maravilhosamente 10 dias antes do prazo." | `elogio_sem_problema` | `atraso_entrega` | ✓ | Same shortcut; the student fixed it here... |
| "Aprovado. Chegou um pouco antes do prazo, isso é bom" | `elogio_sem_problema` | ✓ | `atraso_entrega` | ...and fell for it here. Distillation copies the bias, not each individual answer. |
| "Ainda não usei o produto, chegou lacrado, só espero que seja original." | `elogio_sem_problema` / `baixa` | `nao_recebido` / `media` | `nao_recebido` / `alta` | "Ainda não" + "chegou" read as "not received yet". The student escalates to `alta`. |

**`neutro` vs `positivo`.** Short factual confirmations are the biggest single source of sentiment errors:

| Review | Gold | T | S | Comment |
|---|---|---|---|---|
| "Recebi corretamente." | `neutro` | `positivo` | `positivo` | Gold treats "it arrived as promised" as neutral; both models read it as praise. About 10 similar cases ("Entregou no prazo", "Recebi no prazo e conforme anuncio"...). |
| "O produto chegou no prazo certo." | `neutro` | ✓ | `positivo` | The teacher is inconsistent on the same pattern, so the silver labels are too. |

This boundary is genuinely ambiguous: a human could defend `positivo`. It is a labeling-guide problem as much as a model problem, which matches the bootstrap result that no model beats TF-IDF on sentiment.

**Calm complaints read as negative:**

| Review | Gold | T | S | Comment |
|---|---|---|---|---|
| "Nao vale a pena a reclamação, mas encomendei um carticho preto e outro colorido e recebi dois pretos" | `neutro` | `negativo` | `negativo` | The customer explicitly downplays it; the models see "wrong item" and assume anger. |
| "Gostaria de um vale troca! Aguardando número de postagem." | `neutro` | `negativo` | `negativo` | A request, not a complaint. |

**Urgency over-escalation.** The guide says `alta` only when the customer asks for an action:

| Review | Gold | T | S | Comment |
|---|---|---|---|---|
| "Quase um mês de atraso na entrega" | `media` | `alta` | `alta` | A serious problem, but no request. The models escalate on severity. |
| "Eu comprei 4 rádios e so recebi 3. Estou esperando o restante." | `media` | `alta` | `alta` | Borderline: "estou esperando" is close to a request. A reasonable labeler could say `alta`. |
| "[...] Produto veio com defeito não funciona, bem vou pedir troca [...]" | `alta` | ✓ | `media` | The opposite error: an explicit exchange request the student missed. |

**Tie-break rules not applied:**

| Review | Gold | T | S | Comment |
|---|---|---|---|---|
| "So recebi a cola NF 001063, o iten estufa ainda nao chegou.." | `produto_errado_ou_incompleto` | `nao_recebido` | `nao_recebido` | Rule: part of the order received is "incomplete", not "not received". Neither model applies it. |
| "Nao recebi o meu produto. Mandaram um outro produto [...]" | `nao_recebido` | ✓ | `produto_errado_ou_incompleto` | Two problems; the most severe wins. The teacher gets the priority right, the student does not. |

**`outro` gets forced into a real category:**

| Review | Gold | T | S | Comment |
|---|---|---|---|---|
| "Não é que o produto seja ruim, é que achei que era menor [...] Mas é uma ótima carteira." | `outro` / `positivo` / `baixa` | `produto_errado_ou_incompleto` / `neutro` / `media` | same as T | Personal taste is `outro`. Both models invent a defect and get all three fields wrong. |
| "Péssimo serviço" | `outro` | `atendimento_vendedor` | `atendimento_vendedor` | Too vague to know whose service. The models' guess is plausible; the gold is a judgment call. |

**Gold noise.** At least one gold label looks wrong on re-reading: "Adorei o produto... [as resenhas dizem que] vem com a luva térmica e certificado, porém a minha não veio nada além do manual" is gold `positivo` / `baixa`, models `negativo` / `media`. A missing item is a problem without a request, so by the guide `media` is right and the models are. With one labeler and no second pass, a few such errors are expected; they cut both ways and do not change the comparison.

What did not work, or was not tried:

- The first prompt (`63b2f291`) scored 0.69 macro-F1 on dev sentiment (1 of 8 neutral right). Rewriting the `neutro` definition with concrete phrasings moved it to 0.82 on dev, but the test set shows the problem is only partly solved.
- One training run, no hyperparameter search. The student's ceiling is the teacher's agreement with humans; better silver labels (e.g. teacher self-consistency, or a few hundred human labels) would likely help more than tuning.
- A free demo Space was planned, but Hugging Face now requires a paid plan for Gradio Spaces, so the demo runs locally.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11. GPU steps run on a free Kaggle T4 (about 2 h teacher, 1 h 45 min student).

```bash
uv sync
uv run pytest

# 1. Data and gold labels
uv run python -c "from pathlib import Path; from review_triage.data import download_raw; download_raw(Path('data/raw'))"
uv run python -m review_triage.labeling export   # 300-review blind labeling sheet
uv run python -m review_triage.labeling import   # validated gold_dev.csv / gold_test.csv

# 2. Kaggle bundle (texts only, no gold labels), then run the notebooks on Kaggle
uv run python -m review_triage.kaggle_bundle
#    notebooks/02_teacher_labels.ipynb  -> data/predictions/teacher_full_v2/
#    notebooks/03_train_qlora.ipynb     -> data/predictions/student_v1/

# 3. Baseline and evaluation
uv run python -m review_triage.baseline data/predictions/teacher_full_v2/teacher_silver.csv dev
uv run python -m review_triage.evaluate data/predictions/student_v1/student_dev.csv data/labels/gold_dev.csv
# final test evaluation (run once)
uv run python -m review_triage.baseline data/predictions/teacher_full_v2/teacher_silver.csv test
uv run python -m review_triage.evaluate bootstrap data/labels/gold_test.csv \
    baseline=data/predictions/baseline_test.csv \
    teacher=data/predictions/teacher_full_v2/teacher_test.csv \
    student=data/predictions/student_v1/student_test.csv > results/bootstrap_test.json
uv run python -m review_triage.report   # results/final.json
```

Local demo on CPU (~10 s per review; downloads the base model and the adapter):

```bash
uv run --with-requirements app/requirements.txt python app/app.py
```

## Layout

```
src/review_triage/
  schema.py        output contract (pydantic)
  prompts.py       system prompt, few-shot examples, JSON parser
  data.py          Olist download and review sampling
  labeling.py      gold labeling sheet export/import
  baseline.py      A: TF-IDF + logistic regression
  teacher.py       B: few-shot inference on a T4
  student.py       C: QLoRA training and inference
  evaluate.py      accuracy, macro-F1, valid-JSON rate, paired bootstrap CIs
  report.py        assembles results/final.json
  kaggle_bundle.py dataset upload for Kaggle
notebooks/         Kaggle notebooks for B and C
data/labels/       gold dev/test labels
data/predictions/  every model's predictions and run metadata
results/           final test results and bootstrap CIs
hf/MODEL_CARD.md   Hugging Face model card
app/app.py         Gradio demo
```

## License

Code: MIT ([LICENSE](../LICENSE)). Reviews and derived labels come from the Olist dataset (CC BY-NC-SA 4.0), so the gold labels and the adapter are released for non-commercial use with attribution.
