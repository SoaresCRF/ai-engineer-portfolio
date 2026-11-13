# review-triage-llm

**Can a small LLM fine-tuned with QLoRA match a 4.7× larger open LLM at triaging real Brazilian Portuguese customer reviews into structured JSON, at a fraction of the cost?**

Short answer: on two of three fields, yes. A Qwen3-1.7B student trained on 5,000 labels produced by a few-shot Qwen3-8B teacher ties the teacher on `categoria` and `urgencia`, loses about 4 points on `sentimento`, and runs 3.7× faster in batch. Both LLMs beat a TF-IDF baseline by 5–9 accuracy points. Everything ran on free Kaggle T4s; the whole experiment would cost about $1.30 in rented GPU time.

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
- **Discipline:** the prompt was revised once, on dev. The test set was scored once, after every decision was frozen ([data/predictions/final_test_eval.json](data/predictions/final_test_eval.json)).
- All three approaches share one parser (`review_triage.prompts.parse_output`), so valid-JSON rates are comparable.

## Results

Gold test set, n = 200. Accuracy / macro-F1:

| Approach | categoria | sentimento | urgencia | Valid JSON |
|---|---|---|---|---|
| A. TF-IDF baseline | 0.775 / 0.70 | 0.795 / 0.73 | 0.745 / 0.74 | 100% |
| B. Qwen3-8B teacher | **0.870** / **0.78** | **0.840** / **0.77** | 0.820 / 0.81 | 100% |
| C. Qwen3-1.7B student | 0.855 / 0.77 | 0.800 / 0.69 | **0.825** / **0.82** | 100% |

With 200 reviews, one review is 0.5 points, so gaps under ~3 points are noise. The student agrees with the teacher on 93–94% of test reviews in each field.

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

- **`neutro` sentiment is the hardest class.** Of 40 neutral reviews, the teacher gets 16 right and the student 10. Calm reports of a problem ("ainda estou no aguardo") get read as `negativo`; mild praise as `positivo`. Most of the student's 4-point sentiment gap is here (6 of 8 reviews): it learned the teacher's bias and amplified it.
- **`outro` is under-used.** Recall is 0.28 (teacher) and 0.22 (student): vague or off-topic reviews get forced into a concrete category, mostly `atendimento_vendedor` or `produto_errado_ou_incompleto`. That is why `atendimento_vendedor` precision is only 0.43–0.47.
- **Urgency over-escalates.** About 20 of 68 `media` reviews are predicted `alta` by each model. The rule is "`alta` only if the customer asks for an action"; the models escalate on tone. In a support queue this errs on the safe side, but the predicted `alta` queue is ~45% larger than the true one (65 vs 45).
- **Nothing breaks.** Zero invalid JSON across 300 gold and 5,000 silver reviews, without constrained decoding.

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
  teacher.py       B: zero-shot inference on a T4
  student.py       C: QLoRA training and inference
  evaluate.py      accuracy, macro-F1, valid-JSON rate
  kaggle_bundle.py dataset upload for Kaggle
notebooks/         Kaggle notebooks for B and C
data/labels/       gold dev/test labels
data/predictions/  every model's predictions and run metadata
hf/MODEL_CARD.md   Hugging Face model card
app/app.py         Gradio demo
```

## License

Code: MIT ([LICENSE](../LICENSE)). Reviews and derived labels come from the Olist dataset (CC BY-NC-SA 4.0), so the gold labels and the adapter are released for non-commercial use with attribution.
