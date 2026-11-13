---
license: cc-by-nc-sa-4.0
language:
- pt
base_model: Qwen/Qwen3-1.7B
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- qlora
- text-classification
- structured-output
- customer-reviews
- portuguese
- distillation
---

# Qwen3-1.7B QLoRA: PT-BR e-commerce review triage

A LoRA adapter for [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) that reads a Brazilian Portuguese customer review and returns one line of JSON:

```json
{"categoria": "nao_recebido", "sentimento": "negativo", "urgencia": "alta"}
```

It is the "student" of a teacher/student experiment. [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), prompted zero-shot with a few examples, labeled 5,000 reviews. This 4.7× smaller model was then fine-tuned on those silver labels. Both models were scored against 300 reviews labeled by hand.

The prompt and the output parser ship with this repo in `review_triage/` (`prompts.py`, `schema.py`). Code, labeling guide, notebooks and evaluation: [review-triage-llm on GitHub](https://github.com/SoaresCRF/ai-engineer-portfolio/tree/main/review-triage-llm).

## Labels

| Field | Values |
|---|---|
| `categoria` (main problem) | `atraso_entrega`, `nao_recebido`, `produto_errado_ou_incompleto`, `defeito_ou_qualidade`, `atendimento_vendedor`, `elogio_sem_problema`, `outro` |
| `sentimento` (overall tone) | `positivo`, `neutro`, `negativo` |
| `urgencia` (does the company need to act?) | `alta`, `media`, `baixa` |

When a review mentions several problems, the most severe one wins: `nao_recebido` > `produto_errado_ou_incompleto` > `defeito_ou_qualidade` > `atendimento_vendedor` > `atraso_entrega`.

## Results

Gold test set: 200 Olist reviews labeled blind by a human, scored once, after all prompt and training decisions were final. Accuracy / macro-F1:

| Approach | categoria | sentimento | urgencia | Valid JSON |
|---|---|---|---|---|
| TF-IDF + logistic regression (trained on the same silver labels) | 0.775 / 0.70 | 0.795 / 0.73 | 0.745 / 0.74 | 100% |
| Qwen3-8B 4-bit, few-shot prompt (teacher) | 0.870 / 0.78 | 0.840 / 0.77 | 0.820 / 0.81 | 100% |
| **This adapter (Qwen3-1.7B 4-bit + LoRA, no few-shot)** | **0.855 / 0.77** | **0.800 / 0.69** | **0.825 / 0.82** | **100%** |

Speed on one Kaggle T4 GPU, both models 4-bit:

| | Teacher 8B | Student 1.7B |
|---|---|---|
| Batched throughput | 0.72 reviews/s | 2.6 reviews/s |
| Batch-1 latency p50 / p95 | 3.82 s / 4.05 s | 3.09 s / 3.32 s |

The student matches the teacher on `categoria` and `urgencia` at 3.5× the batched throughput. It is worse on `sentimento`: the `neutro` class is hard for both models (teacher 16/40 correct, student 10/40), and the student pushes more neutral reviews to `negativo` or `positivo`. Batch-1 latency barely improves, because short generations on a T4 are dominated by per-token overhead, not model size.

A paired bootstrap over the 200 test reviews (95% CI) confirms the sentiment gap (teacher +4.0 accuracy points, CI [+1.0, +7.5]) and finds no significant teacher–student difference on `categoria` or `urgencia`.

## Usage

The adapter was trained on the exact prompt used at inference time: the system prompt from `review_triage/prompts.py` (no few-shot examples), the review as the user message, and Qwen3's chat template with `enable_thinking=False`. Other prompts will work worse.

```python
# pip install transformers peft pydantic; download review_triage/ from this repo
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from review_triage.prompts import build_messages, parse_output

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")
model = PeftModel.from_pretrained(
    model, "SoaresCRF/qwen3-1.7b-review-triage-ptbr-lora"
).merge_and_unload()

review = "Comprei há 20 dias e até agora nada. Quero meu dinheiro de volta!"
prompt = tokenizer.apply_chat_template(
    build_messages(review, few_shot=False),
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer(prompt, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=48, do_sample=False)
print(
    parse_output(tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
)
```

Merged into a float32 base, the adapter runs on a laptop CPU at about 10 s per review.

## Training

- Data: 5,000 Olist reviews with text, labeled by the teacher (100% valid JSON). No gold review is in the training set.
- QLoRA: 4-bit NF4 base, LoRA r=16, alpha=32, dropout 0.05 on all linear layers.
- 2 epochs, lr 2e-4 cosine, 3% warmup, effective batch 16, max length 1024, loss on the JSON answer only, paged AdamW 8-bit, fp16, seed 42.
- 625 steps in 96 minutes on one free Kaggle T4. Training loss fell from 0.145 to about 0.015.
- One training run, no hyperparameter search.

## Data and license

Reviews come from the [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), licensed CC BY-NC-SA 4.0. The adapter is trained on derived labels, so it is released under the same non-commercial, share-alike license. The base model is Apache-2.0.

## Limitations

- Trained on one marketplace's reviews from 2016–2018. Other domains (food delivery, apps) will need new labels.
- The student learns the teacher's mistakes. Its ceiling is the teacher's agreement with humans, not human accuracy.
- `neutro` sentiment and the rarer categories (`atendimento_vendedor`, `atraso_entrega`) have few test examples, so their scores are noisy.
- Intended for routing and analytics, not for automated decisions about individual customers.
