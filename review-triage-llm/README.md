# review-triage-llm

Can a small (~0.5–1.5B) LLM fine-tuned with QLoRA match a much larger open LLM used zero-shot at triaging real PT-BR customer reviews into structured JSON (`categoria`, `sentimento`, `urgencia`), at a fraction of the cost and latency?

> Status: **in progress**. Full write-up, results table and demo will land here. Plan: [PLAN.md](PLAN.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv sync
uv run pytest
```

## Data

Public [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (CC BY-NC-SA 4.0).

```bash
uv run python -c "from pathlib import Path; from review_triage.data import download_raw; download_raw(Path('data/raw'))"
uv run python -m review_triage.labeling export   # 300-review blind labeling sheet
uv run python -m review_triage.labeling import   # validated gold_dev.csv / gold_test.csv
```

Labeling rules: [docs/labeling_guide.md](docs/labeling_guide.md).
