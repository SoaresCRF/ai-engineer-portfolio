"""Approach C: QLoRA fine-tune of a small Qwen3 on the teacher's silver labels.

Runs on a Kaggle T4 GPU (see notebooks/03_train_qlora.ipynb). The student sees the same
system prompt as the teacher but no few-shot examples (shorter prompt, faster inference);
the fine-tune replaces them. torch/peft/trl are imported inside functions so the rest of
the package works on machines without them.
"""

import json
import time
from pathlib import Path

import pandas as pd

from review_triage import teacher
from review_triage.prompts import to_json
from review_triage.schema import LABEL_FIELDS, Triage

# 1.7B over 0.6B: seven categories with tie-break rules is a lot for 0.6B, and 1.7B
# still runs on CPU for the demo. Switching to "Qwen/Qwen3-0.6B" needs no other change.
DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
SILVER_FILE = "teacher_silver.csv"
EVAL_SPLITS = ["dev", "test"]
ADAPTER_DIR_NAME = "adapter"

# Starting points from the peft/trl docs; PLAN.md allows one training run, no search.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
EPOCHS = 2
LEARNING_RATE = 2e-4
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2  # effective batch 16
WARMUP_RATIO = 0.03
# System prompt (~450 tokens) + review + answer; no Olist review comes close to this.
MAX_LENGTH = 1024
LOGGING_STEPS = 25
SEED = 42


def training_records(silver: pd.DataFrame, tokenizer) -> list[dict[str, str]]:
    """Prompt-completion pairs from valid silver rows.

    The prompt is rendered exactly as at inference time (non-thinking Qwen3 template, no
    few-shot), so the loss covers only the JSON answer the model has to produce.
    """
    usable = silver[silver["valid_json"].astype(bool)]
    return [
        {
            "prompt": teacher.render_prompt(tokenizer, row.text, few_shot=False),
            "completion": to_json(Triage(**{f: getattr(row, f) for f in LABEL_FIELDS}))
            + tokenizer.eos_token,
        }
        for row in usable.itertuples()
    ]


def train(
    silver: pd.DataFrame,
    output_dir: Path,
    model_id: str = DEFAULT_MODEL,
    quantize_4bit: bool = True,
    max_steps: int = -1,
) -> dict:
    """Fit a LoRA adapter on the silver labels; save it under output_dir/adapter."""
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    tokenizer, model = teacher.load_model(model_id, quantize_4bit)
    dataset = Dataset.from_list(training_records(silver, tokenizer))
    args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=EPOCHS,
        max_steps=max_steps,  # > 0 only for smoke tests
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        warmup_steps=WARMUP_RATIO,  # transformers 5: a float < 1 is a ratio of total steps
        lr_scheduler_type="cosine",
        max_length=MAX_LENGTH,
        completion_only_loss=True,
        gradient_checkpointing=True,
        # TRL defaults to bf16, which the T4 lacks; CPU smoke tests stay in float32.
        bf16=False,
        fp16=quantize_4bit,
        optim="paged_adamw_8bit" if quantize_4bit else "adamw_torch",
        logging_steps=LOGGING_STEPS,
        save_strategy="no",  # one run, keep only the final adapter
        report_to="none",
        seed=SEED,
    )
    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
    )
    # LoRA weights on a 4-bit Qwen3 come out bfloat16 (the checkpoint's dtype), and the
    # fp16 GradScaler can't unscale bf16 grads on a T4. Train the adapter in float32.
    for param in trainer.model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    start = time.perf_counter()
    result = trainer.train()
    seconds = time.perf_counter() - start
    trainer.model.save_pretrained(output_dir / ADAPTER_DIR_NAME)
    tokenizer.save_pretrained(output_dir / ADAPTER_DIR_NAME)
    return {
        "n_train": len(dataset),
        "seconds": round(seconds, 1),
        "train_loss": round(result.training_loss, 4),
        "loss_history": [
            {"step": h["step"], "loss": h["loss"]} for h in trainer.state.log_history if "loss" in h
        ],
        "hyperparameters": {
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "epochs": EPOCHS,
            "max_steps": max_steps,
            "learning_rate": LEARNING_RATE,
            "effective_batch": BATCH_SIZE * GRAD_ACCUM_STEPS,
            "max_length": MAX_LENGTH,
            "seed": SEED,
        },
        "model": trainer.model,
        "tokenizer": tokenizer,
    }


def predict_splits(tokenizer, model, input_dir: Path, output_dir: Path) -> dict:
    """student_<split>.csv for dev and test, with the same parser and timing as the teacher."""
    model.eval()
    splits = {}
    for split in EVAL_SPLITS:
        df = pd.read_csv(input_dir / teacher.SPLIT_FILES[split])
        print(f"{split}: {len(df)} reviews", flush=True)
        start = time.perf_counter()
        raw = teacher.generate(tokenizer, model, df["text"].tolist(), few_shot=False)
        elapsed = time.perf_counter() - start
        preds = teacher.to_prediction_frame(df, raw)
        preds.to_csv(output_dir / f"student_{split}.csv", index=False)
        splits[split] = {
            "n": len(df),
            "seconds": round(elapsed, 1),
            "reviews_per_s": round(len(df) / elapsed, 2),
            "valid_json_rate": round(float(preds["valid_json"].mean()), 4),
        }
        print(f"  {splits[split]}", flush=True)
    return splits


def run(
    input_dir: Path,
    output_dir: Path,
    model_id: str = DEFAULT_MODEL,
    quantize_4bit: bool = True,
    max_steps: int = -1,
) -> dict:
    """Train on silver, predict dev/test, write student_run.json next to the predictions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report = teacher.environment(model_id, quantize_4bit)
    trained = train(
        pd.read_csv(input_dir / SILVER_FILE), output_dir, model_id, quantize_4bit, max_steps
    )
    tokenizer, model = trained.pop("tokenizer"), trained.pop("model")
    report["training"] = trained
    report["splits"] = predict_splits(tokenizer, model, input_dir, output_dir)
    dev_texts = pd.read_csv(input_dir / teacher.SPLIT_FILES["dev"])["text"].tolist()
    report["latency_batch1"] = teacher.single_item_latency(
        tokenizer, model, dev_texts[: teacher.LATENCY_SAMPLE_SIZE], few_shot=False
    )
    (output_dir / "student_run.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "training"}, indent=2))
    return report
