"""Teacher LLM inference: zero-shot predictions on gold and silver labels for training.

Runs on a Kaggle T4 GPU (see notebooks/02_teacher_labels.ipynb). torch/transformers are
imported inside functions so the rest of the package works on machines without them.
"""

import json
import platform
import statistics
import time
from pathlib import Path

import pandas as pd

from review_triage.prompts import build_messages, parse_output, prompt_version
from review_triage.schema import LABEL_FIELDS

DEFAULT_MODEL = "Qwen/Qwen3-8B"
# A one-line JSON answer is ~30 tokens; the cap also stops runaway generations.
MAX_NEW_TOKENS = 48
BATCH_SIZE = 16
LATENCY_SAMPLE_SIZE = 30
LATENCY_P95 = 0.95
PROGRESS_EVERY_BATCHES = 20

SPLIT_FILES = {"dev": "gold_dev.csv", "test": "gold_test.csv", "silver": "silver_pool.csv"}
MODE_SPLITS = {"dev": ["dev"], "full": ["dev", "test", "silver"]}


def load_model(model_id: str = DEFAULT_MODEL, quantize_4bit: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    # T4 (Turing) has no bfloat16, so compute in float16.
    kwargs = {"device_map": "auto", "dtype": torch.float16}
    if quantize_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    return tokenizer, model


def render_prompt(tokenizer, text: str, few_shot: bool = True) -> str:
    return tokenizer.apply_chat_template(
        build_messages(text, few_shot=few_shot),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Qwen3: answer directly, no <think> block
    )


def _generate_batch(tokenizer, model, prompts: list[str]) -> list[str]:
    import torch

    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy: deterministic labels; fine without thinking mode
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)


def generate(tokenizer, model, texts: list[str], batch_size: int = BATCH_SIZE) -> list[str]:
    """Raw completions in input order; batches are length-sorted to minimize padding."""
    prompts = [render_prompt(tokenizer, t) for t in texts]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outputs: list[str] = [""] * len(prompts)
    n_batches = -(-len(order) // batch_size)
    for b, start in enumerate(range(0, len(order), batch_size), start=1):
        idx = order[start : start + batch_size]
        for i, completion in zip(
            idx, _generate_batch(tokenizer, model, [prompts[i] for i in idx]), strict=True
        ):
            outputs[i] = completion
        if b % PROGRESS_EVERY_BATCHES == 0 or b == n_batches:
            print(f"  batch {b}/{n_batches}", flush=True)
    return outputs


def single_item_latency(tokenizer, model, texts: list[str]) -> dict[str, float]:
    """Wall-clock seconds per review at batch size 1 (what one API call would feel like)."""
    import torch

    seconds = []
    for text in texts:
        torch.cuda.synchronize()
        start = time.perf_counter()
        _generate_batch(tokenizer, model, [render_prompt(tokenizer, text)])
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - start)
    seconds.sort()
    return {
        "n": len(seconds),
        "p50_s": statistics.median(seconds),
        "p95_s": seconds[min(len(seconds) - 1, int(LATENCY_P95 * len(seconds)))],
    }


def to_prediction_frame(df: pd.DataFrame, raw: list[str]) -> pd.DataFrame:
    parsed = [parse_output(r) for r in raw]
    out = df[["review_id", "text"]].copy()
    out["raw_output"] = raw
    out["valid_json"] = [p is not None for p in parsed]
    for field in LABEL_FIELDS:
        out[field] = [getattr(p, field).value if p else None for p in parsed]
    return out


def _environment(model_id: str, quantize_4bit: bool) -> dict:
    import bitsandbytes
    import torch
    import transformers

    return {
        "model_id": model_id,
        "quantize_4bit": quantize_4bit,
        "prompt_version": prompt_version(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "run_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def run(
    input_dir: Path,
    output_dir: Path,
    mode: str = "dev",
    model_id: str = DEFAULT_MODEL,
    quantize_4bit: bool = True,
) -> dict:
    """Predict every split for the mode; write teacher_<split>.csv and teacher_run.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_model(model_id, quantize_4bit)
    report = {"mode": mode, **_environment(model_id, quantize_4bit), "splits": {}}

    for split in MODE_SPLITS[mode]:
        df = pd.read_csv(input_dir / SPLIT_FILES[split])
        print(f"{split}: {len(df)} reviews", flush=True)
        start = time.perf_counter()
        raw = generate(tokenizer, model, df["text"].tolist())
        elapsed = time.perf_counter() - start
        preds = to_prediction_frame(df, raw)
        preds.to_csv(output_dir / f"teacher_{split}.csv", index=False)
        report["splits"][split] = {
            "n": len(df),
            "seconds": round(elapsed, 1),
            "reviews_per_s": round(len(df) / elapsed, 2),
            "valid_json_rate": round(float(preds["valid_json"].mean()), 4),
        }
        print(f"  {report['splits'][split]}", flush=True)

    dev_texts = pd.read_csv(input_dir / SPLIT_FILES["dev"])["text"].tolist()
    report["latency_batch1"] = single_item_latency(
        tokenizer, model, dev_texts[:LATENCY_SAMPLE_SIZE]
    )
    (output_dir / "teacher_run.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report
