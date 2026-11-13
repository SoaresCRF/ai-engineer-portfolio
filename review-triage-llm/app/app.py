"""Local Gradio demo for the QLoRA student, on CPU.

Uses the same prompt and parser as training and evaluation (review_triage.prompts).
Run from the project root:
    uv run --with-requirements app/requirements.txt python app/app.py
"""

import os
import time

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from review_triage.prompts import build_messages, parse_output

BASE_MODEL = "Qwen/Qwen3-1.7B"
ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "SoaresCRF/qwen3-1.7b-review-triage-ptbr-lora")
MAX_NEW_TOKENS = 48  # same cap as the evaluation runs

EXAMPLES = [
    "Comprei há 20 dias e até agora nada. Quero meu dinheiro de volta!",
    "Produto excelente, chegou antes do prazo. Recomendo a loja.",
    "Pedi a cor azul e veio preta. Vou ter que devolver.",
    "Chegou certinho, só achei a entrega um pouco demorada.",
    "A tela veio trincada e o vendedor não responde minhas mensagens.",
]

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# 4-bit bitsandbytes needs a GPU; on CPU the adapter is merged into a float32 base.
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
model = PeftModel.from_pretrained(model, ADAPTER_REPO).merge_and_unload().eval()


def triage(review: str) -> tuple[dict, str]:
    if not review.strip():
        raise gr.Error("Digite uma avaliação.")
    prompt = tokenizer.apply_chat_template(
        build_messages(review, few_shot=False),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    seconds = time.perf_counter() - start
    raw = tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    parsed = parse_output(raw)
    result = parsed.model_dump(mode="json") if parsed else {"erro": "JSON inválido", "saida": raw}
    return result, f"{seconds:.1f} s em CPU"


demo = gr.Interface(
    fn=triage,
    inputs=gr.Textbox(label="Avaliação do cliente", lines=4),
    outputs=[gr.JSON(label="Triagem"), gr.Textbox(label="Tempo")],
    examples=EXAMPLES,
    title="Triagem de avaliações PT-BR (Qwen3-1.7B + QLoRA)",
    description=(
        "Classifica uma avaliação de e-commerce em categoria, sentimento e urgência. "
        "Modelo pequeno ajustado com rótulos de um Qwen3-8B. "
        "Em CPU, cada resposta leva cerca de 10 s."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
