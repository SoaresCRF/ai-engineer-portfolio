# PLAN — review-triage-llm

> Status: **concluído** (resultados no [README](README.md))
> Restrições: custo zero (sem API paga, sem GPU própria).

## 1. Objetivo

Responder com números a pergunta:

> **Um modelo pequeno (~0.5–1.5B) ajustado com QLoRA consegue chegar perto de um LLM aberto bem maior (~7–14B) sem treino na triagem de avaliações reais em PT-BR, sendo muito mais barato e rápido?**

Para cada avaliação de cliente, o sistema gera um JSON válido:

```json
{"categoria": "atraso_entrega", "sentimento": "negativo", "urgencia": "alta"}
```

A mesma tarefa é resolvida de três jeitos e todos são medidos no **mesmo conjunto de teste, rotulado à mão e às cegas**:

| # | Abordagem | Papel |
|---|-----------|-------|
| A | TF-IDF + regressão logística (um classificador por campo) | baseline clássica, piso de qualidade |
| B | LLM aberto ~7–14B, zero-shot com instruções e schema | "LLM grande sem treino" e professor |
| C | LLM aberto ~0.5–1.5B + QLoRA, treinado com rótulos gerados por B | aluno (destilação) |

**Por que isso importa para a vaga:** cobre fine-tuning de LLM, avaliação rigorosa, custo e latência, e entrega publicada no Hugging Face. É o eixo "IA/ML com foco em LLMs" da trilha.

## 2. Escopo

### Dentro
- Dataset público da Olist (Kaggle, *Brazilian E-Commerce Public Dataset*), só as avaliações que têm texto.
- Taxonomia fechada + guia de rotulagem.
- **Gold:** 300 avaliações rotuladas à mão, às cegas. **Silver:** ~5 mil rotuladas por B.
- As três abordagens, com avaliação reprodutível.
- Métricas: macro-F1 e acurácia por campo, taxa de JSON válido, latência p50/p95, vazão (avaliações/s) e custo estimado por 1.000 avaliações, com IC 95% por bootstrap.
- Análise de erros: matriz de confusão e 15–20 exemplos comentados, incluindo o que não funcionou.
- Publicação: modelo + model card no HF Hub, dataset gold no HF e demo no HF Spaces (CPU grátis).

### Fora (non-goals)
- Qualquer serviço pago: APIs de LLM, GPU em nuvem.
- API de produção, Docker e CI de deploy: isso é o Projeto 2.
- Busca de hiperparâmetros: uma rodada de treino, no máximo um ajuste.
- Ferramenta de anotação própria: rotular numa planilha basta.

### Anti-goals (fracasso mesmo se "funcionar")
- Medir o modelo no mesmo tipo de rótulo que ele aprendeu (silver) e chamar isso de acurácia real.
- Mostrar só números favoráveis ao modelo ajustado.
- Usar marca, logo ou dados do iFood.

## 3. Decisões de desenho

### 3.1 Rótulos (ponto mais importante)
A Olist só tem `review_score` (nota de 1 a 5), sem categoria nem urgência.

1. **Gold, humano e cego:** 300 avaliações rotuladas por você numa planilha, sem ver a saída de nenhum modelo (~15 s cada, ~1h15 no total).
   - Divididas em **dev (100)**, usado para ajustar o prompt e escolher checkpoint, e **test (200)**, usado **uma única vez** no fim.
   - Amostragem estratificada pela nota, para as reclamações não sumirem no meio dos elogios.
2. **Silver:** ~5 mil avaliações rotuladas por B, usadas para treinar A e C.
3. **A história fica honesta:** C é uma *destilação* de B, e todos são julgados pelo gold. O README também mostra a concordância de B com o gold, ou seja, quão bom é o professor.
4. **Checagens automáticas, sem custo de rotulagem:**
   - sentimento × `review_score`;
   - `atraso_entrega` × atraso real (`order_delivered_customer_date > order_estimated_delivery_date`).

### 3.2 Taxonomia (versão inicial, ajustar depois dos primeiros ~30 exemplos)
- **categoria:** `atraso_entrega`, `nao_recebido`, `produto_errado_ou_incompleto`, `defeito_ou_qualidade`, `atendimento_vendedor`, `elogio_sem_problema`, `outro`
- **sentimento:** `positivo`, `neutro`, `negativo`
- **urgencia:** `baixa`, `media`, `alta`. Regra explícita: `alta` = não recebeu, pede reembolso/troca ou menciona Procon/justiça; `media` = problema sem pedido de ação; `baixa` = elogio ou comentário neutro.

O guia (`docs/labeling_guide.md`) fica pronto **antes** de rotular, com 1–2 exemplos por classe e as regras de desempate.

### 3.3 Modelos e compute (100% grátis)
- **Compute:** Kaggle Notebooks com GPU T4 grátis (cota semanal, exige conta verificada por telefone). O Colab grátis é o reserva.
- **B (professor / zero-shot):** um modelo instruct aberto de ~7–14B com bom PT, como Qwen ou Llama dessa faixa, quantizado em 4-bit (AWQ/GPTQ). Roda com `vLLM`, ou com `transformers` se o vLLM não funcionar na T4. A saída é forçada no schema por geração guiada (JSON schema), quando disponível.
- **C (aluno):** um modelo instruct aberto de ~0.5–1.5B. A escolha sai de um teste zero-shot rápido de 2 candidatos no dev. Preferir o menor que ficar competitivo, porque a demo roda em CPU.
- **QLoRA:** 4-bit (nf4), LoRA nas projeções de atenção/MLP, 1–2 épocas. Os hiperparâmetros iniciais seguem os padrões da documentação do `peft`/`trl`.
- **Custo:** sem nuvem paga, o custo/1k é *estimado* como horas de GPU medidas × preço público de referência de uma T4 (fonte citada no README).
- Todos os IDs e versões de modelos e libs serão verificados na hora de executar.

## 4. Stack

| Camada | Escolha | Motivo |
|--------|---------|--------|
| Linguagem | Python 3.11+ | padrão do ecossistema |
| Ambiente | `uv` + `pyproject.toml` | rápido e com lockfile reprodutível |
| Dados | `pandas` | volume pequeno |
| Baseline | `scikit-learn` | TF-IDF + LogisticRegression; roda no seu PC |
| Schema | `pydantic` | valida o JSON de todas as abordagens com o mesmo código |
| Inferência B | `vLLM` (ou `transformers`) no Kaggle | grátis, sem limite de requisições |
| Fine-tuning C | `transformers`, `peft`, `trl`, `bitsandbytes`, `datasets` no Kaggle | padrão para QLoRA |
| Demo | `gradio` no HF Spaces (CPU grátis) | integração nativa com o HF |
| Qualidade | `pytest`, `ruff` | testes e lint |

Fica de fora: tracking de experimentos (W&B/MLflow). Resultados vão em JSON versionado em `results/`.

## 5. Estrutura de pastas

```
review-triage-llm/
├── PLAN.md
├── README.md                  # problema, arquitetura, como rodar, tabela de resultados, demo
├── pyproject.toml
├── .env.example               # HF_TOKEN=
├── docs/
│   └── labeling_guide.md
├── data/
│   ├── raw/                   # CSVs da Olist (gitignored)
│   ├── interim/               # (gitignored)
│   └── labels/                # gold_dev.csv, gold_test.csv (versionados); silver vai para o HF
├── src/review_triage/
│   ├── schema.py              # enums da taxonomia + modelo pydantic da saída
│   ├── data.py                # carga, limpeza, dedup, amostra estratificada para rotular
│   ├── prompts.py             # prompt único usado por B e no treino de C
│   ├── baseline.py            # abordagem A
│   └── evaluate.py            # métricas, bootstrap, latência, custo → results/*.json
├── notebooks/                 # rodam no Kaggle
│   ├── 01_eda.ipynb
│   ├── 02_teacher_labels.ipynb    # B: zero-shot no dev/test + geração do silver
│   └── 03_train_qlora.ipynb       # C: treino + inferência no dev/test
├── results/                   # métricas versionadas + figuras
├── app/                       # Space do Gradio (app.py + requirements.txt)
└── tests/                     # schema, parsing de JSON, métricas, splits sem vazamento
```

## 6. Critérios de entrega (Definition of Done)

- [x] `results/final.json` com macro-F1, acurácia, JSON válido %, latência p50/p95, vazão e custo/1k estimado para A, B e C, todos no mesmo test gold.
- [x] Test usado uma única vez, com a data registrada no README.
- [x] Modelo publicado no HF com model card: dados, licença, uso pretendido, limitações e métricas.
- [ ] ~~Space público funcionando com exemplos prontos.~~ Cortado: o HF passou a exigir plano pago para Spaces Gradio; a demo roda localmente (`app/app.py`).
- [x] README do projeto com problema, arquitetura, como rodar, métricas, demo e o que não funcionou.
- [x] Testes passando (`pytest`), lint limpo (`ruff`), nenhum segredo no repo.

## 7. Riscos e mitigação

| Risco | Mitigação |
|-------|-----------|
| **Licença:** o dataset da Olist é CC BY-NC-SA 4.0 (confirmar no Kaggle) | dataset e modelo publicados como não comerciais, com atribuição no card |
| PII nos textos, como nomes e telefones | mascarar com regex antes de publicar |
| Test de 200 é pequeno para classes raras | IC 95% por bootstrap + F1 por classe; declarar a limitação |
| Professor de 7–14B não cabe/roda lento na T4 | descer para ~7B em 4-bit; reduzir o silver para ~3k |
| Cota do Kaggle ou fila de GPU | Colab grátis como reserva; aluno de ~0.5B |
| O pequeno empata ou perde para B | resultado válido; o argumento vira qualidade × custo/latência |
| Rotulagem humana inconsistente | guia com exemplos; reler 30 itens no fim e medir concordância consigo mesmo |

## 8. Decisões tomadas

- Custo zero: Kaggle/Colab grátis, LLMs abertos e HF grátis.
- Gold de 300 rotulado à mão e às cegas (dev 100 / test 200).
