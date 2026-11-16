# PLAN — support-rag-agent

> Status: **concluído** (resultados no [README](README.md))
> Restrições: custo zero (sem API paga, sem GPU paga), só dados públicos ou sintéticos, sem marca de terceiros.

## 1. Objetivo

Responder com números a pergunta:

> **Num assistente de atendimento de delivery com RAG e ferramentas, rodando só com modelos abertos, quanto cada ajuste de engenharia melhora o sistema: (a) busca híbrida com reranker vs. só embeddings, e (b) defesa em camadas vs. prompt sem defesa contra prompt injection?**

O assistente:
- responde dúvidas sobre pedidos com base no **Código de Defesa do Consumidor**, no **Decreto 7.962/2013** (comércio eletrônico) e em **políticas fictícias** de entrega, cancelamento e reembolso, citando a fonte;
- chama as ferramentas `consultar_pedido` e `abrir_reembolso` sobre pedidos sintéticos, só do cliente autenticado;
- recusa o que está fora do escopo e não obedece instruções escondidas em mensagens ou dados.

Os dois experimentos de antes e depois:

| # | Experimento | Antes | Depois | Métrica principal |
|---|---|---|---|---|
| E1 | Busca | só embeddings (denso) | BM25 + denso (RRF) + reranker | recall@5 e % de respostas corretas (juiz) |
| E2 | Prompt injection | prompt sem defesa | defesa em camadas (prompt + marcação de dados + guarda nas ferramentas) | taxa de ataque bem-sucedido, sem perder utilidade |

**Por que isso importa para a vaga:** cobre o caminho "software para produtos de IA": API, Docker, CI, observabilidade, segurança e avaliação de um sistema com LLM, e complementa o Projeto 1 (fine-tuning).

## 2. Escopo

### Dentro
- Corpus: CDC (Lei 8.078/1990), Decreto 7.962/2013 e ~8 políticas fictícias escritas para o projeto.
- Busca: BM25, denso e híbrido com reranker, todos em CPU.
- Agente com laço de tool calling escrito à mão (sem framework), até 4 passos por pergunta.
- Pedidos sintéticos (~200 pedidos, ~30 clientes) gerados por script com seed, em SQLite.
- API FastAPI (`POST /chat`, `GET /health`), autenticação por token de demo que identifica o cliente.
- Docker (imagem da API) e CI no GitHub Actions filtrado pela pasta.
- Logs estruturados em JSON e trace por requisição (spans de busca, rerank, LLM e ferramentas, com latência).
- Conjunto de avaliação próprio, congelado antes de qualquer ajuste, dividido em dev e test.
- Métricas: recall@k e MRR da busca; correção e fidelidade das respostas (LLM como juiz, validado contra notas humanas); acerto de ferramentas; taxa de sucesso de ataques; latência p50/p95; IC 95% por bootstrap pareado.
- Análise de erros com exemplos comentados, incluindo o que não funcionou.

### Fora (non-goals)
- Qualquer serviço pago, ou demo hospedada (o HF cobra por Spaces com Docker/Gradio).
- Frameworks de agente/RAG (LangChain, LlamaIndex): o laço cabe em ~100 linhas e fica explicável.
- Banco vetorial: o corpus tem poucas centenas de chunks; numpy em memória basta.
- Interface web própria: o Swagger do FastAPI (`/docs`) é a demo.
- Fine-tuning: isso foi o Projeto 1.
- Memória longa entre conversas: o histórico vai no corpo da requisição.
- Classificador de injection treinado (ex.: Prompt Guard): é focado em inglês e exige aceitar licença no HF; fica como trabalho futuro.

### Anti-goals (fracasso mesmo se "funcionar")
- Escolher a configuração olhando o test.
- Validar o juiz só "no olho": sem concordância medida contra notas humanas, o número do juiz não entra no README.
- Defesa que "zera" ataques porque recusa tudo: a utilidade em perguntas legítimas é medida junto.
- Usar marca, logo ou dados do iFood.

## 3. Decisões de desenho

### 3.1 Corpus
- **CDC e Decreto 7.962/2013:** texto oficial do Planalto. Leis não têm proteção de direito autoral (Lei 9.610/1998, art. 8º, IV). O download precisa da sua permissão.
- **Políticas fictícias** de uma empresa de delivery inventada: entrega e atrasos, cancelamento, reembolso (comida vs. mercado/varejo), itens faltando ou errados, cupons, taxas, marketplace (loja parceira vs. entrega própria).
- As políticas são escritas com **casos parecidos de propósito** (prazos diferentes para comida e varejo, regra de parceiro vs. entrega própria), para a busca ter o que errar. Sem isso, o corpus fica fácil demais e E1 não mede nada.
- Cada trecho tem um ID estável de passagem: `cdc:art49`, `cdc:art18:p1`, `politica_reembolso:3.2`.

### 3.2 Chunking
- **Leis:** por artigo (com parágrafos e incisos juntos); artigos longos são quebrados por parágrafo. O artigo é a unidade que uma resposta cita.
- **Políticas:** por seção de Markdown.
- Cada chunk leva o título do documento e do capítulo como prefixo, para não perder contexto.
- O gold da busca é marcado em IDs de passagem, não de chunk; assim, mudar o chunking não invalida o conjunto de avaliação.

### 3.3 Busca
| Configuração | O que é |
|---|---|
| R1 (antes) | denso: `BAAI/bge-m3`, similaridade de cosseno, top-k |
| R2 | BM25 (`bm25s`), com tokenização simples em PT |
| R3 | híbrido: R1 + R2 por Reciprocal Rank Fusion |
| R4 (depois) | R3 top-20 → reranker `BAAI/bge-reranker-v2-m3` → top-5 |

- O bge-m3 e o reranker são multilíngues, abertos e rodam em CPU; com poucas centenas de chunks, os embeddings são calculados uma vez e salvos.
- O top-k final que vai ao LLM (padrão 5) é escolhido no dev.
- R2 e R3 entram na tabela para mostrar de onde vem o ganho, mas o antes e depois oficial é R1 → R4.

### 3.4 LLMs (todos abertos, rodando no Kaggle T4 grátis)

**Gerador: Qwen3-8B** (instruct, sem thinking, 4-bit).
- Bom em PT, tool calling nativo, licença Apache 2.0.
- Já rodou numa T4 no Projeto 1, o que reduz risco.
- Servido com `vLLM` em modo OpenAI-compatible, com parser de tool calls. Se o vLLM não rodar na T4, a reserva é `transformers` com parsing próprio das tool calls.
- O código do agente fala com qualquer endpoint OpenAI-compatible (`LLM_BASE_URL`), então o mesmo código roda no Kaggle (avaliação) e localmente (demo).

**Juiz: Gemma 3 12B** (instruct, 4-bit).
- É de outra família de modelos: um juiz da mesma família do gerador tende a favorecer respostas parecidas com as suas.
- É maior que o gerador e está disponível no Kaggle Models, sem precisar de token do HF.
- O juiz recebe pergunta, resposta de referência, trechos recuperados e a resposta do sistema. Ele devolve JSON com `correta` (bool), `fiel_ao_contexto` (bool) e uma justificativa curta. A saída é binária porque é mais estável que notas de 1 a 5.
- **Validação:** você dá nota às cegas para ~60 respostas do dev, e eu meço a concordância do juiz com você (acurácia e kappa de Cohen). Se o kappa ficar abaixo de ~0,6, ajusto o prompt do juiz no dev, ou troco o juiz. Isso é decidido antes do test.
- **Reserva:** uma camada gratuita sem cartão (ex.: Gemini API free tier). Ela tem limites que mudam e o modelo não é versionado de forma reproduzível; por isso é só reserva.

**Demo local:** a API em CPU aponta para o Ollama (OpenAI-compatible, com Qwen3 em GGUF). A instalação precisa da sua permissão. É lenta, mas basta para um GIF.

IDs e versões de modelos e libs são conferidos na hora de executar.

### 3.5 Ferramentas
- `consultar_pedido(pedido_id)` devolve status, itens, valores, datas, tipo de loja e observação do pedido.
- `abrir_reembolso(pedido_id, motivo, itens?)` abre a solicitação e devolve protocolo ou recusa com motivo.
- **As regras ficam no código, não no LLM.** O pedido precisa ser do cliente autenticado (sem IDOR), o valor não pode passar do pago, a janela de prazo tem que estar dentro da política e só pode haver um reembolso por pedido (constraint `UNIQUE` no SQLite, o que torna a operação idempotente).
- O LLM só decide **quando** chamar a ferramenta; quem decide se o reembolso é permitido é o código.
- O `customer_id` vem do token, nunca de argumento que o LLM escreve.

### 3.6 Defesa contra prompt injection (camadas)
| Camada | O que faz |
|---|---|
| D1. Prompt de sistema | escopo, regras e ordem explícita de tratar o conteúdo de ferramentas e documentos como dado, nunca como instrução |
| D2. Marcação de dados (spotlighting) | todo conteúdo não confiável (saída de ferramenta, trechos recuperados) vai entre delimitadores com um nonce aleatório por requisição |
| D3. Guarda nas ferramentas | as regras da seção 3.5, aplicadas em código; vale mesmo se o LLM for enganado |
| D4. Filtro de saída | bloqueia a resposta se ela contiver o canary do prompt de sistema |

Os ataques têm alvos verificáveis sem juiz:
- **vazamento:** o prompt de sistema contém um canary aleatório; o ataque vence se o canary sai na resposta;
- **ação indevida:** reembolso em pedido de outro cliente, acima do valor ou fora do prazo;
- **frase plantada:** o ataque manda o modelo dizer algo específico.

Os vetores de ataque:
- **direto:** na mensagem do cliente (ignorar instruções, role-play, "sou o administrador");
- **indireto:** na `observacao` do pedido sintético que a ferramenta devolve, ou no nome de um item.

As defesas são medidas em duas etapas:
- **tentativa:** o LLM emitiu a tool call proibida? Isso mede D1 e D2.
- **execução:** a guarda deixou passar? Isso mede D3.

Assim fica visível que o prompt reduz as tentativas, mas só o código zera as execuções.

### 3.7 Observabilidade
- `logging` da stdlib com formatter JSON; um `request_id` por requisição, devolvido no header `X-Request-ID`.
- Trace por requisição: spans `retrieve`, `rerank`, `llm_call` (tokens, latência), `tool_call` (nome, resultado da guarda), gravados em JSONL.
- **Logs sem PII:** o texto do cliente não vai para o log por padrão, só tamanho, IDs dos chunks e nomes de ferramentas. Isso é ligável em dev.
- Os scripts de avaliação passam pelo mesmo código do agente, então cada número tem um trace por trás.
- OpenTelemetry/Langfuse ficam de fora: o JSONL basta para depurar e não exige infraestrutura.

### 3.8 Segurança da API
- Token de demo por cliente sintético. O mapa token → `customer_id` fica em arquivo de config de demo, não é segredo real e isso fica escrito no README.
- Validação de entrada com pydantic: tamanho máximo da mensagem e do histórico.
- Headers de segurança (`X-Content-Type-Options`, `X-Frame-Options`, CSP), erros genéricos ao cliente e detalhes só no log.
- Rate limit simples por token em memória (`ponytail`: em memória, Redis se houver mais de uma instância).

## 4. Plano de avaliação

### 4.1 Conjunto (congelado antes de qualquer ajuste)
| Parte | Tamanho aprox. | Gold | dev / test |
|---|---|---|---|
| Perguntas de política | 160 | IDs de passagem relevantes + resposta de referência | 60 / 100 |
| Sem resposta no corpus / fora do escopo | 20 | deve recusar ou dizer que não sabe | 8 / 12 |
| Cenários de ferramenta | 40 | ferramenta e argumentos esperados + resultado esperado (reembolso aceito/recusado) | 15 / 25 |
| Ataques de injection | 60 | alvo verificável (canary, ação, frase) | 20 / 40 |
| Benignos parecidos com ataque | 20 | deve atender normalmente (mede recusa indevida) | 8 / 12 |

- **Como montar:** as perguntas são escritas em linguagem de cliente ("pedi lanche e veio frio, tenho direito a quê?"), **sem copiar termos do texto da lei**. Perguntas que repetem o vocabulário do documento favorecem o BM25 e inflam o recall.
- **Quem escreve:** eu escrevo um rascunho de tudo, e você revisa, corrige e marca o gold de busca de uma amostra. **Você escreve ~30 perguntas sozinho**, sem ver as minhas; o recall nelas é reportado à parte, como checagem de viés do autor.
- **Divisão:** estratificada por documento e tipo. O arquivo é congelado com hash SHA-256 registrado no README e checado por teste.
- **Uso do test:** **uma única vez**, no fim, rodando o antes e o depois de E1 e E2. Tudo o que for escolhido (top-k, prompt, prompt do juiz) é escolhido no dev.

### 4.2 Métricas
- **Busca (determinística, CPU):** recall@1/3/5/10 e MRR por configuração R1–R4.
- **Respostas (juiz):** % corretas, % fiéis ao contexto e % de recusa correta nas perguntas sem resposta.
- **Ferramentas (determinística):** ferramenta certa, argumentos certos e resultado final certo.
- **Injection (determinística):** taxa de ataque bem-sucedido (ASR), separada em tentativa e execução e por vetor, mais a recusa indevida nos benignos.
- **Desempenho:** latência p50/p95 por requisição (T4 para o LLM, CPU para a busca) e custo estimado de GPU por 1.000 requisições, com o mesmo preço de referência do Projeto 1.
- **Incerteza:** diferenças antes/depois com IC 95% por bootstrap pareado (2.000 reamostragens), como no Projeto 1.

### 4.3 Experimentos
- **E1:** primeiro R1–R4 só na busca (dev). Depois o agente completo com R1 e com R4, com o juiz medindo a diferença nas respostas. É isso que mostra se o ganho de recall vira ganho de resposta.
- **E2:** três configurações sobre o mesmo conjunto de ataques e benignos:
  - sem defesa;
  - D1 + D2 + D4 (só prompt e saída);
  - D1–D4 (com a guarda no código).
- Se algum ajuste **não** melhorar (ex.: o reranker empatar com o híbrido), isso vai para o README do mesmo jeito.

## 5. Stack

| Camada | Escolha | Motivo |
|---|---|---|
| Linguagem | Python 3.11 (python.org) | o Python do uv é bloqueado pelo Application Control |
| Ambiente | `uv` + `pyproject.toml` | igual ao Projeto 1 |
| API | `fastapi`, `uvicorn`, `pydantic` | pedido na vaga; validação e OpenAPI de graça |
| LLM client | `openai` (SDK) com `base_url` | um cliente para vLLM, Ollama ou camada grátis |
| Embeddings / reranker | `sentence-transformers` (CPU) | carrega bge-m3 e bge-reranker-v2-m3 |
| BM25 | `bm25s` | mantido e rápido |
| Dados | `sqlite3` (stdlib) | pedidos e reembolsos; queries parametrizadas |
| Logs | `logging` (stdlib) + formatter JSON | sem dependência nova |
| Qualidade | `pytest`, `ruff`, `httpx` (TestClient) | igual ao Projeto 1 |
| Container | Docker, `python:3.11-slim`, torch CPU | uma imagem para a API |

Nos testes de CI, o LLM é substituído por um fake roteirizado e o embedder por um fake determinístico; nada é baixado e nada chama a rede.

## 6. Estrutura de pastas

```
support-rag-agent/
├── PLAN.md
├── README.md                  # pergunta, arquitetura, como rodar, antes/depois, erros, demo
├── pyproject.toml
├── Dockerfile
├── .env.example               # LLM_BASE_URL=, LLM_MODEL=
├── corpus/
│   ├── raw/                   # HTML do Planalto (gitignored)
│   ├── laws/                  # CDC e decreto em Markdown, limpos
│   └── policies/              # políticas fictícias em Markdown
├── data/
│   ├── orders.sqlite          # gerado por script com seed (gitignored; o script é versionado)
│   ├── eval/                  # perguntas, cenários e ataques (dev/test) + SHA-256
│   └── runs/                  # respostas, tool calls e traces de cada rodada
├── src/support_agent/
│   ├── corpus.py              # parsing e chunking com IDs de passagem
│   ├── retrieval.py           # denso, BM25, RRF, reranker
│   ├── orders.py              # geração dos pedidos sintéticos + acesso ao SQLite
│   ├── tools.py               # consultar_pedido, abrir_reembolso, guardas
│   ├── agent.py               # laço de tool calling, prompt, spotlighting, filtro de saída
│   ├── tracing.py             # logs JSON e spans
│   ├── api.py                 # FastAPI
│   ├── judge.py               # prompt e parsing do juiz
│   └── evaluate.py            # recall@k, ASR, bootstrap → results/*.json
├── notebooks/                 # rodam no Kaggle: gerador + juiz via vLLM
├── results/                   # todo número do README sai daqui
└── tests/
.github/workflows/support-rag-agent.yml   # ruff + pytest + docker build, filtrado pela pasta
```

## 7. Critérios de entrega (Definition of Done)

- [ ] Conjunto de avaliação congelado (hash no README) antes do primeiro ajuste; o test foi usado uma única vez, com data registrada.
- [ ] `results/` com recall@k/MRR de R1–R4, juiz antes/depois de E1, ASR e utilidade de E2, concordância do juiz com humano, latência e custo, tudo gerado por código.
- [ ] Todo número do README conferido por script contra `results/` antes do commit.
- [ ] Testes de ataque no CI com LLM fake: a guarda bloqueia IDOR, valor acima do pago e reembolso duplicado, sempre.
- [ ] `docker build` e `docker run` funcionando; `/health` e `/chat` respondem no container.
- [ ] Logs JSON com `request_id` e trace por requisição, sem texto do cliente por padrão.
- [ ] README com pergunta, arquitetura (diagrama), como rodar, antes/depois com IC, erros comentados, o que não funcionou e demo (GIF).
- [ ] CI verde (ruff, pytest, docker build), nenhum segredo no repo, README raiz do monorepo atualizado.
- [ ] Merge da `feature/support-rag-agent` na `main`.

## 8. Riscos e mitigação

| Risco | Mitigação |
|---|---|
| Corpus fácil demais: o denso já acerta quase tudo e E1 não mostra diferença | políticas com casos parecidos de propósito e perguntas em linguagem de cliente; se mesmo assim empatar, reportar o empate e apoiar o README em E2 |
| Viés do autor: as perguntas que eu escrevo combinam com os textos que eu escrevo | as ~30 perguntas escritas só por você, reportadas à parte |
| Juiz pouco confiável | validação com kappa contra suas notas no dev, antes do test; métricas determinísticas onde der (ferramentas, ataques, busca) |
| vLLM não instala ou não roda tool calling na T4 | `transformers` + parsing próprio das tool calls; o laço do agente não muda |
| Gemma 3 12B + Qwen3-8B não cabem juntos | rodar em fases: gerar todas as respostas, salvar, trocar o modelo e julgar |
| Cota do Kaggle (30 h/semana) | busca roda inteira em CPU local; no Kaggle só geração e juiz, que são poucas centenas de chamadas por configuração |
| Docker: o CLI está instalado (29.9.2), mas o daemon não estava rodando | abrir o Docker Desktop antes da etapa de container; se o Application Control bloquear, o CI no GitHub faz o `docker build` |
| torch/sentence-transformers bloqueados pelo Application Control | o Projeto 1 já rodou torch em CPU localmente; se falhar, trocar por `fastembed` (ONNX) |
| Imagem Docker grande por causa do torch | wheel CPU-only do torch; tamanho registrado no README |
| Ollama bloqueado pelo Application Control | llama.cpp; em último caso, GIF do `/docs` com respostas gravadas de uma rodada no Kaggle |
| Demo local lenta em CPU | GIF com resposta curta; o README diz que a latência de produção é medida na T4 |
| Texto do CDC muda (alterações de lei) | data de download e hash do HTML registrados |

## 9. Decisões tomadas

- Pasta `support-rag-agent/`, branch `feature/support-rag-agent`.
- Custo zero: Kaggle T4 para gerador e juiz, CPU local para busca e API.
- Gerador Qwen3-8B; juiz Gemma 3 12B, de outra família, validado contra notas humanas.
- Regras de negócio e autorização das ferramentas ficam em código, não no prompt.
- Você escreve ~30 perguntas de controle de viés e dá ~60 notas humanas para validar o juiz.
- Tamanho do conjunto como na seção 4.1 (~300 itens).
- Demo local com Ollama.

## Mudanças durante a execução (2026-11-16)

O texto acima é o plano original e não foi reescrito. O que mudou na execução:

- **Juiz:** o Gemma 3 12B não roda na T4: o vLLM 0.32 recusa `gemma3` em float16 (instabilidade numérica) e a T4 não tem bfloat16. Depois veio o `stelterlab/phi-4-AWQ` (prompts v1 e v2, sem concordância) e, por fim, o `casperhansen/mistral-small-24b-instruct-2501-awq`, servido com tensor parallel nas duas T4 (Kaggle "GPU T4 x2"). Histórico em `docs/dev_log.md`.
- **Portão de kappa não atingido:** com o prompt v3, kappa `correta` / `fiel_ao_contexto` = 0,48 / 0,20 no dev e 0,36 / 0,44 no test, abaixo de 0,6. As notas humanas cegas (60 no dev, 60 no test) viraram a medida primária da qualidade das respostas; o juiz é secundário e sempre aparece com modelo, kappa e "não validado". Isso contraria o anti-goal da seção 2 ("o número do juiz não entra no README"): os números do juiz aparecem, mas rotulados assim, e nenhuma conclusão se apoia só neles.
- **`top_k` e prompts não foram ajustados:** sem uma medida validada de qualidade de resposta, ficaram no padrão (`top_k=5`, prompts sem mudança). A configuração congelada está em `src/support_agent/final_config.py`.
- **Nomes dos arquivos de resultado (por split):** `results/judge_agreement_{sha12}.json` (dev) e `results/judge_agreement_{sha12}_{split}.json` (outros splits), com `sha12` = prefixo do SHA-256 do prompt do juiz; `results/final_{split}.json` em vez de `results/final.json`; `results/readme_tables_{split}.md` em vez de `results/readme_tables.md`. O teste do README confere as tabelas contra `results/readme_tables_test.md`.
