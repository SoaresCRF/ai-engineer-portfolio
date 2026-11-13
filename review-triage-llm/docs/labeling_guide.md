# Guia de rotulagem

Arquivo: `data/labels/gold_labeling.xlsx` (300 avaliações). Preencha as colunas **categoria**, **sentimento** e **urgencia** usando as listas suspensas.

**Regras do jogo**
- Rotule só pelo texto. A nota e os dados de entrega ficam escondidos de propósito.
- Não consulte nenhum LLM. A comparação só vale se o gold for humano.
- Gaste ~15 s por linha. Se ficar em dúvida, aplique as regras de desempate abaixo e siga em frente.
- Não altere o texto nem a ordem das linhas.

Ao terminar, rode `uv run python -m review_triage.labeling import`.

---

## categoria: o problema principal

| Valor | Quando usar | Exemplos reais |
|-------|-------------|----------------|
| `atraso_entrega` | chegou, mas depois do prazo, ou demorou | "o produto foi entregue com atraso"; "demorou no correio para entregar" |
| `nao_recebido` | ainda não chegou, nada chegou ou passou do prazo sem entrega | "Simplesmente não recebi o produto"; "estou no aguardo, ainda do produto" |
| `produto_errado_ou_incompleto` | item, cor, voltagem ou tamanho diferente do pedido; faltou item ou peça | "Produto errado"; "Comprei 5 produtos porém só veio 3"; "O carregador não veio" |
| `defeito_ou_qualidade` | quebrado, não funciona, qualidade ruim, diferente do anunciado em qualidade | "Nao funcionou, parece queimado"; "Qualidade do produto é ruim" |
| `atendimento_vendedor` | problema com loja ou vendedor: não responde, sem contato, cobrança, cancelamento, estorno, troca difícil | "site não disponibiliza 0800 para ligar pedindo estorno"; "me vendeu uma mochila q não tinha no estoque" |
| `elogio_sem_problema` | elogio ou comentário positivo/neutro sem nenhum problema | "Ótimo"; "chegou antes do prazo, recomendo" |
| `outro` | arrependimento ("não gostei da fragrância"), dúvida, texto sem sentido ou que não encaixa | "Só não gostei das fragrâncias" |

**Desempates**
1. **Vários problemas:** vale o mais grave, nesta ordem: `nao_recebido` > `produto_errado_ou_incompleto` > `defeito_ou_qualidade` > `atendimento_vendedor` > `atraso_entrega`.
2. **Elogio com ressalva** ("ótimo, mas demorou"): use a categoria da ressalva.
3. **Recebeu só parte do pedido:** `produto_errado_ou_incompleto`, não `nao_recebido`.
4. **Recebeu algo de uma compra cancelada:** `atendimento_vendedor`.

## sentimento: o tom geral do texto

| Valor | Quando usar |
|-------|-------------|
| `positivo` | satisfeito no geral, mesmo com uma ressalva leve ("ótimo, só demorou um pouco") |
| `neutro` | informativo, misto equilibrado ou sem emoção clara ("Ok", "estou no aguardo") |
| `negativo` | insatisfeito, reclamando |

## urgencia: precisa de ação da empresa?

| Valor | Quando usar |
|-------|-------------|
| `alta` | não recebeu; pede reembolso, estorno, troca ou devolução; menciona Procon, justiça ou Reclame Aqui; cobrança indevida |
| `media` | tem problema, mas não pede ação explícita ("veio na cor errada", "demorou") |
| `baixa` | sem problema a resolver (elogio, comentário neutro, arrependimento leve) |

---

## Exemplos completos

| Texto | categoria | sentimento | urgencia |
|-------|-----------|------------|----------|
| "Já passou 3 dias e o produto não foi entregue." | nao_recebido | negativo | alta |
| "APENAS UM DOS PRODUTOS FOI ENTREGUE COM DEFEITO. COMUNIQUEI AO VENDEDOR E ESTOU HÁ ESPERA DO CONTATO PARA EFETUAR A TROCA" | defeito_ou_qualidade | negativo | alta |
| "Gostei da mercadoria, chegou rápido, mas não veio na voltagem certa" | produto_errado_ou_incompleto | neutro | media |
| "O produto é maravilhoso, recebi tudo direitinho. Só achei demorar um pouco a entrega" | atraso_entrega | positivo | baixa |
| "Pedi a devolução, pois o produto veio errado" | produto_errado_ou_incompleto | negativo | alta |
| "Excelente, recebi um dia antes do prazo" | elogio_sem_problema | positivo | baixa |
| "Só não gostei das fragrâncias, bem enjoativas e doces." | outro | negativo | baixa |

> No 4º exemplo, a ressalva é leve e não pede ação. Por isso o sentimento é positivo e a urgência é baixa, mesmo com a categoria de atraso.

## Checagem de consistência

Depois do import, uma planilha extra com 30 dessas avaliações, sem os seus rótulos, será gerada para você rotular de novo. A taxa de acordo entre as duas rodadas entra no README como concordância consigo mesmo.
