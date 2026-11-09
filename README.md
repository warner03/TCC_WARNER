# TCC_WARNER — Monte Carlo × Rede Neural | Copa do Mundo FIFA 2026

Repositório da parte prática do TCC *Estudo Comparativo com Modelos Probabilístico e Preditivo em Cenários de Previsão de Campeonato de Futebol*.

## O princípio metodológico do projeto

1. **Histórico completo desde 1872.** `data/results.csv` e `data/former_names.csv` são as fontes da **base histórica**. `data/base_de_dadosTCC.csv.gz` é a base **original enviada pelo autor**, com **49.308 jogos**, de **1872-11-30 a 2026-06-09**. Ela é a entrada **padrão do treinamento** da RNA e não será substituída silenciosamente.
2. **Cada previsão usa somente o passado.** Os oito atributos numéricos são calculados **antes** da partida. O nome da seleção identifica o histórico correto, mas nunca faz parte do vetor de entrada dos modelos.
3. **Apenas placares reais alteram atributos.** O Elo Rating, a média de pontos e o saldo dos últimos cinco jogos são atualizados depois do jogo **com o placar observado**, nunca com probabilidades ou placares gerados pelos modelos. Sem horário confiável, jogos de uma mesma data usam o estado do início do dia.
4. **104 partidas, não só 72.** A tabela tem 72 jogos de grupos (identificados desde o `results.csv`) mais 32 jogos eliminatórios de numeração oficial M73–M104. Os oponentes eliminatórios são definidos **pelos resultados reais**; não são inventados quando ainda não há classificação.
5. **Duas avaliações distintas.** O teste cronológico da RNA na base histórica é separado da avaliação das partidas da Copa. As métricas da Copa são calculadas **somente nos jogos com placar real inserido**.

## Estrutura

```text
TCC_WARNER/
├── data/
│   ├── results.csv                     # resultados brutos desde 1872; 72 jogos da Copa sem placar
│   ├── former_names.csv                # nomes antigos de seleções
│   ├── base_de_dadosTCC.csv.gz         # 49.308 exemplos históricos originais (gzip, sem perda)
│   ├── copa_2026_104_jogos.csv          # 104 slots oficiais, com identificador FIFA
│   ├── annex_c_2026.csv                # 495 combinações FIFA para os oito terceiros
│   ├── placares_reais_2026_MODELO.csv   # estrutura para inserir resultados reais
│   ├── classificacao_oficial_MODELO.csv # opção de desempate (cartões/ranking)
│   └── README.md
├── tcc_warner/
│   ├── core.py                         # Elo desde 1872, últimas cinco partidas e dataset
│   ├── calendar.py                     # datas, 72 grupos + 32 mata-mata
│   ├── annex.py                        # validação das 495 combinações do Anexo C
│   ├── bracket.py                      # classificação e propagação de vencedores reais
│   ├── models.py                       # Monte Carlo / Poisson e inferência da RNA
│   ├── neural.py                       # treinamento Keras, split temporal 90/10
│   ├── full_workflow.py                # replay sequencial dos 104 jogos
│   ├── workflow.py                     # métricas Brier, log-loss e acurácia
│   └── cli.py                          # comandos do terminal
├── tests/
│   ├── test_pipeline.py
│   └── test_full_cup.py
├── requirements.txt
├── requirements-ml.txt
└── README.md
```

## Instalação — Fedora/Linux

No terminal, dentro do repositório:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Para **treinar e carregar a Rede Neural** é necessário instalar também o TensorFlow:

```bash
pip install -r requirements-ml.txt
```

### 1. Conferir o histórico e reconstruir os atributos (opcional)

```bash
python -m tcc_warner.cli prepare
```

Produz `outputs/base_recalculada.csv` desde **1872**, reproduzindo as 49.308 linhas e a ordem da base original. O treinamento **não depende** dessa reconstrução: por padrão, lê `data/base_de_dadosTCC.csv.gz`.

**Reprodutibilidade:** o script original ordenava o histórico duas vezes com `sort_values` padrão, o que define uma ordem específica para partidas na mesma data. A reconstrução mantém essa ordem, para preservar exatamente o Elo e demais atributos da base entregue. Na Copa, sem informação confiável de horário, as previsões de um mesmo dia usam atributos do **início do dia**, antes de atualizar qualquer partida daquele dia.

### 2. Gerar ou conferir o calendário dos 104 jogos

```bash
python -m tcc_warner.cli calendar
```

Produz/atualiza `data/copa_2026_104_jogos.csv` (os IDs 1 a 104 são os da FIFA). Em **três partidas** (`M51`, `M53`, `M59`), o arquivo `results.csv` registra `home_team` e `away_team` na ordem inversa de uma versão externa da tabela: o projeto preserva **a orientação do CSV enviado pelo autor**, incluindo a posição dos placares e o indicador de mando, para que o código não troque a seleção A pela B.

### 3. Fazer as previsões do Monte Carlo

```bash
python -m tcc_warner.cli cup
```

Produz `outputs/previsoes_copa_104.csv`, com 104 linhas. Se nenhum placar real for fornecido, faz **72 previsões** de grupos e mantém os **32 jogos eliminatórios pendentes**, sem fabricar os classificados. O Monte Carlo continua utilizando os parâmetros heurísticos definidos no TCC: Poisson; λ-base 1,35; Elo; diferença recente de pontos; saldo dos últimos cinco; +0,05 de vantagem de mando para a equipe A quando `Campo_Neutro=0`; 10.000 amostras por partida. A variável `Peso_Jogo=7` é aplicada às atualizações reais de Elo, mas não diretamente ao λ.

### 4. Treinar a Rede Neural com a base ORIGINAL

```bash
python -m tcc_warner.cli train
```

Lê por padrão `data/base_de_dadosTCC.csv.gz` (a base original de treinamento em formato gzip; o pandas descompacta automaticamente) — **não** `outputs/base_recalculada.csv`. Utiliza 8 atributos, MLP 64–32–3 com ReLU, Dropout 0,2 e Softmax; separação cronológica aproximada de 90% para desenvolvimento e 10% para teste. A divisão de desenvolvimento ainda separa validação temporal, e o `StandardScaler` é ajustado somente na parte de treinamento (sem vazamento no teste). Os pesos de classe auxiliam o tratamento do desbalanceamento de empates.

Salva os artefatos de treinamento em `outputs/modelo/` e relata **acurácia, log-loss e Brier Score multiclasse** do conjunto histórico de teste. Requer TensorFlow instalado e execução do comando — **nenhum valor foi inventado**.

### 5. Preencher os placares REAIS da Copa e comparar os dois métodos

Faça uma cópia de `data/placares_reais_2026_MODELO.csv` como `data/placares_reais_2026.csv` e preencha apenas partidas confirmadas. Colunas:

| Campo | Significado |
|---|---|
| `match_id` | Identificador FIFA do jogo (1–104) |
| `home_score_90` | Gols reais da seleção A nos **90 minutos**, sem prorrogação |
| `away_score_90` | Gols reais da seleção B nos **90 minutos**, sem prorrogação |
| `winner` | Nos jogos eliminatórios terminados empatados após 90 minutos, vencedor real da classificação (`A`, `B` ou nome da seleção). Nos outros casos, pode ficar vazio |

**Regra importante:** Para o cálculo das três classes (vitória A, empate, vitória B), utilize **somente os 90 minutos**. Se uma partida eliminatória empatar nos 90 minutos, informe em `winner` qual seleção **realmente avançou** após prorrogação/pênaltis. O `winner` serve exclusivamente para conectar o chaveamento; não modifica o rótulo de empate nos 90 minutos.

Com o modelo treinado:

```bash
python -m tcc_warner.cli cup \
  --real-scores data/placares_reais_2026.csv \
  --model outputs/modelo/modelo_tcc_warner.keras \
  --scaler outputs/modelo/scaler_tcc_warner.joblib
```

Sem a RNA (só Monte Carlo):

```bash
python -m tcc_warner.cli cup --real-scores data/placares_reais_2026.csv
```

O sistema usa as **495 combinações oficiais do Anexo C** para definir o adversário de cada campeão de grupo que enfrenta um terceiro colocado, sem arbitrariedade. Depois usa as referências `W73...W102` e `L101/L102` para criar os jogos seguintes. Jogos com participantes ainda desconhecidos permanecem pendentes. Os únicos placares que atualizam Elo e formas recentes são os **reais e conhecidos**, depois de gerar as previsões da data.

**Desempates:** a classificação dentro do grupo aplica pontos e confronto direto antes dos demais critérios. Quando persistir empate exigindo cartões e Ranking FIFA, o sistema **para com erro explicativo** em vez de inventar um desempate. Pode-se criar `data/classificacao_oficial.csv` seguindo `data/classificacao_oficial_MODELO.csv` e passar `--ranking-override data/classificacao_oficial.csv` com a ordem oficial do grupo afetado. Também é necessário resolver oficialmente empates no corte entre o 8º e o 9º terceiro (pontos, saldo e gols iguais).

As métricas da Copa (`outputs/metricas_copa.csv`) são **acurácia, log-loss e Brier Score multiclasse**, apenas para jogos com resultado real. Não confundir com as métricas de teste da RNA em partidas anteriores à Copa.

### 6. Testar

```bash
pytest -q
```

Os testes incluem um calendário completo com 104 partidas, validação das 495 linhas do Anexo C, histórico desde 1872, bloqueio de vazamento de resultados, atualização dinâmica após jogos reais e propagação do chaveamento até a final com **placares sintéticos apenas no ambiente de testes**. Nenhum desses placares sintéticos aparece no CSV de resultados reais.

## Limites e decisões para a análise científica

- **A experiência é observacional/replay:** não existe simulação de campeões da Copa baseada em resultados inventados. O Monte Carlo simula *placares de uma partida para estimar probabilidades*, não os jogos seguintes para atualizar o estado.
- **Ajuste do Elo:** inicial 1500; `K=20×Peso_Jogo`, conforme o código do autor, com 7 para Copa. Esta é uma escolha heurística a justificar no texto científico.
- **Recorte histórico:** a base de aprendizado começa em **1872**, não em 2006. O estado das seleções antes da Copa também considera o histórico completo.
- **Mando de campo:** nos 72 jogos, usa-se `neutral` do `results.csv` original; nas fases eliminatórias o calendário pressupõe campo neutro até revisão da vantagem efetiva de anfitriões, o que merece análise de sensibilidade.
- **Limite dos dados:** os arquivos enviados não contêm os placares reais dos jogos da Copa; nenhum foi importado automaticamente de sites externos. Para obter avaliação com métricas da Copa, é preciso um registro confiável dos resultados reais nos **90 minutos**. O treinamento TensorFlow também não foi executado neste ambiente.
- **Resultados históricos do mesmo dia:** a base original tinha uma ordem de registros sem horário. Ela foi mantida para preservar 100% da base de treinamento, mas o estudo deve explicitar essa limitação cronológica na redação.
- **Empates eliminatórios:** o vencedor definido por prorrogação ou pênaltis não altera a classe observada do jogo nos 90 minutos.

## Fontes para validação de datas e chaveamento (não para importar resultados)

- [FIFA — Calendário dos 104 jogos](https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/match-schedule-fixtures-results-teams-stadiums)
- [FIFA — Regulamento 2026, artigos 12–13 e Anexo C](https://digitalhub.fifa.com/m/636f5c9c6f29771f/original/FWC2026_regulations_EN.pdf)
- [Tabela estruturada de terceiros (495 combinações, referência cruzada)](https://github.com/manganite/wm2026/blob/main/thirdPlaceAssignments.mjs)
- [Lista independente de jogos/IDs usada como conferência](https://github.com/mjwebmaster/world-cup-2026-schedule-data)

**Autor:** Warner Pereira Cardoso Júnior — Ciência da Computação, UFJ, 2026.