# TCC_WARNER — Monte Carlo × Rede Neural na Copa de 2026

Projeto de pesquisa: **Estudo Comparativo com Modelos Probabilístico e Preditivo em Cenários de Previsão de Campeonato de Futebol**.

## Regra metodológica mais importante

**O Elo Rating, a média de pontos e o saldo dos últimos cinco jogos são atualizados SOMENTE com placares reais.** O Monte Carlo amostra gols para estimar probabilidades, mas esses gols **nunca** atualizam as seleções. A RNA prevê as probabilidades de empate (0), vitória da equipe A (1) e vitória da equipe B (2), e também **nunca** atualiza o histórico com suas previsões.

Os nomes só identificam internamente quais estados atualizar; **não entram no vetor de oito características** de nenhum modelo. Para datas sem horário, todas as partidas do mesmo dia são previstas com o mesmo estado pré-dia; os placares reais são aplicados ao final do dia. Assim, nenhum confronto utiliza informação de um jogo ainda não observado.

## Arquivos e estrutura

- `tcc_warner/core.py`: leitura, normalização, Elo, estado dos cinco jogos e dataset cronológico.
- `tcc_warner/models.py`: Monte Carlo/Poisson e inferência da RNA.
- `tcc_warner/neural.py`: treinamento da MLP Keras 64→32→3, Dropout 0,2.
- `tcc_warner/workflow.py`: previsões da Copa, atualização com resultados reais, métricas.
- `tcc_warner/cli.py`: comandos do terminal.
- `tests/test_pipeline.py`: testes de invariantes cronológicas e vazamento temporal.
- `data/`: veja `data/README.md` para carregar `results.csv`, `former_names.csv` e placares reais.

## Instalação (Fedora/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
# Para treinar/prever com a RNA Keras:
pip install -r requirements-ml.txt
```

## Executar

1. Coloque em `data/` os arquivos originais `results.csv` e `former_names.csv` enviados para este projeto. Dados da Copa sem placar são considerados **não conhecidos**, e não como empate.
2. Gere os exemplos de treino (Elo calculado em sequência desde os jogos históricos; só exemplos de 2006 em diante):

```bash
python -m tcc_warner.cli prepare
```

3. Treine a RNA (TensorFlow necessário):

```bash
python -m tcc_warner.cli train
```

4. Faça previsões do torneio. Com apenas `results.csv`, partidas sem placar geram previsões, mas **não** atualizam estado:

```bash
python -m tcc_warner.cli cup
```

5. Quando tiver os **resultados reais confirmados**, crie `data/real_scores_2026.csv` com o cabeçalho e placares (um jogo por linha) e execute:

```bash
python -m tcc_warner.cli cup --real-scores data/real_scores_2026.csv --model outputs/modelo/modelo_tcc_warner.keras --scaler outputs/modelo/scaler_tcc_warner.joblib
```

O programa reprocessa cronologicamente todo o histórico para que o estado de cada seleção dependa apenas dos jogos reais anteriores e exporta `outputs/previsoes_copa.csv`. Apenas jogos com resultado real são usados nas métricas (acurácia, log-loss, Brier). Em Copa, mede-se exclusivamente **tempo regulamentar**: não confundir classificação nos pênaltis com a classe vitória.

6. Testes:

```bash
pytest -q
```

## Delimitações e cuidados científicos

- A base original enviada contém **72 confrontos da Copa 2026 sem placares**, até 27/06/2026. Não há resultados reais da Copa nesse arquivo: para uma avaliação retrospectiva, complete o arquivo complementar apenas com fontes verificadas. Jogos do mata-mata não presentes na tabela original precisam ser incluídos ao calendário original antes de rodar o replay.
- O arquivo histórico inclui jogos desde 1872, empregados para aquecer o Elo. A base **de aprendizado** inicia em 2006, conforme o escopo do TCC. A parametrização histórica do Elo é a do projeto original: Elo inicial 1500, K=20 multiplicado pelo peso do torneio; os pesos são heurísticos, e devem ser justificados no texto do TCC.
- Monte Carlo usa taxa-base heurística 1,35 gol/equipe, ajuste de razão Elo, média de pontos, saldo e vantagem de mandante (+0,05 para equipe A se o campo não for neutro), com λ restrito a [0,15; 4,0]. Os limites são hipótese de implementação a validar, não um resultado estimado.
- A RNA usa split temporal de ~90% treino/10% teste e reserva a parte final do treino para validação, ajustando o StandardScaler **somente no treino de ajuste**. Balancear classes altera a calibração das probabilidades: acurácia sozinha não basta; interpretar log-loss/Brier com cautela e considerar calibração posterior.
- Os modelos partilham as mesmas oito variáveis de entrada, mas o Monte Carlo é heurístico e não usa explicitamente `Peso_Jogo` na função λ (ele é usado na atualização real do Elo). Essa diferença deve ser documentada no TCC.
- Previsões de confrontos futuros não alteram o histórico; só os placares de jogos reais, inseridos pelo pesquisador, podem fazê-lo. **Não há simulação de tabela mata-mata com resultados inventados.**
- As métricas do modelo no conjunto de teste cronológico durante o treinamento e as métricas do torneio são conjuntos de avaliação distintos; não confundi-los. O último deve ser mantido fora do treinamento.