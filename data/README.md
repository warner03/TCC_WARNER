# Bases e rastreabilidade

- `results.csv`: arquivo original enviado pelo autor, inclui jogos de 1872 a 2026 (e os 72 jogos de grupos da Copa SEM placar). Na Copa, só placares reais preenchem os resultados.
- `former_names.csv`: padronização original de nomes antigos das seleções.
- `base_de_dadosTCC.csv.gz`: base original de treinamento compactada **sem perda**, 49.308 linhas de 1872-11-30 a 2026-06-09. **Não modificar ao treinar.** Pode-se reconstruir uma cópia via `python -m tcc_warner.cli prepare`.
- `copa_2026_104_jogos.csv`: tabela completa dos 104 confrontos identificados por `match_id` de 1 a 104; na fase final os nomes são referências condicionais (`1A`, `2B`, `T1E`, `W73`, `L101`).
- `annex_c_2026.csv`: 495 combinações do regulamento oficial 2026 (colunas A B D E G I K L indicam o grupo de origem do terceiro oponente de cada campeão), para resolver os cruzamentos sem adivinhação.
- `placares_reais_2026_MODELO.csv`: **cabeçalho apenas**; copiar e preencher com fatos verificados, sem placares simulados.
- `classificacao_oficial_MODELO.csv`: cabeçalho para desempates que exijam cartões e ranking FIFA, não disponíveis na base de gols.

Não misture o conjunto histórico usado para treinar a RNA com placares da Copa que serão utilizados como avaliação independente. Confronte as fontes externas e registre a origem e data da coleta em seu relatório.

Para obter CSV descompactado: `gzip -dkc data/base_de_dadosTCC.csv.gz > data/base_de_dadosTCC.csv`. O Pandas lê `.csv.gz` diretamente, sem esta etapa.