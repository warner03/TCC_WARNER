"""Calendário de 104 partidas: 72 jogos já conhecidos + chave FIFA 73-104.

Os identificadores FIFA (1..104) são a chave estável dos confrontos. Os
nomes são apenas metadados de identidade, não variáveis dos modelos.
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path

# match_id|grupo|data|seleção A|seleção B (nomes conforme results.csv)
# Fonte de conferência: calendário FIFA 2026, versão publicada 2026-03-31.
_GROUPS = '''
1|A|2026-06-11|Mexico|South Africa
2|A|2026-06-11|South Korea|Czech Republic
3|B|2026-06-12|Canada|Bosnia and Herzegovina
4|D|2026-06-12|United States|Paraguay
5|C|2026-06-13|Haiti|Scotland
6|D|2026-06-13|Australia|Turkey
7|C|2026-06-13|Brazil|Morocco
8|B|2026-06-13|Qatar|Switzerland
9|E|2026-06-14|Ivory Coast|Ecuador
10|E|2026-06-14|Germany|Curaçao
11|F|2026-06-14|Netherlands|Japan
12|F|2026-06-14|Sweden|Tunisia
13|H|2026-06-15|Saudi Arabia|Uruguay
14|H|2026-06-15|Spain|Cape Verde
15|G|2026-06-15|Iran|New Zealand
16|G|2026-06-15|Belgium|Egypt
17|I|2026-06-16|France|Senegal
18|I|2026-06-16|Iraq|Norway
19|J|2026-06-16|Argentina|Algeria
20|J|2026-06-16|Austria|Jordan
21|L|2026-06-17|Ghana|Panama
22|L|2026-06-17|England|Croatia
23|K|2026-06-17|Portugal|DR Congo
24|K|2026-06-17|Uzbekistan|Colombia
25|A|2026-06-18|Czech Republic|South Africa
26|B|2026-06-18|Switzerland|Bosnia and Herzegovina
27|B|2026-06-18|Canada|Qatar
28|A|2026-06-18|Mexico|South Korea
29|C|2026-06-19|Brazil|Haiti
30|C|2026-06-19|Scotland|Morocco
31|D|2026-06-19|Turkey|Paraguay
32|D|2026-06-19|United States|Australia
33|E|2026-06-20|Germany|Ivory Coast
34|E|2026-06-20|Ecuador|Curaçao
35|F|2026-06-20|Netherlands|Sweden
36|F|2026-06-20|Tunisia|Japan
37|H|2026-06-21|Uruguay|Cape Verde
38|H|2026-06-21|Spain|Saudi Arabia
39|G|2026-06-21|Belgium|Iran
40|G|2026-06-21|New Zealand|Egypt
41|I|2026-06-22|Norway|Senegal
42|I|2026-06-22|France|Iraq
43|J|2026-06-22|Argentina|Austria
44|J|2026-06-22|Jordan|Algeria
45|L|2026-06-23|England|Ghana
46|L|2026-06-23|Panama|Croatia
47|K|2026-06-23|Portugal|Uzbekistan
48|K|2026-06-23|Colombia|DR Congo
49|C|2026-06-24|Scotland|Brazil
50|C|2026-06-24|Morocco|Haiti
51|B|2026-06-24|Switzerland|Canada
52|B|2026-06-24|Bosnia and Herzegovina|Qatar
53|A|2026-06-24|Czech Republic|Mexico
54|A|2026-06-24|South Africa|South Korea
55|E|2026-06-25|Curaçao|Ivory Coast
56|E|2026-06-25|Ecuador|Germany
57|F|2026-06-25|Japan|Sweden
58|F|2026-06-25|Tunisia|Netherlands
59|D|2026-06-25|Turkey|United States
60|D|2026-06-25|Paraguay|Australia
61|I|2026-06-26|Norway|France
62|I|2026-06-26|Senegal|Iraq
63|G|2026-06-26|Egypt|Iran
64|G|2026-06-26|New Zealand|Belgium
65|H|2026-06-26|Cape Verde|Saudi Arabia
66|H|2026-06-26|Uruguay|Spain
67|L|2026-06-27|Panama|England
68|L|2026-06-27|Croatia|Ghana
69|J|2026-06-27|Algeria|Austria
70|J|2026-06-27|Jordan|Argentina
71|K|2026-06-27|Colombia|Portugal
72|K|2026-06-27|DR Congo|Uzbekistan
'''.strip()

# 1A = campeão grupo A; 2A = vice grupo A; 3:A=terceiro do A,
# T1A = terceiro correspondente ao adversário do campeão A no Anexo C.
# W89/L101 = vencedor/perdedor da partida indicada.
_KNOCKOUT = '''
73|2026-06-28|16avos|2A|2B
74|2026-06-29|16avos|1E|T1E
75|2026-06-29|16avos|1F|2C
76|2026-06-29|16avos|1C|2F
77|2026-06-30|16avos|1I|T1I
78|2026-06-30|16avos|2E|2I
79|2026-06-30|16avos|1A|T1A
80|2026-07-01|16avos|1L|T1L
81|2026-07-01|16avos|1D|T1D
82|2026-07-01|16avos|1G|T1G
83|2026-07-02|16avos|2K|2L
84|2026-07-02|16avos|1H|2J
85|2026-07-02|16avos|1B|T1B
86|2026-07-03|16avos|1J|2H
87|2026-07-03|16avos|1K|T1K
88|2026-07-03|16avos|2D|2G
89|2026-07-04|oitavas|W74|W77
90|2026-07-04|oitavas|W73|W75
91|2026-07-05|oitavas|W76|W78
92|2026-07-05|oitavas|W79|W80
93|2026-07-06|oitavas|W83|W84
94|2026-07-06|oitavas|W81|W82
95|2026-07-07|oitavas|W86|W88
96|2026-07-07|oitavas|W85|W87
97|2026-07-09|quartas|W89|W90
98|2026-07-10|quartas|W93|W94
99|2026-07-11|quartas|W91|W92
100|2026-07-11|quartas|W95|W96
101|2026-07-14|semifinais|W97|W98
102|2026-07-15|semifinais|W99|W100
103|2026-07-18|terceiro_lugar|L101|L102
104|2026-07-19|final|W101|W102
'''.strip()


def create_calendar(historical_matches: pd.DataFrame) -> pd.DataFrame:
    """Confere 72 jogos com results.csv e acrescenta 32 cruzamentos condicionais."""
    cup = historical_matches.loc[
        historical_matches.tournament.eq('FIFA World Cup') &
        historical_matches.date.eq(pd.Timestamp('2026-06-11')) |
        (historical_matches.tournament.eq('FIFA World Cup') &
         historical_matches.date.between('2026-06-12', '2026-06-27'))
    ].copy()
    if len(cup) != 72:
        raise ValueError(f"Esperados 72 jogos de grupos em results.csv; encontrados {len(cup)}")
    key_columns = ['date', 'home_team', 'away_team']
    if cup.duplicated(key_columns).any():
        raise ValueError('Duplicação dos jogos da fase de grupos na base')
    entries = []
    for line in _GROUPS.splitlines():
        i, group, d, home, away = line.split('|')
        matched = cup[(cup.date == pd.Timestamp(d)) &
                      (((cup.home_team == home) & (cup.away_team == away)) |
                       ((cup.home_team == away) & (cup.away_team == home)))]
        if len(matched) != 1:
            raise ValueError(f'Calendário FIFA não encontrado em results.csv: {i}: {d} {home} x {away}')
        game = matched.iloc[0]
        # Para manter as colunas A/B e os placares alinhados à base do autor,
        # prevalece a orientação home/away do results.csv em 3 confrontos
        # nos quais a tabela externa apresenta ordem inversa.
        entries.append({'match_id': int(i), 'date': d, 'fase': 'grupos', 'grupo': group,
                        'home_slot': game.home_team, 'away_slot': game.away_team,
                        'Campo_Neutro': int(game.neutral)})
    for line in _KNOCKOUT.splitlines():
        i, date, stage, home, away = line.split('|')
        entries.append({'match_id': int(i), 'date': date, 'fase': stage, 'grupo': '',
                        'home_slot': home, 'away_slot': away, 'Campo_Neutro': 1})
    df = pd.DataFrame(entries).sort_values('match_id').reset_index(drop=True)
    assert df.match_id.tolist() == list(range(1, 105))
    return df


def load_calendar(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    needed = {'match_id', 'date', 'fase', 'grupo', 'home_slot', 'away_slot', 'Campo_Neutro'}
    if not needed <= set(df.columns):
        raise ValueError(f'Calendário sem colunas: {sorted(needed-set(df.columns))}')
    df['match_id'] = pd.to_numeric(df.match_id, errors='raise').astype(int)
    if sorted(df.match_id.tolist()) != list(range(1, 105)) or len(df) != 104:
        raise ValueError('Calendário deve ter identificadores FIFA 1..104 sem repetições')
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    return df.sort_values(['date', 'match_id']).reset_index(drop=True)