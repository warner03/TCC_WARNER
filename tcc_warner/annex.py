"""Leitura e validação do Anexo C da FIFA (495 combinações de oito terceiros)."""
from __future__ import annotations
from itertools import combinations
from pathlib import Path
import pandas as pd

WINNERS = ('A', 'B', 'D', 'E', 'G', 'I', 'K', 'L')
ALLOWED = {'A': 'CEFHI', 'B': 'EFGIJ', 'D': 'BEFIJ', 'E': 'ABCDF',
           'G': 'AEHIJ', 'I': 'CDFGH', 'K': 'DEIJL', 'L': 'EHIJK'}


def load_annex(path: str | Path) -> dict[str, dict[str, str]]:
    """Chave: conjunto ordenado dos grupos qualificados; valor: campeão -> terceiro."""
    table = pd.read_csv(path, dtype=str)
    if not {'grupos_terceiros', *WINNERS} <= set(table.columns):
        raise ValueError('Anexo C incompleto; são necessárias as colunas grupos_terceiros,A,B,D,E,G,I,K,L')
    result = {}
    for row in table.to_dict('records'):
        key = ''.join(sorted(str(row['grupos_terceiros'])))
        value = {winner: str(row[winner]) for winner in WINNERS}
        if (len(key) != 8 or len(set(key)) != 8 or any(v not in key for v in value.values())
            or len(set(value.values())) != 8 or any(value[w] not in ALLOWED[w] for w in WINNERS)):
            raise ValueError(f'Linha inválida no Anexo C: {key}')
        if key in result:
            raise ValueError(f'Anexo C duplicado: {key}')
        result[key] = value
    expected = {''.join(c) for c in combinations('ABCDEFGHIJKL', 8)}
    if set(result) != expected:
        raise ValueError(f'Anexo C com {len(result)} combinações; necessárias 495, sem lacunas')
    return result