"""Classificação da Copa 2026 baseada SOMENTE nos placares oficiais inseridos.

O desempate entre seleções com mesmos pontos começa por confronto direto,
conforme art. 13 do regulamento da FIFA. Sem cartões/Ranking FIFA, uma
igualdade persistente é sinalizada, e não resolvida arbitrariamente.
"""
from __future__ import annotations
from collections import defaultdict
import pandas as pd


class UnresolvedRanking(ValueError):
    """Desempate que necessita de cartões, ranking FIFA ou ordem verificada."""


def _game_stats(games: list[dict], teams: set[str] | None = None) -> dict:
    data = defaultdict(lambda: {'pts': 0, 'gf': 0, 'ga': 0, 'jogos': 0})
    for g in games:
        a, b = g['home_team'], g['away_team']
        if teams is not None and (a not in teams or b not in teams):
            continue
        ga, gb = int(g['home_score']), int(g['away_score'])
        data[a]['gf'] += ga; data[a]['ga'] += gb; data[a]['jogos'] += 1
        data[b]['gf'] += gb; data[b]['ga'] += ga; data[b]['jogos'] += 1
        if ga > gb:
            data[a]['pts'] += 3
        elif ga < gb:
            data[b]['pts'] += 3
        else:
            data[a]['pts'] += 1; data[b]['pts'] += 1
    for v in data.values():
        v['sg'] = v['gf'] - v['ga']
    return dict(data)


def _group_by(items: list[str], score_fn) -> list[list[str]]:
    groupings = defaultdict(list)
    for team in items:
        groupings[score_fn(team)].append(team)
    return [groupings[k] for k in sorted(groupings, reverse=True)]


def _separate_tie(tied: list[str], games: list[dict], overall: dict) -> list[str]:
    """Confronto direto primeiro, reavaliando se só parte do empate foi desfeita."""
    if len(tied) == 1:
        return tied
    h2h = _game_stats(games, set(tied))
    keys = lambda t: (h2h[t]['pts'], h2h[t]['sg'], h2h[t]['gf'])
    separated = _group_by(tied, keys)
    if len(separated) > 1:
        return [t for block in separated for t in _separate_tie(block, games, overall)]
    # Confronto direto não desfez esta igualdade; critérios de todo o grupo.
    separated = _group_by(tied, lambda t: (overall[t]['sg'], overall[t]['gf']))
    if any(len(block) > 1 for block in separated):
        raise UnresolvedRanking('Desempate exige cartões/Ranking FIFA ou classificação oficial')
    return [t for block in separated for t in block]


def group_standings(calendar: pd.DataFrame, played: dict[int, dict], overrides: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    """Retorna 1º a 4º por grupo somente após seus seis jogos reais."""
    result = {}
    for letter in 'ABCDEFGHIJKL':
        fixture = calendar[(calendar.fase == 'grupos') & (calendar.grupo == letter)]
        if len(fixture) != 6:
            raise ValueError(f'Grupo {letter} tem {len(fixture)} jogos; esperado 6')
        ids = fixture.match_id.tolist()
        if not all(i in played for i in ids):
            continue
        games = [played[i] for i in ids]
        standings = _game_stats(games)
        if len(standings) != 4 or any(x['jogos'] != 3 for x in standings.values()):
            raise ValueError(f'Jogos inválidos no grupo {letter}')
        try:
            blocks = _group_by(list(standings), lambda team: standings[team]['pts'])
            ordered = [team for block in blocks for team in _separate_tie(block, games, standings)]
        except UnresolvedRanking as exc:
            if not overrides or letter not in overrides:
                raise UnresolvedRanking(f'Grupo {letter}: {exc}') from exc
            ordered = overrides[letter]
            if set(ordered) != set(standings) or len(ordered) != 4:
                raise ValueError(f'Classificação oficial inválida para grupo {letter}')
        result[letter] = ordered
    return result


def best_thirds(calendar: pd.DataFrame, played: dict[int, dict], standings: dict[str, list[str]]) -> list[str] | None:
    """Grupos dos 8 melhores terceiros. Empate no corte exige dado oficial."""
    if len(standings) != 12:
        return None
    ranked = []
    for letter in 'ABCDEFGHIJKL':
        games = [played[int(i)] for i in calendar[(calendar.fase == 'grupos') &
                                                  (calendar.grupo == letter)].match_id]
        s = _game_stats(games)[standings[letter][2]]
        ranked.append((letter, (s['pts'], s['sg'], s['gf'])))
    ranked.sort(key=lambda x: x[1], reverse=True)
    if ranked[7][1] == ranked[8][1]:
        raise UnresolvedRanking('Oitavo terceiro empatado com o nono: informar classificação oficial')
    return [name for name, score in ranked[:8]]


def load_ranking_override(path: str | None) -> dict[str, list[str]]:
    """CSV opcional: grupo,primeiro,segundo,terceiro,quarto."""
    if not path:
        return {}
    df = pd.read_csv(path)
    required = {'grupo', 'primeiro', 'segundo', 'terceiro', 'quarto'}
    if not required.issubset(df.columns) or df.grupo.duplicated().any():
        raise ValueError('Classificação oficial deve ter colunas grupo,primeiro,segundo,terceiro,quarto')
    return {r.grupo: [r.primeiro, r.segundo, r.terceiro, r.quarto] for r in df.itertuples(index=False)}


def resolve_slot(slot: str, standings: dict[str, list[str]],
                 best_groups: list[str] | None, annex: dict,
                 winner: dict[int, str], loser: dict[int, str]) -> str | None:
    """Resolve referências FIFA com dados reais, nunca com previsões."""
    if len(slot) == 2 and slot[0] in '12' and slot[1] in 'ABCDEFGHIJKL':
        teams = standings.get(slot[1]); return teams[int(slot[0])-1] if teams else None
    if len(slot) == 3 and slot[:2] in {'T1'} and slot[2] in 'ABDEGIKL':
        if best_groups is None or len(standings) != 12:
            return None
        key = ''.join(sorted(best_groups))
        third_group = annex[key][slot[2]]
        return standings[third_group][2]
    if slot.startswith('W') and slot[1:].isdigit():
        return winner.get(int(slot[1:]))
    if slot.startswith('L') and slot[1:].isdigit():
        return loser.get(int(slot[1:]))
    return slot  # Nome literal: fase de grupos