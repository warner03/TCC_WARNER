"""Replay dos 104 jogos: previsões *antes* dos placares reais e atualização *depois*.

Parciais ou totais, os placares vêm somente do pesquisador. Nunca são usados
resultados simulados para atualizar a história ou avançar no mata-mata.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .core import RealMatchState, FEATURES
from .models import poisson_monte_carlo, neural_probabilities
from .bracket import group_standings, best_thirds, resolve_slot, load_ranking_override
from .annex import load_annex
from .workflow import evaluate_known

_SCORE_COLS = ('match_id', 'home_score_90', 'away_score_90')


def load_real_scores(path: str | Path | None) -> dict[int, dict]:
    """match_id,home_score_90,away_score_90,winner (optional em mata-mata)."""
    if not path:
        return {}
    df = pd.read_csv(path, dtype={'winner': str})
    if not set(_SCORE_COLS) <= set(df.columns):
        raise ValueError('Placares exigem match_id,home_score_90,away_score_90')
    if df.match_id.duplicated().any():
        raise ValueError('Placares duplicados no mesmo match_id')
    out = {}
    for row in df.to_dict('records'):
        i = int(row['match_id'])
        if i not in range(1, 105):
            raise ValueError(f'Partida inexistente: {i}')
        score_a, score_b = row['home_score_90'], row['away_score_90']
        if any(pd.isna(v) or float(v) < 0 or float(v) % 1 for v in (score_a, score_b)):
            raise ValueError(f'Placar inválido na partida {i}')
        w = row.get('winner')
        out[i] = {'home_score': int(score_a), 'away_score': int(score_b),
                  'winner': None if pd.isna(w) or str(w).strip() == '' else str(w).strip()}
    return out


def replay_104(matches: pd.DataFrame, calendar: pd.DataFrame,
               annex: dict, real_scores_file: str | Path | None = None,
               ranking_override: str | Path | None = None,
               model=None, scaler=None, iterations: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Retorna sempre 104 linhas, inclusive partidas não resolvidas.

    - Treino é feito separadamente em base_de_dadosTCC.csv (nunca aqui).
    - Atributos são recalculados com histórico completo desde 1872.
    - Não antecipa resultados da mesma data.
    - Fases eliminatórias só são liberadas quando os participantes são conhecidos.
    """
    if (model is None) != (scaler is None):
        raise ValueError('Modelo e scaler precisam ser fornecidos juntos')
    real = load_real_scores(real_scores_file)
    # Resultados oficiais já presentes no results.csv podem ser usados, mas nunca
    # entram como fonte para partidas de mata-mata não cadastradas na base.
    raw_group = matches[(matches.tournament == 'FIFA World Cup') &
                        matches.date.between('2026-06-11', '2026-06-27')]
    for m in calendar[calendar.fase == 'grupos'].itertuples(index=False):
        old = raw_group[(raw_group.date == m.date) &
                        (raw_group.home_team == m.home_slot) &
                        (raw_group.away_team == m.away_slot)]
        if len(old) != 1:
            raise ValueError(f'Partida do grupo divergente no CSV original: {m.match_id}')
        r = old.iloc[0]
        if pd.notna(r.home_score):
            existing = {'home_score': int(r.home_score), 'away_score': int(r.away_score), 'winner': None}
            if m.match_id in real and (real[m.match_id]['home_score'], real[m.match_id]['away_score']) != (existing['home_score'], existing['away_score']):
                raise ValueError(f'Placar complementar contradiz a base histórica: M{m.match_id}')
            real.setdefault(m.match_id, existing)
    old_games = matches[matches.date < pd.Timestamp('2026-06-11')]
    state = RealMatchState()
    # Mesma lógica de reprocessamento da base_de_dadosTCC original, desde 1872.
    state.apply_sequential(old_games)
    standings = {}; ranked_best = None
    winners, losers, completed = {}, {}, {}
    overrides = load_ranking_override(str(ranking_override) if ranking_override else None)
    forecasts = []
    for date, day in calendar.groupby('date', sort=True):
        standings = group_standings(calendar, completed, overrides)
        ranked_best = best_thirds(calendar, completed, standings)
        daily_updates = []
        for match in day.itertuples(index=False):
            a = resolve_slot(match.home_slot, standings, ranked_best, annex, winners, losers)
            b = resolve_slot(match.away_slot, standings, ranked_best, annex, winners, losers)
            record = {'match_id': match.match_id, 'date': date.date().isoformat(),
                      'Fase': match.fase, 'Grupo': match.grupo, 'Slot_A': match.home_slot,
                      'Slot_B': match.away_slot, 'Time_A': a or '', 'Time_B': b or '',
                      'Situacao': 'aguardando_participantes',
                      'Gols_Reais_A': np.nan, 'Gols_Reais_B': np.nan,
                      'Resultado_Real': np.nan, 'Vencedor_Real': ''}
            if a is None or b is None:
                if match.match_id in real:
                    raise ValueError(f'M{match.match_id}: placar inserido antes de identificar os participantes reais')
                forecasts.append(record)
                continue
            if a == b:
                raise ValueError(f'M{match.match_id}: mesma equipe em dois lados')
            features = state.feature_row(a, b, 'FIFA World Cup', bool(match.Campo_Neutro))
            mc = poisson_monte_carlo(features, iterations=iterations, seed=seed+int(match.match_id))
            record.update(features)
            record.update({'MC_Empate': mc['prob_empate'], 'MC_A': mc['prob_a'], 'MC_B': mc['prob_b'],
                           'Lambda_A': mc['lambda_a'], 'Lambda_B': mc['lambda_b']})
            if model is not None:
                nn = neural_probabilities(features, model, scaler)
                record.update({'RNA_Empate': nn['prob_empate'], 'RNA_A': nn['prob_a'], 'RNA_B': nn['prob_b']})
            record['Situacao'] = 'previsto_sem_placar'
            if match.match_id in real:
                result = real[match.match_id]
                gh, ga = result['home_score'], result['away_score']
                record.update({'Gols_Reais_A': gh, 'Gols_Reais_B': ga,
                               'Resultado_Real': 1 if gh>ga else 2 if gh<ga else 0,
                               'Situacao': 'avaliado'})
                if match.fase != 'grupos':
                    actual = a if gh > ga else b if gh < ga else None
                    requested = result['winner']
                    if requested in {'A', 'a'}:
                        requested = a
                    elif requested in {'B', 'b'}:
                        requested = b
                    if (gh == ga and requested not in {a, b}) or (actual is not None and requested not in {None, actual}):
                        raise ValueError(f'M{match.match_id}: informar vencedor real consistente (campo winner)')
                    actual = requested or actual
                    winners[match.match_id] = actual
                    losers[match.match_id] = b if actual == a else a
                    record['Vencedor_Real'] = actual
                completed[match.match_id] = {'home_team': a, 'away_team': b,
                                               'home_score': gh, 'away_score': ga}
                daily_updates.append({'home_team': a, 'away_team': b, 'home_score': gh,
                                      'away_score': ga, 'tournament': 'FIFA World Cup'})
            forecasts.append(record)
        if daily_updates:
            state.apply_day(pd.DataFrame(daily_updates))
    result = pd.DataFrame(forecasts).sort_values('match_id').reset_index(drop=True)
    if len(result) != 104:
        raise AssertionError('O replay precisa gerar 104 slots FIFA')
    return result