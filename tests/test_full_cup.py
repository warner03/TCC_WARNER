from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from tcc_warner.annex import load_annex
from tcc_warner.bracket import group_standings, best_thirds, UnresolvedRanking, resolve_slot
from tcc_warner.calendar import load_calendar, create_calendar
from tcc_warner.core import load_matches, generate_training_data
from tcc_warner.full_workflow import replay_104, load_real_scores

DATA = Path(__file__).resolve().parents[1] / 'data'


@pytest.fixture(scope='module')
def calendar():
    return load_calendar(DATA / 'copa_2026_104_jogos.csv')


@pytest.fixture(scope='module')
def annex():
    return load_annex(DATA / 'annex_c_2026.csv')


@pytest.fixture(scope='module')
def small_matches(calendar):
    # Simulação de base histórica mínima para testes rápidos: jogos da Copa
    # ainda SEM resultados, mais dois históricos do mesmo time.
    records = [dict(date=pd.Timestamp('1872-11-30'), home_team='Mexico', away_team='Canada',
                    home_score=2, away_score=0, tournament='Friendly', neutral=True)]
    for m in calendar[calendar.fase == 'grupos'].itertuples(index=False):
        records.append(dict(date=m.date, home_team=m.home_slot, away_team=m.away_slot,
                            home_score=np.nan, away_score=np.nan,
                            tournament='FIFA World Cup', neutral=bool(m.Campo_Neutro)))
    return pd.DataFrame(records)


def synthetic_group_results(calendar):
    """Placar artificial APENAS para testes, sem relação com a Copa real."""
    rows = []
    for gi, (letter, group) in enumerate(calendar[calendar.fase == 'grupos'].groupby('grupo')):
        teams = sorted(set(group.home_slot) | set(group.away_slot))
        ranking = {t: i for i, t in enumerate(teams)}
        for match in group.itertuples(index=False):
            ha, aw = ranking[match.home_slot], ranking[match.away_slot]
            winner = match.home_slot if ha > aw else match.away_slot
            # Terceiros distintos em saldo, eliminando empate no corte.
            margin = 2 + (gi if sorted([ha, aw]) == [0, 1] else 0)
            rows.append(dict(match_id=match.match_id,
                             home_score_90=margin if winner == match.home_slot else 0,
                             away_score_90=0 if winner == match.home_slot else margin,
                             winner=''))
    return rows


def write_scores(tmp_path, rows):
    p = tmp_path / 'reais.csv'
    pd.DataFrame(rows, columns=['match_id','home_score_90','away_score_90','winner']).to_csv(p, index=False)
    return p


def test_104_calendar_and_annex(calendar, annex):
    assert sorted(calendar.match_id) == list(range(1, 105))
    assert calendar.fase.value_counts().to_dict() == {'grupos':72,'16avos':16,'oitavas':8,
                                                       'quartas':4,'semifinais':2,'terceiro_lugar':1,'final':1}
    assert len(annex) == 495
    assert set(annex['BDEFIJKL'].values()) == set('BDEFIJKL')


def test_group_calendar_matches_user_input(calendar):
    raw = load_matches(DATA / 'results.csv', DATA / 'former_names.csv')
    derived = create_calendar(raw)
    assert derived.equals(calendar.sort_values('match_id').reset_index(drop=True).assign(
        date=lambda x: x.date.dt.strftime('%Y-%m-%d')))


def test_no_real_scores_means_72_forecasts_and_32_pending(small_matches, calendar, annex):
    report = replay_104(small_matches, calendar, annex, iterations=100)
    assert len(report) == 104
    assert (report.iloc[:72].Situacao == 'previsto_sem_placar').all()
    assert (report.iloc[72:].Situacao == 'aguardando_participantes').all()
    assert report.Resultado_Real.isna().all()
    assert report.MC_A.iloc[0] + report.MC_Empate.iloc[0] + report.MC_B.iloc[0] == pytest.approx(1)


def test_real_result_updates_only_later_games(small_matches, calendar, annex, tmp_path):
    before = replay_104(small_matches, calendar, annex, iterations=100)
    scorefile = write_scores(tmp_path, [dict(match_id=1, home_score_90=4, away_score_90=0, winner='')])
    after = replay_104(small_matches, calendar, annex, real_scores_file=scorefile, iterations=100)
    assert after.loc[after.match_id.eq(1), 'Resultado_Real'].iloc[0] == 1
    # O mesmo dia tem outro confronto, que não usa o futuro.
    assert before.loc[before.match_id.eq(1), 'Elo_A_Pre_Jogo'].iloc[0] == after.loc[after.match_id.eq(1), 'Elo_A_Pre_Jogo'].iloc[0]
    # México (A) x Coreia (B) em 18/06 já reflete o placar real de M1.
    next_match = 28
    b = before.loc[before.match_id.eq(next_match), 'Elo_A_Pre_Jogo'].iloc[0]
    a = after.loc[after.match_id.eq(next_match), 'Elo_A_Pre_Jogo'].iloc[0]
    assert a > b
    assert after.loc[after.match_id.eq(next_match), 'Saldo_Gols_Ult_5_A'].iloc[0] == 6
    # O Brasil nem participou desse jogo: não pode ser alterado por previsão.
    brazil_id = 7
    assert before.loc[before.match_id.eq(brazil_id), 'Elo_A_Pre_Jogo'].iloc[0] == after.loc[after.match_id.eq(brazil_id), 'Elo_A_Pre_Jogo'].iloc[0]


def test_all_group_results_unlock_16_games(small_matches, calendar, annex, tmp_path):
    scores=synthetic_group_results(calendar)
    report = replay_104(small_matches, calendar, annex,
                        real_scores_file=write_scores(tmp_path, scores), iterations=100)
    assert (report.iloc[:72].Situacao == 'avaliado').all()
    assert (report.iloc[72:88].Situacao == 'previsto_sem_placar').all()
    assert (report.iloc[88:].Situacao == 'aguardando_participantes').all()
    assert report.loc[72:87, 'Time_A'].notna().all()
    assert report.loc[72:87, 'Time_B'].notna().all()


def test_full_real_bracket_104_and_winner_chain(small_matches, calendar, annex, tmp_path):
    scores = synthetic_group_results(calendar)
    # Como os 32 confrontos são compostos dinamicamente de vencedores anteriores,
    # cada um poderá avançar no teste selecionando a equipe A (placar real teste 1x0).
    scores += [dict(match_id=i,home_score_90=1,away_score_90=0,winner='') for i in range(73,105)]
    report = replay_104(small_matches, calendar, annex,
                        real_scores_file=write_scores(tmp_path, scores), iterations=100)
    assert len(report) == 104
    assert report.Situacao.eq('avaliado').sum() == 104
    w101 = report.loc[report.match_id.eq(101),'Time_A'].iloc[0]
    w102 = report.loc[report.match_id.eq(102),'Time_A'].iloc[0]
    assert report.loc[report.match_id.eq(104),'Time_A'].iloc[0] == w101
    assert report.loc[report.match_id.eq(104),'Time_B'].iloc[0] == w102
    assert report.loc[report.match_id.eq(103),'Time_A'].iloc[0] != w101


def test_knockout_draw_needs_real_winner(small_matches, calendar, annex, tmp_path):
    scores = synthetic_group_results(calendar)
    scores += [dict(match_id=73,home_score_90=1,away_score_90=1,winner='')]
    with pytest.raises(ValueError, match='vencedor real'):
        replay_104(small_matches, calendar, annex,
                   real_scores_file=write_scores(tmp_path, scores), iterations=100)


def test_bad_score_and_invalid_schedule(tmp_path, calendar):
    with pytest.raises(ValueError, match='Placar inválido'):
        load_real_scores(write_scores(tmp_path,[dict(match_id=1, home_score_90=-1,away_score_90=0,winner='')]))
    broken = calendar.iloc[:-1].copy()
    p=tmp_path/'cal.csv';broken.to_csv(p,index=False)
    with pytest.raises(ValueError,match='104'):
        load_calendar(p)