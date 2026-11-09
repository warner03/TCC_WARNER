import numpy as np
import pandas as pd
import pytest
from tcc_warner.core import RealMatchState, generate_training_data, incorporate_real_scores, FEATURES
from tcc_warner.models import poisson_monte_carlo, neural_probabilities
from tcc_warner.workflow import replay_world_cup, evaluate_known


def fixtures():
    return pd.DataFrame([
        ["2026-06-09", "A", "B", 2, 0, "Friendly", True],
        ["2026-06-11", "A", "C", np.nan, np.nan, "FIFA World Cup", True],
        ["2026-06-11", "B", "D", np.nan, np.nan, "FIFA World Cup", True],
        ["2026-06-15", "A", "D", np.nan, np.nan, "FIFA World Cup", True],
    ], columns=["date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"]).assign(
        date=lambda x: pd.to_datetime(x.date))


def patch_csv(tmp_path):
    f = tmp_path / "real.csv"
    f.write_text("date,home_team,away_team,home_score,away_score\n"
                 "2026-06-11,A,C,0,1\n2026-06-11,B,D,2,2\n", encoding="utf-8")
    return f


def test_scoreless_does_not_update():
    state = RealMatchState()
    before = state.feature_row("A", "B", "FIFA World Cup", True)
    state.apply_day(fixtures().iloc[[1]])
    after = state.feature_row("A", "B", "FIFA World Cup", True)
    assert before == after


def test_only_real_updates_next_day(tmp_path):
    games = fixtures()
    without = replay_world_cup(games, iterations=100, seed=10)
    with_real = replay_world_cup(games, real_scores_file=patch_csv(tmp_path), iterations=100, seed=10)
    assert len(with_real) == 3
    assert without.Resultado_Real.isna().all()
    assert with_real.Resultado_Real.notna().sum() == 2
    # Dia 11 não conhece os resultados do próprio dia.
    assert np.allclose(without.iloc[:2].Elo_A_Pre_Jogo, with_real.iloc[:2].Elo_A_Pre_Jogo)
    # Dia 15 usa o 0x1 de A contra C, mas sem resultado não há alteração.
    assert with_real.iloc[2].Elo_A_Pre_Jogo < without.iloc[2].Elo_A_Pre_Jogo
    assert with_real.iloc[2].Saldo_Gols_Ult_5_A == 1  # +2 anterior e -1 real
    assert without.iloc[2].Saldo_Gols_Ult_5_A == 2


def test_training_features_are_pregame():
    games = fixtures().iloc[[0]].copy()
    games.loc[1] = [pd.Timestamp("2026-06-10"), "A", "B", 0, 1, "Friendly", True]
    dataset = generate_training_data(games, "2026-01-01")
    assert len(dataset) == 2
    assert dataset.iloc[0].Elo_A_Pre_Jogo == 1500
    assert dataset.iloc[1].Elo_A_Pre_Jogo > dataset.iloc[0].Elo_A_Pre_Jogo
    assert dataset.iloc[1].Saldo_Gols_Ult_5_A == 2


def test_same_day_examples_use_same_prior_state():
    games = pd.DataFrame([
        [pd.Timestamp('2026-06-10'), "A", "B", 2, 0, "Friendly", True],
        [pd.Timestamp('2026-06-10'), "A", "C", 1, 0, "Friendly", True],
    ], columns=fixtures().columns)
    ds = generate_training_data(games, "2026-01-01")
    assert ds.iloc[0].Elo_A_Pre_Jogo == ds.iloc[1].Elo_A_Pre_Jogo == 1500


def test_monte_carlo_probabilities_sum_one():
    x = RealMatchState().feature_row("A", "B", "FIFA World Cup", True)
    p = poisson_monte_carlo(x, iterations=1000, seed=123)
    assert p['prob_a'] + p['prob_b'] + p['prob_empate'] == pytest.approx(1)
    assert p == poisson_monte_carlo(x, iterations=1000, seed=123)


def test_neural_uses_exact_eight_features_only():
    class Scaler:
        def transform(self, x):
            assert list(x.columns) == FEATURES
            return x.to_numpy()
    class Model:
        def predict(self, x, verbose=0):
            assert x.shape == (1,8)
            return np.array([[.2, .5, .3]])
    x = RealMatchState().feature_row("A", "B", "FIFA World Cup", True)
    assert neural_probabilities(x, Model(), Scaler())['prob_a'] == .5


def test_conflicts_rejected(tmp_path):
    games = fixtures()
    file = tmp_path / "bad.csv"
    file.write_text("date,home_team,away_team,home_score,away_score\n2026-06-09,A,B,0,7\n")
    with pytest.raises(ValueError, match="contradiz"):
        incorporate_real_scores(games, file)


def test_evaluation_only_real_matches(tmp_path):
    rep = replay_world_cup(fixtures(), real_scores_file=patch_csv(tmp_path), iterations=100)
    metrics = evaluate_known(rep)
    assert metrics.iloc[0].jogos == 2
    assert metrics.iloc[0].modelo == "Monte Carlo"