"""Previsão cronológica da Copa com atualização de estado somente pelo placar real."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .core import RealMatchState, incorporate_real_scores
from .models import poisson_monte_carlo, neural_probabilities


def replay_world_cup(matches: pd.DataFrame, year: int = 2026, real_scores_file: str | Path | None = None,
                     model=None, scaler=None, iterations: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Backtest por data; resultado verdadeiro entra SÓ depois das previsões.

    Se placares não forem fornecidos, ainda cria previsões, mas não atualiza
    atributos com palpites. Para jogar Copa inteira, alimente resultados reais.
    """
    matches = incorporate_real_scores(matches, real_scores_file)
    cup = (matches.tournament.eq("FIFA World Cup") & matches.date.dt.year.eq(year))
    if not cup.any():
        raise ValueError(f"Nenhuma partida FIFA World Cup no ano {year}")
    cup_start = matches.loc[cup, "date"].min()
    # Apenas resultados ANTERIORES ao começo do torneio constroem o estado inicial.
    state = RealMatchState()
    rows = []
    # Histórico desde 1872 é processado para manter compatibilidade com Elo original.
    for day, group in matches.groupby("date", sort=True):
        if day < cup_start:
            state.apply_day(group)
            continue
        target = group[cup.loc[group.index]]
        # Previsões de TODAS as partidas do dia antes de qualquer atualização.
        for i, match in target.iterrows():
            features = state.feature_row(match.home_team, match.away_team,
                                         match.tournament, match.neutral)
            mc = poisson_monte_carlo(features, iterations=iterations, seed=seed + int(i))
            record = {"date": day.date().isoformat(), "Time_A": match.home_team,
                      "Time_B": match.away_team, **features,
                      "MC_Empate": mc["prob_empate"], "MC_A": mc["prob_a"], "MC_B": mc["prob_b"],
                      "Lambda_A": mc["lambda_a"], "Lambda_B": mc["lambda_b"]}
            if (model is None) != (scaler is None):
                raise ValueError("Forneça model e scaler juntos")
            if model is not None:
                neural = neural_probabilities(features, model, scaler)
                record.update({"RNA_Empate": neural["prob_empate"],
                               "RNA_A": neural["prob_a"], "RNA_B": neural["prob_b"]})
            if pd.notna(match.home_score) and pd.notna(match.away_score):
                record.update({"Gols_Reais_A": int(match.home_score),
                               "Gols_Reais_B": int(match.away_score),
                               "Resultado_Real": 1 if match.home_score > match.away_score else
                               (2 if match.home_score < match.away_score else 0)})
            else:
                record.update({"Gols_Reais_A": np.nan, "Gols_Reais_B": np.nan,
                               "Resultado_Real": np.nan})
            rows.append(record)
        # Só resultados reais são aplicados. Jogos futuros sem placar são ignorados.
        state.apply_day(group)
    return pd.DataFrame(rows)


def evaluate_known(report: pd.DataFrame) -> pd.DataFrame:
    """Acurácia, log-loss e Brier multiclasse apenas dos jogos com placar real."""
    known = report.loc[report.Resultado_Real.notna()]
    if len(known) == 0:
        return pd.DataFrame(columns=["modelo", "jogos", "acuracia", "log_loss", "brier"])
    classes = known.Resultado_Real.astype(int).to_numpy()
    truth = np.eye(3)[classes]
    items = []
    for name, cols in (("Monte Carlo", ["MC_Empate", "MC_A", "MC_B"]),
                       ("Rede Neural", ["RNA_Empate", "RNA_A", "RNA_B"])):
        if not set(cols).issubset(known.columns):
            continue
        probs = known[cols].to_numpy(dtype=float)
        pred = probs.argmax(axis=1)
        items.append({"modelo": name, "jogos": len(known),
                      "acuracia": float(np.mean(pred == classes)),
                      "log_loss": float(-np.log(np.clip(probs[np.arange(len(classes)), classes], 1e-15, 1)).mean()),
                      "brier": float(np.mean(np.sum((probs - truth) ** 2, axis=1)))})
    return pd.DataFrame(items)