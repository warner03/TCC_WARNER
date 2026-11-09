"""Monte Carlo/Poisson e Rede Neural Keras com as MESMAS oito entradas numéricas."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .core import FEATURES


def poisson_monte_carlo(features: dict, iterations: int = 10_000, seed: int = 42) -> dict:
    """Probabilidades de desfechos em 90 minutos; placares são APENAS amostras.

    Reproduz os pesos heurísticos do código inicial do TCC: razão Elo,
    5% por ponto recente, 1% por gol de saldo e 5% de mando (se não neutro).
    O valor 1,35 foi adotado pelo autor; não foi estimado novamente aqui.
    """
    if iterations <= 0:
        raise ValueError("iterations deve ser positivo")
    elo_a, elo_b = float(features["Elo_A_Pre_Jogo"]), float(features["Elo_B_Pre_Jogo"])
    if min(elo_a, elo_b) <= 0:
        raise ValueError("Elo deve ser maior que zero")
    momentum = float(features["Media_Pts_Ult_5_A"]) - float(features["Media_Pts_Ult_5_B"])
    gd = float(features["Saldo_Gols_Ult_5_A"]) - float(features["Saldo_Gols_Ult_5_B"])
    mul_a = elo_a / elo_b + .05 * momentum + .01 * gd
    mul_b = elo_b / elo_a - .05 * momentum - .01 * gd
    if not int(features["Campo_Neutro"]):
        mul_a += .05  # A é mandante no formato home_team/away_team
    lambda_a = float(np.clip(1.35 * max(.1, mul_a), .15, 4.0))
    lambda_b = float(np.clip(1.35 * max(.1, mul_b), .15, 4.0))
    rng = np.random.default_rng(seed)
    goals_a = rng.poisson(lambda_a, iterations)
    goals_b = rng.poisson(lambda_b, iterations)
    return {
        "prob_empate": float(np.mean(goals_a == goals_b)),
        "prob_a": float(np.mean(goals_a > goals_b)),
        "prob_b": float(np.mean(goals_a < goals_b)),
        "lambda_a": lambda_a, "lambda_b": lambda_b,
    }


def load_neural(model_file: str | Path, scaler_file: str | Path):
    """Carrega um modelo treinado. TensorFlow só é necessário para a RNA."""
    import joblib
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("Instale as dependências de requirements-ml.txt para utilizar a RNA") from exc
    return tf.keras.models.load_model(model_file), joblib.load(scaler_file)


def neural_probabilities(features: dict, model, scaler) -> dict:
    """A entrada da rede contém APENAS FEATURES, nunca nomes das seleções."""
    import pandas as pd
    x = pd.DataFrame([[features[k] for k in FEATURES]], columns=FEATURES)
    y = np.asarray(model.predict(scaler.transform(x), verbose=0))[0]
    return {"prob_empate": float(y[0]), "prob_a": float(y[1]), "prob_b": float(y[2])}