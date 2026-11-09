"""Estado cronológico compartilhado, atualizado APENAS com placares reais.

As seleções são identificadas internamente, mas seus nomes NÃO são features.
Quando não há horário, todos os jogos do mesmo dia usam o estado do começo
 desse dia; só depois todos os resultados conhecidos são aplicados.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd

FEATURES = [
    "Peso_Jogo", "Campo_Neutro", "Elo_A_Pre_Jogo", "Elo_B_Pre_Jogo",
    "Media_Pts_Ult_5_A", "Media_Pts_Ult_5_B",
    "Saldo_Gols_Ult_5_A", "Saldo_Gols_Ult_5_B",
]
UNOFFICIAL = {"Aymara", "Mapuche", "Arameans Suryoye", "Isle of Man", "Ellan Vannin"}


def tournament_weight(name: str) -> int:
    name = str(name).lower()
    if name == "fifa world cup":
        return 7
    if "qualification" in name or "qualifying" in name:
        return 3
    if "nations league" in name:
        return 2
    if any(x in name for x in (
        "copa américa", "copa america", "uefa euro", "african cup of nations",
        "afc asian cup", "gold cup", "oceania nations cup", "confederations cup",
        "conmebol–uefa cup of champions", "asian cup", "arab cup", "nations cup"
    )):
        return 4
    return 1


def load_matches(path: str | Path, former_names: str | Path | None = None) -> pd.DataFrame:
    """Carrega jogos reais e/ou tabela futura, sem converter NaN em resultados."""
    df = pd.read_csv(path)
    must = {"date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"}
    missing = must.difference(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")
    if former_names:
        mapping = pd.read_csv(former_names)
        aliases = dict(zip(mapping["former"], mapping["current"]))
        for col in ("home_team", "away_team"):
            df[col] = df[col].replace(aliases)
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    df = df.loc[~(df.home_team.isin(UNOFFICIAL) | df.away_team.isin(UNOFFICIAL))].copy()
    for col in ("home_score", "away_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if (df["home_score"].notna() != df["away_score"].notna()).any():
        raise ValueError("Partida com apenas um placar preenchido")
    valid_scores = df["home_score"].notna()
    if ((df.loc[valid_scores, ["home_score", "away_score"]] < 0).any().any() or
        (df.loc[valid_scores, ["home_score", "away_score"]] % 1 != 0).any().any()):
        raise ValueError("Placar real deve conter inteiros não negativos")
    if df["neutral"].dtype == object:
        raw = df["neutral"].astype(str).str.lower()
        if not raw.isin(["true", "false", "0", "1"]).all():
            raise ValueError("Campo neutral deve ser True/False ou 1/0")
        df["neutral"] = raw.isin(["true", "1"])
    else:
        df["neutral"] = df["neutral"].astype(bool)
    # Ordenação estável: não introduz resultados de partidas da mesma data.
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def incorporate_real_scores(matches: pd.DataFrame, scores_file: str | Path | None) -> pd.DataFrame:
    """Substitui apenas placares disponíveis por um CSV de resultados verificáveis.

    Chave: date, home_team, away_team. Dados conflitantes ou partidas ausentes
    provocam erro; nada é inferido a partir das previsões dos modelos.
    """
    if not scores_file:
        return matches.copy()
    patch = pd.read_csv(scores_file)
    required = ["date", "home_team", "away_team", "home_score", "away_score"]
    if not set(required).issubset(patch.columns):
        raise ValueError(f"CSV de placares exige: {required}")
    patch = patch[required].copy()
    patch["date"] = pd.to_datetime(patch["date"]).dt.normalize()
    if patch.duplicated(["date", "home_team", "away_team"]).any():
        raise ValueError("Resultados reais duplicados no CSV complementar")
    if (patch.home_score.isna() | patch.away_score.isna()).any():
        raise ValueError("CSV complementar contém resultado incompleto")
    for c in ("home_score", "away_score"):
        patch[c] = pd.to_numeric(patch[c], errors="raise")
        if ((patch[c] < 0) | (patch[c] % 1 != 0)).any():
            raise ValueError("Placar deve ser inteiro não negativo")
    keys = ["date", "home_team", "away_team"]
    if matches.duplicated(keys).any():
        raise ValueError("Tabela com partidas duplicadas por data e seleções")
    original = matches.copy()
    original["_pos"] = range(len(original))
    new = original.merge(patch, on=keys, how="left", validate="one_to_one", indicator=True, suffixes=("", "_novo"))
    found = patch.merge(original[keys], on=keys, how="left", indicator=True)
    if (found["_merge"] == "left_only").any():
        raise ValueError("Há resultados de partidas que não estão na tabela original")
    valid = new["home_score_novo"].notna()
    conflicting = valid & new["home_score"].notna() & (
        (new["home_score"] != new["home_score_novo"]) |
        (new["away_score"] != new["away_score_novo"])
    )
    if conflicting.any():
        raise ValueError("O arquivo complementar contradiz um placar já existente")
    for c in ("home_score", "away_score"):
        new[c] = new[f"{c}_novo"].combine_first(new[c])
    new = new.sort_values("_pos", kind="stable")
    return new[matches.columns].reset_index(drop=True)


@dataclass
class Team:
    elo: float = 1500.0
    points: deque = field(default_factory=lambda: deque(maxlen=5))
    gd: deque = field(default_factory=lambda: deque(maxlen=5))


class RealMatchState:
    """Estado mutável, com atualização invocada SOMENTE após placar real."""

    def __init__(self, k: float = 20.0):
        self.k = k
        self.teams: dict[str, Team] = defaultdict(Team)

    def feature_row(self, home: str, away: str, tournament: str, neutral: bool) -> dict:
        a, b = self.teams[home], self.teams[away]
        return dict(zip(FEATURES, [
            tournament_weight(tournament), int(bool(neutral)), a.elo, b.elo,
            sum(a.points) / len(a.points) if a.points else 0.0,
            sum(b.points) / len(b.points) if b.points else 0.0,
            sum(a.gd), sum(b.gd),
        ]))

    def apply_day(self, games: pd.DataFrame) -> None:
        """Calcula todos os deltas com estado ANTERIOR ao dia e depois aplica.

        Não aplicar a jogos sem placar. Isso evita usar o resultado de uma
        partida futura ou do mesmo dia sem conhecer o horário real.
        """
        changes = defaultdict(float)
        memories = []
        for game in games.itertuples(index=False):
            if pd.isna(game.home_score) or pd.isna(game.away_score):
                continue
            h, a = game.home_team, game.away_team
            ha, aw = self.teams[h], self.teams[a]
            weight = tournament_weight(game.tournament)
            expected = 1 / (1 + 10 ** ((aw.elo - ha.elo) / 400))
            score = 1.0 if game.home_score > game.away_score else (0.5 if game.home_score == game.away_score else 0.0)
            delta = self.k * weight * (score - expected)
            changes[h] += delta
            changes[a] -= delta
            pts_h = 3 if score == 1 else 1 if score == .5 else 0
            pts_a = 3 if score == 0 else 1 if score == .5 else 0
            gd = int(game.home_score - game.away_score)
            memories.append((h, a, pts_h, pts_a, gd))
        for h, a, ph, pa, gd in memories:
            self.teams[h].points.append(ph)
            self.teams[a].points.append(pa)
            self.teams[h].gd.append(gd)
            self.teams[a].gd.append(-gd)
        for team, delta in changes.items():
            self.teams[team].elo += delta


def generate_training_data(matches: pd.DataFrame, from_date: str = "2006-01-01") -> pd.DataFrame:
    """Cria exemplos com atributos ANTES da partida e classe observada.

    O Elo é aquecido usando jogos anteriores a 2006. Datas sem horário usam
    o mesmo estado pré-dia. Nunca utiliza os placares do próprio confronto
    nem de jogos futuros ao gerar seus atributos.
    """
    start = pd.Timestamp(from_date)
    state = RealMatchState()
    examples = []
    for day, group in matches.groupby("date", sort=True):
        if day >= start:
            for game in group.itertuples(index=False):
                if pd.isna(game.home_score):
                    continue
                x = state.feature_row(game.home_team, game.away_team, game.tournament, game.neutral)
                result = 1 if game.home_score > game.away_score else (2 if game.home_score < game.away_score else 0)
                examples.append({"date": day.date().isoformat(), "Time_A": game.home_team, "Time_B": game.away_team,
                                 **x, "Resultado": result})
        state.apply_day(group)
    return pd.DataFrame(examples, columns=["date", "Time_A", "Time_B", *FEATURES, "Resultado"])