"""Terminal: preparar base, treinar RNA e prever a Copa com resultados reais."""
from __future__ import annotations
import argparse
from pathlib import Path
from .core import load_matches, generate_training_data
from .workflow import replay_world_cup, evaluate_known
from .neural import train_model
from .models import load_neural


def main() -> None:
    parser = argparse.ArgumentParser(description="TCC Warner - Monte Carlo versus Rede Neural")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Construir atributos pré-jogo sem vazamento temporal")
    prepare.add_argument("--results", default="data/results.csv")
    prepare.add_argument("--aliases", default="data/former_names.csv")
    prepare.add_argument("--from-date", default="2006-01-01")
    prepare.add_argument("--output", default="outputs/base_modelagem.csv")
    train = sub.add_parser("train", help="Treinar RNA Keras com corte cronológico")
    train.add_argument("--dataset", default="outputs/base_modelagem.csv")
    train.add_argument("--output-dir", default="outputs/modelo")
    train.add_argument("--epochs", type=int, default=40)
    cup = sub.add_parser("cup", help="Prever confrontos da Copa antes do resultado real")
    cup.add_argument("--results", default="data/results.csv")
    cup.add_argument("--aliases", default="data/former_names.csv")
    cup.add_argument("--real-scores", default=None, help="CSV com placares reais confirmados (opcional)")
    cup.add_argument("--year", type=int, default=2026)
    cup.add_argument("--model", default=None, help="Arquivo .keras (opcional)")
    cup.add_argument("--scaler", default=None, help="Arquivo .joblib (obrigatório com --model)")
    cup.add_argument("--iterations", type=int, default=10000)
    cup.add_argument("--seed", type=int, default=42)
    cup.add_argument("--output", default="outputs/previsoes_copa.csv")
    args = parser.parse_args()
    if args.command == "prepare":
        matches = load_matches(args.results, args.aliases)
        data = generate_training_data(matches, from_date=args.from_date)
        dest = Path(args.output); dest.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(dest, index=False)
        print(f"Gerados {len(data)} exemplos em {dest}")
    elif args.command == "train":
        print(train_model(args.dataset, args.output_dir, epochs=args.epochs))
    else:
        if bool(args.model) != bool(args.scaler):
            parser.error("--model e --scaler precisam ser informados juntos")
        model, scaler = load_neural(args.model, args.scaler) if args.model else (None, None)
        matches = load_matches(args.results, args.aliases)
        report = replay_world_cup(matches, year=args.year,
                                  real_scores_file=args.real_scores, model=model, scaler=scaler,
                                  iterations=args.iterations, seed=args.seed)
        dest = Path(args.output); dest.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(dest, index=False)
        scored = int(report.Resultado_Real.notna().sum())
        print(f"Previsões: {len(report)} | jogos com resultado real: {scored} | relatório: {dest}")
        metrics = evaluate_known(report)
        if not metrics.empty:
            print(metrics.to_string(index=False))
        else:
            print("Ainda não há resultados reais da Copa nesta base. Nenhum resultado foi inventado.")


if __name__ == "__main__":
    main()