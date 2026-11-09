"""Interface de terminal reproduzível para o experimento completo do TCC."""
from __future__ import annotations
import argparse
from pathlib import Path
from .core import load_matches, generate_training_data
from .calendar import create_calendar, load_calendar
from .annex import load_annex
from .neural import train_model
from .models import load_neural
from .full_workflow import replay_104
from .workflow import evaluate_known


def main() -> None:
    parser = argparse.ArgumentParser(description='TCC: Monte Carlo × Rede Neural — Copa 2026 (104 partidas)')
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='RECONSTRUIR base de 1872 até 2026 para auditar a original')
    prep.add_argument('--results', default='data/results.csv')
    prep.add_argument('--aliases', default='data/former_names.csv')
    prep.add_argument('--from-date', default='1872-11-30')
    prep.add_argument('--output', default='outputs/base_recalculada.csv')

    cal = sub.add_parser('calendar', help='Montar 104 jogos FIFA a partir das 72 partidas em results.csv')
    cal.add_argument('--results', default='data/results.csv')
    cal.add_argument('--aliases', default='data/former_names.csv')
    cal.add_argument('--output', default='data/copa_2026_104_jogos.csv')

    tr = sub.add_parser('train', help='Treinar RNA com base_de_dadosTCC.csv ORIGINAL (1872-2026)')
    tr.add_argument('--dataset', default='data/base_de_dadosTCC.csv.gz')
    tr.add_argument('--output-dir', default='outputs/modelo')
    tr.add_argument('--epochs', type=int, default=40)
    tr.add_argument('--seed', type=int, default=42)

    cup = sub.add_parser('cup', help='Comparar previsões nos 104 jogos usando apenas resultados REAIS')
    cup.add_argument('--results', default='data/results.csv')
    cup.add_argument('--aliases', default='data/former_names.csv')
    cup.add_argument('--calendar', default='data/copa_2026_104_jogos.csv')
    cup.add_argument('--annex', default='data/annex_c_2026.csv')
    cup.add_argument('--real-scores', default=None, help='CSV: match_id, home_score_90, away_score_90, winner')
    cup.add_argument('--ranking-override', default=None, help='CSV de classificação oficial quando desempate exigir')
    cup.add_argument('--model', default=None, help='Modelo .keras treinado')
    cup.add_argument('--scaler', default=None, help='Arquivo .joblib correspondente ao modelo')
    cup.add_argument('--iterations', type=int, default=10000)
    cup.add_argument('--seed', type=int, default=42)
    cup.add_argument('--output', default='outputs/previsoes_copa_104.csv')
    cup.add_argument('--metrics', default='outputs/metricas_copa.csv')
    args = parser.parse_args()
    if args.command == 'prepare':
        games = load_matches(args.results, args.aliases)
        dataset = generate_training_data(games, from_date=args.from_date)
        path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(path, index=False)
        print(f'Base reconstruída: {len(dataset)} jogos, de {dataset.date.min()} até {dataset.date.max()} em {path}')
    elif args.command == 'calendar':
        df = create_calendar(load_matches(args.results, args.aliases))
        p = Path(args.output); p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        print(f'Calendário gerado: {len(df)} partidas: 72 grupos + 32 mata-mata, em {p}')
    elif args.command == 'train':
        print(train_model(args.dataset, args.output_dir, epochs=args.epochs, seed=args.seed))
    else:
        if bool(args.model) != bool(args.scaler):
            parser.error('--model e --scaler são obrigatórios juntos')
        model, scaler = load_neural(args.model, args.scaler) if args.model else (None, None)
        report = replay_104(load_matches(args.results, args.aliases),
                            load_calendar(args.calendar), load_annex(args.annex),
                            real_scores_file=args.real_scores, ranking_override=args.ranking_override,
                            model=model, scaler=scaler, iterations=args.iterations, seed=args.seed)
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output, index=False, encoding='utf-8-sig')
        metrics = evaluate_known(report)
        if not metrics.empty:
            p = Path(args.metrics); p.parent.mkdir(parents=True, exist_ok=True)
            metrics.to_csv(p, index=False)
        n_known = int(report.Resultado_Real.notna().sum())
        n_forecast = int((report.Situacao == 'previsto_sem_placar').sum())
        n_pending = int((report.Situacao == 'aguardando_participantes').sum())
        print(f'104 jogos | avaliados {n_known} | previsões sem placar {n_forecast} | aguardando participantes {n_pending}')
        print(f'Relatório: {output}')
        if metrics.empty:
            print('Sem resultados reais inseridos. Nenhuma acurácia da Copa foi inventada.')
        else:
            print(metrics.to_string(index=False))


if __name__ == '__main__':
    main()