#!/usr/bin/env python3
"""Script principal do trabalho da MLP."""

from __future__ import annotations

import argparse
from pathlib import Path

from core_mlp import MLP, calculate_metrics
from io_utils import (
    extract_features,
    extract_targets,
    load_samples,
    render_history_table,
    render_metrics_table,
    render_predictions_table,
    save_predictions,
)


DATA_FILE = Path(__file__).with_name("dados_frutas.csv")
PREDICTIONS_FILE = Path(__file__).with_name("predicoes_frutas.csv")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treinamento de uma MLP simples para classificar frutas."
    )
    parser.add_argument("--hidden-size", type=int, default=5, help="Neuronios na camada escondida.")
    parser.add_argument("--epochs", type=int, default=20000, help="Numero de epocas.")
    parser.add_argument("--learning-rate", type=float, default=0.5, help="Taxa de aprendizado.")
    parser.add_argument("--report-every", type=int, default=2000, help="Intervalo do historico.")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatoria.")
    return parser.parse_args()


def main() -> None:
    # Le o conjunto de dados e separa entradas e saidas.
    args = parse_arguments()
    samples = load_samples(DATA_FILE)
    features = extract_features(samples)
    targets = extract_targets(samples)

    # Cria a rede neural com os parametros informados.
    network = MLP(
        input_size=2,
        hidden_size=args.hidden_size,
        output_size=1,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    history = network.train(
        features=features,
        targets=targets,
        epochs=args.epochs,
        report_every=args.report_every,
    )

    # Gera as previsoes e mede o desempenho da rede.
    probabilities = network.predict_proba(features)
    predictions = network.predict(features)
    summary = calculate_metrics(targets, predictions)

    print("Rede utilizada: 2 entradas ->", args.hidden_size, "neuronios escondidos -> 1 saida")
    print("Historico do treinamento:")
    print(render_history_table(history))

    print("\nPrevisoes da MLP:")
    print(render_predictions_table(samples, probabilities, predictions))

    print("\nMetricas finais:")
    print(render_metrics_table(summary))

    save_predictions(PREDICTIONS_FILE, samples, probabilities, predictions)
    print(f"\nArquivo salvo: {PREDICTIONS_FILE.name}")


if __name__ == "__main__":
    main()
