"""Utilitarios de leitura, escrita e formatacao do trabalho da MLP."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sample:
    """Representa uma linha do conjunto de frutas."""

    entrada_1: float
    entrada_2: float
    saida: int
    descricao: str


def load_samples(path: Path) -> list[Sample]:
    """Le o arquivo CSV e monta as amostras do problema."""
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [
            Sample(
                entrada_1=float(row["entrada_1"]),
                entrada_2=float(row["entrada_2"]),
                saida=int(row["saida"]),
                descricao=row["descricao"],
            )
            for row in reader
        ]


def extract_features(samples: list[Sample]) -> list[list[float]]:
    """Separa apenas os atributos de entrada para a rede."""
    return [[sample.entrada_1, sample.entrada_2] for sample in samples]


def extract_targets(samples: list[Sample]) -> list[int]:
    """Separa apenas as classes desejadas."""
    return [sample.saida for sample in samples]


def save_predictions(
    path: Path,
    samples: list[Sample],
    probabilities: list[float],
    predictions: list[int],
) -> None:
    """Salva as previsoes finais em um arquivo CSV."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "entrada_1",
                "entrada_2",
                "saida_real",
                "probabilidade",
                "classe_prevista",
                "acertou",
                "descricao",
            ]
        )
        for sample, probability, prediction in zip(samples, probabilities, predictions):
            writer.writerow(
                [
                    f"{sample.entrada_1:.2f}",
                    f"{sample.entrada_2:.2f}",
                    sample.saida,
                    f"{probability:.6f}",
                    prediction,
                    int(sample.saida == prediction),
                    sample.descricao,
                ]
            )


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Monta uma tabela de texto simples para saida no terminal."""
    widths = [len(header) for header in headers]
    for row in rows:
        for column_index, value in enumerate(row):
            widths[column_index] = max(widths[column_index], len(value))

    lines = [
        " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "-+-".join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append(
            " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        )
    return "\n".join(lines)


def render_history_table(history: list[tuple[int, float]]) -> str:
    """Formata o historico do treinamento."""
    rows = [[str(epoch), f"{mse:.6f}"] for epoch, mse in history]
    return format_table(["Epoca", "Erro medio quadratico"], rows)


def render_predictions_table(
    samples: list[Sample],
    probabilities: list[float],
    predictions: list[int],
) -> str:
    """Formata as previsoes feitas pela MLP."""
    rows = []
    for sample, probability, prediction in zip(samples, probabilities, predictions):
        rows.append(
            [
                f"{sample.entrada_1:.2f}",
                f"{sample.entrada_2:.2f}",
                str(sample.saida),
                f"{probability:.4f}",
                str(prediction),
                "sim" if sample.saida == prediction else "nao",
                sample.descricao,
            ]
        )

    return format_table(
        ["Entrada 1", "Entrada 2", "Real", "Probabilidade", "Previsto", "Acertou", "Descricao"],
        rows,
    )


def render_metrics_table(summary: dict[str, float]) -> str:
    """Formata as metricas finais do treinamento."""
    return format_table(
        ["Metrica", "Valor"],
        [
            ["Acuracia", f"{summary['accuracy']:.2f}"],
            ["Precisao", f"{summary['precision']:.2f}"],
            ["Recall", f"{summary['recall']:.2f}"],
            ["TP", str(int(summary["true_positive"]))],
            ["TN", str(int(summary["true_negative"]))],
            ["FP", str(int(summary["false_positive"]))],
            ["FN", str(int(summary["false_negative"]))],
        ],
    )
