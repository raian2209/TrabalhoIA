#!/usr/bin/env python3
"""Carrega o modelo salvo e executa inferencia em dados de teste."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from core_mlp import MLP  # noqa: E402
from io_utils import format_table  # noqa: E402


MODEL_FILE = Path(__file__).with_name("modelo_mlp_frutas.json")
TEST_FILE = Path(__file__).with_name("dados_teste.csv")
OUTPUT_FILE = Path(__file__).with_name("predicoes_teste.csv")


@dataclass
class TestSample:
    entrada_1: float
    entrada_2: float
    descricao: str


def load_model(path: Path) -> MLP:
    model_data = json.loads(path.read_text(encoding="utf-8"))
    architecture = model_data["architecture"]
    training = model_data["training"]

    network = MLP(
        input_size=architecture["input_size"],
        hidden_size=architecture["hidden_size"],
        output_size=architecture["output_size"],
        learning_rate=training["learning_rate"],
        seed=training["seed"],
    )

    network.weights_input_hidden = model_data["weights_input_hidden"]
    network.bias_hidden = model_data["bias_hidden"]
    network.weights_hidden_output = model_data["weights_hidden_output"]
    network.bias_output = model_data["bias_output"]
    return network


def load_test_samples(path: Path) -> list[TestSample]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [
            TestSample(
                entrada_1=float(row["entrada_1"]),
                entrada_2=float(row["entrada_2"]),
                descricao=row["descricao"],
            )
            for row in reader
        ]


def save_predictions(
    samples: list[TestSample],
    probabilities: list[float],
    predictions: list[int],
) -> None:
    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["entrada_1", "entrada_2", "probabilidade", "classe_prevista", "descricao"])
        for sample, probability, prediction in zip(samples, probabilities, predictions):
            writer.writerow(
                [
                    f"{sample.entrada_1:.2f}",
                    f"{sample.entrada_2:.2f}",
                    f"{probability:.6f}",
                    prediction,
                    sample.descricao,
                ]
            )


def main() -> None:
    network = load_model(MODEL_FILE)
    samples = load_test_samples(TEST_FILE)
    features = [[sample.entrada_1, sample.entrada_2] for sample in samples]

    probabilities = network.predict_proba(features)
    predictions = network.predict(features)
    save_predictions(samples, probabilities, predictions)

    rows = []
    for sample, probability, prediction in zip(samples, probabilities, predictions):
        rows.append(
            [
                f"{sample.entrada_1:.2f}",
                f"{sample.entrada_2:.2f}",
                f"{probability:.6f}",
                str(prediction),
                sample.descricao,
            ]
        )

    print("Inferencia usando modelo salvo:")
    print(
        format_table(
            ["Entrada 1", "Entrada 2", "Probabilidade", "Classe", "Descricao"],
            rows,
        )
    )
    print(f"\nPredicoes salvas em: {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
