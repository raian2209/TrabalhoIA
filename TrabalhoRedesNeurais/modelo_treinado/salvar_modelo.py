#!/usr/bin/env python3
"""Treina a MLP do trabalho e salva os pesos em JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from core_mlp import MLP, calculate_metrics  # noqa: E402
from io_utils import extract_features, extract_targets, load_samples  # noqa: E402


DATA_FILE = PROJECT_DIR / "dados_frutas.csv"
MODEL_FILE = Path(__file__).with_name("modelo_mlp_frutas.json")


def main() -> None:
    samples = load_samples(DATA_FILE)
    features = extract_features(samples)
    targets = extract_targets(samples)

    network = MLP(
        input_size=2,
        hidden_size=5,
        output_size=1,
        learning_rate=0.5,
        seed=42,
    )

    history = network.train(
        features=features,
        targets=targets,
        epochs=20000,
        report_every=2000,
    )

    predictions = network.predict(features)
    metrics = calculate_metrics(targets, predictions)

    model_data = {
        "description": "MLP treinada para classificacao binaria de frutas.",
        "architecture": {
            "input_size": 2,
            "hidden_size": 5,
            "output_size": 1,
            "activation": "sigmoid",
        },
        "training": {
            "learning_rate": 0.5,
            "epochs": 20000,
            "report_every": 2000,
            "seed": 42,
            "final_mse": history[-1][1],
            "threshold": 0.5,
        },
        "metrics": metrics,
        "weights_input_hidden": network.weights_input_hidden,
        "bias_hidden": network.bias_hidden,
        "weights_hidden_output": network.weights_hidden_output,
        "bias_output": network.bias_output,
    }

    MODEL_FILE.write_text(
        json.dumps(model_data, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"Modelo salvo em: {MODEL_FILE}")


if __name__ == "__main__":
    main()
