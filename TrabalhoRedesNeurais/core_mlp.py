"""Nucleo da MLP simples com uma camada escondida."""

from __future__ import annotations

import math
import random


def sigmoid(value: float) -> float:
    """Aplica a funcao sigmoide sobre um valor escalar."""
    return 1.0 / (1.0 + math.exp(-value))


def sigmoid_derivative(activation: float) -> float:
    """Calcula a derivada da sigmoide a partir da propria ativacao."""
    return activation * (1.0 - activation)


class MLP:
    """Implementa o perceptron multicamadas com uma camada escondida."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        learning_rate: float,
        seed: int,
    ) -> None:
        self.learning_rate = learning_rate
        rng = random.Random(seed)

        # Pesos entre a camada de entrada e a camada escondida.
        self.weights_input_hidden = [
            [rng.uniform(-0.5, 0.5) for _ in range(hidden_size)]
            for _ in range(input_size)
        ]
        self.bias_hidden = [rng.uniform(-0.5, 0.5) for _ in range(hidden_size)]

        # Pesos entre a camada escondida e a camada de saida.
        self.weights_hidden_output = [
            [rng.uniform(-0.5, 0.5) for _ in range(output_size)]
            for _ in range(hidden_size)
        ]
        self.bias_output = [rng.uniform(-0.5, 0.5) for _ in range(output_size)]

    def forward(self, inputs: list[float]) -> tuple[list[float], list[float]]:
        """Executa a propagacao direta de um exemplo."""
        hidden_outputs: list[float] = []
        for hidden_index in range(len(self.bias_hidden)):
            weighted_sum = self.bias_hidden[hidden_index]
            for input_index, input_value in enumerate(inputs):
                weighted_sum += (
                    input_value * self.weights_input_hidden[input_index][hidden_index]
                )
            hidden_outputs.append(sigmoid(weighted_sum))

        final_outputs: list[float] = []
        for output_index in range(len(self.bias_output)):
            weighted_sum = self.bias_output[output_index]
            for hidden_index, hidden_value in enumerate(hidden_outputs):
                weighted_sum += (
                    hidden_value * self.weights_hidden_output[hidden_index][output_index]
                )
            final_outputs.append(sigmoid(weighted_sum))

        return hidden_outputs, final_outputs

    def train(
        self,
        features: list[list[float]],
        targets: list[int],
        epochs: int,
        report_every: int,
    ) -> list[tuple[int, float]]:
        """Treina a rede pelo algoritmo backpropagation."""
        history: list[tuple[int, float]] = []

        for epoch in range(1, epochs + 1):
            squared_error_sum = 0.0

            # Atualiza a rede amostra por amostra.
            for inputs, target in zip(features, targets):
                hidden_outputs, final_outputs = self.forward(inputs)
                output_value = final_outputs[0]
                target_value = float(target)
                error_output = target_value - output_value
                squared_error_sum += error_output * error_output

                # Calcula o erro local na camada de saida.
                delta_output = error_output * sigmoid_derivative(output_value)

                # Retropropaga o erro para a camada escondida.
                hidden_deltas: list[float] = []
                for hidden_index, hidden_value in enumerate(hidden_outputs):
                    propagated_error = (
                        delta_output * self.weights_hidden_output[hidden_index][0]
                    )
                    hidden_deltas.append(
                        propagated_error * sigmoid_derivative(hidden_value)
                    )

                # Ajusta os pesos da camada escondida para a saida.
                for hidden_index, hidden_value in enumerate(hidden_outputs):
                    self.weights_hidden_output[hidden_index][0] += (
                        self.learning_rate * delta_output * hidden_value
                    )
                self.bias_output[0] += self.learning_rate * delta_output

                # Ajusta os pesos da camada de entrada para a camada escondida.
                for input_index, input_value in enumerate(inputs):
                    for hidden_index, hidden_delta in enumerate(hidden_deltas):
                        self.weights_input_hidden[input_index][hidden_index] += (
                            self.learning_rate * hidden_delta * input_value
                        )

                for hidden_index, hidden_delta in enumerate(hidden_deltas):
                    self.bias_hidden[hidden_index] += self.learning_rate * hidden_delta

            mean_squared_error = squared_error_sum / len(features)
            if epoch == 1 or epoch % report_every == 0 or epoch == epochs:
                history.append((epoch, mean_squared_error))

        return history

    def predict_proba(self, features: list[list[float]]) -> list[float]:
        """Calcula a probabilidade prevista para cada entrada."""
        probabilities = []
        for inputs in features:
            _, outputs = self.forward(inputs)
            probabilities.append(outputs[0])
        return probabilities

    def predict(self, features: list[list[float]], threshold: float = 0.5) -> list[int]:
        """Converte as probabilidades em classes binarias."""
        return [1 if value >= threshold else 0 for value in self.predict_proba(features)]


def calculate_metrics(targets: list[int], predictions: list[int]) -> dict[str, float]:
    """Resume o desempenho da rede no conjunto de dados."""
    true_positive = sum(
        1
        for target, prediction in zip(targets, predictions)
        if target == 1 and prediction == 1
    )
    true_negative = sum(
        1
        for target, prediction in zip(targets, predictions)
        if target == 0 and prediction == 0
    )
    false_positive = sum(
        1
        for target, prediction in zip(targets, predictions)
        if target == 0 and prediction == 1
    )
    false_negative = sum(
        1
        for target, prediction in zip(targets, predictions)
        if target == 1 and prediction == 0
    )

    accuracy = (true_positive + true_negative) / len(targets)
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative)
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "true_positive": float(true_positive),
        "true_negative": float(true_negative),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
    }
