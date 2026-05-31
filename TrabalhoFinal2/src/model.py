"""Rede neural convolucional (CNN) para classificacao de digitos do MNIST.

A arquitetura segue o estilo LeNet/VGG simplificado: dois blocos convolucionais
seguidos de um classificador totalmente conectado. As convolucoes extraem
caracteristicas locais (bordas, tracos, curvaturas) com invariancia a pequenas
translacoes, o que e adequado a digitos manuscritos. BatchNorm acelera a
convergencia e Dropout reduz o overfitting.

Entrada:  tensor [N, 1, 28, 28] (escala de cinza normalizada).
Saida:    tensor [N, 10] com os logits de cada digito (0 a 9).
"""

from __future__ import annotations

import torch
from torch import nn


class DigitCNN(nn.Module):
    """CNN compacta para reconhecimento de digitos manuscritos (10 classes)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Bloco 1: 1 -> 32 canais, 28x28 -> 14x14 apos o pooling.
        # Bloco 2: 32 -> 64 canais, 14x14 -> 7x7 apos o pooling.
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

    def count_parameters(self) -> int:
        """Numero de parametros treinaveis da rede."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
