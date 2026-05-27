"""CNN alvo para classificacao binaria gato-vs-cachorro (CIFAR-10).

Esta rede e a "vitima" do ataque adversarial. Apos treinada, seus pesos sao
congelados e o ataque branco (white-box) tem acesso aos seus gradientes.

A arquitetura e propositalmente compacta para treinar em poucos minutos no
CPU. Reduzir a complexidade do classificador NAO compromete a validade do
experimento: a propria literatura mostra que mesmo redes muito profundas
(AllConv, NiN, VGG16 no paper de Su et al.) sao vulneraveis a alteracoes de
poucos pixels.
"""

from __future__ import annotations

import torch
from torch import nn


class CatDogClassifier(nn.Module):
    """CNN pequena com BatchNorm para CIFAR-10 binario (32x32 RGB)."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
