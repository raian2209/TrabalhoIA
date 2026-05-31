"""Carregamento do conjunto MNIST de digitos manuscritos (0 a 9).

O MNIST contem 70.000 imagens em tons de cinza de 28x28 pixels (60.000 para
treino e 10.000 para teste), cada uma rotulada com o digito correspondente.
Mantemos a normalizacao classica (media 0.1307, desvio 0.3081) calculada sobre
o conjunto de treino, o que acelera e estabiliza a convergencia da rede.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

# Estatisticas padrao do MNIST (canal unico de intensidade).
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

CLASS_NAMES = tuple(str(d) for d in range(10))


def build_transform() -> transforms.Compose:
    """Converte a imagem PIL para tensor [1, 28, 28] e normaliza."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )


# Tamanho do conjunto de validacao retirado das 60.000 imagens de treino.
# Resultado: 50.000 treino + 10.000 validacao + 10.000 teste (oficial).
VAL_SIZE = 10_000


def get_mnist_loaders(
    batch_size: int = 128,
    num_workers: int = 0,
    root: Path | str = DATA_ROOT,
    val_size: int = VAL_SIZE,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """DataLoaders de treino, validacao e teste do MNIST (10 classes, 28x28).

    O conjunto oficial de treino (60.000) e dividido em treino e validacao com
    uma particao reproduzivel (semente fixa). O conjunto de teste (10.000)
    permanece intacto e e usado apenas na avaliacao final.
    """
    transform = build_transform()
    full_train = datasets.MNIST(root=str(root), train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=str(root), train=False, download=True, transform=transform)

    n_train = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [n_train, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader, test_loader


def fix_seed(seed: int = 42) -> None:
    """Fixa a semente para reprodutibilidade dos experimentos."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
