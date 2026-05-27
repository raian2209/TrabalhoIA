"""Carregamento do subconjunto gato-vs-cachorro do CIFAR-10.

Mantemos as imagens em [0, 1] (RGB, 3 x 32 x 32) sem normalizacao adicional,
para que a manipulacao de pixels pelo atacante seja diretamente interpretavel.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

CIFAR_CAT = 3
CIFAR_DOG = 5
CLASS_NAMES = ("gato", "cachorro")


class _RelabelDataset(torch.utils.data.Dataset):
    """Aplica um mapa de rotulos do CIFAR-10 (3 ou 5) para binario (0 ou 1)."""

    def __init__(self, base: torch.utils.data.Dataset, label_map: dict[int, int]) -> None:
        self.base = base
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        return x, self.label_map[int(y)]


def _filter_indices(targets, keep: tuple[int, ...]) -> list[int]:
    keep_set = set(keep)
    return [i for i, t in enumerate(targets) if int(t) in keep_set]


def get_catdog_loaders(
    batch_size: int = 128,
    num_workers: int = 0,
    root: Path | str = DATA_ROOT,
) -> tuple[DataLoader, DataLoader]:
    """DataLoaders binarios: 0 = gato, 1 = cachorro. CIFAR-10 com 32x32 RGB."""
    transform = transforms.ToTensor()
    train_ds = datasets.CIFAR10(root=str(root), train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10(root=str(root), train=False, download=True, transform=transform)

    label_map = {CIFAR_CAT: 0, CIFAR_DOG: 1}
    train_subset = Subset(train_ds, _filter_indices(train_ds.targets, (CIFAR_CAT, CIFAR_DOG)))
    test_subset = Subset(test_ds, _filter_indices(test_ds.targets, (CIFAR_CAT, CIFAR_DOG)))
    train_subset = _RelabelDataset(train_subset, label_map)
    test_subset = _RelabelDataset(test_subset, label_map)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, test_loader


def fix_seed(seed: int = 42) -> None:
    """Fixa a semente para reprodutibilidade."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
