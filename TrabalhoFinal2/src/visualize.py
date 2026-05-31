"""Geracao das figuras usadas no artigo a partir dos artefatos de treino.

Produz:
  - figures/amostras.png        : grade de digitos do MNIST com seus rotulos.
  - figures/curvas.png          : curvas de perda e acuracia por epoca.
  - figures/matriz_confusao.png : matriz de confusao do conjunto de teste.
  - figures/predicoes.png       : exemplos de predicao (acertos e erros).

Uso:
    python -m src.visualize
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import CLASS_NAMES, get_mnist_loaders
from .model import DigitCNN

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"
CHECKPOINTS = ROOT / "checkpoints"

# Desnormalizacao para exibir as imagens (inverte data.build_transform).
MEAN, STD = 0.1307, 0.3081


def _denorm(img: torch.Tensor) -> np.ndarray:
    return (img.squeeze().cpu().numpy() * STD + MEAN).clip(0, 1)


def plot_samples(test_loader) -> None:
    x, y = next(iter(test_loader))
    fig, axes = plt.subplots(2, 8, figsize=(10, 3))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(_denorm(x[i]), cmap="gray")
        ax.set_title(f"{CLASS_NAMES[int(y[i])]}", fontsize=10)
        ax.axis("off")
    fig.suptitle("Amostras do MNIST (28x28, escala de cinza)")
    fig.tight_layout()
    fig.savefig(FIGURES / "amostras.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_curves(history: list[dict]) -> None:
    ep = [h["epoca"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(ep, [h["perda_treino"] for h in history], "o-", label="treino")
    ax1.plot(ep, [h["perda_validacao"] for h in history], "s-", label="validacao")
    ax1.set_xlabel("Epoca")
    ax1.set_ylabel("Perda (cross-entropy)")
    ax1.set_title("Curva de perda")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(ep, [h["acc_treino"] for h in history], "o-", label="treino")
    ax2.plot(ep, [h["acc_validacao"] for h in history], "s-", label="validacao")
    ax2.set_xlabel("Epoca")
    ax2.set_ylabel("Acuracia")
    ax2.set_title("Curva de acuracia")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "curvas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(confusion: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    ax.set_title("Matriz de confusao (conjunto de teste)")
    thresh = confusion.max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(
                j,
                i,
                int(confusion[i, j]),
                ha="center",
                va="center",
                color="white" if confusion[i, j] > thresh else "black",
                fontsize=7,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES / "matriz_confusao.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def plot_predictions(model, test_loader, device) -> None:
    model.eval()
    x, y = next(iter(test_loader))
    x, y = x.to(device), y.to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1)
    fig, axes = plt.subplots(2, 8, figsize=(12, 4))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(_denorm(x[i]), cmap="gray")
        ok = int(pred[i]) == int(y[i])
        color = "green" if ok else "red"
        ax.set_title(
            f"pred {int(pred[i])} ({probs[i, pred[i]]*100:.0f}%)\nreal {int(y[i])}",
            fontsize=8,
            color=color,
        )
        ax.axis("off")
    fig.suptitle("Predicoes da CNN (verde = acerto, vermelho = erro)")
    fig.tight_layout()
    fig.savefig(FIGURES / "predicoes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = json.loads((OUTPUTS / "metrics.json").read_text())

    _, _, test_loader = get_mnist_loaders(batch_size=128)
    model = DigitCNN().to(device)
    model.load_state_dict(
        torch.load(CHECKPOINTS / "digit_cnn_best.pt", map_location=device)
    )

    plot_samples(test_loader)
    plot_curves(metrics["history"])
    plot_confusion(np.array(metrics["matriz_confusao"]))
    plot_predictions(model, test_loader, device)
    print(f"Figuras salvas em {FIGURES}")


if __name__ == "__main__":
    main()
