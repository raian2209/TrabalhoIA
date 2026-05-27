"""Gera figuras comparando imagens originais e adversariais.

Para cada K em {1, 3, 5} gera um grid 4 linhas x N colunas:
  linha 1: imagem original  (rotulo verdadeiro)
  linha 2: imagem atacada    (rotulo previsto, com circulo destacando o pixel)
  linha 3: diferenca amplificada |x_adv - x|
  linha 4: mapa do gradiente d(CE)/dx (saliencia)

Tambem gera uma figura de barras com a taxa de sucesso por K (NN vs aleatorio).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle
from torch.nn import functional as F

from attacker import KPixelAttacker, compute_input_gradient
from classifier import CatDogClassifier
from data import CLASS_NAMES, fix_seed, get_catdog_loaders

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def _to_image(t: torch.Tensor) -> np.ndarray:
    """(C, H, W) tensor -> (H, W, C) numpy in [0, 1]."""
    return t.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)


def _pick_successful_samples(
    classifier, attacker, loader, device, n_samples: int
) -> list[tuple[torch.Tensor, int, int, int, torch.Tensor, torch.Tensor]]:
    """Retorna ate n_samples tuplas (x, y, clean_pred, adv_pred, mask, grad)."""
    out = []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        clean = classifier(x_batch).argmax(dim=1)
        grad = compute_input_gradient(classifier, x_batch, y_batch)
        x_adv, mask, value_map = attacker.attack_deterministic(x_batch, grad)
        adv = classifier(x_adv).argmax(dim=1)
        for i in range(x_batch.size(0)):
            if clean[i] == y_batch[i] and adv[i] != y_batch[i]:
                out.append(
                    (
                        x_batch[i].cpu(),
                        int(y_batch[i].item()),
                        int(clean[i].item()),
                        int(adv[i].item()),
                        mask[i].cpu(),
                        grad[i].cpu(),
                    )
                )
                if len(out) >= n_samples:
                    return out
    return out


def visualize_attacker_grid(classifier, k: int, n_samples: int, device, loader, path: Path):
    att = KPixelAttacker(num_pixels=k).to(device)
    att.load_state_dict(torch.load(CHECKPOINT_DIR / f"attacker_k{k}.pt", map_location=device))
    att.eval()

    samples = _pick_successful_samples(classifier, att, loader, device, n_samples)
    if not samples:
        print(f"K={k}: nenhum ataque bem-sucedido para visualizar.")
        return

    n = len(samples)
    fig, axes = plt.subplots(4, n, figsize=(2.4 * n, 9.5))
    if n == 1:
        axes = axes.reshape(4, 1)
    fig.suptitle(
        f"Ataque white-box com K = {k} pixel(s) | rede atacante + gradiente do classificador",
        fontsize=12,
    )

    for col, (x, y, clean, adv, mask, grad) in enumerate(samples):
        with torch.no_grad():
            xt = x.unsqueeze(0).to(device)
            mt = mask.unsqueeze(0).to(device)
            grad_t = grad.unsqueeze(0).to(device)
            att_x, _, _ = att.attack_deterministic(xt, grad_t)
            x_adv = att_x[0].cpu()
        diff = (x_adv - x).abs()

        axes[0, col].imshow(_to_image(x))
        axes[0, col].set_title(f"Original\n(rotulo: {CLASS_NAMES[y]})", fontsize=9)
        axes[0, col].axis("off")

        axes[1, col].imshow(_to_image(x_adv))
        axes[1, col].set_title(
            f"Atacada\n(predito: {CLASS_NAMES[adv]})", fontsize=9, color="red"
        )
        axes[1, col].axis("off")
        ys, xs = torch.where(mask[0] > 0.5)
        for ry, rx in zip(ys.tolist(), xs.tolist()):
            axes[1, col].add_patch(Circle((rx, ry), radius=1.2, fill=False, color="yellow", lw=1.5))

        amp = diff.max().item() + 1e-8
        axes[2, col].imshow((_to_image(diff / amp)))
        axes[2, col].set_title("|x_adv - x| (amplificada)", fontsize=9)
        axes[2, col].axis("off")

        sal = grad.abs().mean(dim=0).numpy()
        axes[3, col].imshow(sal, cmap="hot")
        axes[3, col].set_title("Saliencia (|grad|)", fontsize=9)
        axes[3, col].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"K={k}: grid salvo em {path}")


def plot_summary(metrics: list[tuple[int, float, float]], path: Path) -> None:
    """Plota barras de sucesso por K (NN+grad x aleatorio)."""
    ks = [m[0] for m in metrics]
    nn_rates = [m[1] for m in metrics]
    rnd_rates = [m[2] for m in metrics]

    x = np.arange(len(ks))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(x - width / 2, nn_rates, width, label="NN + gradiente (este trabalho)", color="#d62728")
    ax.bar(x + width / 2, rnd_rates, width, label="K pixels aleatorios", color="#7f7f7f")
    ax.set_xticks(x)
    ax.set_xticklabels([f"K = {k}" for k in ks])
    ax.set_ylabel("Taxa de sucesso (gato-vs-cachorro)")
    ax.set_title("Taxa de sucesso do ataque por numero de pixels")
    ax.set_ylim(0.0, max(max(nn_rates), max(rnd_rates)) * 1.25 + 0.02)
    for xi, v in zip(x - width / 2, nn_rates):
        ax.text(xi, v + 0.005, f"{v:.1%}", ha="center", fontsize=9)
    for xi, v in zip(x + width / 2, rnd_rates):
        ax.text(xi, v + 0.005, f"{v:.1%}", ha="center", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Sumario salvo em {path}")


def main() -> None:
    fix_seed(0)
    device = torch.device("cpu")

    classifier = CatDogClassifier().to(device)
    classifier.load_state_dict(torch.load(CHECKPOINT_DIR / "classifier.pt", map_location=device))
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad_(False)

    _, test_loader = get_catdog_loaders(batch_size=128)

    summary: list[tuple[int, float, float]] = []
    for k in (1, 3, 5):
        visualize_attacker_grid(
            classifier, k=k, n_samples=6, device=device, loader=test_loader,
            path=OUTPUT_DIR / f"grid_k{k}.png",
        )

    # Reutiliza o evaluate.py para nao recalcular tudo
    from evaluate import evaluate_attacker, evaluate_random_baseline

    for k in (1, 3, 5):
        att = KPixelAttacker(num_pixels=k).to(device)
        att.load_state_dict(torch.load(CHECKPOINT_DIR / f"attacker_k{k}.pt", map_location=device))
        m = evaluate_attacker(classifier, att, test_loader, device, k)
        r = evaluate_random_baseline(classifier, test_loader, device, k)
        summary.append((k, m["success_rate"], r["success_rate"]))

    plot_summary(summary, OUTPUT_DIR / "summary_success.png")


if __name__ == "__main__":
    main()
