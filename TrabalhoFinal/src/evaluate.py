"""Avaliacao consolidada dos atacantes (K = 1, 3, 5) contra o classificador.

Metricas calculadas (sobre as imagens originalmente corretas):
  - Taxa de sucesso (fracao em que o ataque alterou a predicao)
  - Confianca media na classe errada apos o ataque (Adv. Prob. Labels do paper)
  - Mudanca media |x_adv - x| L1 por imagem
  - Comparacao com um BASELINE aleatorio: K pixels em posicoes aleatorias
    com valores aleatorios em [0, 1].

Os resultados sao impressos e gravados em outputs/metrics.txt.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F

from attacker import KPixelAttacker, compute_input_gradient
from classifier import CatDogClassifier
from data import CLASS_NAMES, fix_seed, get_catdog_loaders

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def random_pixel_attack(x: torch.Tensor, k: int) -> torch.Tensor:
    """Baseline: substitui K pixels aleatorios por valores RGB aleatorios."""
    b, c, h, w = x.shape
    x_adv = x.clone()
    for i in range(b):
        idx = torch.randperm(h * w)[:k]
        rows = idx // w
        cols = idx % w
        for r, co in zip(rows.tolist(), cols.tolist()):
            x_adv[i, :, r, co] = torch.rand(c, device=x.device)
    return x_adv


@torch.no_grad()
def _classifier_eval(classifier, x):
    return classifier(x)


def evaluate_attacker(classifier, attacker, loader, device, k):
    attacker.eval()
    classifier.eval()
    total_correct = 0
    total_succ = 0
    total_change = 0.0
    total_conf_wrong = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        clean_preds = _classifier_eval(classifier, x).argmax(dim=1)
        grad = compute_input_gradient(classifier, x, y)
        x_adv, mask, value_map = attacker.attack_deterministic(x, grad)
        adv_logits = _classifier_eval(classifier, x_adv)
        adv_preds = adv_logits.argmax(dim=1)
        adv_probs = F.softmax(adv_logits, dim=1)

        was_correct = clean_preds == y
        is_wrong = adv_preds != y
        successes = was_correct & is_wrong

        total_correct += was_correct.sum().item()
        total_succ += successes.sum().item()
        total_change += (x_adv - x).abs().sum().item()
        if successes.any():
            wrong_class = adv_preds[successes]
            total_conf_wrong += adv_probs[successes].gather(1, wrong_class.unsqueeze(1)).sum().item()

    success_rate = total_succ / max(total_correct, 1)
    avg_change = total_change / max(len(loader.dataset), 1)
    avg_conf_wrong = total_conf_wrong / max(total_succ, 1)
    return {
        "k": k,
        "success_rate": success_rate,
        "avg_l1_change": avg_change,
        "avg_conf_wrong": avg_conf_wrong,
        "successes": total_succ,
        "correct_total": total_correct,
    }


def evaluate_random_baseline(classifier, loader, device, k):
    classifier.eval()
    total_correct = 0
    total_succ = 0
    total_change = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        clean_preds = _classifier_eval(classifier, x).argmax(dim=1)
        x_adv = random_pixel_attack(x, k)
        adv_preds = _classifier_eval(classifier, x_adv).argmax(dim=1)

        was_correct = clean_preds == y
        is_wrong = adv_preds != y
        total_correct += was_correct.sum().item()
        total_succ += (was_correct & is_wrong).sum().item()
        total_change += (x_adv - x).abs().sum().item()

    return {
        "k": k,
        "success_rate": total_succ / max(total_correct, 1),
        "avg_l1_change": total_change / max(len(loader.dataset), 1),
    }


def main() -> None:
    fix_seed(0)
    device = torch.device("cpu")

    classifier = CatDogClassifier().to(device)
    classifier.load_state_dict(torch.load(CHECKPOINT_DIR / "classifier.pt", map_location=device))
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad_(False)

    _, test_loader = get_catdog_loaders(batch_size=128)

    clean_acc = sum(
        (_classifier_eval(classifier, x.to(device)).argmax(dim=1) == y.to(device)).sum().item()
        for x, y in test_loader
    ) / len(test_loader.dataset)

    lines: list[str] = []
    lines.append(f"Classes: {CLASS_NAMES}")
    lines.append(f"Acuracia do classificador sobre o teste limpo: {clean_acc:.4f}")
    lines.append("")
    lines.append(f"{'K':>3} | {'metodo':>10} | {'sucesso':>8} | {'conf_err':>8} | {'mud_L1':>8}")
    lines.append("-" * 56)

    for k in (1, 3, 5):
        ckpt = CHECKPOINT_DIR / f"attacker_k{k}.pt"
        if not ckpt.exists():
            lines.append(f"{k:>3} | (atacante ausente em {ckpt})")
            continue
        att = KPixelAttacker(num_pixels=k).to(device)
        att.load_state_dict(torch.load(ckpt, map_location=device))

        m = evaluate_attacker(classifier, att, test_loader, device, k)
        r = evaluate_random_baseline(classifier, test_loader, device, k)

        lines.append(
            f"{k:>3} | {'NN+grad':>10} | "
            f"{m['success_rate']:>8.4f} | {m['avg_conf_wrong']:>8.4f} | {m['avg_l1_change']:>8.4f}"
        )
        lines.append(
            f"{k:>3} | {'aleatorio':>10} | "
            f"{r['success_rate']:>8.4f} | {'-':>8} | {r['avg_l1_change']:>8.4f}"
        )

    report = "\n".join(lines)
    print(report)
    (OUTPUT_DIR / "metrics.txt").write_text(report + "\n")
    print(f"\nMetricas salvas em {OUTPUT_DIR / 'metrics.txt'}")


if __name__ == "__main__":
    main()
