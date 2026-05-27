"""Treinamento da rede atacante (K pixels) por adversarial training white-box.

Procedimento:
  1. Carrega o classificador (gato vs cachorro) ja treinado e congela seus pesos.
  2. Para cada batch de imagens, calcula d(CE)/dx (gradiente da perda em
     relacao a entrada) usando o classificador congelado. Esse gradiente
     entra como canal adicional do atacante (saliencia tipo FGSM/JSMA).
  3. O atacante propoe K (posicao, valor RGB) que sao aplicados via mascara
     Gumbel-Softmax sem reposicao.
  4. As imagens perturbadas passam pelo classificador congelado e o
     gradiente da perda adversarial flui de volta SOMENTE para o atacante.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import optim

from attacker import KPixelAttacker, compute_input_gradient, margin_loss
from classifier import CatDogClassifier
from data import fix_seed, get_catdog_loaders

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treinamento da rede atacante.")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kappa", type=float, default=5.0)
    parser.add_argument("--tau-start", type=float, default=1.0)
    parser.add_argument("--tau-end", type=float, default=0.3)
    parser.add_argument("--k", type=int, default=1, help="Numero de pixels a atacar.")
    parser.add_argument(
        "--classifier-path",
        type=str,
        default=str(CHECKPOINT_DIR / "classifier.pt"),
    )
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def attack_success_rate(attacker, classifier, loader, device) -> tuple[float, float]:
    """Avalia taxa de sucesso (sobre imagens originalmente corretas)."""
    attacker.eval()
    classifier.eval()
    successes = 0
    total_correct = 0
    total_change = 0.0
    total_seen = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            clean_preds = classifier(x).argmax(dim=1)
        grad = compute_input_gradient(classifier, x, y)
        x_adv, mask, value_map = attacker.attack_deterministic(x, grad)
        with torch.no_grad():
            adv_preds = classifier(x_adv).argmax(dim=1)

        was_correct = clean_preds == y
        is_wrong_now = adv_preds != y
        successes += (was_correct & is_wrong_now).sum().item()
        total_correct += was_correct.sum().item()
        total_change += (x_adv - x).abs().sum().item()
        total_seen += y.size(0)

    success_rate = successes / max(total_correct, 1)
    avg_change_per_image = total_change / max(total_seen, 1)
    return success_rate, avg_change_per_image


def main() -> None:
    args = parse_args()
    if args.output is None:
        args.output = str(CHECKPOINT_DIR / f"attacker_k{args.k}.pt")

    fix_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device} | K = {args.k}")

    train_loader, test_loader = get_catdog_loaders(batch_size=args.batch_size)

    classifier = CatDogClassifier().to(device)
    classifier.load_state_dict(torch.load(args.classifier_path, map_location=device))
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad_(False)

    attacker = KPixelAttacker(num_pixels=args.k).to(device)
    optimizer = optim.Adam(attacker.parameters(), lr=args.lr)

    num_steps = args.epochs * len(train_loader)
    step = 0

    for epoch in range(1, args.epochs + 1):
        attacker.train()
        running_loss = 0.0
        running_total = 0
        running_succ = 0
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)

            progress = step / max(num_steps - 1, 1)
            tau = args.tau_start + progress * (args.tau_end - args.tau_start)
            step += 1

            grad = compute_input_gradient(classifier, x, y)
            x_adv, mask, value_map = attacker(x, grad, tau=tau, hard=True)
            logits = classifier(x_adv)

            loss = margin_loss(logits, y, kappa=args.kappa)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            running_total += y.size(0)
            running_succ += (logits.argmax(dim=1) != y).sum().item()

            if batch_idx % 25 == 0:
                avg_loss = running_loss / running_total
                fool_rate = running_succ / running_total
                print(
                    f"K={args.k} epoch {epoch} batch {batch_idx}/{len(train_loader)} "
                    f"tau={tau:.3f} loss={avg_loss:.4f} fool={fool_rate:.4f}"
                )

        succ_rate, avg_change = attack_success_rate(attacker, classifier, test_loader, device)
        print(
            f"== K={args.k} epoca {epoch} | sucesso (teste, deterministico) = "
            f"{succ_rate:.4f} | mudanca media |x_adv - x| = {avg_change:.4f}"
        )

    torch.save(attacker.state_dict(), args.output)
    print(f"Atacante K={args.k} salvo em {args.output}")


if __name__ == "__main__":
    main()
