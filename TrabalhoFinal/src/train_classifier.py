"""Treina o classificador CNN binario gato-vs-cachorro sobre CIFAR-10.

Esta e a primeira rede do experimento: a "vitima" do ataque.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn, optim

from classifier import CatDogClassifier
from data import fix_seed, get_catdog_loaders

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treinamento do classificador gato-vs-cachorro.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=str(CHECKPOINT_DIR / "classifier.pt"))
    return parser.parse_args()


def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def main() -> None:
    args = parse_args()
    fix_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    train_loader, test_loader = get_catdog_loaders(batch_size=args.batch_size)
    print(f"Treino: {len(train_loader.dataset)} amostras | Teste: {len(test_loader.dataset)} amostras")

    model = CatDogClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            running_correct += (logits.argmax(dim=1) == y).sum().item()
            running_total += y.size(0)

            if batch_idx % 25 == 0:
                avg_loss = running_loss / running_total
                acc = running_correct / running_total
                print(
                    f"epoch {epoch} batch {batch_idx}/{len(train_loader)} "
                    f"loss={avg_loss:.4f} acc={acc:.4f}"
                )

        scheduler.step()
        test_acc = evaluate(model, test_loader, device)
        print(f"== epoca {epoch} concluida | acuracia teste = {test_acc:.4f}")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), args.output)
            print(f"   (novo melhor, salvo em {args.output})")

    print(f"\nMelhor acuracia teste: {best_acc:.4f}")


if __name__ == "__main__":
    main()
