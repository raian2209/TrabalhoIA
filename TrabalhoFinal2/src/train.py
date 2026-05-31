"""Treino e avaliacao da CNN no MNIST, com registro de metricas e artefatos.

Executa o laco de treino com Adam + cross-entropy, avalia a cada epoca no
conjunto de teste e salva: checkpoint dos pesos, historico de metricas (JSON),
matriz de confusao e relatorio por classe. Esses artefatos alimentam o artigo.

Uso:
    python -m src.train --epochs 8 --batch-size 128 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import CLASS_NAMES, fix_seed, get_mnist_loaders
from .model import DigitCNN

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
CHECKPOINTS = ROOT / "checkpoints"


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str) -> tuple[float, float, np.ndarray]:
    """Retorna (perda media, acuracia, matriz de confusao) no conjunto dado."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, total = 0.0, 0, 0
    confusion = np.zeros((10, 10), dtype=np.int64)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
            confusion[t, p] += 1
    return total_loss / total, correct / total, confusion


def train_one_epoch(model, loader, optimizer, criterion, device) -> tuple[float, float]:
    """Uma passada de treino. Retorna (perda media, acuracia de treino)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


def per_class_report(confusion: np.ndarray) -> dict[str, dict[str, float]]:
    """Precisao, revocacao e F1 por classe a partir da matriz de confusao."""
    report: dict[str, dict[str, float]] = {}
    for c in range(10):
        tp = confusion[c, c]
        fn = confusion[c, :].sum() - tp
        fp = confusion[:, c].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        report[CLASS_NAMES[c]] = {
            "precisao": round(float(precision), 4),
            "revocacao": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "suporte": int(confusion[c, :].sum()),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino da CNN no MNIST")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUTS.mkdir(exist_ok=True)
    CHECKPOINTS.mkdir(exist_ok=True)
    fix_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, test_loader = get_mnist_loaders(
        batch_size=args.batch_size, seed=args.seed
    )
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    model = DigitCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Dispositivo: {device} | Parametros treinaveis: {model.count_parameters():,}")
    print(f"Particao -> treino: {n_train} | validacao: {n_val} | teste: {n_test}")

    history: list[dict[str, float]] = []
    best_val_acc = 0.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        # Selecao de modelo feita na VALIDACAO; o teste so e tocado no final.
        val_loss, val_acc, _ = evaluate(model, val_loader, device)
        history.append(
            {
                "epoca": epoch,
                "perda_treino": round(tr_loss, 4),
                "acc_treino": round(tr_acc, 4),
                "perda_validacao": round(val_loss, 4),
                "acc_validacao": round(val_acc, 4),
            }
        )
        print(
            f"Epoca {epoch:02d}/{args.epochs} | "
            f"treino: perda {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"validacao: perda {val_loss:.4f} acc {val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINTS / "digit_cnn_best.pt")

    elapsed = time.time() - t0

    # Avaliacao final, no conjunto de TESTE, com o melhor checkpoint (escolhido na validacao).
    model.load_state_dict(torch.load(CHECKPOINTS / "digit_cnn_best.pt", map_location=device))
    final_loss, final_acc, confusion = evaluate(model, test_loader, device)
    report = per_class_report(confusion)

    metrics = {
        "config": {
            "epocas": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "otimizador": "Adam",
            "funcao_perda": "CrossEntropyLoss",
            "dispositivo": device,
            "parametros_treinaveis": model.count_parameters(),
            "tempo_treino_s": round(elapsed, 1),
            "n_treino": n_train,
            "n_validacao": n_val,
            "n_teste": n_test,
        },
        "acuracia_teste_final": round(final_acc, 4),
        "perda_teste_final": round(final_loss, 4),
        "melhor_acuracia_validacao": round(best_val_acc, 4),
        "history": history,
        "matriz_confusao": confusion.tolist(),
        "relatorio_por_classe": report,
    }
    (OUTPUTS / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nAcuracia final no teste: {final_acc:.4f} | tempo total: {elapsed:.1f}s")
    print(f"Metricas salvas em {OUTPUTS / 'metrics.json'}")


if __name__ == "__main__":
    main()
