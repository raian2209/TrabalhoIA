"""Site para demonstracao da CNN treinada no MNIST.

Sorteia uma imagem aleatoria do conjunto de teste, executa a rede neural
treinada e exibe o digito previsto, a probabilidade e a distribuicao completa.

Uso:
    python app.py
    # abra http://127.0.0.1:5000 no navegador
"""

from __future__ import annotations

import base64
import io
import random
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template
from PIL import Image
from torchvision import datasets

from src.data import CLASS_NAMES, build_transform
from src.model import DigitCNN

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "checkpoints" / "digit_cnn_best.pt"
DATA_ROOT = ROOT / "data"

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Carrega uma unica vez: modelo treinado e base de teste (imagens cruas PIL).
_transform = build_transform()
_test_raw = datasets.MNIST(root=str(DATA_ROOT), train=False, download=True)

_model = DigitCNN().to(device)
if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint nao encontrado em {CHECKPOINT}. "
        "Treine o modelo antes: python -m src.train"
    )
_model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
_model.eval()


def _pil_to_base64(img: Image.Image, scale: int = 8) -> str:
    """Amplia a imagem 28x28 e retorna um data URI PNG para exibir no navegador."""
    big = img.resize((28 * scale, 28 * scale), Image.NEAREST)
    buffer = io.BytesIO()
    big.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict")
def predict():
    """Sorteia uma imagem de teste, classifica e devolve o resultado em JSON."""
    idx = random.randrange(len(_test_raw))
    img, true_label = _test_raw[idx]  # img: PIL Image (L), true_label: int

    tensor = _transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    pred = int(probs.argmax().item())
    confidence = float(probs[pred].item())

    return jsonify(
        {
            "indice": idx,
            "imagem": _pil_to_base64(img),
            "predicao": CLASS_NAMES[pred],
            "confianca": round(confidence * 100, 2),
            "rotulo_verdadeiro": CLASS_NAMES[int(true_label)],
            "acertou": pred == int(true_label),
            "distribuicao": [round(float(p) * 100, 2) for p in probs],
        }
    )


if __name__ == "__main__":
    print(f"Modelo carregado ({device}). Acesse http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
