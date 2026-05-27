# Trabalho Final — Ataque adversarial de K pixels (gato vs cachorro)

Replicacao em Python (PyTorch) das ideias dos artigos:

- Su, Vargas, Sakurai (2019), **"One Pixel Attack for Fooling Deep Neural Networks"** (arXiv 1710.08864).
- Huang et al. (2017), **"Adversarial Attacks on Neural Network Policies"** (arXiv 1702.02284).

Sao duas redes neurais:

1. **Classificador (vitima)** — uma CNN treinada para distinguir **gato** e
   **cachorro** no subconjunto correspondente do CIFAR-10.
2. **Atacante** — uma segunda rede neural treinada por *adversarial training*
   white-box. A perda flui pelo classificador congelado e ajusta apenas o
   atacante, que aprende a perturbar APENAS K pixels (K ∈ {1, 3, 5}) para
   inverter a predicao do classificador.

A selecao discreta de pixels e diferenciavel via **Gumbel-Softmax sem
reposicao** (variante Plackett-Luce do top-K). O gradiente da perda de
classificacao em relacao a entrada e ALIMENTADO como canal extra do atacante,
fornecendo a informacao de saliencia (caracteristica do ataque white-box,
no espirito de FGSM/JSMA).

## Estrutura do projeto

```
TrabalhoFinal/
├── 1702.02284v1.pdf          # paper #2 (Huang et al.)
├── 1710.08864v7.pdf          # paper #1 (Su et al.)
├── src/
│   ├── data.py               # CIFAR-10 cat/dog -> DataLoaders binarios
│   ├── classifier.py         # CNN vitima (3x32x32 RGB -> 2 classes)
│   ├── attacker.py           # Rede atacante + Gumbel-Softmax top-K
│   ├── train_classifier.py   # Treino da vitima
│   ├── train_attacker.py     # Treino white-box do atacante (por K)
│   ├── evaluate.py           # Metricas vs baseline aleatorio
│   └── visualize.py          # Geracao de grids e graficos
├── checkpoints/              # Pesos treinados (.pt)
├── outputs/                  # metrics.txt + figuras .png
├── requirements.txt
├── README.md
└── RELATORIO.md              # Relatorio tecnico
```

## Como reproduzir

```bash
# 1. Ambiente
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Treinar o classificador gato vs cachorro
.venv/bin/python src/train_classifier.py --epochs 10

# 3. Treinar os 3 atacantes (1, 3 e 5 pixels)
.venv/bin/python src/train_attacker.py --epochs 5 --k 1
.venv/bin/python src/train_attacker.py --epochs 5 --k 3
.venv/bin/python src/train_attacker.py --epochs 5 --k 5

# 4. Avaliar (gera outputs/metrics.txt)
.venv/bin/python src/evaluate.py

# 5. Visualizar (gera outputs/grid_k*.png e outputs/summary_success.png)
.venv/bin/python src/visualize.py
```

## Resultados (resumidos)

Sobre o conjunto de teste (2000 imagens gato/cachorro do CIFAR-10):

| K | Atacante NN+grad | Aleatorio (mesmo K) | Confianca media na classe errada |
|---|------------------|---------------------|----------------------------------|
| 1 | 14.2%            | 1.1%                | 71.8%                            |
| 3 | 25.6%            | 1.8%                | 82.9%                            |
| 5 | 33.6%            | 2.9%                | 84.9%                            |

(Acuracia do classificador no teste limpo: 83.75%.)

Detalhes, discussao e ligacao com os papers em `RELATORIO.md`.
