# Reconhecimento de Dígitos Manuscritos com CNN (MNIST)

Estudo de caso de uma rede neural convolucional (CNN) para reconhecimento de
dígitos manuscritos da base **MNIST**, com artigo no padrão **SBC** preenchido
com os resultados reais do treino.

## Estrutura

```
src/
  data.py        # carregamento e pré-processamento do MNIST
  model.py       # arquitetura da rede (DigitCNN)
  train.py       # laço de treino, avaliação e métricas
  visualize.py   # geração das figuras do artigo
gerar_artigo.py  # gera o artigo .docx no estilo SBC com os resultados
app.py           # site Flask: sorteia uma imagem e mostra a predição da rede
templates/
  index.html     # interface web da demonstração
outputs/         # metrics.json, log de treino (gerado)
figures/         # figuras do artigo (gerado)
checkpoints/     # pesos do melhor modelo (gerado)
artigo_sbc_mnist.docx  # artigo final (gerado)
```

## Como executar

```bash
pip install -r requirements.txt

python -m src.train --epochs 8 --batch-size 128 --lr 1e-3
python -m src.visualize
python gerar_artigo.py
```

## Site de demonstração

Após treinar (gera `checkpoints/digit_cnn_best.pt`), suba o site:

```bash
python app.py
# abra http://127.0.0.1:5000 no navegador
```

Clique em **"Sortear imagem"**: o servidor escolhe aleatoriamente uma imagem do
conjunto de teste do MNIST, a rede prevê o dígito e a página mostra a imagem
ampliada, o número previsto, a confiança, o rótulo verdadeiro (acertou/errou) e
a distribuição de probabilidade entre os 10 dígitos.

## Divisão dos dados

As 60.000 imagens de treino do MNIST são divididas em **50.000 treino + 10.000
validação** (partição reproduzível, semente fixa); as **10.000 de teste**
oficiais ficam intactas. A validação seleciona o melhor modelo; o teste é usado
só na avaliação final.

## Resultado obtido

| Métrica                          | Valor                |
|----------------------------------|----------------------|
| Acurácia no teste (10.000)       | ver `outputs/metrics.json` |
| Parâmetros treináveis            | 468.202              |
| Otimizador / perda               | Adam / Cross-Entropy |
| Divisão treino/val/teste         | 50.000 / 10.000 / 10.000 |
| Tempo de treino (CPU, 8 ép.)     | ~7 min               |

O artigo final está em `artigo_sbc_mnist.docx`.
