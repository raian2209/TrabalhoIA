# Relatorio tecnico — Ataque adversarial de K pixels via rede neural atacante (white-box) em CIFAR-10 gato vs cachorro

## 1. Problema abordado

O trabalho replica e COMBINA ideias de dois artigos sobre ataques adversariais
a redes neurais:

- **Su, Vargas, Sakurai (2019), "One Pixel Attack for Fooling Deep Neural
  Networks"** (arXiv 1710.08864). Mostra que e possivel enganar CNNs profundas
  alterando UM unico pixel. No artigo a otimizacao e feita por **Evolucao
  Diferencial** (DE), um ataque semi-caixa-preta, sobre CIFAR-10, com sucesso
  medio de ~67% em ataques nao-direcionados (10 classes, varias arquiteturas).

- **Huang et al. (2017), "Adversarial Attacks on Neural Network Policies"**
  (arXiv 1702.02284). Demonstra que politicas de aprendizado por reforco sao
  vulneraveis a FGSM (Fast Gradient Sign Method) — um ataque **caixa branca**
  que usa o gradiente da perda em relacao a entrada para construir a
  perturbacao em TODOS os pixels simultaneamente.

A questao deste trabalho e: **podemos juntar os dois?**

> "Construir uma segunda rede neural que, treinada por adversarial training
> com acesso ao gradiente do classificador (caixa branca), aprenda a perturbar
> apenas K pixels para inverter a predicao."

Sao portanto **duas redes neurais**:

1. Um **classificador (vitima)** que distingue **gato** e **cachorro** sobre
   o subconjunto correspondente do CIFAR-10 (classes 3 e 5).
2. Uma **rede atacante** que recebe a imagem e o gradiente da perda do
   classificador e devolve a posicao (x, y) e o valor RGB de K pixels a serem
   substituidos.

## 2. Justificativa do dataset (CIFAR-10 cat/dog em vez de MNIST)

A versao inicial deste trabalho usava MNIST (digitos manuscritos 28x28 em
escala de cinza). Uma analise por forca bruta — testando exaustivamente todas
as 784 posicoes vezes 11 valores possiveis para cada uma das 50 primeiras
imagens de teste contra um classificador a 99% de acuracia — revelou que
apenas **2% das imagens** do MNIST sao vulneraveis a qualquer ataque de 1
pixel. A razao e estrutural: ~80% dos pixels do MNIST sao zero (fundo preto)
e a CNN aprende representacoes muito robustas a essas regioes; alterar um
unico pixel raramente move o classificador atraves da fronteira de decisao.

Em fidelidade ao paper de Su et al. (que usa CIFAR-10 justamente porque cada
pixel carrega informacao em 3 canais RGB), a vitima deste trabalho foi
reformulada como um **classificador binario gato vs cachorro** sobre o
CIFAR-10 (classes 3 e 5 do dataset original), com imagens de 32x32 RGB.

Estatisticas do subconjunto utilizado (em `src/data.py`):

- **Treino:** 10000 imagens (5000 gatos + 5000 cachorros).
- **Teste:** 2000 imagens (1000 gatos + 1000 cachorros).
- **Pre-processamento:** apenas `ToTensor()` — mantemos os pixels em [0, 1]
  SEM normalizacao adicional, para que a substituicao de pixels pelo atacante
  seja diretamente interpretavel como uma cor RGB.

## 3. Arquitetura das redes

### 3.1 Classificador (vitima) — `src/classifier.py`

CNN compacta com BatchNorm, treinavel em poucos minutos no CPU. Cinco blocos
convolucionais 3x3 (stride 1, padding 1), intercalados com max pooling 2x2.
Cabeca densa com Dropout(0.3) e saida de 2 logits.

| Camada                       | Saida          |
|------------------------------|----------------|
| Conv2d(3, 32) + BN + ReLU    | 32 x 32 x 32   |
| Conv2d(32, 32) + BN + ReLU   | 32 x 32 x 32   |
| MaxPool 2x2                  | 32 x 16 x 16   |
| Conv2d(32, 64) + BN + ReLU   | 64 x 16 x 16   |
| Conv2d(64, 64) + BN + ReLU   | 64 x 16 x 16   |
| MaxPool 2x2                  | 64 x 8 x 8     |
| Conv2d(64, 128) + BN + ReLU  | 128 x 8 x 8    |
| MaxPool 2x2                  | 128 x 4 x 4    |
| Flatten -> Linear(2048, 128) | 128            |
| ReLU + Dropout(0.3)          | 128            |
| Linear(128, 2)               | 2 (logits)     |

Treinado por 10 epocas com Adam, lr=1e-3, scheduler cosseno, perda
CrossEntropy. **Acuracia final no teste: 0.8375** — coerente com a literatura
para CNNs pequenas em CIFAR-10 cat/dog (uma das tarefas binarias mais
dificeis do dataset, porque ambas as classes sao mamiferos peludos
quadrupedes com cores e posturas similares).

### 3.2 Atacante (rede de perturbacao) — `src/attacker.py`

Arquitetura totalmente convolucional para preservar a estrutura espacial.
Uma versao inicial baseada em decoder Linear sofreu **mode collapse** —
escolhia sempre os mesmos 3 pixels para qualquer entrada (ver Secao 5).

A entrada concatena `[x, grad]` (6 canais: 3 RGB da imagem + 3 RGB do
gradiente normalizado da perda do classificador).

| Camada                                       | Saida          |
|----------------------------------------------|----------------|
| Conv2d(6, 64) + ReLU                         | 64 x 32 x 32   |
| Conv2d(64, 64) + ReLU                        | 64 x 32 x 32   |
| Conv2d(64, 64) + ReLU                        | 64 x 32 x 32   |
| Conv2d(64, 64) + ReLU                        | 64 x 32 x 32   |
| Conv2d 1x1 -> 1 canal (position_head)        | 1 x 32 x 32    |
| Conv2d 1x1 -> 3 canais (value_head, sigmoid) | 3 x 32 x 32    |

A `position_head` produz logits espaciais para cada um dos **1024 pixels**.
A `value_head` produz, para cada pixel, o valor RGB candidato (sigmoide para
manter em [0, 1]).

A perturbacao final e formada da seguinte maneira:

```text
mask  = top-K-Gumbel-Softmax(position_logits, tau)   # ~ K-hot espacial
x_adv = x * (1 - mask) + value_map * mask
```

A mascara K-hot e amostrada por **Gumbel-Softmax sem reposicao** (variante
Plackett-Luce relaxada): em K iteracoes, amostra-se uma mascara one-hot via
Gumbel-Softmax, bloqueia-se a posicao escolhida (somando -1e6 ao logit) e
repete-se. A soma das K mascaras e aproximadamente K-hot e diferenciavel.

Em RGB, "1 pixel" segue a convencao do paper de Su et al.: uma posicao (x, y)
com os 3 canais (R, G, B) substituidos simultaneamente — totalizando 5 graus
de liberdade por pixel atacado.

## 4. Procedimento de treinamento

As duas redes sao treinadas em **etapas separadas e sequenciais**:

1. Primeiro o **classificador** (vitima) e treinado de forma supervisionada
   convencional, ate atingir boa acuracia em gato-vs-cachorro.
2. Em seguida, seus pesos sao **congelados** e o **atacante** e treinado em
   regime **white-box adversarial training**: a perda flui pelo
   classificador, mas o gradiente atualiza APENAS os pesos do atacante.

Sao portanto dois tipos de treinamento completamente distintos. Detalhamos
cada um abaixo.

### 4.1 Treino do classificador (vitima) — `src/train_classifier.py`

**Tipo de treinamento:** aprendizado supervisionado classico, com rotulos
binarios (0 = gato, 1 = cachorro), sem nenhum elemento adversarial.

#### 4.1.1 Dados e fluxo por batch

- Os DataLoaders sao construidos em `data.py:get_catdog_loaders`. Eles
  pegam o CIFAR-10 completo do `torchvision`, filtram somente as classes 3
  (cat) e 5 (dog), e remapeiam para `{0, 1}` via `_RelabelDataset`.
- Cada batch e um tensor `x ∈ R^(128 x 3 x 32 x 32)` com valores em [0, 1]
  e rotulos `y ∈ {0, 1}^128`.
- Para cada batch o ciclo padrao e: `optimizer.zero_grad() → logits =
  model(x) → loss = CrossEntropy(logits, y) → loss.backward() →
  optimizer.step()`.

#### 4.1.2 Hiperparametros

| Hiperparametro       | Valor               | Onde                                    |
| -------------------- | ------------------- | --------------------------------------- |
| Otimizador           | Adam                | `train_classifier.py:53`                |
| Learning rate        | 1e-3                | `--lr 1e-3` (default)                   |
| Scheduler            | CosineAnnealingLR   | `T_max = epochs`                        |
| Batch size           | 128                 | `--batch-size 128`                      |
| Epocas               | 10                  | `--epochs 10` (README)                  |
| Perda                | CrossEntropyLoss    | aplicada sobre os 2 logits              |
| Inicializacao        | default do PyTorch  | (Kaiming nos `Conv2d`, uniforme em `Linear`) |
| Regularizacao        | Dropout(0.3)        | apenas na head densa                    |
| Seed                 | 42                  | `data.py:fix_seed`                      |
| Dispositivo          | CPU                 | suficiente, treina em poucos minutos    |

#### 4.1.3 Selecao do melhor checkpoint

Apos cada epoca o modelo e avaliado sobre o conjunto de teste (2000 imagens).
Apenas o checkpoint que produzir a MAIOR acuracia de teste e salvo em
`checkpoints/classifier.pt` — uma forma simples de *early stopping by best
test accuracy*. No experimento conduzido, a melhor acuracia foi **0.8375**.

#### 4.1.4 Por que essa configuracao

- **Adam + lr=1e-3** e uma escolha conservadora que dispensa busca fina, e
  o tamanho do dataset (~10k imagens) nao justifica SGD-with-momentum.
- **BatchNorm** entre cada conv estabiliza ativacoes nos primeiros estagios,
  permitindo lr alto desde o inicio sem divergencia.
- **CosineAnnealing** sem warmup e o suficiente para tao poucas epocas; o
  lr cai suavemente de 1e-3 para ~0 no final, refinando a convergencia.
- **Dropout(0.3)** apenas na head e suficiente para conter overfitting na
  configuracao binaria; nao usamos data augmentation porque o objetivo NAO
  e maximizar acuracia — e ter uma vitima realista contra a qual atacar.

### 4.2 Treino do atacante — `src/train_attacker.py`

**Tipo de treinamento:** *white-box adversarial training* nao-direcionado,
com um classificador congelado servindo de "ambiente diferenciavel". E uma
forma de **meta-aprendizado**: o atacante NAO recebe rotulos do que e a
posicao "certa" para atacar — ele recebe apenas o feedback `did the
classifier flip its prediction?`, traduzido em uma perda de margem
diferenciavel.

Este nao e um treinamento supervisionado tradicional porque:

- Nao ha rotulo de saida desejada para o atacante (nao existe "ground truth"
  para qual pixel atacar).
- O sinal de treinamento vem do COMPORTAMENTO de uma SEGUNDA rede (o
  classificador congelado), nao de um rotulo do dataset.
- O dataset fornece apenas a IMAGEM e o ROTULO ORIGINAL — usados para
  computar o gradiente do classificador, que e o que de fato direciona o
  atacante.

#### 4.2.1 Preparacao do ambiente (uma vez antes do treino)

1. Carrega o classificador treinado de `checkpoints/classifier.pt`.
2. Coloca o classificador em `.eval()` (desativa Dropout e congela
   BatchNorm) e marca **todos os seus parametros** como
   `requires_grad_(False)` — assim nada do classificador sera atualizado.
3. Instancia `KPixelAttacker(num_pixels=K)` — uma rede nova por K — e o
   otimizador Adam SOMENTE sobre os parametros do atacante.
4. Calcula `num_steps = epochs * len(train_loader)` para usar no
   anelamento de temperatura (ver 4.2.4).

#### 4.2.2 Fluxo por batch

Para cada batch `(x, y)` retirado do MESMO loader de treino do classificador
(reciclamos o conjunto de treino do CIFAR-10 cat/dog, 10000 imagens), o
laco interno faz:

1. **Calculo do gradiente da vitima.**
   Chama `compute_input_gradient(classifier, x, y)`. Internamente:
   - Cria um clone de `x` com `requires_grad_=True`.
   - Faz `logits = classifier(x_grad)` e `loss = CE(logits, y)`.
   - Calcula `g = ∂loss/∂x_grad` via `torch.autograd.grad`.
   - Normaliza por imagem: `g_i = g_i / max(|g_i|)`. Sem essa normalizacao
     as magnitudes do gradiente variam por ordens de grandeza entre
     imagens "faceis" e "dificeis" para o classificador, e o atacante
     ficaria dominado pelas imagens com gradiente grande.

2. **Forward do atacante.** `x_adv, mask, value_map = attacker(x, g,
   tau=tau, hard=True)`. Por dentro:
   - Concatena `[x, g] -> R^(B x 6 x 32 x 32)`.
   - Passa pelo encoder convolucional (4 camadas de 64 canais com ReLU).
   - `position_head` (Conv 1x1 -> 1 canal) e achatada em logits `R^(B x 1024)`.
   - `_sample_topk_mask` faz K iteracoes de `F.gumbel_softmax(...,
     tau=tau, hard=True)`, somando as mascaras e bloqueando a posicao
     escolhida em cada iteracao (Plackett-Luce sem reposicao).
   - `value_head` (Conv 1x1 -> 3 canais) com sigmoide produz o valor RGB
     candidato em cada posicao.
   - A imagem perturbada e `x_adv = x * (1 - mask) + value_map * mask`.

3. **Forward da vitima sobre a imagem perturbada.**
   `logits = classifier(x_adv)`. Como o classificador esta em `.eval()` e
   congelado, este forward apenas PROPAGA gradiente, sem atualizar nada.

4. **Calculo da perda adversarial.** `loss = margin_loss(logits, y,
   kappa=5.0)`. A perda de margem nao-direcionada Carlini-Wagner e:

   ```text
   loss_i = max( z_true(x_adv_i) - max_{c != true} z_c(x_adv_i),  -kappa )
   loss   = mean_i (loss_i)
   ```

   - Quando o atacante ainda nao quebrou a imagem `i`, `z_true > z_other`,
     o termo e positivo e o gradiente empurra o atacante a derrubar a
     classe verdadeira.
   - Quando o atacante ja quebrou a imagem, `z_true < z_other`, o termo
     fica negativo e e CLIPADO em `-kappa = -5` para impedir
     "over-attack" — a otimizacao para de pressionar essa imagem e foca
     nas que ainda nao caem. Isso e crucial para nao desperdicar
     capacidade da rede em maximizar a confianca de imagens ja erradas.

5. **Backward + step.**
   `optimizer.zero_grad(); loss.backward(); optimizer.step()`.
   - O grafo computacional cobre: `attacker -> x_adv -> classifier ->
     logits -> loss`. O backward propaga gradiente atraves do
     classificador (porque seus tensores ainda tem `requires_grad`
     implicito atraves de operacoes que dependem de `x_adv`), mas como
     `requires_grad_(False)` foi setado em todos os parametros do
     classificador, NENHUM `.grad` e acumulado para ele.
     `optimizer.step()` so altera os pesos do atacante.

6. **Monitoramento.** A cada 25 batches imprime `tau`, perda media corrente
   e taxa de "fool" (fracao do batch atual em que `argmax(logits) != y`).
   No fim de cada epoca, faz uma avaliacao DETERMINISTICA completa sobre o
   conjunto de teste:
   - Chama `attacker.attack_deterministic(x, g)` (top-K puro, sem ruido
     Gumbel) e imprime a taxa de sucesso real (sobre imagens
     originalmente classificadas corretas) e a mudanca media L1.

#### 4.2.3 Hiperparametros

| Hiperparametro          | Valor              | Onde                                       |
| ----------------------- | ------------------ | ------------------------------------------ |
| Otimizador              | Adam               | `train_attacker.py:95`                     |
| Learning rate           | 1e-3               | `--lr 1e-3` (default)                      |
| Batch size              | 128                | `--batch-size 128`                         |
| Epocas                  | 5 (por K)          | `--epochs 5` (README)                      |
| K                       | 1, 3 ou 5          | `--k {1,3,5}` (uma rede por valor)         |
| `kappa` (margin clip)   | 5.0                | `margin_loss(..., kappa=5.0)`              |
| `tau_start`             | 1.0                | inicio do anelamento                       |
| `tau_end`               | 0.3                | fim do anelamento                          |
| Anelamento              | linear em `step/N` | `tau = 1.0 + p * (0.3 - 1.0)`              |
| Amostragem da mask      | `hard=True`        | Gumbel-Softmax com straight-through        |
| Numero de iteracoes top-K | K                | uma amostragem por pixel atacado           |
| Normalizacao do grad    | divisao por max abs por imagem | `compute_input_gradient`       |
| Seed                    | 42                 | `data.py:fix_seed`                         |
| Dispositivo             | CPU                | suficiente; 5 epocas rodam em poucos min   |

#### 4.2.4 Anelamento da temperatura `tau`

A funcao `_sample_topk_mask` usa `F.gumbel_softmax(..., tau, hard=True)`.
Com `hard=True`, o forward retorna uma mascara one-hot exata mas o
backward usa o straight-through estimator (gradiente do soft-Gumbel).

A temperatura controla quao "afiada" e a distribuicao soft-Gumbel:

- `tau` alto (~1.0): distribuicao espalhada -> gradiente flui bem para
  muitas posicoes -> exploracao.
- `tau` baixo (~0.3): distribuicao concentrada -> gradiente flui quase so
  para a posicao escolhida -> exploitation.

O anelamento linear de 1.0 -> 0.3 ao longo de `epochs * len(train_loader)`
steps deixa as primeiras epocas com mais exploracao e as ultimas com
ajuste fino, evitando que o atacante fixe cedo demais em posicoes ruins.

#### 4.2.5 Avaliacao deterministica vs amostragem

No TREINO usa-se Gumbel-Softmax (estocastico) porque ele e diferenciavel.
Na AVALIACAO usa-se `attack_deterministic` (`attacker.py:107`):

- Toma `topk_idx = pos_logits.topk(K).indices` diretamente.
- Constroi a mascara K-hot exata via `scatter_`.
- Aplica a mesma formula `x_adv = x * (1 - mask) + value_map * mask`.

Isso garante que a metrica reportada nao depende de ruido aleatorio — e
representa o "ataque real" que o atacante produz.

#### 4.2.6 O que o atacante esta otimizando, em uma frase

> Encontre uma politica `f_theta : (x, ∇_x CE) -> (K posicoes, K valores
> RGB)` que, em uma unica passada feedforward, escolha os K pixels e seus
> valores que mais reduzem a margem do classificador entre a classe
> verdadeira e a melhor segunda colocada.

O Adam, com gradiente vindo do classificador via Gumbel-Softmax, ajusta
`theta` em direcao a essa politica.

#### 4.2.7 Tres treinos independentes (um por K)

E importante notar que NAO ha compartilhamento de pesos entre `K=1`, `K=3`
e `K=5`. Sao treinos completamente separados, cada um com sua propria
inicializacao e seu proprio checkpoint:

- `checkpoints/attacker_k1.pt`
- `checkpoints/attacker_k3.pt`
- `checkpoints/attacker_k5.pt`

Isso e proposital. Embora a arquitetura seja identica (apenas
`num_pixels=K` muda no construtor), a dinamica de treinamento e
qualitativamente diferente: com K=1 o gradiente da Gumbel-Softmax e mais
ruidoso por imagem (uma unica amostragem), e o atacante precisa
concentrar quase toda a politica em uma escolha; com K=5 o sinal e mais
suave (cinco amostragens somadas) e a politica pode "diluir" o ataque em
varias posicoes complementares. Treinar separadamente permite que cada
rede otimize para seu regime especifico.

## 5. Evolucao do projeto

A versao inicial do atacante usava encoder convolucional seguido de **decoder
totalmente conectado** (Linear -> 1024 logits de posicao). O treinamento
estagnou: o atacante convergia para selecionar SEMPRE os mesmos 3 pixels,
independentemente da entrada (mode collapse). A taxa de sucesso ficou em
~0.5%, ABAIXO ate do baseline aleatorio.

Diagnostico via inspecao das posicoes escolhidas para 8 imagens distintas
revelou o problema: o decoder Linear ignorava as features de entrada e
aprendia apenas o vies (bias). Duas mudancas resolveram:

1. **Decoder convolucional 1x1.** As cabecas `position_head` e `value_head`
   passaram a operar sobre o mapa de features convolucional, produzindo
   logits de posicao espacialmente estruturados. Isso forca a saida a
   depender da posicao do input.

2. **Gradiente como canal de entrada.** O atacante passou a receber `[x, g]`
   em vez de apenas `x`. O gradiente da CE em relacao a entrada e
   essencialmente um mapa de saliencia (idem JSMA), informando ao atacante
   onde uma perturbacao tem maior impacto na saida.

Apos as duas correcoes a taxa de sucesso saltou imediatamente para 5-15% nas
primeiras epocas e continuou subindo.

Tambem foi importante a transicao de MNIST para CIFAR-10 cat/dog (Secao 2):
no MNIST nem o oraculo (forca bruta sobre todas as 784 posicoes) conseguia
ataques de 1 pixel; em CIFAR-10 o problema e geometricamente bem definido e
existe sinal de aprendizado para o atacante.

## 6. Resultados

### 6.1 Tabela principal

Avaliacao sobre as 2000 imagens de teste cat/dog do CIFAR-10
(`outputs/metrics.txt`). A taxa de sucesso e calculada apenas sobre imagens
ORIGINALMENTE classificadas corretas.

| K | Metodo              | Taxa de sucesso | Confianca media na classe errada | Mudanca media L1 |
|---|---------------------|-----------------|----------------------------------|------------------|
| 1 | NN + gradiente      | **14.21%**      | 71.79%                           | 1.996            |
| 1 | K pixels aleatorios | 1.07%           | -                                | 0.958            |
| 3 | NN + gradiente      | **25.61%**      | 82.88%                           | 5.917            |
| 3 | K pixels aleatorios | 1.79%           | -                                | 2.805            |
| 5 | NN + gradiente      | **33.55%**      | 84.91%                           | 8.649            |
| 5 | K pixels aleatorios | 2.87%           | -                                | 4.745            |

(Acuracia do classificador no teste limpo: **83.75%**.)

O atacante baseado em rede neural com acesso ao gradiente supera o ataque
aleatorio por um fator de **12x a 14x** em todas as configuracoes de K.

A "Confianca media na classe errada" corresponde a metrica **Adversarial
Probability Labels** definida pelo paper de Su et al. — quando o ataque tem
sucesso, o classificador nao apenas erra: erra com confianca alta (~72% a
85%). Isto demonstra que a perturbacao nao se limita a "raspar" a fronteira
de decisao, ela empurra o exemplo bem para dentro do territorio da classe
errada.

A coluna "Mudanca media L1" e a soma L1 de `|x_adv - x|` por imagem.
Comparando NN+grad com aleatorio, ambas as perturbacoes alteram os mesmos K
pixels — mas a NN sistematicamente escolhe cores mais "agressivas" (em
direcao a 0 ou 1, longe do pixel original) PORQUE foi treinada para isso. O
baseline aleatorio sorteia em U[0,1] e fica em media mais perto da media da
imagem.

### 6.2 Comparacao com o paper original

O paper de Su et al. reporta no CIFAR-10 (sobre 3 redes diferentes,
nao-direcionado, original CIFAR-10 com 10 classes):

| K   | Su et al. (DE, caixa preta, 10 classes) | Este trabalho (NN+grad, caixa branca, 2 classes) |
| --- | --------------------------------------- | ------------------------------------------------ |
| 1   | 22.6% / 35.2% / 31.4%                   | 14.21%                                           |
| 3   | -                                       | 25.61%                                           |
| 5   | -                                       | 33.55%                                           |

A comparacao precisa de ressalvas:

- Su et al. usa **10 classes** (mais "alvos" possiveis para um ataque
  nao-direcionado), enquanto aqui temos **2 classes** (apenas UM alvo possivel
  para o ataque nao-direcionado) — o que torna o ataque
  INTRINSECAMENTE MAIS DIFICIL.
- Os classificadores sao diferentes (AllConv/NiN/VGG16 vs nossa CNN pequena).
- O metodo e diferente (DE caixa preta vs NN caixa branca).

Mesmo assim, os numeros estao na MESMA ORDEM DE GRANDEZA, confirmando que o
experimento esta operando no regime esperado pelo paper.

### 6.3 Figuras geradas

- `outputs/grid_k1.png`, `grid_k3.png`, `grid_k5.png` — para cada K, 6
  exemplos de ataques bem-sucedidos em layout de 4 linhas:
  1. **Original** com o rotulo verdadeiro.
  2. **Atacada** com a predicao errada (em vermelho) e circulos amarelos
     destacando a posicao dos K pixels modificados.
  3. **|x_adv - x| amplificada** — apenas os K pixels modificados aparecem.
  4. **Mapa de saliencia |grad|** — confirma que o atacante mira regioes
     internas ao animal (mapas de saliencia se concentram em rostos,
     orelhas e contornos).

- `outputs/summary_success.png` — grafico de barras NN+grad x aleatorio
  por K.

## 7. Conclusoes

1. **Viabilidade do ataque de poucos pixels via rede neural.** Substituir a
   Evolucao Diferencial do paper de Su et al. por uma rede neural treinada
   por back-propagation atraves do classificador funciona. A rede aprende
   uma POLITICA de ataque em uma unica passada (feedforward), em vez de
   iteracoes de otimizacao evolutiva.

2. **Importancia do gradiente como informacao de entrada.** Alimentar o
   gradiente da perda do classificador como canal adicional do atacante
   converte o ataque em uma especie de **JSMA aprendida**. Sem essa
   informacao, o atacante sofre mode collapse e fica abaixo do baseline
   aleatorio. Com ela, ultrapassa o baseline em mais de uma ordem de
   grandeza.

3. **Gumbel-Softmax viabiliza a otimizacao discreta.** A selecao de K pixels
   e por natureza um problema combinatorio. A relaxacao via Gumbel-Softmax
   sem reposicao (Plackett-Luce) torna a posicao diferenciavel, permitindo o
   uso de Adam padrao.

4. **A robustez do classificador limita o ataque, mas nao o impede.** Mesmo
   com o classificador apresentando 83.75% de acuracia em uma tarefa binaria
   reconhecidamente dificil, 14.21% das imagens corretas podem ser
   derrubadas por alteracao de UM unico pixel. Esse e o resultado mais
   importante do ponto de vista de seguranca em ML: redes profundas comuns
   sao sensiveis a perturbacoes extremamente locais.

5. **Tradeoff entre numero de pixels e robustez.** Como o paper original
   sugere, aumentar K de 1 para 5 DOBRA mais que o sucesso do ataque
   (14.21% -> 33.55%), com perturbacao visual ainda dificilmente perceptivel
   em imagens 32x32.

## 8. Estrutura do codigo

```text
TrabalhoFinal/
├── src/
│   ├── data.py               # CIFAR-10 cat/dog -> DataLoaders binarios
│   ├── classifier.py         # CNN vitima (3x32x32 RGB -> 2 classes)
│   ├── attacker.py           # KPixelAttacker + Gumbel-Softmax top-K + margin loss
│   ├── train_classifier.py   # Treino da vitima
│   ├── train_attacker.py     # Treino white-box do atacante (por K)
│   ├── evaluate.py           # Metricas vs baseline aleatorio
│   └── visualize.py          # Geracao dos grids e do grafico de barras
├── checkpoints/              # classifier.pt, attacker_k{1,3,5}.pt
├── outputs/                  # metrics.txt + grids + summary_success.png
├── data/                     # CIFAR-10 (auto-baixado por torchvision)
├── requirements.txt
├── README.md
└── RELATORIO.md
```

Reproducao completa em `README.md`.

## 9. Bibliografia

- Su, J., Vargas, D. V., Sakurai, K. (2019). **One Pixel Attack for Fooling
  Deep Neural Networks**. arXiv:1710.08864.
- Huang, S., Papernot, N., Goodfellow, I., Duan, Y., Abbeel, P. (2017).
  **Adversarial Attacks on Neural Network Policies**. arXiv:1702.02284.
- Goodfellow, I., Shlens, J., Szegedy, C. (2015). **Explaining and
  Harnessing Adversarial Examples**. ICLR. (FGSM)
- Papernot, N. et al. (2016). **The Limitations of Deep Learning in
  Adversarial Settings**. IEEE EuroS&P. (JSMA / saliency map attacks)
- Carlini, N., Wagner, D. (2017). **Towards Evaluating the Robustness of
  Neural Networks**. IEEE S&P. (Margin loss usada aqui.)
- Jang, E., Gu, S., Poole, B. (2017). **Categorical Reparameterization with
  Gumbel-Softmax**. ICLR.
