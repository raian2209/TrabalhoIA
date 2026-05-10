# Relatorio tecnico do trabalho de Redes Neurais

## 1. Problema abordado

O trabalho implementa uma rede neural artificial do tipo perceptron multicamadas
(MLP) para resolver um problema simples de classificacao binaria de frutas. A
tarefa da rede e receber duas caracteristicas numericas de uma fruta e decidir
se ela pertence a classe `0` ou a classe `1`.

O conjunto de dados esta no arquivo `dados_frutas.csv` e foi transcrito do
enunciado em PDF do trabalho. Cada linha possui:

- `entrada_1`: primeira caracteristica numerica da fruta.
- `entrada_2`: segunda caracteristica numerica da fruta.
- `saida`: classe desejada, sendo `0` ou `1`.
- `descricao`: texto descritivo da amostra, usado apenas para interpretacao.

As entradas ja estao em escala pequena, aproximadamente entre `0.10` e `0.95`.
Isso facilita o treinamento com sigmoide, pois evita valores muito grandes nas
somas ponderadas. A saida desejada e binaria: valores baixos ou combinacoes
ruins das entradas tendem a pertencer a classe `0`; valores mais altos e
combinacoes consideradas boas tendem a pertencer a classe `1`.

O conjunto tem 20 amostras:

- 10 exemplos da classe `0`.
- 10 exemplos da classe `1`.

Exemplos da classe `0` incluem frutas descritas como "Muito pequena e leve",
"Borda ruim", "Leve mas larga" e "Pesada mas fina". Exemplos da classe `1`
incluem "Acima da media", "Boa", "Muito boa", "Excelente" e "Borda boa".

## 2. Restricoes do trabalho

O enunciado pede a implementacao manual de uma MLP e do algoritmo
backpropagation, sem utilizar bibliotecas prontas de redes neurais. A solucao
segue essa restricao: nao usa TensorFlow, PyTorch, scikit-learn, Weka, Matlab
Toolbox ou qualquer framework de aprendizado de maquina.

Foram usadas apenas bibliotecas padrao do Python:

- `math`, para calcular a funcao exponencial usada na sigmoide.
- `random`, para inicializar pesos e biases.
- `csv`, para ler e salvar arquivos CSV.
- `argparse`, para configurar parametros pela linha de comando.
- `pathlib`, para montar caminhos de arquivos.
- `dataclasses`, para representar uma amostra do conjunto de dados.

## 3. Arquitetura da rede neural

A arquitetura implementada no arquivo `core_mlp.py` e uma MLP totalmente
conectada com uma camada escondida:

```text
2 entradas -> 5 neuronios escondidos -> 1 neuronio de saida
```

Na execucao padrao, os parametros sao definidos em `mlp_frutas.py`:

- `input_size = 2`
- `hidden_size = 5`
- `output_size = 1`
- `learning_rate = 0.5`
- `epochs = 20000`
- `report_every = 2000`
- `seed = 42`

A quantidade de neuronios escondidos, a taxa de aprendizado, o numero de epocas,
o intervalo de relatorio e a semente podem ser alterados por argumentos de linha
de comando. Por exemplo:

```bash
python3 mlp_frutas.py --hidden-size 8 --epochs 30000 --learning-rate 0.3
```

### 3.1 Camada de entrada

A camada de entrada possui 2 valores:

```text
x1 = entrada_1
x2 = entrada_2
```

Esses valores sao extraidos diretamente do CSV pela funcao `extract_features`,
em `io_utils.py`, que transforma cada amostra em uma lista `[entrada_1,
entrada_2]`.

### 3.2 Camada escondida

A camada escondida possui 5 neuronios na configuracao padrao. Cada neuronio
escondido recebe as duas entradas e possui:

- Um peso vindo de `entrada_1`.
- Um peso vindo de `entrada_2`.
- Um bias proprio.
- Funcao de ativacao sigmoide.

Para cada neuronio escondido `j`, a soma ponderada e:

```text
z_hidden_j = bias_hidden_j + x1 * w_input_hidden_1j + x2 * w_input_hidden_2j
```

A ativacao do neuronio escondido e:

```text
h_j = sigmoid(z_hidden_j)
```

A funcao sigmoide usada no codigo e:

```text
sigmoid(v) = 1 / (1 + exp(-v))
```

### 3.3 Camada de saida

A camada de saida possui 1 neuronio, pois o problema e de classificacao binaria.
Esse neuronio recebe as ativacoes dos 5 neuronios escondidos e tambem usa
sigmoide.

A soma ponderada da saida e:

```text
z_output = bias_output + h1 * w_hidden_output_1 + ... + h5 * w_hidden_output_5
```

A saida final da rede e:

```text
y_pred = sigmoid(z_output)
```

Como a sigmoide retorna um valor entre `0` e `1`, esse valor e interpretado como
probabilidade da classe `1`. A classe final e obtida com limiar `0.5`:

```text
classe = 1, se y_pred >= 0.5
classe = 0, se y_pred < 0.5
```

## 4. Implementacao do core MLP

O nucleo da rede esta no arquivo `core_mlp.py`. Ele concentra a logica da MLP,
incluindo inicializacao dos pesos, propagacao direta, treinamento por
backpropagation, predicao e calculo de metricas.

### 4.1 Funcoes de ativacao

O arquivo define duas funcoes matematicas:

```python
def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))
```

Essa funcao comprime qualquer valor real para o intervalo `(0, 1)`, permitindo
que a saida da rede seja usada como probabilidade.

Tambem foi implementada a derivada da sigmoide:

```python
def sigmoid_derivative(activation: float) -> float:
    return activation * (1.0 - activation)
```

A derivada recebe a propria ativacao ja calculada. Isso evita recalcular a
sigmoide durante o backpropagation.

### 4.2 Inicializacao da rede

A classe principal e `MLP`. Seu construtor recebe:

- `input_size`: quantidade de entradas.
- `hidden_size`: quantidade de neuronios escondidos.
- `output_size`: quantidade de neuronios de saida.
- `learning_rate`: taxa de aprendizado usada na atualizacao dos pesos.
- `seed`: semente aleatoria para tornar os resultados reproduziveis.

Os pesos e biases sao inicializados com valores aleatorios uniformes no intervalo
`[-0.5, 0.5]`.

As estruturas principais sao:

- `weights_input_hidden`: matriz de pesos da entrada para a camada escondida.
- `bias_hidden`: lista de biases da camada escondida.
- `weights_hidden_output`: matriz de pesos da camada escondida para a saida.
- `bias_output`: lista de biases da camada de saida.

Com a arquitetura padrao `2 -> 5 -> 1`, essas estruturas possuem os seguintes
tamanhos:

- `weights_input_hidden`: `2 x 5`, total de 10 pesos.
- `bias_hidden`: 5 biases.
- `weights_hidden_output`: `5 x 1`, total de 5 pesos.
- `bias_output`: 1 bias.

Portanto, a rede padrao possui 21 parametros treinaveis:

```text
10 pesos entrada-escondida
+ 5 biases escondidos
+ 5 pesos escondida-saida
+ 1 bias de saida
= 21 parametros
```

### 4.3 Propagacao direta

A propagacao direta esta no metodo `forward`. Ele recebe uma amostra de entrada,
calcula as ativacoes da camada escondida e depois calcula a saida final.

O metodo retorna duas listas:

- `hidden_outputs`: ativacoes dos neuronios escondidos.
- `final_outputs`: ativacoes da camada de saida.

O retorno da camada escondida e necessario porque o backpropagation usa esses
valores para calcular os gradientes e atualizar os pesos.

Fluxo do `forward`:

1. Para cada neuronio escondido, comeca a soma pelo bias.
2. Soma cada entrada multiplicada pelo peso correspondente.
3. Aplica sigmoide na soma ponderada.
4. Para o neuronio de saida, comeca a soma pelo bias de saida.
5. Soma cada ativacao escondida multiplicada pelo peso correspondente.
6. Aplica sigmoide novamente para obter a saida final.

## 5. Treinamento com backpropagation

O treinamento esta no metodo `train`, em `core_mlp.py`. Ele recebe:

- `features`: lista de entradas.
- `targets`: lista de saidas desejadas.
- `epochs`: numero total de epocas.
- `report_every`: intervalo usado para registrar o erro medio quadratico.

O treinamento e feito de forma online, isto e, amostra por amostra. A cada
exemplo, a rede faz a propagacao direta, calcula o erro, retropropaga esse erro
e atualiza imediatamente pesos e biases.

### 5.1 Erro da saida

Para cada amostra, o codigo calcula:

```text
erro = target - output
```

Esse erro tambem e acumulado ao quadrado para calcular o erro medio quadratico
da epoca:

```text
MSE = soma(erro^2) / quantidade_de_amostras
```

### 5.2 Delta da camada de saida

O delta da saida representa o erro local do neuronio de saida considerando a
derivada da funcao de ativacao:

```text
delta_output = erro * sigmoid_derivative(output)
```

Como a derivada da sigmoide e calculada por `output * (1 - output)`, o codigo
usa:

```text
delta_output = (target - output) * output * (1 - output)
```

### 5.3 Deltas da camada escondida

Depois de calcular o delta da saida, o erro e propagado para cada neuronio
escondido. Para cada neuronio escondido `j`, o codigo calcula:

```text
erro_propagado_j = delta_output * peso_hidden_output_j
delta_hidden_j = erro_propagado_j * h_j * (1 - h_j)
```

Assim, cada neuronio escondido recebe uma parcela do erro proporcional ao peso
que o liga ao neuronio de saida.

### 5.4 Atualizacao dos pesos e biases

A atualizacao segue a regra classica do gradiente com taxa de aprendizado fixa.

Pesos da camada escondida para a saida:

```text
w_hidden_output_j = w_hidden_output_j + learning_rate * delta_output * h_j
```

Bias da saida:

```text
bias_output = bias_output + learning_rate * delta_output
```

Pesos da entrada para a camada escondida:

```text
w_input_hidden_ij = w_input_hidden_ij + learning_rate * delta_hidden_j * x_i
```

Biases da camada escondida:

```text
bias_hidden_j = bias_hidden_j + learning_rate * delta_hidden_j
```

No codigo, a atualizacao usa soma porque o erro foi definido como `target -
output`. Com essa convencao de sinal, somar o termo de correcao move o peso na
direcao que reduz o erro.

## 6. Fluxo geral da aplicacao

O arquivo `mlp_frutas.py` e o script principal. Ele organiza a execucao do
trabalho:

1. Le argumentos de linha de comando com `argparse`.
2. Carrega `dados_frutas.csv` usando `load_samples`.
3. Separa entradas com `extract_features`.
4. Separa classes desejadas com `extract_targets`.
5. Cria a rede `MLP(input_size=2, hidden_size=5, output_size=1, ...)`.
6. Treina a rede com `network.train`.
7. Calcula probabilidades com `network.predict_proba`.
8. Converte probabilidades em classes com `network.predict`.
9. Calcula metricas com `calculate_metrics`.
10. Mostra tabelas no terminal.
11. Salva o resultado final em `predicoes_frutas.csv`.

O arquivo `io_utils.py` separa a parte de entrada, saida e formatacao. Essa
separacao deixa `core_mlp.py` responsavel apenas pela rede neural, enquanto
`io_utils.py` cuida de CSV, tabelas e representacao das amostras.

## 7. Metricas implementadas

A funcao `calculate_metrics`, em `core_mlp.py`, compara classes reais e classes
previstas. Ela calcula:

- Verdadeiros positivos (`TP`): classe real `1` prevista como `1`.
- Verdadeiros negativos (`TN`): classe real `0` prevista como `0`.
- Falsos positivos (`FP`): classe real `0` prevista como `1`.
- Falsos negativos (`FN`): classe real `1` prevista como `0`.
- Acuracia: proporcao total de acertos.
- Precisao: proporcao de previsoes positivas que estavam corretas.
- Recall: proporcao dos positivos reais que foram encontrados.

As formulas usadas sao:

```text
acuracia = (TP + TN) / total
precisao = TP / (TP + FP)
recall = TP / (TP + FN)
```

Quando o denominador de precisao ou recall e zero, o codigo retorna `0.0` para
evitar divisao por zero.

## 8. Resultado obtido

Com os parametros padrao, a execucao foi:

```bash
python3 TrabalhoRedesNeurais/mlp_frutas.py
```

A rede utilizada foi:

```text
2 entradas -> 5 neuronios escondidos -> 1 saida
```

O erro medio quadratico diminuiu ao longo do treinamento:

| Epoca | Erro medio quadratico |
| ---: | ---: |
| 1 | 0.259400 |
| 2000 | 0.001819 |
| 4000 | 0.000616 |
| 6000 | 0.000345 |
| 8000 | 0.000233 |
| 10000 | 0.000173 |
| 12000 | 0.000136 |
| 14000 | 0.000112 |
| 16000 | 0.000095 |
| 18000 | 0.000082 |
| 20000 | 0.000072 |

As metricas finais foram:

| Metrica | Valor |
| --- | ---: |
| Acuracia | 1.00 |
| Precisao | 1.00 |
| Recall | 1.00 |
| TP | 10 |
| TN | 10 |
| FP | 0 |
| FN | 0 |

O arquivo `predicoes_frutas.csv` registra, para cada amostra, a saida real, a
probabilidade calculada pela rede, a classe prevista e se a previsao acertou.
Na execucao padrao, todas as 20 amostras foram classificadas corretamente.

## 9. Como executar com parametros customizados

O programa principal e o arquivo `mlp_frutas.py`. Para executar com os
parametros padrao, use:

```bash
cd TrabalhoRedesNeurais
python3 mlp_frutas.py
```

Tambem e possivel alterar os principais parametros da rede pela linha de
comando. Os argumentos disponiveis sao:

- `--hidden-size`: quantidade de neuronios na camada escondida.
- `--epochs`: numero de epocas de treinamento.
- `--learning-rate`: taxa de aprendizado usada na atualizacao dos pesos.
- `--report-every`: intervalo de epocas usado para mostrar o erro no historico.
- `--seed`: semente aleatoria usada na inicializacao dos pesos e biases.

Exemplo aumentando a camada escondida para 8 neuronios e treinando por 30000
epocas:

```bash
python3 mlp_frutas.py --hidden-size 8 --epochs 30000
```

Exemplo alterando a taxa de aprendizado e a semente aleatoria:

```bash
python3 mlp_frutas.py --learning-rate 0.3 --seed 7
```

Exemplo completo com todos os parametros customizados:

```bash
python3 mlp_frutas.py --hidden-size 8 --epochs 30000 --learning-rate 0.3 --report-every 3000 --seed 7
```

Ao final da execucao, o programa imprime o historico do erro, as previsoes, as
metricas finais e salva as predicoes atualizadas no arquivo
`predicoes_frutas.csv`.

## 10. Conclusao

A implementacao atende ao objetivo do trabalho: construir manualmente uma MLP
com uma camada escondida e treina-la usando backpropagation para classificar o
conjunto de frutas. A rede usa uma arquitetura simples, adequada ao tamanho do
problema, e consegue separar corretamente os exemplos fornecidos.

O nucleo da solucao esta em `core_mlp.py`, onde a MLP e implementada sem
dependencia de bibliotecas externas de aprendizado de maquina. O script
`mlp_frutas.py` apenas coordena a leitura dos dados, o treinamento, a avaliacao
e a gravacao das predicoes finais.
