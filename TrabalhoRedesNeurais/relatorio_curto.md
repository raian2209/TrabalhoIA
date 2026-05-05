# Relatorio Curto

## Tema

Implementacao de uma rede neural do tipo perceptron multicamadas para classificar frutas em duas classes: `0` e `1`.

## Estrutura da rede

A rede implementada possui:

- 2 neuronios de entrada
- 1 camada escondida com 5 neuronios
- 1 neuronio de saida

A funcao de ativacao usada na camada escondida e na saida foi a sigmoide.

## Algoritmo utilizado

Foi utilizado somente o algoritmo classico de `MLP + Backpropagation`, como pedido no trabalho.

Passos executados:

1. Propagacao direta das entradas ate a saida.
2. Calculo do erro entre a saida desejada e a saida produzida.
3. Retropropagacao do erro da camada de saida para a camada escondida.
4. Atualizacao dos pesos e bias com taxa de aprendizado fixa.

## O que nao foi utilizado

Nao foram usados algoritmos mais complexos, como:

- redes convolucionais
- algoritmos geneticos
- SVM
- otimizadores como Adam
- bibliotecas de redes neurais como TensorFlow, PyTorch ou scikit-learn

## Resultado

No conjunto de frutas fornecido, a rede conseguiu aprender a separacao entre os exemplos e gerar as previsoes finais no arquivo `predicoes_frutas.csv`.
