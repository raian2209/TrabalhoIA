# Relatorio Curto

## Objetivo

Implementar um algoritmo genetico binario para encontrar o maximo global da funcao:

`f(x, y) = |e^(-x) - y^2 + 1| + 10^-4`

com busca no intervalo `[-10, 10]` e precisao de `0.005`.

## Representacao do individuo

Cada individuo foi representado por um cromossomo binario com 24 bits:

- 12 bits para representar `x`
- 12 bits para representar `y`

Essa quantidade de bits produz uma resolucao aproximada de `0.004884`, que atende a precisao pedida no enunciado.

## Etapas implementadas

O algoritmo contempla todos os requisitos pedidos:

1. Inicializacao aleatoria da populacao.
2. Decodificacao binaria dos individuos para valores reais.
3. Calculo da funcao de aptidao usando a propria funcao objetivo.
4. Selecao de pais por torneio.
5. Cruzamento de um ponto.
6. Mutacao por inversao de bits.
7. Criterio de parada por numero maximo de geracoes ou estagnacao.

## Parametros escolhidos

- Populacao: 80 individuos
- Taxa de crossover: 0.90
- Taxa de mutacao: 0.01
- Torneio: 3 individuos
- Elitismo: 2 individuos
- Maximo de geracoes: 250
- Estagnacao: 60 geracoes sem melhora

## Resultado esperado

Como a funcao cresce fortemente quando `x` se aproxima de `-10`, o algoritmo tende a encontrar solucoes muito proximas dessa regiao e valores de `y` proximos de `0`, o que produz fitness elevado.

Os arquivos `historico_ag.csv` e `resultado_ag.txt` sao gerados para registrar o comportamento do algoritmo e a melhor solucao encontrada.
