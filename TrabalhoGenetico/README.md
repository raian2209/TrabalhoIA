# Trabalho de Algoritmo Genetico

Implementacao de um algoritmo genetico binario para maximizar a funcao:

`f(x, y) = |e^(-x) - y^2 + 1| + 10^-4`

no intervalo `[-10, 10]`, com precisao minima de `0.005`.

## Arquivos

- `TrabalhoAGs.pdf`: enunciado original.
- `ag_funcao_binaria.py`: script principal da execucao.
- `core_ag.py`: logica central do algoritmo genetico.
- `io_utils.py`: funcoes de persistencia e formatacao da saida.
- `historico_ag.csv`: desempenho por geracao, gerado apos a execucao.
- `resultado_ag.txt`: resumo final da melhor solucao encontrada.
- `relatorio_curto.md`: explicacao curta das escolhas feitas.

## Escolhas do algoritmo

- Codificacao binaria:
  - 12 bits para `x`
  - 12 bits para `y`
  - 24 bits no cromossomo completo
- Populacao inicial: 80 individuos
- Selecao: torneio com 3 individuos
- Crossover: um ponto com taxa `0.90`
- Mutacao: inversao de bits com taxa `0.01`
- Elitismo: 2 melhores individuos preservados
- Criterio de parada:
  - maximo de 250 geracoes
  - ou 60 geracoes sem melhora no melhor fitness

## Como executar

```bash
cd TrabalhoGenetico
python3 ag_funcao_binaria.py
```

## Observacao

O codigo foi escrito em Python puro e segue exatamente os elementos pedidos no trabalho:
codificacao binaria, inicializacao da populacao, fitness, selecao, crossover, mutacao e criterio de parada.
