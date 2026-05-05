# Trabalho de Redes Neurais

Implementacao de uma MLP simples com uma camada escondida e treinamento por backpropagation.

## Arquivos

- `dados_frutas.csv`: conjunto de dados do problema.
- `mlp_frutas.py`: script principal da execucao.
- `core_mlp.py`: logica central da rede neural.
- `io_utils.py`: leitura, escrita e formatacao das saidas.
- `predicoes_frutas.csv`: saida gerada pelo programa.
- `relatorio_curto.md`: resumo tecnico do trabalho.

## Como executar

```bash
cd TrabalhoRedesNeurais
python3 mlp_frutas.py
```

## Observacao

O codigo usa apenas a ideia classica de perceptron multicamadas:

- 2 neuronios de entrada
- 1 camada escondida
- 1 neuronio de saida
- funcao sigmoide
- algoritmo backpropagation
- atualizacao simples dos pesos por taxa de aprendizado

Nao foi usado framework de rede neural nem algoritmo mais avancado de treinamento.
