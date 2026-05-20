# Trabalho de Algoritmo Genético

Implementação de um algoritmo genético binário para maximizar a função:

`f(x, y) = |e^(-x) - y^2 + 1| + 10^-4`

no intervalo `[-10, 10]`, com precisão mínima de `0.005`.

## Arquivos

- `TrabalhoAGs.pdf`: enunciado original.
- `ag_funcao_binaria.py`: script principal da execução (parsing de argumentos, chamada ao núcleo e geração das saídas).
- `core_ag.py`: lógica central do algoritmo genético (codificação, fitness, operadores e laço evolutivo).
- `io_utils.py`: funções de persistência (CSV/TXT) e formatação da saída no terminal.
- `historico_ag.csv`: desempenho por geração, gerado após a execução.
- `resultado_ag.txt`: resumo final da melhor solução encontrada.
- `relatorio_curto.md`: explicação curta das escolhas feitas.

## Escolhas do algoritmo

- Codificação binária:
  - 12 bits para `x`
  - 12 bits para `y`
  - 24 bits no cromossomo completo
- População inicial: 80 indivíduos
- Seleção: torneio com 3 indivíduos
- Crossover: um ponto, com taxa `0.90`
- Mutação: inversão de bits, com taxa `0.01`
- Elitismo: os 2 melhores indivíduos são preservados
- Critério de parada:
  - máximo de 250 gerações; ou
  - 60 gerações sem melhora no melhor fitness

## Como executar

```bash
cd TrabalhoGenetico
python3 ag_funcao_binaria.py
```

Também é possível sobrescrever os parâmetros pela linha de comando, por exemplo:

```bash
python3 ag_funcao_binaria.py --population-size 120 --mutation-rate 0.02 --seed 7
```

---

## Detalhamento da lógica de implementação (`core_ag.py`)

A seguir, cada bloco do `core_ag.py` é comentado para explicar **o que faz**, **por que foi feito assim** e **como se encaixa no ciclo do AG**.

### 1. Constantes do problema

```python
INTERVALO_MINIMO = -10.0
INTERVALO_MAXIMO = 10.0
PRECISAO_DESEJADA = 0.005
BITS_POR_VARIAVEL = math.ceil(
    math.log2(((INTERVALO_MAXIMO - INTERVALO_MINIMO) / PRECISAO_DESEJADA) + 1.0)
)
TOTAL_BITS = BITS_POR_VARIAVEL * 2
```

- `INTERVALO_MINIMO` e `INTERVALO_MAXIMO` definem o domínio `[-10, 10]` da busca, exatamente como pedido no enunciado.
- `PRECISAO_DESEJADA` é a precisão mínima `0.005` exigida pelo problema.
- `BITS_POR_VARIAVEL` calcula automaticamente quantos bits são necessários para representar uma variável com a precisão desejada. A fórmula vem do critério clássico de discretização binária: `2^n >= (b - a) / p + 1`, isolando `n` por `log2` e arredondando para cima. Com `b - a = 20` e `p = 0.005`, obtém-se `n = 12` bits por variável.
- `TOTAL_BITS` é o tamanho do cromossomo completo, que concatena `x` e `y` (24 bits no total).

Esse cálculo **automático** garante que, se alguém trocar a precisão no código, o número de bits se ajusta sozinho.

### 2. Estrutura do indivíduo

```python
@dataclass
class Individuo:
    cromossomo: list[int]
    x: float = 0.0
    y: float = 0.0
    fitness: float = 0.0

    def clone(self) -> "Individuo":
        return Individuo(self.cromossomo.copy(), self.x, self.y, self.fitness)
```

- O decorador `@dataclass` evita o trabalho repetitivo de escrever um `__init__` manual.
- O indivíduo carrega **simultaneamente** sua representação binária (`cromossomo`) e o fenótipo decodificado (`x`, `y`, `fitness`), evitando decodificar várias vezes o mesmo indivíduo.
- O método `clone()` faz uma cópia segura do cromossomo (com `list.copy()`), impedindo que dois indivíduos compartilhem a mesma lista por referência — o que estragaria mutações e crossovers.

### 3. Função objetivo

```python
def funcao_objetivo(x: float, y: float) -> float:
    return abs(math.exp(-x) - (y * y) + 1.0) + (10.0 ** -4)
```

- Implementação literal de `f(x, y) = |e^(-x) - y^2 + 1| + 10^-4`.
- O termo `+ 10^-4` evita fitness igual a zero (o que poderia atrapalhar operadores baseados em proporção). Aqui o torneio não precisa dessa garantia, mas o termo é mantido por fidelidade ao enunciado.
- Como o objetivo é **maximizar**, a própria função objetivo serve diretamente como fitness, sem transformações.

### 4. Codificação e decodificação

```python
def resolucao_codificacao() -> float:
    return (INTERVALO_MAXIMO - INTERVALO_MINIMO) / ((2**BITS_POR_VARIAVEL) - 1)
```

- Retorna o "passo" entre dois números reais consecutivos representáveis. Com 12 bits e intervalo de 20, dá `~0.004884`, ficando abaixo da precisão exigida.

```python
def bits_para_inteiro(bits: list[int]) -> int:
    valor = 0
    for bit in bits:
        valor = (valor << 1) | bit
    return valor
```

- Converte a lista de bits em inteiro, do bit mais significativo (esquerda) para o menos significativo (direita), usando deslocamentos.

```python
def decodificar_variavel(bits: list[int]) -> float:
    inteiro = bits_para_inteiro(bits)
    return INTERVALO_MINIMO + (inteiro * resolucao_codificacao())
```

- Aplica a transformação linear de inteiro para real: `valor_real = mínimo + inteiro * passo`.
- O inteiro varia de `0` a `2^12 - 1 = 4095`, mapeando para `[-10, +10]`.

```python
def avaliar_individuo(individuo: Individuo) -> None:
    bits_x = individuo.cromossomo[:BITS_POR_VARIAVEL]
    bits_y = individuo.cromossomo[BITS_POR_VARIAVEL:]
    individuo.x = decodificar_variavel(bits_x)
    individuo.y = decodificar_variavel(bits_y)
    individuo.fitness = funcao_objetivo(individuo.x, individuo.y)
```

- Reparte o cromossomo nos dois genes (12 + 12) e calcula `x`, `y` e `fitness`.
- Atualiza o próprio indivíduo no lugar (efeito colateral controlado), o que é barato e suficiente para o tamanho do problema.

### 5. Inicialização da população

```python
def criar_individuo_aleatorio(rng: random.Random) -> Individuo:
    cromossomo = [rng.randint(0, 1) for _ in range(TOTAL_BITS)]
    individuo = Individuo(cromossomo)
    avaliar_individuo(individuo)
    return individuo

def inicializar_populacao(tamanho_populacao: int, rng: random.Random) -> list[Individuo]:
    return [criar_individuo_aleatorio(rng) for _ in range(tamanho_populacao)]
```

- A população inicial é gerada **uniformemente aleatória**, sem viés, o que garante diversidade no primeiro passo.
- Todo indivíduo já é criado com o `fitness` calculado, simplificando o código do laço principal.
- O `rng` (uma instância de `random.Random` com semente) é passado explicitamente, garantindo **reprodutibilidade**: a mesma semente gera os mesmos resultados.

### 6. Seleção por torneio

```python
def selecao_por_torneio(populacao, tamanho_torneio, rng):
    competidores = rng.sample(populacao, tamanho_torneio)
    return max(competidores, key=lambda individuo: individuo.fitness).clone()
```

- Sorteia `k = 3` indivíduos **sem reposição** (`rng.sample`) e devolve o de maior fitness.
- O torneio foi escolhido em vez da roleta porque:
  - é robusto a escalas absurdas de fitness;
  - permite controlar a pressão seletiva pelo tamanho do torneio (`k` maior implica pressão maior e exploração menor);
  - não exige normalização.
- A cópia (`.clone()`) garante que alterar o filho não altere o pai original que ainda está na população.

### 7. Crossover de um ponto

```python
def crossover_um_ponto(pai_1, pai_2, taxa_crossover, rng):
    if rng.random() > taxa_crossover:
        return pai_1.clone(), pai_2.clone()
    ponto_corte = rng.randint(1, TOTAL_BITS - 1)
    filho_1_bits = pai_1.cromossomo[:ponto_corte] + pai_2.cromossomo[ponto_corte:]
    filho_2_bits = pai_2.cromossomo[:ponto_corte] + pai_1.cromossomo[ponto_corte:]
    return Individuo(filho_1_bits), Individuo(filho_2_bits)
```

- Com probabilidade `1 - taxa_crossover` (10% no padrão), os pais são apenas **copiados** para a próxima geração, o que ajuda a preservar diversidade mantendo indivíduos competentes.
- Quando o crossover ocorre, sorteia-se um ponto entre `1` e `TOTAL_BITS - 1` (os extremos são excluídos para evitar "corte vazio").
- Os dois filhos são construídos trocando os segmentos a partir do ponto de corte, garantindo:
  - troca de material genético entre os pais;
  - cromossomos válidos com o mesmo tamanho.

### 8. Mutação por inversão de bits

```python
def mutar(individuo, taxa_mutacao, rng):
    for indice, bit in enumerate(individuo.cromossomo):
        if rng.random() < taxa_mutacao:
            individuo.cromossomo[indice] = 1 - bit
```

- Percorre todo o cromossomo e, **bit a bit**, decide se inverte (`0 -> 1` ou `1 -> 0`) com a probabilidade `taxa_mutacao`.
- Com taxa `0.01` e 24 bits, espera-se em média `0.24` mutação por indivíduo: suficiente para introduzir diversidade sem destruir o material genético já encontrado.
- A mutação bit a bit é a forma clássica em codificação binária e respeita o que foi pedido no enunciado.

### 9. Geração da nova população (com elitismo)

```python
def gerar_nova_populacao(populacao_atual, taxa_crossover, taxa_mutacao,
                        tamanho_torneio, elitismo, rng):
    populacao_ordenada = sorted(populacao_atual, key=lambda i: i.fitness, reverse=True)
    nova_populacao = [individuo.clone() for individuo in populacao_ordenada[:elitismo]]

    while len(nova_populacao) < len(populacao_atual):
        pai_1 = selecao_por_torneio(populacao_atual, tamanho_torneio, rng)
        pai_2 = selecao_por_torneio(populacao_atual, tamanho_torneio, rng)
        filho_1, filho_2 = crossover_um_ponto(pai_1, pai_2, taxa_crossover, rng)
        mutar(filho_1, taxa_mutacao, rng)
        mutar(filho_2, taxa_mutacao, rng)
        avaliar_individuo(filho_1)
        avaliar_individuo(filho_2)
        nova_populacao.append(filho_1)
        if len(nova_populacao) < len(populacao_atual):
            nova_populacao.append(filho_2)
    return nova_populacao
```

Esta função orquestra **uma geração completa**:

1. **Ordenação e elitismo**: a população é ordenada por fitness (em ordem decrescente) e os `elitismo = 2` melhores são copiados diretamente para a nova geração. Isso impede que mutações ou crossovers ruins joguem fora a melhor solução já encontrada.
2. **Reprodução em pares**: enquanto a nova população não atinge o tamanho original, dois pais são escolhidos por torneio, geram dois filhos por crossover e cada filho passa por mutação.
3. **Reavaliação**: como o material genético dos filhos foi alterado, é obrigatório chamar `avaliar_individuo()` antes de adicioná-los à população.
4. **Verificação de tamanho ímpar**: se o tamanho da população for ímpar, o segundo filho do último par pode não caber. O `if` extra evita estourar o tamanho da população.

### 10. Registro do histórico

```python
def registrar_geracao(historico, geracao, populacao):
    melhor = max(populacao, key=lambda i: i.fitness)
    pior = min(populacao, key=lambda i: i.fitness)
    media = mean(individuo.fitness for individuo in populacao)
    historico.append({
        "geracao": geracao,
        "melhor_fitness": melhor.fitness,
        "fitness_medio": media,
        "pior_fitness": pior.fitness,
        "melhor_x": melhor.x,
        "melhor_y": melhor.y,
        "melhor_cromossomo": "".join(str(bit) for bit in melhor.cromossomo),
    })
```

- Para cada geração, salva:
  - melhor, pior e média do fitness (permite visualizar convergência e diversidade);
  - melhor `x`, `y` e a string binária do melhor cromossomo.
- É esse histórico que vira o `historico_ag.csv`, útil para gerar gráficos ou auditar a evolução.

### 11. Laço principal do algoritmo

```python
def executar_algoritmo_genetico(...):
    rng = random.Random(seed)
    populacao = inicializar_populacao(tamanho_populacao, rng)
    historico = []
    melhor_global = max(populacao, key=lambda i: i.fitness).clone()
    registrar_geracao(historico, 0, populacao)
    geracoes_sem_melhora = 0

    for geracao in range(1, maximo_geracoes + 1):
        populacao = gerar_nova_populacao(...)
        melhor_geracao = max(populacao, key=lambda i: i.fitness)
        registrar_geracao(historico, geracao, populacao)

        if melhor_geracao.fitness > melhor_global.fitness:
            melhor_global = melhor_geracao.clone()
            geracoes_sem_melhora = 0
        else:
            geracoes_sem_melhora += 1

        if geracoes_sem_melhora >= limite_estagnacao:
            return melhor_global, historico, geracao

    return melhor_global, historico, maximo_geracoes
```

Esse é o núcleo do ciclo evolutivo. Pontos importantes:

- **Semente única**: cria uma única instância de `random.Random(seed)` e propaga para todas as funções. Resultado: a execução é totalmente reprodutível.
- **Geração 0**: a população inicial já entra no histórico como geração `0`, e o melhor dela vira o `melhor_global` inicial.
- **Atualização do melhor global**: o melhor de cada geração é comparado com o melhor global e só o substitui **se houver melhora estrita**. O `clone()` evita que mutações futuras na população alterem o registro do melhor.
- **Critério de parada duplo**:
  - **Máximo de gerações** (`250`) garante a terminação.
  - **Estagnação** (`60` gerações sem melhora) evita rodar à toa quando o AG já convergiu.
- **Retorno**: além do melhor indivíduo, devolve o histórico completo e o número real de gerações executadas, para que o `ag_funcao_binaria.py` consiga relatar isso ao usuário.

---

## Como as peças se encaixam

O fluxo geral de execução é:

1. `ag_funcao_binaria.py` faz o parsing dos argumentos e chama `executar_algoritmo_genetico` (`core_ag.py`).
2. `core_ag.py` inicializa a população, evolui geração após geração usando torneio, crossover, mutação e elitismo, e mantém o histórico.
3. `io_utils.py` recebe os resultados e grava:
   - `historico_ag.csv` (uma linha por geração);
   - `resultado_ag.txt` (resumo final mais os parâmetros utilizados).
4. O script principal imprime no terminal um resumo das gerações selecionadas e a melhor solução encontrada.

## Observação

O código foi escrito em Python puro e segue exatamente os elementos pedidos no trabalho: codificação binária, inicialização da população, fitness, seleção, crossover, mutação e critério de parada.
