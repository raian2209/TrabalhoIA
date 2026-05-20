"""Núcleo do algoritmo genético binário do trabalho.

Este módulo concentra toda a lógica do AG:
- constantes do problema (intervalo, precisão, bits por variável);
- representação do indivíduo (cromossomo binário + fenótipo decodificado);
- operadores genéticos (seleção por torneio, crossover de um ponto, mutação bit a bit);
- elitismo e ciclo evolutivo com critério de parada por geração máxima ou estagnação.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import mean


# ---------------------------------------------------------------------------
# Constantes matemáticas do problema (vindas do enunciado).
# ---------------------------------------------------------------------------
# Intervalo de busca para x e y.
INTERVALO_MINIMO = -10.0
INTERVALO_MAXIMO = 10.0

# Precisão mínima exigida pelo enunciado para a discretização.
PRECISAO_DESEJADA = 0.005

# ---------------------------------------------------------------------------
# BITS_POR_VARIAVEL — quantos bits são usados para representar UMA variável
# (x ou y) em forma binária.
#
# Como o cromossomo é binário, cada variável vira um inteiro entre 0 e 2^n - 1,
# que depois é mapeado linearmente para o intervalo real [INTERVALO_MINIMO,
# INTERVALO_MAXIMO]. Para garantir a precisão exigida (p), o número de bits
# precisa ser grande o suficiente para que o "passo" entre dois valores reais
# consecutivos seja menor ou igual a p.
#
# Derivação da fórmula:
#   - Com n bits temos 2^n valores distintos, formando 2^n - 1 intervalos
#     entre eles.
#   - Para cobrir o domínio com passo p, precisamos de pelo menos
#     (b - a) / p intervalos, ou seja:
#         2^n - 1  >=  (b - a) / p
#         2^n      >=  (b - a) / p + 1            <- por isso o "+ 1.0"
#   - Aplicando log2 dos dois lados e arredondando para cima:
#         n = ceil(log2((b - a) / p + 1))
#
# Cálculo no nosso caso:
#   (b - a) / p + 1 = 20 / 0.005 + 1 = 4001
#   log2(4001) ≈ 11.966   ->   ceil(11.966) = 12 bits.
#
# Verificação: com 12 bits, a resolução real é
#     passo = 20 / (2^12 - 1) ≈ 0.004884   (< 0.005, atende o enunciado)
# Com 11 bits daria ≈ 0.00977, que NÃO atenderia.
# ---------------------------------------------------------------------------
BITS_POR_VARIAVEL = math.ceil(
    math.log2(((INTERVALO_MAXIMO - INTERVALO_MINIMO) / PRECISAO_DESEJADA) + 1.0)
)

# Cromossomo completo: concatenação dos genes de x e y.
TOTAL_BITS = BITS_POR_VARIAVEL * 2


@dataclass
class Individuo:
    """Representa um cromossomo binário e sua avaliação.

    O indivíduo carrega ao mesmo tempo o genótipo (lista de bits) e o
    fenótipo já decodificado (x, y) com o respectivo fitness. Guardar tudo
    junto evita decodificar o mesmo indivíduo várias vezes durante a evolução.
    """

    cromossomo: list[int]
    x: float = 0.0
    y: float = 0.0
    fitness: float = 0.0

    def clone(self) -> "Individuo":
        """Cria uma cópia independente do indivíduo.

        É essencial chamar `cromossomo.copy()` para que pai e filho não
        compartilhem a mesma lista por referência — caso contrário, uma
        mutação em um afetaria o outro silenciosamente.
        """
        return Individuo(self.cromossomo.copy(), self.x, self.y, self.fitness)


def funcao_objetivo(x: float, y: float) -> float:
    """Calcula a aptidão segundo a função do enunciado.

    f(x, y) = |e^(-x) - y^2 + 1| + 10^-4

    O termo +10^-4 evita fitness igual a zero (útil para operadores baseados
    em proporção, como roleta). Como queremos maximizar f, ela serve
    diretamente como fitness, sem nenhuma transformação extra.
    """
    return abs(math.exp(-x) - (y * y) + 1.0) + (10.0 ** -4)


def resolucao_codificacao() -> float:
    """Retorna a resolução real obtida com a quantidade de bits escolhida.

    É o "passo" entre dois valores reais consecutivos representáveis.
    Com 12 bits e intervalo de 20, dá ~0.004884 (abaixo da precisão exigida).
    """
    return (INTERVALO_MAXIMO - INTERVALO_MINIMO) / ((2**BITS_POR_VARIAVEL) - 1)


def bits_para_inteiro(bits: list[int]) -> int:
    """Converte uma lista de bits no inteiro correspondente.

    O bit mais significativo está à esquerda (índice 0). A conversão é feita
    por deslocamentos: a cada bit, o acumulador é deslocado uma posição para
    a esquerda e o bit novo é OR-ado na posição menos significativa.
    """
    valor = 0
    for bit in bits:
        valor = (valor << 1) | bit
    return valor


def decodificar_variavel(bits: list[int]) -> float:
    """Transforma um gene binário em valor real dentro do intervalo permitido.

    Aplica a transformação linear:
        valor_real = INTERVALO_MINIMO + inteiro * resolução
    O inteiro varia de 0 a 2^BITS_POR_VARIAVEL - 1, mapeando para [-10, +10].
    """
    inteiro = bits_para_inteiro(bits)
    return INTERVALO_MINIMO + (inteiro * resolucao_codificacao())


def avaliar_individuo(individuo: Individuo) -> None:
    """Decodifica o cromossomo e calcula o fitness do indivíduo.

    Reparte o cromossomo nos dois genes (primeiros BITS_POR_VARIAVEL bits
    formam x, o restante forma y) e atualiza x, y e fitness no próprio objeto.
    Deve ser chamado sempre que o cromossomo é criado ou alterado por crossover
    ou mutação.
    """
    bits_x = individuo.cromossomo[:BITS_POR_VARIAVEL]
    bits_y = individuo.cromossomo[BITS_POR_VARIAVEL:]

    individuo.x = decodificar_variavel(bits_x)
    individuo.y = decodificar_variavel(bits_y)
    individuo.fitness = funcao_objetivo(individuo.x, individuo.y)


def criar_individuo_aleatorio(rng: random.Random) -> Individuo:
    """Gera um cromossomo binário aleatório, já com o fitness calculado.

    Usa o `rng` recebido como parâmetro (em vez do `random` global) para
    manter o algoritmo totalmente reprodutível a partir de uma semente.
    """
    cromossomo = [rng.randint(0, 1) for _ in range(TOTAL_BITS)]
    individuo = Individuo(cromossomo)
    avaliar_individuo(individuo)
    return individuo


def inicializar_populacao(tamanho_populacao: int, rng: random.Random) -> list[Individuo]:
    """Monta a população inicial do algoritmo genético.

    Distribuição uniformemente aleatória, sem viés inicial: cada bit é 0 ou 1
    com igual probabilidade. Isso garante boa diversidade no ponto de partida.
    """
    return [criar_individuo_aleatorio(rng) for _ in range(tamanho_populacao)]


def selecao_por_torneio(
    populacao: list[Individuo], tamanho_torneio: int, rng: random.Random
) -> Individuo:
    """Seleciona um pai usando o método do torneio.

    Sorteia `tamanho_torneio` indivíduos sem reposição (`rng.sample`) e
    devolve uma cópia do de maior fitness. O torneio é robusto a escalas
    arbitrárias de fitness e a pressão seletiva é controlada pelo tamanho do
    torneio: quanto maior k, maior a pressão e menor a exploração.

    Retorna `.clone()` para que mutações posteriores no filho não alterem o
    indivíduo original que ainda está na população.
    """
    competidores = rng.sample(populacao, tamanho_torneio)
    return max(competidores, key=lambda individuo: individuo.fitness).clone()


def crossover_um_ponto(
    pai_1: Individuo,
    pai_2: Individuo,
    taxa_crossover: float,
    rng: random.Random,
) -> tuple[Individuo, Individuo]:
    """Executa crossover de um ponto no cromossomo.

    - Com probabilidade `1 - taxa_crossover`, os pais são apenas copiados
      para a próxima geração (preserva indivíduos competentes).
    - Caso contrário, sorteia um ponto entre 1 e TOTAL_BITS - 1 (extremos
      excluídos para evitar "corte vazio") e troca os segmentos dos pais
      a partir desse ponto, produzindo dois filhos com mesmo tamanho.
    """
    if rng.random() > taxa_crossover:
        return pai_1.clone(), pai_2.clone()

    ponto_corte = rng.randint(1, TOTAL_BITS - 1)
    filho_1_bits = pai_1.cromossomo[:ponto_corte] + pai_2.cromossomo[ponto_corte:]
    filho_2_bits = pai_2.cromossomo[:ponto_corte] + pai_1.cromossomo[ponto_corte:]

    return Individuo(filho_1_bits), Individuo(filho_2_bits)


def mutar(individuo: Individuo, taxa_mutacao: float, rng: random.Random) -> None:
    """Aplica mutação por inversão de bits, bit a bit.

    Para cada bit do cromossomo, com probabilidade `taxa_mutacao`, inverte o
    valor (0 vira 1 e vice-versa). Com taxa 0.01 e 24 bits, espera-se em
    média ~0.24 mutações por indivíduo: o suficiente para gerar diversidade
    sem destruir o material genético já encontrado.
    """
    for indice, bit in enumerate(individuo.cromossomo):
        if rng.random() < taxa_mutacao:
            individuo.cromossomo[indice] = 1 - bit


def gerar_nova_populacao(
    populacao_atual: list[Individuo],
    taxa_crossover: float,
    taxa_mutacao: float,
    tamanho_torneio: int,
    elitismo: int,
    rng: random.Random,
) -> list[Individuo]:
    """Produz a próxima geração a partir da atual.

    Etapas:
      1. Ordena a população por fitness (decrescente) e copia os `elitismo`
         melhores diretamente para a nova geração (preserva as melhores
         soluções já encontradas).
      2. Em pares: seleciona dois pais por torneio, gera dois filhos por
         crossover, aplica mutação em cada filho e os reavalia (avaliar o
         filho é obrigatório, pois seu cromossomo é novo).
      3. Repete até preencher a nova população. O `if` extra evita estourar
         o tamanho da população quando ela tem tamanho ímpar.
    """
    populacao_ordenada = sorted(
        populacao_atual, key=lambda individuo: individuo.fitness, reverse=True
    )

    # Mantém os melhores indivíduos para preservar boas soluções (elitismo).
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
        # Evita ultrapassar o tamanho da população quando ela é ímpar.
        if len(nova_populacao) < len(populacao_atual):
            nova_populacao.append(filho_2)

    return nova_populacao


def registrar_geracao(
    historico: list[dict[str, float | int | str]],
    geracao: int,
    populacao: list[Individuo],
) -> None:
    """Guarda um resumo da geração atual no histórico.

    Para cada geração, armazena melhor/pior/médio do fitness e os dados do
    melhor indivíduo (x, y e o cromossomo como string binária). Esse histórico
    é depois convertido para CSV e serve para auditar a convergência.
    """
    melhor = max(populacao, key=lambda individuo: individuo.fitness)
    pior = min(populacao, key=lambda individuo: individuo.fitness)
    media = mean(individuo.fitness for individuo in populacao)

    historico.append(
        {
            "geracao": geracao,
            "melhor_fitness": melhor.fitness,
            "fitness_medio": media,
            "pior_fitness": pior.fitness,
            "melhor_x": melhor.x,
            "melhor_y": melhor.y,
            "melhor_cromossomo": "".join(str(bit) for bit in melhor.cromossomo),
        }
    )


def executar_algoritmo_genetico(
    tamanho_populacao: int,
    taxa_crossover: float,
    taxa_mutacao: float,
    tamanho_torneio: int,
    elitismo: int,
    maximo_geracoes: int,
    limite_estagnacao: int,
    seed: int,
) -> tuple[Individuo, list[dict[str, float | int | str]], int]:
    """Executa o ciclo completo do algoritmo genético.

    Fluxo:
      1. Cria um `rng` a partir da semente — garante reprodutibilidade total
         em todas as funções que recebem `rng`.
      2. Gera a população inicial aleatória e registra como geração 0.
      3. Para cada geração subsequente:
         - produz uma nova população (torneio + crossover + mutação + elitismo);
         - registra estatísticas no histórico;
         - atualiza o melhor global apenas se houver melhora estrita
           (o `clone()` blinda o melhor contra mutações posteriores);
         - encerra antecipadamente se passar `limite_estagnacao` gerações
           sem melhora.
      4. Retorna o melhor indivíduo, o histórico e o número real de gerações
         executadas.

    Critério de parada duplo:
      - máximo de gerações: garante terminação;
      - estagnação: evita rodar à toa quando o AG já convergiu.
    """
    rng = random.Random(seed)
    populacao = inicializar_populacao(tamanho_populacao, rng)
    historico: list[dict[str, float | int | str]] = []

    # Inicializa o melhor global com o melhor da população inicial.
    melhor_global = max(populacao, key=lambda individuo: individuo.fitness).clone()
    registrar_geracao(historico, 0, populacao)
    geracoes_sem_melhora = 0

    for geracao in range(1, maximo_geracoes + 1):
        populacao = gerar_nova_populacao(
            populacao_atual=populacao,
            taxa_crossover=taxa_crossover,
            taxa_mutacao=taxa_mutacao,
            tamanho_torneio=tamanho_torneio,
            elitismo=elitismo,
            rng=rng,
        )

        melhor_geracao = max(populacao, key=lambda individuo: individuo.fitness)
        registrar_geracao(historico, geracao, populacao)

        # Atualiza o melhor global apenas se houver melhora estrita.
        if melhor_geracao.fitness > melhor_global.fitness:
            melhor_global = melhor_geracao.clone()
            geracoes_sem_melhora = 0
        else:
            geracoes_sem_melhora += 1

        # Parada antecipada por estagnação.
        if geracoes_sem_melhora >= limite_estagnacao:
            return melhor_global, historico, geracao

    return melhor_global, historico, maximo_geracoes
