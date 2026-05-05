"""Nucleo do algoritmo genetico binario do trabalho."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import mean


# Configuracoes matematicas do problema fornecido no enunciado.
INTERVALO_MINIMO = -10.0
INTERVALO_MAXIMO = 10.0
PRECISAO_DESEJADA = 0.005
BITS_POR_VARIAVEL = math.ceil(
    math.log2(((INTERVALO_MAXIMO - INTERVALO_MINIMO) / PRECISAO_DESEJADA) + 1.0)
)
TOTAL_BITS = BITS_POR_VARIAVEL * 2


@dataclass
class Individuo:
    """Representa um cromossomo binario e sua avaliacao."""

    cromossomo: list[int]
    x: float = 0.0
    y: float = 0.0
    fitness: float = 0.0

    def clone(self) -> "Individuo":
        """Cria uma copia para preservar o individuo original."""
        return Individuo(self.cromossomo.copy(), self.x, self.y, self.fitness)


def funcao_objetivo(x: float, y: float) -> float:
    """Calcula a aptidao segundo a funcao do enunciado."""
    return abs(math.exp(-x) - (y * y) + 1.0) + (10.0 ** -4)


def resolucao_codificacao() -> float:
    """Retorna a resolucao real obtida com a quantidade de bits escolhida."""
    return (INTERVALO_MAXIMO - INTERVALO_MINIMO) / ((2**BITS_POR_VARIAVEL) - 1)


def bits_para_inteiro(bits: list[int]) -> int:
    """Converte uma lista de bits para seu valor inteiro."""
    valor = 0
    for bit in bits:
        valor = (valor << 1) | bit
    return valor


def decodificar_variavel(bits: list[int]) -> float:
    """Transforma um gene binario em valor real dentro do intervalo permitido."""
    inteiro = bits_para_inteiro(bits)
    return INTERVALO_MINIMO + (inteiro * resolucao_codificacao())


def avaliar_individuo(individuo: Individuo) -> None:
    """Decodifica o cromossomo e calcula o fitness do individuo."""
    bits_x = individuo.cromossomo[:BITS_POR_VARIAVEL]
    bits_y = individuo.cromossomo[BITS_POR_VARIAVEL:]

    individuo.x = decodificar_variavel(bits_x)
    individuo.y = decodificar_variavel(bits_y)
    individuo.fitness = funcao_objetivo(individuo.x, individuo.y)


def criar_individuo_aleatorio(rng: random.Random) -> Individuo:
    """Gera um cromossomo binario aleatorio."""
    cromossomo = [rng.randint(0, 1) for _ in range(TOTAL_BITS)]
    individuo = Individuo(cromossomo)
    avaliar_individuo(individuo)
    return individuo


def inicializar_populacao(tamanho_populacao: int, rng: random.Random) -> list[Individuo]:
    """Monta a populacao inicial do algoritmo genetico."""
    return [criar_individuo_aleatorio(rng) for _ in range(tamanho_populacao)]


def selecao_por_torneio(
    populacao: list[Individuo], tamanho_torneio: int, rng: random.Random
) -> Individuo:
    """Seleciona um pai por torneio."""
    competidores = rng.sample(populacao, tamanho_torneio)
    return max(competidores, key=lambda individuo: individuo.fitness).clone()


def crossover_um_ponto(
    pai_1: Individuo,
    pai_2: Individuo,
    taxa_crossover: float,
    rng: random.Random,
) -> tuple[Individuo, Individuo]:
    """Executa crossover de um ponto no cromossomo."""
    if rng.random() > taxa_crossover:
        return pai_1.clone(), pai_2.clone()

    ponto_corte = rng.randint(1, TOTAL_BITS - 1)
    filho_1_bits = pai_1.cromossomo[:ponto_corte] + pai_2.cromossomo[ponto_corte:]
    filho_2_bits = pai_2.cromossomo[:ponto_corte] + pai_1.cromossomo[ponto_corte:]

    return Individuo(filho_1_bits), Individuo(filho_2_bits)


def mutar(individuo: Individuo, taxa_mutacao: float, rng: random.Random) -> None:
    """Aplica mutacao por inversao de bits."""
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
    """Produz a proxima geracao a partir da atual."""
    populacao_ordenada = sorted(
        populacao_atual, key=lambda individuo: individuo.fitness, reverse=True
    )

    # Mantem os melhores individuos para preservar boas solucoes.
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


def registrar_geracao(
    historico: list[dict[str, float | int | str]],
    geracao: int,
    populacao: list[Individuo],
) -> None:
    """Guarda um resumo da geracao atual."""
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
    """Executa o ciclo completo do algoritmo genetico."""
    rng = random.Random(seed)
    populacao = inicializar_populacao(tamanho_populacao, rng)
    historico: list[dict[str, float | int | str]] = []

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

        if melhor_geracao.fitness > melhor_global.fitness:
            melhor_global = melhor_geracao.clone()
            geracoes_sem_melhora = 0
        else:
            geracoes_sem_melhora += 1

        if geracoes_sem_melhora >= limite_estagnacao:
            return melhor_global, historico, geracao

    return melhor_global, historico, maximo_geracoes
