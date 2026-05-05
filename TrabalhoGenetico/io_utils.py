"""Utilitarios de formatacao e persistencia do trabalho genetico."""

from __future__ import annotations

import csv
from pathlib import Path

from core_ag import (
    BITS_POR_VARIAVEL,
    INTERVALO_MAXIMO,
    INTERVALO_MINIMO,
    PRECISAO_DESEJADA,
    Individuo,
    resolucao_codificacao,
)


def salvar_historico_csv(caminho: Path, historico: list[dict[str, float | int | str]]) -> None:
    """Salva o historico geracao a geracao em CSV."""
    with caminho.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(
            arquivo,
            fieldnames=[
                "geracao",
                "melhor_fitness",
                "fitness_medio",
                "pior_fitness",
                "melhor_x",
                "melhor_y",
                "melhor_cromossomo",
            ],
        )
        escritor.writeheader()
        escritor.writerows(historico)


def salvar_resultado_txt(
    caminho: Path,
    melhor: Individuo,
    geracoes_executadas: int,
    parametros: dict[str, float | int],
) -> None:
    """Registra em texto o melhor individuo encontrado."""
    linhas = [
        "Algoritmo Genetico Binario - Resultado Final",
        "",
        "Funcao: f(x, y) = |e^(-x) - y^2 + 1| + 10^-4",
        f"Intervalo de busca: [{INTERVALO_MINIMO}, {INTERVALO_MAXIMO}]",
        f"Precisao desejada: {PRECISAO_DESEJADA}",
        f"Resolucao obtida pela codificacao: {resolucao_codificacao():.6f}",
        f"Bits por variavel: {BITS_POR_VARIAVEL}",
        f"Total de bits do cromossomo: {BITS_POR_VARIAVEL * 2}",
        "",
        "Parametros usados:",
    ]

    for nome, valor in parametros.items():
        linhas.append(f"- {nome}: {valor}")

    linhas.extend(
        [
            "",
            f"Geracoes executadas: {geracoes_executadas}",
            f"Melhor cromossomo: {''.join(str(bit) for bit in melhor.cromossomo)}",
            f"Melhor x: {melhor.x:.6f}",
            f"Melhor y: {melhor.y:.6f}",
            f"Melhor fitness: {melhor.fitness:.6f}",
        ]
    )

    with caminho.open("w", encoding="utf-8") as arquivo:
        arquivo.write("\n".join(linhas))


def formatar_tabela(cabecalhos: list[str], linhas: list[list[str]]) -> str:
    """Monta uma tabela textual simples para o terminal."""
    larguras = [len(cabecalho) for cabecalho in cabecalhos]
    for linha in linhas:
        for indice, valor in enumerate(linha):
            larguras[indice] = max(larguras[indice], len(valor))

    saida = [
        " | ".join(
            cabecalho.ljust(larguras[indice])
            for indice, cabecalho in enumerate(cabecalhos)
        ),
        "-+-".join("-" * largura for largura in larguras),
    ]

    for linha in linhas:
        saida.append(
            " | ".join(valor.ljust(larguras[indice]) for indice, valor in enumerate(linha))
        )

    return "\n".join(saida)


def selecionar_resumo_geracoes(
    historico: list[dict[str, float | int | str]]
) -> list[list[str]]:
    """Escolhe algumas geracoes para resumir a evolucao do AG."""
    resumo_geracoes = [
        historico[0],
        historico[min(1, len(historico) - 1)],
        historico[min(5, len(historico) - 1)],
        historico[-1],
    ]

    linhas_tabela = []
    indices_usados: set[int] = set()
    for resumo in resumo_geracoes:
        geracao = int(resumo["geracao"])
        if geracao in indices_usados:
            continue
        indices_usados.add(geracao)
        linhas_tabela.append(
            [
                str(geracao),
                f"{float(resumo['melhor_fitness']):.6f}",
                f"{float(resumo['fitness_medio']):.6f}",
                f"{float(resumo['melhor_x']):.6f}",
                f"{float(resumo['melhor_y']):.6f}",
            ]
        )

    return linhas_tabela


def render_melhor_solucao(melhor: Individuo) -> str:
    """Formata a melhor solucao final encontrada."""
    return formatar_tabela(
        ["x", "y", "fitness", "cromossomo"],
        [
            [
                f"{melhor.x:.6f}",
                f"{melhor.y:.6f}",
                f"{melhor.fitness:.6f}",
                "".join(str(bit) for bit in melhor.cromossomo),
            ]
        ],
    )
