#!/usr/bin/env python3
"""Script principal do trabalho do algoritmo genetico."""

from __future__ import annotations

import argparse
from pathlib import Path

from core_ag import (
    BITS_POR_VARIAVEL,
    executar_algoritmo_genetico,
    resolucao_codificacao,
)
from io_utils import (
    formatar_tabela,
    render_melhor_solucao,
    salvar_historico_csv,
    salvar_resultado_txt,
    selecionar_resumo_geracoes,
)


ARQUIVO_HISTORICO = Path(__file__).with_name("historico_ag.csv")
ARQUIVO_RESULTADO = Path(__file__).with_name("resultado_ag.txt")


def parse_args() -> argparse.Namespace:
    """Permite alterar os parametros do AG via linha de comando."""
    parser = argparse.ArgumentParser(
        description="Algoritmo genetico binario para maximizar f(x, y)."
    )
    parser.add_argument("--population-size", type=int, default=80, help="Tamanho da populacao.")
    parser.add_argument("--crossover-rate", type=float, default=0.90, help="Taxa de crossover.")
    parser.add_argument("--mutation-rate", type=float, default=0.01, help="Taxa de mutacao por bit.")
    parser.add_argument("--tournament-size", type=int, default=3, help="Quantidade de individuos por torneio.")
    parser.add_argument("--elitism", type=int, default=2, help="Quantidade de elites mantidos por geracao.")
    parser.add_argument("--generations", type=int, default=250, help="Numero maximo de geracoes.")
    parser.add_argument("--stagnation-limit", type=int, default=60, help="Parada por estagnacao.")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatoria.")
    return parser.parse_args()


def main() -> None:
    """Executa o AG, mostra o resumo e salva os arquivos auxiliares."""
    args = parse_args()

    # Roda o nucleo do algoritmo genetico com os parametros escolhidos.
    melhor, historico, geracoes_executadas = executar_algoritmo_genetico(
        tamanho_populacao=args.population_size,
        taxa_crossover=args.crossover_rate,
        taxa_mutacao=args.mutation_rate,
        tamanho_torneio=args.tournament_size,
        elitismo=args.elitism,
        maximo_geracoes=args.generations,
        limite_estagnacao=args.stagnation_limit,
        seed=args.seed,
    )

    salvar_historico_csv(ARQUIVO_HISTORICO, historico)
    salvar_resultado_txt(
        ARQUIVO_RESULTADO,
        melhor,
        geracoes_executadas,
        {
            "tamanho_populacao": args.population_size,
            "taxa_crossover": args.crossover_rate,
            "taxa_mutacao": args.mutation_rate,
            "tamanho_torneio": args.tournament_size,
            "elitismo": args.elitism,
            "maximo_geracoes": args.generations,
            "limite_estagnacao": args.stagnation_limit,
            "seed": args.seed,
        },
    )

    # Prepara as tabelas e resumos para exibicao no terminal.
    linhas_tabela = selecionar_resumo_geracoes(historico)

    print("Algoritmo genetico binario para maximizar f(x, y)")
    print(f"Funcao: f(x, y) = |e^(-x) - y^2 + 1| + 10^-4")
    print(f"Bits por variavel: {BITS_POR_VARIAVEL}")
    print(f"Resolucao da codificacao: {resolucao_codificacao():.6f}")
    print(
        f"Parada: maximo de {args.generations} geracoes ou "
        f"{args.stagnation_limit} geracoes sem melhora"
    )
    print("\nResumo de algumas geracoes:")
    print(
        formatar_tabela(
            ["Geracao", "Melhor fitness", "Fitness medio", "Melhor x", "Melhor y"],
            linhas_tabela,
        )
    )
    print("\nMelhor solucao encontrada:")
    print(render_melhor_solucao(melhor))
    print(f"\nHistorico salvo em: {ARQUIVO_HISTORICO.name}")
    print(f"Resumo final salvo em: {ARQUIVO_RESULTADO.name}")


if __name__ == "__main__":
    main()
