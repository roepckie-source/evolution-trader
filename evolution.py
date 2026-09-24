"""
Evolution Trader
================
Evolution Engine

Verbindet:

Market Data
    ↓
Population
    ↓
Backtester
    ↓
Fitness
    ↓
Survivors
    ↓
Next Generation

V1:
- BTC/USD
- 5m
- 96 Strategien
- 4 Familien
- mehrere Generationen
- Paper / Backtest only
- keine Live-Orders

WICHTIG:
Diese Datei verwendet zunächst den kompletten
Datensatz als Trainingsdatensatz.

Der echte 70/30-OOS- und Walk-Forward-Test kommt
in walk_forward.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import config

from backtester import (
    BacktestResult,
    Backtester,
)

from fitness import (
    FitnessResult,
    calculate_fitness,
)

from market_data import (
    CSV_FILE,
    download_history,
    load_csv,
    save_csv,
    validate_data,
)

from population import (
    Individual,
    Population,
)


# ============================================================
# GENERATION RESULT
# ============================================================

@dataclass
class GenerationResult:
    """
    Ergebnis einer kompletten Generation.
    """

    generation: int

    total_strategies: int

    survivors: int

    best_strategy_id: Optional[str]

    best_fitness: Optional[float]

    best_return: Optional[float]

    best_drawdown: Optional[float]

    best_win_rate: Optional[float]

    best_profit_factor: Optional[float]


# ============================================================
# EVOLUTION ENGINE
# ============================================================

class EvolutionEngine:
    """
    Hauptmaschine der Evolution.
    """

    def __init__(
        self,
        candles,
    ):
        self.candles = candles

        self.population = Population(
            generation=1
        )

        self.history: List[
            GenerationResult
        ] = []

    # ========================================================
    # EVALUATE STRATEGY
    # ========================================================

    def evaluate_strategy(
        self,
        individual: Individual,
    ) -> tuple[
        BacktestResult,
        FitnessResult,
    ]:
        """
        Backtestet eine Strategie und berechnet
        anschließend deren Fitness.
        """

        backtester = Backtester(
            candles=self.candles,
            genome=individual.genome,
            starting_capital=(
                config.STARTING_CAPITAL
            ),
        )

        result = backtester.run()

        fitness = calculate_fitness(
            result
        )

        return result, fitness

    # ========================================================
    # EVALUATE POPULATION
    # ========================================================

    def evaluate_population(
        self,
    ) -> dict[
        str,
        tuple[
            BacktestResult,
            FitnessResult,
        ],
    ]:
        """
        Testet jede Strategie der aktuellen Population.
        """

        results = {}

        print()
        print("=" * 70)

        print(
            f"EVALUATING GENERATION "
            f"{self.population.generation}"
        )

        print("=" * 70)

        total = len(
            self.population.individuals
        )

        for index, individual in enumerate(
            self.population.individuals,
            start=1,
        ):

            result, fitness = (
                self.evaluate_strategy(
                    individual
                )
            )

            results[
                individual.strategy_id
            ] = (
                result,
                fitness,
            )

            self.population.set_fitness(
                strategy_id=(
                    individual.strategy_id
                ),

                fitness=fitness.score,

                survives=fitness.survives,

                rejection_reason=(
                    fitness.rejection_reason
                ),
            )

            print(
                f"[{index:03d}/{total:03d}] "
                f"{individual.strategy_id} "
                f"{individual.genome.family:<24} "
                f"Return={result.total_return:>8.2%} "
                f"DD={result.max_drawdown:>7.2%} "
                f"WR={result.win_rate:>7.2%} "
                f"PF={result.profit_factor:>6.2f} "
                f"Fitness={fitness.score:>8.3f} "
                f"{'SURVIVES' if fitness.survives else 'DEAD'}"
            )

        return results

    # ========================================================
    # GENERATION SUMMARY
    # ========================================================

    def summarize_generation(
        self,
        results,
    ) -> GenerationResult:
        """
        Erzeugt eine Zusammenfassung.
        """

        ranked = self.population.ranked()

        survivors = (
            self.population.survivors()
        )

        best_strategy_id = None
        best_fitness = None
        best_return = None
        best_drawdown = None
        best_win_rate = None
        best_profit_factor = None

        if survivors:

            # Nur echte Survivors betrachten.
            survivor_ranked = sorted(
                survivors,
                key=lambda individual: (
                    individual.fitness
                    if individual.fitness is not None
                    else -float("inf")
                ),
                reverse=True,
            )

            best = survivor_ranked[0]

            best_strategy_id = (
                best.strategy_id
            )

            best_fitness = (
                best.fitness
            )

            result, _ = results[
                best.strategy_id
            ]

            best_return = (
                result.total_return
            )

            best_drawdown = (
                result.max_drawdown
            )

            best_win_rate = (
                result.win_rate
            )

            best_profit_factor = (
                result.profit_factor
            )

        generation_result = GenerationResult(
            generation=(
                self.population.generation
            ),

            total_strategies=len(
                self.population.individuals
            ),

            survivors=len(
                survivors
            ),

            best_strategy_id=(
                best_strategy_id
            ),

            best_fitness=best_fitness,

            best_return=best_return,

            best_drawdown=best_drawdown,

            best_win_rate=best_win_rate,

            best_profit_factor=(
                best_profit_factor
            ),
        )

        self.history.append(
            generation_result
        )

        return generation_result

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    @staticmethod
    def print_generation_summary(
        summary: GenerationResult,
    ) -> None:
        """
        Gibt die wichtigsten Werte aus.
        """

        print()
        print("=" * 70)

        print(
            f"GENERATION "
            f"{summary.generation} SUMMARY"
        )

        print("=" * 70)

        print(
            f"Strategies:       "
            f"{summary.total_strategies}"
        )

        print(
            f"Survivors:        "
            f"{summary.survivors}"
        )

        if summary.best_strategy_id:

            print(
                f"Best Strategy:    "
                f"{summary.best_strategy_id}"
            )

            print(
                f"Fitness:          "
                f"{summary.best_fitness:.4f}"
            )

            print(
                f"Return:           "
                f"{summary.best_return:.2%}"
            )

            print(
                f"Drawdown:         "
                f"{summary.best_drawdown:.2%}"
            )

            print(
                f"Win Rate:         "
                f"{summary.best_win_rate:.2%}"
            )

            print(
                f"Profit Factor:    "
                f"{summary.best_profit_factor:.2f}"
            )

        else:

            print(
                "Best Strategy:    NONE"
            )

        print("=" * 70)

    # ========================================================
    # CREATE NEXT GENERATION
    # ========================================================

    def next_generation(
        self,
    ) -> None:
        """
        Erzeugt die nächste Population.
        """

        self.population = (
            self.population.create_next_generation()
        )

    # ========================================================
    # RUN
    # ========================================================

    def run(
        self,
        generations: int = 3,
    ) -> List[GenerationResult]:
        """
        Führt mehrere Generationen aus.

        Für den ersten Test verwenden wir nur wenige
        Generationen.

        Später kann dies auf 100+ erweitert werden.
        """

        if generations <= 0:

            raise ValueError(
                "generations muss > 0 sein."
            )

        # ----------------------------------------------------
        # GENERATION 1
        # ----------------------------------------------------

        self.population.create_initial_population()

        for generation_index in range(
            generations
        ):

            print()
            print(
                "#" * 70
            )

            print(
                f"EVOLUTION CYCLE "
                f"{generation_index + 1}"
                f"/{generations}"
            )

            print(
                "#" * 70
            )

            # -----------------------------------------------
            # EVALUATE
            # -----------------------------------------------

            results = (
                self.evaluate_population()
            )

            # -----------------------------------------------
            # SUMMARY
            # -----------------------------------------------

            summary = (
                self.summarize_generation(
                    results
                )
            )

            self.print_generation_summary(
                summary
            )

            # -----------------------------------------------
            # STOP IF NO SURVIVORS
            # -----------------------------------------------

            if not self.population.survivors():

                print()
                print(
                    "WARNING:"
                )

                print(
                    "No strategies survived."
                )

                print(
                    "Evolution stopped."
                )

                break

            # -----------------------------------------------
            # NEXT GENERATION
            # -----------------------------------------------

            if (
                generation_index
                < generations - 1
            ):

                self.next_generation()

        return self.history


# ============================================================
# MARKET DATA LOADER
# ============================================================

def load_market_data(
    days: int = 30,
):
    """
    Lädt vorhandene CSV-Daten.

    Falls keine CSV vorhanden ist,
    werden die Daten öffentlich heruntergeladen.
    """

    if CSV_FILE.exists():

        print(
            f"Loading existing market data: "
            f"{CSV_FILE}"
        )

        candles = load_csv(
            CSV_FILE
        )

    else:

        print(
            "No market data CSV found."
        )

        print(
            f"Downloading {days} days..."
        )

        candles = download_history(
            days=days
        )

        save_csv(
            candles,
            CSV_FILE,
        )

    validate_data(
        candles
    )

    return candles


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Startpunkt für den manuellen Evolutionstest.
    """

    print()
    print("=" * 70)
    print("EVOLUTION TRADER")
    print("=" * 70)

    print(
        f"Symbol:      {config.SYMBOL}"
    )

    print(
        f"Timeframe:   {config.TIMEFRAME}"
    )

    print(
        f"Population:  {config.POPULATION_SIZE}"
    )

    print(
        "Mode:        PAPER / BACKTEST ONLY"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    candles = load_market_data(
        days=30
    )

    print()
    print(
        f"Loaded {len(candles)} candles."
    )

    # --------------------------------------------------------
    # ENGINE
    # --------------------------------------------------------

    engine = EvolutionEngine(
        candles=candles
    )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    history = engine.run(
        generations=3
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EVOLUTION COMPLETE")
    print("=" * 70)

    for result in history:

        print(
            f"Generation "
            f"{result.generation}: "
            f"{result.survivors}/"
            f"{result.total_strategies} survivors"
        )

        if result.best_strategy_id:

            print(
                f"  Best: "
                f"{result.best_strategy_id}"
            )

            print(
                f"  Return: "
                f"{result.best_return:.2%}"
            )

            print(
                f"  DD: "
                f"{result.best_drawdown:.2%}"
            )

    print("=" * 70)


if __name__ == "__main__":
    main()
