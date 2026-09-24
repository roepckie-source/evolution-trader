"""
Evolution Trader
Walk-Forward / Out-of-Sample Test

V1

Workflow:

1. Load historical BTC 5m data
2. Split chronologically into:
       70% TRAIN
       30% OOS
3. Evolve strategies ONLY on TRAIN data
4. Freeze the final TRAIN population
5. Evaluate the frozen population on OOS data
6. Report TRAIN vs OOS results

IMPORTANT:
- OOS data is never used during evolution
- OOS data does not influence mutation
- OOS data does not influence parent selection
- OOS data does not influence fitness
- No live trading
"""

from dataclasses import dataclass
from typing import List, Tuple

import config

from backtester import Backtester
from fitness import calculate_fitness
from market_data import load_candles
from population import Population, Individual


# ============================================================
# SETTINGS
# ============================================================

TRAIN_PERCENT = 0.70
OOS_PERCENT = 0.30

EVOLUTION_GENERATIONS = 3

MIN_OOS_RETURN = config.MIN_OOS_RETURN


# ============================================================
# RESULT
# ============================================================

@dataclass
class OOSResult:
    strategy_id: str
    family: str

    train_fitness: float
    train_return: float
    train_drawdown: float
    train_win_rate: float
    train_profit_factor: float
    train_trades: int

    oos_return: float
    oos_drawdown: float
    oos_win_rate: float
    oos_profit_factor: float
    oos_trades: int

    oos_survives: bool


# ============================================================
# DATA SPLIT
# ============================================================

def split_data(
    candles,
) -> Tuple[list, list]:

    if len(candles) < 100:
        raise ValueError(
            "Not enough candles for train/OOS split."
        )

    split_index = int(
        len(candles) * TRAIN_PERCENT
    )

    train = candles[:split_index]
    oos = candles[split_index:]

    if not train:
        raise ValueError(
            "TRAIN dataset is empty."
        )

    if not oos:
        raise ValueError(
            "OOS dataset is empty."
        )

    return train, oos


# ============================================================
# EVALUATE ONE STRATEGY
# ============================================================

def evaluate_strategy(
    individual: Individual,
    candles,
):
    """
    Run one genome through the backtester.
    """

    backtester = Backtester(
        candles=candles,
        genome=individual.genome,
        starting_capital=config.STARTING_CAPITAL,
    )

    result = backtester.run()

    fitness_result = calculate_fitness(
        result
    )

    return result, fitness_result


# ============================================================
# TRAIN EVOLUTION
# ============================================================

def evolve_on_train(
    train_candles,
) -> Population:

    print()
    print("=" * 70)
    print("TRAINING EVOLUTION")
    print("=" * 70)

    print(
        f"TRAIN candles: {len(train_candles)}"
    )

    print(
        f"Generations:   {EVOLUTION_GENERATIONS}"
    )

    print(
        f"Population:    {config.POPULATION_SIZE}"
    )

    print()

    population = Population(
        generation=1
    )

    population.create_initial_population()

    for generation in range(
        1,
        EVOLUTION_GENERATIONS + 1,
    ):

        print()
        print("#" * 70)
        print(
            f"TRAIN GENERATION "
            f"{generation}/{EVOLUTION_GENERATIONS}"
        )
        print("#" * 70)

        # ----------------------------------------------------
        # Evaluate population
        # ----------------------------------------------------

        for index, individual in enumerate(
            population.individuals,
            start=1,
        ):

            result, fitness_result = (
                evaluate_strategy(
                    individual,
                    train_candles,
                )
            )

            population.set_fitness(
                individual.strategy_id,
                fitness_result.score,
                fitness_result.survives,
                fitness_result.rejection_reason,
            )

            print(
                f"[{index:03d}/"
                f"{len(population.individuals):03d}] "
                f"{individual.strategy_id:<12} "
                f"{individual.genome.family:<22} "
                f"Return="
                f"{result.total_return * 100:7.2f}% "
                f"DD="
                f"{result.max_drawdown * 100:6.2f}% "
                f"WR="
                f"{result.win_rate * 100:6.2f}% "
                f"PF="
                f"{result.profit_factor:6.2f} "
                f"Fitness="
                f"{fitness_result.score:8.3f} "
                f"{'SURVIVES' if fitness_result.survives else 'DEAD'}"
            )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        ranked = population.ranked()

        survivors = population.survivors()

        print()
        print("=" * 70)
        print(
            f"TRAIN GENERATION "
            f"{generation} SUMMARY"
        )
        print("=" * 70)

        print(
            f"Strategies: "
            f"{len(population.individuals)}"
        )

        print(
            f"Survivors:  "
            f"{len(survivors)}"
        )

        if ranked:

            best = ranked[0]

            print(
                f"Best Strategy: "
                f"{best.strategy_id}"
            )

            print(
                f"Family:        "
                f"{best.genome.family}"
            )

            print(
                f"Fitness:       "
                f"{best.fitness:.4f}"
            )

        print("=" * 70)

        # ----------------------------------------------------
        # No survivors
        # ----------------------------------------------------

        if not survivors:

            print()
            print(
                "NO SURVIVING STRATEGIES."
            )

            print(
                "Evolution stopped."
            )

            break

        # ----------------------------------------------------
        # Create next generation
        # ----------------------------------------------------

        if generation < EVOLUTION_GENERATIONS:

            population = (
                population.create_next_generation()
            )

    return population


# ============================================================
# OOS EVALUATION
# ============================================================

def evaluate_oos(
    population: Population,
    oos_candles,
) -> List[OOSResult]:

    print()
    print("=" * 70)
    print("OUT-OF-SAMPLE EVALUATION")
    print("=" * 70)

    print(
        f"OOS candles: "
        f"{len(oos_candles)}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The OOS results are NOT used "
        "to modify the population."
    )

    print(
        "No mutation."
    )

    print(
        "No crossover."
    )

    print(
        "No parent selection."
    )

    print()

    # --------------------------------------------------------
    # Use TRAIN survivors only.
    #
    # We deliberately do not select the best OOS strategy.
    # --------------------------------------------------------

    candidates = population.survivors()

    print(
        f"Frozen TRAIN survivors: "
        f"{len(candidates)}"
    )

    print()

    results: List[OOSResult] = []

    for index, individual in enumerate(
        candidates,
        start=1,
    ):

        # ----------------------------------------------------
        # Backtest OOS
        # ----------------------------------------------------

        backtester = Backtester(
            candles=oos_candles,
            genome=individual.genome,
            starting_capital=config.STARTING_CAPITAL,
        )

        result = backtester.run()

        # ----------------------------------------------------
        # OOS survival rules
        #
        # These are evaluated independently from TRAIN.
        # ----------------------------------------------------

        oos_survives = True

        reasons = []

        if result.total_trades < config.MIN_TRADES:

            oos_survives = False

            reasons.append(
                "MIN_TRADES"
            )

        if result.win_rate < config.MIN_WIN_RATE:

            oos_survives = False

            reasons.append(
                "WIN_RATE"
            )

        if result.max_drawdown > config.MAX_DRAWDOWN:

            oos_survives = False

            reasons.append(
                "MAX_DRAWDOWN"
            )

        if result.total_return <= MIN_OOS_RETURN:

            oos_survives = False

            reasons.append(
                "OOS_RETURN"
            )

        if result.profit_factor <= 1.0:

            oos_survives = False

            reasons.append(
                "PROFIT_FACTOR"
            )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        result_row = OOSResult(

            strategy_id=individual.strategy_id,

            family=individual.genome.family,

            train_fitness=individual.fitness,

            train_return=0.0,

            train_drawdown=0.0,

            train_win_rate=0.0,

            train_profit_factor=0.0,

            train_trades=0,

            oos_return=result.total_return,

            oos_drawdown=result.max_drawdown,

            oos_win_rate=result.win_rate,

            oos_profit_factor=result.profit_factor,

            oos_trades=result.total_trades,

            oos_survives=oos_survives,
        )

        results.append(
            result_row
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        pf_text = (
            "INF"
            if result.profit_factor == float("inf")
            else f"{result.profit_factor:.2f}"
        )

        status = (
            "OOS SURVIVES"
            if oos_survives
            else "OOS DEAD"
        )

        reason_text = (
            ""
            if oos_survives
            else " | "
            + ",".join(reasons)
        )

        print(
            f"[{index:03d}/"
            f"{len(candidates):03d}] "
            f"{individual.strategy_id:<12} "
            f"{individual.genome.family:<22} "
            f"TRAIN Fitness="
            f"{individual.fitness:6.3f} "
            f"OOS Return="
            f"{result.total_return * 100:7.2f}% "
            f"DD="
            f"{result.max_drawdown * 100:6.2f}% "
            f"WR="
            f"{result.win_rate * 100:6.2f}% "
            f"PF="
            f"{pf_text:>6} "
            f"Trades="
            f"{result.total_trades:4d} "
            f"{status}"
            f"{reason_text}"
        )

    return results


# ============================================================
# REPORT
# ============================================================

def print_oos_report(
    results: List[OOSResult],
) -> None:

    print()
    print("#" * 70)
    print("FINAL OUT-OF-SAMPLE REPORT")
    print("#" * 70)

    if not results:

        print()
        print(
            "No TRAIN survivors available "
            "for OOS testing."
        )

        return

    survivors = [
        result
        for result in results
        if result.oos_survives
    ]

    print()
    print(
        f"TRAIN survivors tested: "
        f"{len(results)}"
    )

    print(
        f"OOS survivors:          "
        f"{len(survivors)}"
    )

    print()

    # --------------------------------------------------------
    # Best TRAIN fitness
    # --------------------------------------------------------

    best_train = max(
        results,
        key=lambda x: x.train_fitness,
    )

    # --------------------------------------------------------
    # Best OOS result
    #
    # This is only a REPORTING metric.
    # It is NOT used to modify the population.
    # --------------------------------------------------------

    best_oos = max(
        results,
        key=lambda x: x.oos_return,
    )

    print(
        "Best TRAIN fitness:"
    )

    print(
        f"  Strategy: "
        f"{best_train.strategy_id}"
    )

    print(
        f"  Fitness:  "
        f"{best_train.train_fitness:.4f}"
    )

    print()

    print(
        "Highest OOS return:"
    )

    print(
        f"  Strategy: "
        f"{best_oos.strategy_id}"
    )

    print(
        f"  Return:   "
        f"{best_oos.oos_return * 100:.2f}%"
    )

    print(
        f"  DD:       "
        f"{best_oos.oos_drawdown * 100:.2f}%"
    )

    print(
        f"  WR:       "
        f"{best_oos.oos_win_rate * 100:.2f}%"
    )

    print(
        f"  PF:       "
        f"{best_oos.oos_profit_factor}"
    )

    print()

    # --------------------------------------------------------
    # OOS survivors
    # --------------------------------------------------------

    print("=" * 70)
    print("OOS SURVIVORS")
    print("=" * 70)

    if not survivors:

        print(
            "NONE"
        )

    else:

        for result in sorted(
            survivors,
            key=lambda x: x.oos_return,
            reverse=True,
        ):

            pf_text = (
                "INF"
                if result.oos_profit_factor == float("inf")
                else f"{result.oos_profit_factor:.2f}"
            )

            print(
                f"{result.strategy_id:<12} "
                f"{result.family:<22} "
                f"Return="
                f"{result.oos_return * 100:7.2f}% "
                f"DD="
                f"{result.oos_drawdown * 100:6.2f}% "
                f"WR="
                f"{result.oos_win_rate * 100:6.2f}% "
                f"PF="
                f"{pf_text}"
            )

    print("=" * 70)

    # --------------------------------------------------------
    # Family statistics
    # --------------------------------------------------------

    print()
    print("OOS FAMILY RESULTS")
    print("=" * 70)

    families = {}

    for result in results:

        if result.family not in families:

            families[result.family] = {
                "tested": 0,
                "survived": 0,
            }

        families[result.family]["tested"] += 1

        if result.oos_survives:

            families[result.family]["survived"] += 1

    for family, stats in sorted(
        families.items()
    ):

        print(
            f"{family:<24} "
            f"tested={stats['tested']:3d} "
            f"survived={stats['survived']:3d}"
        )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("EVOLUTION TRADER - WALK FORWARD TEST")
    print("=" * 70)

    print(
        f"Symbol:       {config.SYMBOL}"
    )

    print(
        f"Timeframe:    {config.TIMEFRAME}"
    )

    print(
        f"Train split:  {TRAIN_PERCENT * 100:.0f}%"
    )

    print(
        f"OOS split:    {OOS_PERCENT * 100:.0f}%"
    )

    print(
        f"Generations:  {EVOLUTION_GENERATIONS}"
    )

    print(
        f"Population:   {config.POPULATION_SIZE}"
    )

    print(
        "Mode:         PAPER / BACKTEST ONLY"
    )

    print("=" * 70)

    # ========================================================
    # LOAD DATA
    # ========================================================

    print()
    print("Loading market data...")

    candles = load_candles()

    print(
        f"Loaded candles: "
        f"{len(candles)}"
    )

    # ========================================================
    # SPLIT
    # ========================================================

    train_candles, oos_candles = (
        split_data(candles)
    )

    print()
    print("=" * 70)
    print("DATA SPLIT")
    print("=" * 70)

    print(
        f"Total: "
        f"{len(candles)} candles"
    )

    print(
        f"TRAIN: "
        f"{len(train_candles)} candles "
        f"({TRAIN_PERCENT * 100:.0f}%)"
    )

    print(
        f"OOS:   "
        f"{len(oos_candles)} candles "
        f"({OOS_PERCENT * 100:.0f}%)"
    )

    print()

    print(
        "TRAIN and OOS are strictly chronological."
    )

    print(
        "OOS data will not influence evolution."
    )

    # ========================================================
    # EVOLUTION
    # ========================================================

    final_population = evolve_on_train(
        train_candles
    )

    # ========================================================
    # OOS
    # ========================================================

    oos_results = evaluate_oos(
        final_population,
        oos_candles,
    )

    # ========================================================
    # REPORT
    # ========================================================

    print_oos_report(
        oos_results
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print()
    print("=" * 70)
    print("WALK-FORWARD TEST COMPLETE")
    print("=" * 70)

    print(
        f"TRAIN candles: "
        f"{len(train_candles)}"
    )

    print(
        f"OOS candles:   "
        f"{len(oos_candles)}"
    )

    print(
        f"TRAIN survivors: "
        f"{len(final_population.survivors())}"
    )

    print(
        f"OOS survivors:   "
        f"{sum(1 for x in oos_results if x.oos_survives)}"
    )

    print()
    print(
        "NO LIVE TRADING"
    )

    print(
        "NO API KEYS"
    )

    print(
        "NO REAL ORDERS"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
