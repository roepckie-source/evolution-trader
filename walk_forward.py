"""
Evolution Trader
Walk-Forward / Out-of-Sample Test

V1

Workflow:

1. Load 30 days of BTC/USD 5m data
2. Split chronologically:
       70% TRAIN
       30% OOS
3. Evolve strategies ONLY on TRAIN
4. Freeze final TRAIN population
5. Test TRAIN survivors on OOS
6. Report OOS survival

IMPORTANT:
- OOS data is NEVER used during evolution
- OOS data does NOT influence fitness
- OOS data does NOT influence mutation
- OOS data does NOT influence crossover
- OOS data does NOT influence parent selection
- NO live trading
"""

from dataclasses import dataclass
from typing import List, Tuple

import config

from backtester import Backtester
from fitness import calculate_fitness
from market_data import (
    download_history,
    save_csv,
    load_csv,
    validate_data,
)
from population import Population, Individual


# ============================================================
# SETTINGS
# ============================================================

DATA_DAYS = 30

TRAIN_PERCENT = 0.70
OOS_PERCENT = 0.30

EVOLUTION_GENERATIONS = 3


# ============================================================
# RESULT
# ============================================================

@dataclass
class OOSResult:
    strategy_id: str
    family: str

    train_fitness: float

    oos_return: float
    oos_drawdown: float
    oos_win_rate: float
    oos_profit_factor: float
    oos_trades: int

    oos_survives: bool

    rejection_reason: str


# ============================================================
# LOAD MARKET DATA
# ============================================================

def load_market_data() -> list:
    """
    Load existing 30-day CSV if available.

    If no CSV exists, download 30 days from Coinbase,
    validate them and save them to CSV.
    """

    try:

        print()
        print("=" * 70)
        print("LOADING MARKET DATA")
        print("=" * 70)

        candles = load_csv()

        print(
            f"Existing CSV found: "
            f"{len(candles)} candles"
        )

        validate_data(candles)

        # ----------------------------------------------------
        # We need enough data for the configured test.
        # ----------------------------------------------------

        minimum_required = 100

        if len(candles) < minimum_required:

            print(
                "CSV contains too few candles."
            )

            print(
                "Downloading fresh 30-day dataset..."
            )

            candles = download_history(
                days=DATA_DAYS
            )

            validate_data(candles)

            save_csv(candles)

        return candles

    except FileNotFoundError:

        print(
            "No market-data CSV found."
        )

        print(
            f"Downloading {DATA_DAYS} days..."
        )

        candles = download_history(
            days=DATA_DAYS
        )

        validate_data(candles)

        save_csv(candles)

        return candles


# ============================================================
# SPLIT DATA
# ============================================================

def split_data(
    candles: list,
) -> Tuple[list, list]:
    """
    Chronological 70/30 split.

    TRAIN:
        first 70%

    OOS:
        final 30%

    No shuffling.
    """

    if not candles:

        raise ValueError(
            "No market data available."
        )

    split_index = int(
        len(candles) * TRAIN_PERCENT
    )

    if split_index <= 0:

        raise ValueError(
            "TRAIN dataset is empty."
        )

    if split_index >= len(candles):

        raise ValueError(
            "OOS dataset is empty."
        )

    train_candles = candles[
        :split_index
    ]

    oos_candles = candles[
        split_index:
    ]

    return (
        train_candles,
        oos_candles,
    )


# ============================================================
# EVALUATE STRATEGY
# ============================================================

def evaluate_strategy(
    individual: Individual,
    candles: list,
):
    """
    Backtest one strategy.

    Returns:
        backtest result
        fitness result
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

    return (
        result,
        fitness_result,
    )


# ============================================================
# TRAIN EVOLUTION
# ============================================================

def evolve_on_train(
    train_candles: list,
) -> Population:
    """
    Run the evolutionary process ONLY on TRAIN data.

    OOS data is not available here.
    """

    print()
    print("=" * 70)
    print("TRAINING EVOLUTION")
    print("=" * 70)

    print(
        f"TRAIN candles: "
        f"{len(train_candles)}"
    )

    print(
        f"Population:    "
        f"{config.POPULATION_SIZE}"
    )

    print(
        f"Generations:   "
        f"{EVOLUTION_GENERATIONS}"
    )

    print()

    # --------------------------------------------------------
    # INITIAL POPULATION
    # --------------------------------------------------------

    population = Population()

    population.create_initial_population()

    # ========================================================
    # GENERATIONS
    # ========================================================

    for generation in range(
        1,
        EVOLUTION_GENERATIONS + 1,
    ):

        print()
        print("#" * 70)

        print(
            f"TRAIN GENERATION "
            f"{generation}/"
            f"{EVOLUTION_GENERATIONS}"
        )

        print("#" * 70)

        # ----------------------------------------------------
        # EVALUATE ALL STRATEGIES
        # ----------------------------------------------------

        total = len(
            population.individuals
        )

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

            if result.profit_factor == float(
                "inf"
            ):

                pf_text = "INF"

            else:

                pf_text = (
                    f"{result.profit_factor:.2f}"
                )

            status = (
                "SURVIVES"
                if fitness_result.survives
                else "DEAD"
            )

            print(
                f"[{index:03d}/{total:03d}] "
                f"{individual.strategy_id:<12} "
                f"{individual.genome.family:<22} "
                f"Return="
                f"{result.total_return * 100:7.2f}% "
                f"DD="
                f"{result.max_drawdown * 100:6.2f}% "
                f"WR="
                f"{result.win_rate * 100:6.2f}% "
                f"PF="
                f"{pf_text:>6} "
                f"Fitness="
                f"{fitness_result.score:8.3f} "
                f"{status}"
            )

        # ----------------------------------------------------
        # SUMMARY
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
        # STOP IF NOTHING SURVIVES
        # ----------------------------------------------------

        if not survivors:

            print()
            print(
                "NO TRAIN SURVIVORS."
            )

            print(
                "Evolution stopped."
            )

            break

        # ----------------------------------------------------
        # CREATE NEXT GENERATION
        # ----------------------------------------------------

        if generation < EVOLUTION_GENERATIONS:

            population = (
                population.create_next_generation()
            )

    return population


# ============================================================
# OOS TEST
# ============================================================

def evaluate_oos(
    population: Population,
    oos_candles: list,
) -> List[OOSResult]:
    """
    Evaluate frozen TRAIN survivors on OOS.

    IMPORTANT:

    OOS results are NEVER fed back into the
    evolutionary process.
    """

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
        "TRAIN population is now FROZEN."
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

    print(
        "No evolution using OOS results."
    )

    print()

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Only TRAIN survivors are allowed into OOS.
    # --------------------------------------------------------

    candidates = population.survivors()

    print(
        f"Frozen TRAIN survivors: "
        f"{len(candidates)}"
    )

    print()

    results: List[OOSResult] = []

    # ========================================================
    # TEST EACH FROZEN STRATEGY
    # ========================================================

    for index, individual in enumerate(
        candidates,
        start=1,
    ):

        backtester = Backtester(
            candles=oos_candles,
            genome=individual.genome,
            starting_capital=config.STARTING_CAPITAL,
        )

        result = backtester.run()

        # ----------------------------------------------------
        # OOS SURVIVAL GATES
        # ----------------------------------------------------

        survives = True

        rejection_reasons = []

        if result.total_trades < config.MIN_TRADES:

            survives = False

            rejection_reasons.append(
                "MIN_TRADES"
            )

        if result.win_rate < config.MIN_WIN_RATE:

            survives = False

            rejection_reasons.append(
                "WIN_RATE"
            )

        if result.max_drawdown > config.MAX_DRAWDOWN:

            survives = False

            rejection_reasons.append(
                "MAX_DRAWDOWN"
            )

        if result.total_return <= config.MIN_OOS_RETURN:

            survives = False

            rejection_reasons.append(
                "OOS_RETURN"
            )

        # ----------------------------------------------------
        # Profit factor robustness gate
        # ----------------------------------------------------

        if result.profit_factor <= 1.0:

            survives = False

            rejection_reasons.append(
                "PROFIT_FACTOR"
            )

        # ----------------------------------------------------
        # Rejection text
        # ----------------------------------------------------

        if rejection_reasons:

            rejection_reason = ",".join(
                rejection_reasons
            )

        else:

            rejection_reason = ""

        # ----------------------------------------------------
        # PF display
        # ----------------------------------------------------

        if result.profit_factor == float(
            "inf"
        ):

            pf_text = "INF"

        else:

            pf_text = (
                f"{result.profit_factor:.2f}"
            )

        status = (
            "OOS SURVIVES"
            if survives
            else "OOS DEAD"
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        oos_result = OOSResult(

            strategy_id=(
                individual.strategy_id
            ),

            family=(
                individual.genome.family
            ),

            train_fitness=(
                individual.fitness
            ),

            oos_return=(
                result.total_return
            ),

            oos_drawdown=(
                result.max_drawdown
            ),

            oos_win_rate=(
                result.win_rate
            ),

            oos_profit_factor=(
                result.profit_factor
            ),

            oos_trades=(
                result.total_trades
            ),

            oos_survives=(
                survives
            ),

            rejection_reason=(
                rejection_reason
            ),
        )

        results.append(
            oos_result
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

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
            + (
                f" [{rejection_reason}]"
                if rejection_reason
                else ""
            )
        )

    return results


# ============================================================
# FINAL OOS REPORT
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
            "No TRAIN survivors were available."
        )

        print(
            "Therefore no OOS test was possible."
        )

        return

    # --------------------------------------------------------
    # OOS survivors
    # --------------------------------------------------------

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

    # ========================================================
    # BEST TRAIN FITNESS
    # ========================================================

    best_train = max(
        results,
        key=lambda result:
        result.train_fitness,
    )

    print(
        "Best TRAIN strategy:"
    )

    print(
        f"  Strategy: "
        f"{best_train.strategy_id}"
    )

    print(
        f"  Family:   "
        f"{best_train.family}"
    )

    print(
        f"  Fitness:  "
        f"{best_train.train_fitness:.4f}"
    )

    # ========================================================
    # BEST OOS RETURN
    # ========================================================

    best_oos = max(
        results,
        key=lambda result:
        result.oos_return,
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
        f"  Family:   "
        f"{best_oos.family}"
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
        f"  Winrate:  "
        f"{best_oos.oos_win_rate * 100:.2f}%"
    )

    if best_oos.oos_profit_factor == float(
        "inf"
    ):

        print(
            "  PF:       INF"
        )

    else:

        print(
            f"  PF:       "
            f"{best_oos.oos_profit_factor:.2f}"
        )

    print(
        f"  Trades:   "
        f"{best_oos.oos_trades}"
    )

    print()

    # ========================================================
    # OOS SURVIVORS
    # ========================================================

    print("=" * 70)
    print("OOS SURVIVORS")
    print("=" * 70)

    if not survivors:

        print(
            "NONE"
        )

    else:

        sorted_survivors = sorted(
            survivors,
            key=lambda result:
            result.oos_return,
            reverse=True,
        )

        for result in sorted_survivors:

            if result.oos_profit_factor == float(
                "inf"
            ):

                pf_text = "INF"

            else:

                pf_text = (
                    f"{result.oos_profit_factor:.2f}"
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
                f"{pf_text:>6} "
                f"Trades="
                f"{result.oos_trades}"
            )

    print("=" * 70)

    # ========================================================
    # FAMILY ANALYSIS
    # ========================================================

    print()
    print("OOS FAMILY RESULTS")
    print("=" * 70)

    families = {}

    for result in results:

        family = result.family

        if family not in families:

            families[family] = {
                "tested": 0,
                "survived": 0,
            }

        families[family]["tested"] += 1

        if result.oos_survives:

            families[family]["survived"] += 1

    for family in sorted(
        families.keys()
    ):

        stats = families[family]

        print(
            f"{family:<24} "
            f"tested="
            f"{stats['tested']:3d} "
            f"survived="
            f"{stats['survived']:3d}"
        )

    print("=" * 70)

    # ========================================================
    # GENERALIZATION RATE
    # ========================================================

    print()

    train_count = len(
        results
    )

    oos_count = len(
        survivors
    )

    if train_count > 0:

        generalization_rate = (
            oos_count
            / train_count
        )

    else:

        generalization_rate = 0.0

    print(
        "TRAIN → OOS GENERALIZATION"
    )

    print(
        f"TRAIN survivors: "
        f"{train_count}"
    )

    print(
        f"OOS survivors:   "
        f"{oos_count}"
    )

    print(
        f"Survival rate:   "
        f"{generalization_rate * 100:.2f}%"
    )

    print()

    print(
        "This percentage is descriptive only."
    )

    print(
        "It is NOT used to modify the strategy population."
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("EVOLUTION TRADER")
    print("70/30 WALK-FORWARD OOS TEST")
    print("=" * 70)

    print(
        f"Symbol:       "
        f"{config.SYMBOL}"
    )

    print(
        f"Timeframe:    "
        f"{config.TIMEFRAME}"
    )

    print(
        f"Data:         "
        f"{DATA_DAYS} days"
    )

    print(
        f"Train:        "
        f"{TRAIN_PERCENT * 100:.0f}%"
    )

    print(
        f"OOS:          "
        f"{OOS_PERCENT * 100:.0f}%"
    )

    print(
        f"Population:   "
        f"{config.POPULATION_SIZE}"
    )

    print(
        f"Generations:  "
        f"{EVOLUTION_GENERATIONS}"
    )

    print(
        "Mode:         "
        "PAPER / BACKTEST ONLY"
    )

    print(
        "Live trading: "
        "DISABLED"
    )

    print("=" * 70)

    # ========================================================
    # LOAD DATA
    # ========================================================

    candles = load_market_data()

    print()
    print(
        f"Total candles loaded: "
        f"{len(candles)}"
    )

    # ========================================================
    # SPLIT
    # ========================================================

    train_candles, oos_candles = (
        split_data(
            candles
        )
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

    # ========================================================
    # TRAIN
    # ========================================================

    final_population = (
        evolve_on_train(
            train_candles
        )
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
        f"TRAIN candles:     "
        f"{len(train_candles)}"
    )

    print(
        f"OOS candles:       "
        f"{len(oos_candles)}"
    )

    print(
        f"TRAIN survivors:   "
        f"{len(final_population.survivors())}"
    )

    print(
        f"OOS survivors:     "
        f"{sum(1 for result in oos_results if result.oos_survives)}"
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
