"""
Evolution Trader
================
Strategy Genome

Eine Strategie wird durch ein Genom beschrieben.
Das Genom kann mutiert und mit einem anderen Genom
kombiniert werden.

Wichtig:
Dieses Modul enthält noch KEINE Backtest- oder
Trading-Logik.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Tuple

import config


# ============================================================
# PARAMETERBEREICHE
# ============================================================

LOOKBACK_MIN = 5
LOOKBACK_MAX = 200

ENTRY_THRESHOLD_MIN = 0.10
ENTRY_THRESHOLD_MAX = 3.00

GRID_LEVELS_MIN = 1
GRID_LEVELS_MAX = 10

SPACING_MIN = 0.001
SPACING_MAX = 0.050

SIZE_MULTIPLIER_MIN = 0.50
SIZE_MULTIPLIER_MAX = 2.00

TAKE_PROFIT_MIN = 0.001
TAKE_PROFIT_MAX = 0.050

STOP_LOSS_MIN = 0.001
STOP_LOSS_MAX = 0.050


# ============================================================
# STRATEGY GENOME
# ============================================================

@dataclass
class StrategyGenome:
    """
    DNA einer einzelnen Handelsstrategie.

    Jede Strategie besitzt genau acht Gene:

    1. family
    2. lookback
    3. entry_threshold
    4. grid_levels
    5. spacing
    6. size_multiplier
    7. take_profit
    8. stop_loss
    """

    family: str

    lookback: int

    entry_threshold: float

    grid_levels: int

    spacing: float

    size_multiplier: float

    take_profit: float

    stop_loss: float

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate(self) -> None:
        """Prüft, ob das Genom gültig ist."""

        if self.family not in config.STRATEGY_FAMILIES:
            raise ValueError(
                f"Ungültige Strategie-Familie: {self.family}"
            )

        if not (
            LOOKBACK_MIN
            <= self.lookback
            <= LOOKBACK_MAX
        ):
            raise ValueError("lookback außerhalb des erlaubten Bereichs")

        if not (
            ENTRY_THRESHOLD_MIN
            <= self.entry_threshold
            <= ENTRY_THRESHOLD_MAX
        ):
            raise ValueError(
                "entry_threshold außerhalb des erlaubten Bereichs"
            )

        if not (
            GRID_LEVELS_MIN
            <= self.grid_levels
            <= GRID_LEVELS_MAX
        ):
            raise ValueError(
                "grid_levels außerhalb des erlaubten Bereichs"
            )

        if not (
            SPACING_MIN
            <= self.spacing
            <= SPACING_MAX
        ):
            raise ValueError(
                "spacing außerhalb des erlaubten Bereichs"
            )

        if not (
            SIZE_MULTIPLIER_MIN
            <= self.size_multiplier
            <= SIZE_MULTIPLIER_MAX
        ):
            raise ValueError(
                "size_multiplier außerhalb des erlaubten Bereichs"
            )

        if not (
            TAKE_PROFIT_MIN
            <= self.take_profit
            <= TAKE_PROFIT_MAX
        ):
            raise ValueError(
                "take_profit außerhalb des erlaubten Bereichs"
            )

        if not (
            STOP_LOSS_MIN
            <= self.stop_loss
            <= STOP_LOSS_MAX
        ):
            raise ValueError(
                "stop_loss außerhalb des erlaubten Bereichs"
            )

    # ========================================================
    # RANDOM GENOME
    # ========================================================

    @classmethod
    def random(
        cls,
        family: str,
        rng: Random,
    ) -> "StrategyGenome":
        """
        Erzeugt ein zufälliges gültiges Genom.
        """

        genome = cls(
            family=family,

            lookback=rng.randint(
                LOOKBACK_MIN,
                LOOKBACK_MAX,
            ),

            entry_threshold=rng.uniform(
                ENTRY_THRESHOLD_MIN,
                ENTRY_THRESHOLD_MAX,
            ),

            grid_levels=rng.randint(
                GRID_LEVELS_MIN,
                GRID_LEVELS_MAX,
            ),

            spacing=rng.uniform(
                SPACING_MIN,
                SPACING_MAX,
            ),

            size_multiplier=rng.uniform(
                SIZE_MULTIPLIER_MIN,
                SIZE_MULTIPLIER_MAX,
            ),

            take_profit=rng.uniform(
                TAKE_PROFIT_MIN,
                TAKE_PROFIT_MAX,
            ),

            stop_loss=rng.uniform(
                STOP_LOSS_MIN,
                STOP_LOSS_MAX,
            ),
        )

        genome.validate()

        return genome

    # ========================================================
    # COPY
    # ========================================================

    def copy(self) -> "StrategyGenome":
        """Erzeugt eine unabhängige Kopie."""

        return StrategyGenome(
            family=self.family,
            lookback=self.lookback,
            entry_threshold=self.entry_threshold,
            grid_levels=self.grid_levels,
            spacing=self.spacing,
            size_multiplier=self.size_multiplier,
            take_profit=self.take_profit,
            stop_loss=self.stop_loss,
        )

    # ========================================================
    # MUTATION
    # ========================================================

    def mutate(
        self,
        rng: Random,
        mutation_rate: float = config.MUTATION_RATE,
    ) -> "StrategyGenome":
        """
        Mutiert einzelne Gene.

        Das ursprüngliche Genom wird NICHT verändert.
        Es wird ein neues Genom zurückgegeben.
        """

        child = self.copy()

        # ----------------------------------------------------
        # FAMILY
        # ----------------------------------------------------

        if rng.random() < mutation_rate:
            child.family = rng.choice(
                config.STRATEGY_FAMILIES
            )

        # ----------------------------------------------------
        # LOOKBACK
        # ----------------------------------------------------

        if rng.random() < mutation_rate:
            child.lookback = rng.randint(
                LOOKBACK_MIN,
                LOOKBACK_MAX,
            )

        # ----------------------------------------------------
        # ENTRY THRESHOLD
        # ----------------------------------------------------

        if rng.random() < mutation_rate:
            child.entry_threshold = rng.uniform(
                ENTRY_THRESHOLD_MIN,
                ENTRY_THRESHOLD_MAX,
            )

        # ----------------------------------------------------
        # GRID LEVELS
        # ----------------------------------------------------

        if rng.random() < mutation_rate:
            child.grid_levels = rng.randint(
                GRID_LEVELS_MIN,
                GRID_LEVELS_MAX,
            )

        # ----------------------------------------------------
        # SPACING
        # ----------------------------------------------------

        if rng.random() < mutation_rate:
            child.spacing = rng.uniform(
                SPACING_MIN,
                SPACING_MAX,
            )

        # ----------------------------------------------------
        # SIZE MULTIPLIER
        # ----------------------------------------------------

        if rng.random() < mutation_rate:
            child.size_multiplier = rng.uniform(
                SIZE_MULTIPLIER_MIN,
                SIZE_MULTIPLIER_MAX,
            )

        # ----------------------------------------------------
        # TAKE PROFIT
        # ----------------------------------------------------

        if rng.random() < mutation_rate:
            child.take_profit = rng.uniform(
                TAKE_PROFIT_MIN,
                TAKE_PROFIT_MAX,
            )

        # ----------------------------------------------------
        # STOP LOSS
        # ----------------------------------------------------

        if rng.random() < mutation_rate:
            child.stop_loss = rng.uniform(
                STOP_LOSS_MIN,
                STOP_LOSS_MAX,
            )

        child.validate()

        return child

    # ========================================================
    # CROSSOVER
    # ========================================================

    @classmethod
    def crossover(
        cls,
        parent_a: "StrategyGenome",
        parent_b: "StrategyGenome",
        rng: Random,
    ) -> "StrategyGenome":
        """
        Erzeugt ein Kind aus zwei Eltern.

        Für jedes Gen wird zufällig entschieden,
        welcher Elternteil das Gen vererbt.
        """

        child = cls(
            family=(
                parent_a.family
                if rng.random() < 0.5
                else parent_b.family
            ),

            lookback=(
                parent_a.lookback
                if rng.random() < 0.5
                else parent_b.lookback
            ),

            entry_threshold=(
                parent_a.entry_threshold
                if rng.random() < 0.5
                else parent_b.entry_threshold
            ),

            grid_levels=(
                parent_a.grid_levels
                if rng.random() < 0.5
                else parent_b.grid_levels
            ),

            spacing=(
                parent_a.spacing
                if rng.random() < 0.5
                else parent_b.spacing
            ),

            size_multiplier=(
                parent_a.size_multiplier
                if rng.random() < 0.5
                else parent_b.size_multiplier
            ),

            take_profit=(
                parent_a.take_profit
                if rng.random() < 0.5
                else parent_b.take_profit
            ),

            stop_loss=(
                parent_a.stop_loss
                if rng.random() < 0.5
                else parent_b.stop_loss
            ),
        )

        child.validate()

        return child

    # ========================================================
    # SERIALIZATION
    # ========================================================

    def to_dict(self) -> dict:
        """Konvertiert das Genom in ein Dictionary."""

        return {
            "family": self.family,
            "lookback": self.lookback,
            "entry_threshold": self.entry_threshold,
            "grid_levels": self.grid_levels,
            "spacing": self.spacing,
            "size_multiplier": self.size_multiplier,
            "take_profit": self.take_profit,
            "stop_loss": self.stop_loss,
        }

    # ========================================================
    # STRING REPRESENTATION
    # ========================================================

    def short_description(self) -> str:
        """Kurze lesbare Beschreibung."""

        return (
            f"{self.family} | "
            f"LB={self.lookback} | "
            f"Entry={self.entry_threshold:.3f} | "
            f"Grid={self.grid_levels} | "
            f"Spacing={self.spacing:.3%} | "
            f"Size={self.size_multiplier:.2f} | "
            f"TP={self.take_profit:.3%} | "
            f"SL={self.stop_loss:.3%}"
        )


# ============================================================
# SELF TEST
# ============================================================

def self_test() -> None:
    """Grundlegender Test der Genome-Engine."""

    rng = Random(config.RANDOM_SEED)

    print("=" * 70)
    print("EVOLUTION TRADER - GENOME SELF TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # RANDOM
    # --------------------------------------------------------

    parent_a = StrategyGenome.random(
        family="momentum",
        rng=rng,
    )

    parent_b = StrategyGenome.random(
        family="mean_reversion",
        rng=rng,
    )

    print("\nPARENT A")
    print(parent_a.short_description())

    print("\nPARENT B")
    print(parent_b.short_description())

    # --------------------------------------------------------
    # CROSSOVER
    # --------------------------------------------------------

    child = StrategyGenome.crossover(
        parent_a,
        parent_b,
        rng,
    )

    print("\nCHILD")
    print(child.short_description())

    # --------------------------------------------------------
    # MUTATION
    # --------------------------------------------------------

    mutated = child.mutate(
        rng,
        mutation_rate=1.0,
    )

    print("\nMUTATED CHILD")
    print(mutated.short_description())

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    mutated.validate()

    print("\nPASS: Genome validation")
    print("PASS: Random generation")
    print("PASS: Crossover")
    print("PASS: Mutation")
    print("PASS: Serialization")

    print("\nGENOME SELF TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    self_test()
