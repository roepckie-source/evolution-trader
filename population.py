"""
Evolution Trader
================
Population Engine

Verwaltet die Population der Handelsstrategien.

V1:
- 96 Strategien
- 24 pro Strategie-Familie
- zufällige Generation 1
- Elite-Auswahl
- Crossover
- Mutation
- Random Injection

Wichtig:
Dieses Modul entscheidet noch NICHT, welche Strategie
wirtschaftlich gut ist.

Das macht später die Kombination aus:

Backtester
    ↓
Fitness
    ↓
Population Evolution
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import List, Optional

import config

from strategy_genome import StrategyGenome


# ============================================================
# EVOLUTION INDIVIDUAL
# ============================================================

@dataclass
class Individual:
    """
    Eine Strategie innerhalb der Population.

    genome:
        DNA der Strategie

    fitness:
        Wird erst nach dem Backtest gesetzt.

    generation:
        Generation, aus der die Strategie stammt.

    strategy_id:
        Eindeutige Kennung.
    """

    genome: StrategyGenome

    strategy_id: str

    generation: int

    fitness: Optional[float] = None

    survives: Optional[bool] = None

    rejection_reason: str = ""

    # --------------------------------------------------------
    # DESCRIPTION
    # --------------------------------------------------------

    def description(self) -> str:
        """Lesbare Beschreibung."""

        fitness_text = (
            f"{self.fitness:.4f}"
            if self.fitness is not None
            else "N/A"
        )

        return (
            f"{self.strategy_id} | "
            f"Gen={self.generation} | "
            f"Fitness={fitness_text} | "
            f"{self.genome.short_description()}"
        )


# ============================================================
# POPULATION
# ============================================================

class Population:
    """
    Verwaltet eine komplette Generation.
    """

    def __init__(
        self,
        generation: int = 1,
        rng: Optional[Random] = None,
    ):
        self.generation = generation

        self.rng = (
            rng
            if rng is not None
            else Random(config.RANDOM_SEED)
        )

        self.individuals: List[
            Individual
        ] = []

    # ========================================================
    # GENERATION 1
    # ========================================================

    def create_initial_population(
        self,
    ) -> List[Individual]:
        """
        Erzeugt Generation 1.

        Gleichmäßige Verteilung über alle Familien.
        """

        self.individuals = []

        counter = 1

        for family in config.STRATEGY_FAMILIES:

            for _ in range(
                config.STRATEGIES_PER_FAMILY
            ):

                genome = StrategyGenome.random(
                    family=family,
                    rng=self.rng,
                )

                strategy_id = (
                    f"G{self.generation:03d}"
                    f"-S{counter:03d}"
                )

                individual = Individual(
                    genome=genome,
                    strategy_id=strategy_id,
                    generation=self.generation,
                )

                self.individuals.append(
                    individual
                )

                counter += 1

        if len(self.individuals) != config.POPULATION_SIZE:
            raise RuntimeError(
                "Population size mismatch: "
                f"{len(self.individuals)} "
                f"!= "
                f"{config.POPULATION_SIZE}"
            )

        return self.individuals

    # ========================================================
    # SET FITNESS
    # ========================================================

    def set_fitness(
        self,
        strategy_id: str,
        fitness: float,
        survives: bool,
        rejection_reason: str = "",
    ) -> None:
        """
        Setzt das Ergebnis eines Backtests.
        """

        for individual in self.individuals:

            if individual.strategy_id == strategy_id:

                individual.fitness = fitness
                individual.survives = survives
                individual.rejection_reason = (
                    rejection_reason
                )

                return

        raise ValueError(
            f"Strategy not found: {strategy_id}"
        )

    # ========================================================
    # SORT
    # ========================================================

    def ranked(
        self,
    ) -> List[Individual]:
        """
        Gibt die Population nach Fitness sortiert zurück.

        Unbewertete Strategien landen unten.
        """

        return sorted(
            self.individuals,
            key=lambda individual: (
                individual.fitness
                if individual.fitness is not None
                else -float("inf")
            ),
            reverse=True,
        )

    # ========================================================
    # SURVIVORS
    # ========================================================

    def survivors(
        self,
    ) -> List[Individual]:
        """
        Gibt nur überlebende Strategien zurück.
        """

        return [
            individual
            for individual in self.individuals
            if individual.survives is True
        ]

    # ========================================================
    # ELITE
    # ========================================================

    def elite(
        self,
        count: Optional[int] = None,
    ) -> List[Individual]:
        """
        Gibt die besten Überlebenden zurück.
        """

        if count is None:
            count = config.ELITE_COUNT

        survivors = self.survivors()

        survivors.sort(
            key=lambda individual: (
                individual.fitness
                if individual.fitness is not None
                else -float("inf")
            ),
            reverse=True,
        )

        return survivors[:count]

    # ========================================================
    # SELECT PARENT
    # ========================================================

    def select_parent(
        self,
        candidates: List[Individual],
    ) -> Individual:
        """
        Wählt einen Elternteil.

        V1 verwendet ein einfaches Fitness-gewichtetes
        Auswahlverfahren.

        Die besten Strategien haben dadurch eine höhere
        Wahrscheinlichkeit, Eltern zu werden.

        Gleichzeitig können schwächere Survivors noch
        ausgewählt werden.
        """

        if not candidates:
            raise ValueError(
                "Keine Parent-Kandidaten."
            )

        # Nur positive Fitnesswerte verwenden.
        weights = []

        for candidate in candidates:

            fitness = (
                candidate.fitness
                if candidate.fitness is not None
                else 0.0
            )

            # Mindestgewicht verhindert, dass einzelne
            # Strategien vollständig ausgeschlossen werden.
            weights.append(
                max(fitness, 0.01)
            )

        return self.rng.choices(
            candidates,
            weights=weights,
            k=1,
        )[0]

    # ========================================================
    # RANDOM INJECTION
    # ========================================================

    def create_random_individual(
        self,
        index: int,
    ) -> Individual:
        """
        Erzeugt einen komplett neuen Zufallskandidaten.
        """

        family = self.rng.choice(
            config.STRATEGY_FAMILIES
        )

        genome = StrategyGenome.random(
            family=family,
            rng=self.rng,
        )

        strategy_id = (
            f"G{self.generation + 1:03d}"
            f"-R{index:03d}"
        )

        return Individual(
            genome=genome,
            strategy_id=strategy_id,
            generation=self.generation + 1,
        )

    # ========================================================
    # CHILD
    # ========================================================

    def create_child(
        self,
        parent_a: Individual,
        parent_b: Individual,
        child_index: int,
    ) -> Individual:
        """
        Erzeugt ein Kind durch Crossover + Mutation.
        """

        # ----------------------------------------------------
        # CROSSOVER
        # ----------------------------------------------------

        if (
            self.rng.random()
            < config.CROSSOVER_RATE
        ):

            genome = StrategyGenome.crossover(
                parent_a.genome,
                parent_b.genome,
                self.rng,
            )

        else:

            # Kein Crossover:
            # zufällig einen Elternteil klonen.
            genome = (
                parent_a.genome.copy()
                if self.rng.random() < 0.5
                else parent_b.genome.copy()
            )

        # ----------------------------------------------------
        # MUTATION
        # ----------------------------------------------------

        genome = genome.mutate(
            self.rng,
            mutation_rate=config.MUTATION_RATE,
        )

        strategy_id = (
            f"G{self.generation + 1:03d}"
            f"-C{child_index:03d}"
        )

        return Individual(
            genome=genome,
            strategy_id=strategy_id,
            generation=self.generation + 1,
        )

    # ========================================================
    # NEXT GENERATION
    # ========================================================

    def create_next_generation(
        self,
    ) -> "Population":
        """
        Erzeugt die nächste Generation.

        Struktur:

        1. Elite direkt übernehmen
        2. 8 Random Injection
        3. Rest durch Crossover + Mutation
        """

        survivors = self.survivors()

        if not survivors:

            raise RuntimeError(
                "Keine Survivor vorhanden. "
                "Eine neue Generation kann nicht "
                "aus toten Strategien gezüchtet werden."
            )

        elite = self.elite(
            config.ELITE_COUNT
        )

        next_population = Population(
            generation=self.generation + 1,
            rng=self.rng,
        )

        new_individuals: List[
            Individual
        ] = []

        # ----------------------------------------------------
        # 1. ELITE
        # ----------------------------------------------------

        for index, individual in enumerate(
            elite,
            start=1,
        ):

            copied_genome = (
                individual.genome.copy()
            )

            elite_id = (
                f"G{self.generation + 1:03d}"
                f"-E{index:03d}"
            )

            new_individuals.append(
                Individual(
                    genome=copied_genome,
                    strategy_id=elite_id,
                    generation=(
                        self.generation + 1
                    ),
                )
            )

        # ----------------------------------------------------
        # 2. RANDOM INJECTION
        # ----------------------------------------------------

        for index in range(
            1,
            config.RANDOM_INJECTIONS + 1,
        ):

            new_individuals.append(
                next_population.create_random_individual(
                    index
                )
            )

        # ----------------------------------------------------
        # 3. CHILDREN
        # ----------------------------------------------------

        child_index = 1

        while len(new_individuals) < config.POPULATION_SIZE:

            parent_a = self.select_parent(
                survivors
            )

            parent_b = self.select_parent(
                survivors
            )

            child = self.create_child(
                parent_a,
                parent_b,
                child_index,
            )

            new_individuals.append(
                child
            )

            child_index += 1

        # ----------------------------------------------------
        # FINAL SIZE
        # ----------------------------------------------------

        new_individuals = new_individuals[
            :config.POPULATION_SIZE
        ]

        next_population.individuals = (
            new_individuals
        )

        if len(
            next_population.individuals
        ) != config.POPULATION_SIZE:

            raise RuntimeError(
                "Next generation size mismatch."
            )

        return next_population

    # ========================================================
    # FAMILY DISTRIBUTION
    # ========================================================

    def family_distribution(
        self,
    ) -> dict[str, int]:
        """
        Zählt Strategien pro Familie.
        """

        distribution = {
            family: 0
            for family in config.STRATEGY_FAMILIES
        }

        for individual in self.individuals:

            family = (
                individual.genome.family
            )

            distribution[family] = (
                distribution.get(
                    family,
                    0,
                )
                + 1
            )

        return distribution

    # ========================================================
    # SUMMARY
    # ========================================================

    def print_summary(
        self,
    ) -> None:
        """Gibt eine Zusammenfassung aus."""

        print("=" * 70)
        print(
            f"POPULATION GENERATION "
            f"{self.generation}"
        )
        print("=" * 70)

        print(
            f"Total strategies: "
            f"{len(self.individuals)}"
        )

        print()

        print(
            "Family distribution:"
        )

        for family, count in (
            self.family_distribution()
            .items()
        ):

            print(
                f"  {family:<24} "
                f"{count}"
            )

        print()

        print(
            f"Survivors: "
            f"{len(self.survivors())}"
        )

        print(
            f"Elite: "
            f"{len(self.elite())}"
        )

        print("=" * 70)


# ============================================================
# SELF TEST
# ============================================================

def self_test() -> None:
    """
    Testet die Population Engine.
    """

    print("=" * 70)
    print("EVOLUTION TRADER - POPULATION SELF TEST")
    print("=" * 70)

    rng = Random(
        config.RANDOM_SEED
    )

    # --------------------------------------------------------
    # CREATE GENERATION 1
    # --------------------------------------------------------

    population = Population(
        generation=1,
        rng=rng,
    )

    individuals = (
        population.create_initial_population()
    )

    assert len(
        individuals
    ) == config.POPULATION_SIZE

    print()
    print(
        f"PASS: Created "
        f"{len(individuals)} strategies"
    )

    # --------------------------------------------------------
    # FAMILY DISTRIBUTION
    # --------------------------------------------------------

    distribution = (
        population.family_distribution()
    )

    for family in config.STRATEGY_FAMILIES:

        assert (
            distribution[family]
            == config.STRATEGIES_PER_FAMILY
        )

    print(
        "PASS: Family distribution"
    )

    # --------------------------------------------------------
    # SIMULATE FITNESS
    # --------------------------------------------------------

    # Wir simulieren Ergebnisse nur für den
    # Population-Test.

    for index, individual in enumerate(
        population.individuals
    ):

        # Einige überleben.
        survives = (
            index < 20
        )

        fitness = (
            10.0 - index * 0.1
            if survives
            else -1000.0
        )

        population.set_fitness(
            strategy_id=individual.strategy_id,
            fitness=fitness,
            survives=survives,
            rejection_reason=(
                ""
                if survives
                else "test_rejection"
            ),
        )

    # --------------------------------------------------------
    # SURVIVORS
    # --------------------------------------------------------

    survivors = (
        population.survivors()
    )

    assert len(
        survivors
    ) == 20

    print(
        f"PASS: "
        f"{len(survivors)} survivors"
    )

    # --------------------------------------------------------
    # ELITE
    # --------------------------------------------------------

    elite = population.elite()

    assert len(
        elite
    ) == min(
        config.ELITE_COUNT,
        len(survivors),
    )

    print(
        f"PASS: "
        f"{len(elite)} elite"
    )

    # --------------------------------------------------------
    # NEXT GENERATION
    # --------------------------------------------------------

    next_population = (
        population.create_next_generation()
    )

    assert len(
        next_population.individuals
    ) == config.POPULATION_SIZE

    print(
        "PASS: Next generation created"
    )

    # --------------------------------------------------------
    # RANDOM INJECTION
    # --------------------------------------------------------

    random_count = sum(
        1
        for individual
        in next_population.individuals
        if "-R" in individual.strategy_id
    )

    assert random_count == (
        config.RANDOM_INJECTIONS
    )

    print(
        f"PASS: "
        f"{random_count} random injections"
    )

    # --------------------------------------------------------
    # GENERATION NUMBER
    # --------------------------------------------------------

    assert (
        next_population.generation
        == 2
    )

    print(
        "PASS: Generation increment"
    )

    # --------------------------------------------------------
    # ALL GENOMES VALID
    # --------------------------------------------------------

    for individual in (
        next_population.individuals
    ):

        individual.genome.validate()

    print(
        "PASS: All child genomes valid"
    )

    print()
    print(
        "POPULATION SELF TEST PASSED"
    )

    print("=" * 70)


if __name__ == "__main__":
    self_test()
