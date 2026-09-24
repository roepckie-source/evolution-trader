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
- Fallback-Selektion, falls eine Generation
  keine echten Survivors besitzt

Wichtig:

Dieses Modul entscheidet noch NICHT, welche Strategie
wirtschaftlich gut ist.

Das macht später die Kombination aus:

Backtester
    ↓
Fitness
    ↓
Population Evolution

Wichtig zur Fallback-Selektion:

Ein Fallback-Kandidat ist KEIN echter Survivor.

Er darf nur als evolutionärer Elternteil verwendet werden,
wenn eine Generation überhaupt keinen echten Survivor besitzt.

Das verhindert, dass die Evolution komplett ausstirbt,
ohne die eigentlichen Fitness-Gates aufzuweichen.
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

    survives:
        True = echter Fitness-Survivor.

        False = hat die normalen Hard-Gates nicht erfüllt.

        Wichtig:
        Fallback-Kandidaten werden NICHT auf True gesetzt.
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
        Gibt nur echte Fitness-Überlebende zurück.

        Wichtig:

        Fallback-Kandidaten werden hier NICHT zurückgegeben.
        """

        return [
            individual
            for individual in self.individuals
            if individual.survives is True
        ]

    # ========================================================
    # EVOLUTION PARENTS
    # ========================================================

    def evolution_parents(
        self,
    ) -> List[Individual]:
        """
        Gibt die Kandidaten zurück, die für die Erzeugung
        der nächsten Generation verwendet werden dürfen.

        Normalfall:
            echte Survivors

        Fallback:
            Wenn keine echten Survivors existieren,
            werden die bestbewerteten Kandidaten verwendet.

        Wichtig:

        Die Fallback-Kandidaten werden NICHT zu echten
        Survivors gemacht.

        Damit bleiben die Fitness-Gates vollständig erhalten.
        """

        survivors = self.survivors()

        if survivors:
            return survivors

        ranked = self.ranked()

        evaluated = [
            individual
            for individual in ranked
            if individual.fitness is not None
        ]

        if not evaluated:
            raise RuntimeError(
                "Keine bewerteten Strategien vorhanden. "
                "Evolution kann nicht fortgesetzt werden."
            )

        # ----------------------------------------------------
        # FALLBACK SIZE
        # ----------------------------------------------------
        #
        # Wir verwenden maximal 20 % der Population,
        # mindestens aber 4 Kandidaten.
        #
        # Bei 96 Strategien:
        #
        # 96 * 0.20 = 19.2
        #
        # => 19 Fallback-Eltern.
        # ----------------------------------------------------

        fallback_count = max(
            4,
            int(
                len(self.individuals)
                * 0.20
            ),
        )

        fallback_count = min(
            fallback_count,
            len(evaluated),
        )

        fallback = evaluated[
            :fallback_count
        ]

        return fallback

    # ========================================================
    # ELITE
    # ========================================================

    def elite(
        self,
        count: Optional[int] = None,
    ) -> List[Individual]:
        """
        Gibt die besten echten Überlebenden zurück.

        Fallback-Kandidaten werden NICHT als Elite
        betrachtet.
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
    # FALLBACK ELITE
    # ========================================================

    def fallback_elite(
        self,
        count: Optional[int] = None,
    ) -> List[Individual]:
        """
        Gibt die besten Kandidaten für eine Fallback-
        Evolution zurück.

        Diese Strategien sind NICHT echte Survivors.
        """

        if count is None:
            count = min(
                config.ELITE_COUNT,
                4,
            )

        parents = self.evolution_parents()

        parents = sorted(
            parents,
            key=lambda individual: (
                individual.fitness
                if individual.fitness is not None
                else -float("inf")
            ),
            reverse=True,
        )

        return parents[:count]

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

        Gleichzeitig können schwächere Kandidaten noch
        ausgewählt werden.
        """

        if not candidates:
            raise ValueError(
                "Keine Parent-Kandidaten."
            )

        # ----------------------------------------------------
        # FITNESS-GEWICHTE
        # ----------------------------------------------------

        weights = []

        minimum_weight = 0.01

        for candidate in candidates:

            fitness = (
                candidate.fitness
                if candidate.fitness is not None
                else 0.0
            )

            # ------------------------------------------------
            # Negative Fitnesswerte können nicht sinnvoll
            # als Gewicht verwendet werden.
            #
            # Alle Fallback-Kandidaten können z.B. -1000
            # besitzen.
            #
            # Deshalb verwenden wir einen positiven Rang-
            # basierten Anteil.
            # ------------------------------------------------

            if fitness <= 0:
                weights.append(
                    minimum_weight
                )
            else:
                weights.append(
                    max(
                        fitness,
                        minimum_weight,
                    )
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

            # ------------------------------------------------
            # Kein Crossover:
            # zufällig einen Elternteil klonen.
            # ------------------------------------------------

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

        Normalfall:

        1. echte Elite übernehmen
        2. Random Injection
        3. Crossover + Mutation

        Fallback:

        Wenn keine echten Survivors existieren:

        1. keine echte Elite
        2. beste Fallback-Kandidaten klonen
        3. Random Injection
        4. Crossover + Mutation

        Wichtig:

        Die Fallback-Kandidaten bleiben als Objekte der
        alten Generation gekennzeichnet.

        Sie werden dadurch nicht automatisch zu Survivors.
        """

        survivors = self.survivors()

        using_fallback = not bool(
            survivors
        )

        # ----------------------------------------------------
        # ELTERN BESTIMMEN
        # ----------------------------------------------------

        if using_fallback:

            parents = self.evolution_parents()

            elite = self.fallback_elite(
                min(
                    config.ELITE_COUNT,
                    4,
                )
            )

        else:

            parents = survivors

            elite = self.elite(
                config.ELITE_COUNT
            )

        # ----------------------------------------------------
        # NÄCHSTE POPULATION
        # ----------------------------------------------------

        next_population = Population(
            generation=self.generation + 1,
            rng=self.rng,
        )

        new_individuals: List[
            Individual
        ] = []

        # ----------------------------------------------------
        # 1. ELITE / FALLBACK ELITE
        # ----------------------------------------------------

        for index, individual in enumerate(
            elite,
            start=1,
        ):

            copied_genome = (
                individual.genome.copy()
            )

            if using_fallback:

                elite_id = (
                    f"G{self.generation + 1:03d}"
                    f"-F{index:03d}"
                )

            else:

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
                parents
            )

            parent_b = self.select_parent(
                parents
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

        # ----------------------------------------------------
        # FALLBACK STATUS
        # ----------------------------------------------------

        if not self.survivors():

            try:

                fallback_count = len(
                    self.evolution_parents()
                )

            except RuntimeError:

                fallback_count = 0

            print(
                f"Fallback parents: "
                f"{fallback_count}"
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
    print(
        "EVOLUTION TRADER - POPULATION SELF TEST"
    )
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

    # ========================================================
    # FALLBACK TEST
    # ========================================================

    print()
    print(
        "TEST: Zero-survivor fallback"
    )

    fallback_population = Population(
        generation=1,
        rng=Random(
            config.RANDOM_SEED
        ),
    )

    fallback_population.create_initial_population()

    # --------------------------------------------------------
    # Alle Strategien werden als NICHT überlebend markiert.
    # --------------------------------------------------------

    for index, individual in enumerate(
        fallback_population.individuals
    ):

        # Unterschiedliche negative Fitnesswerte,
        # damit das Ranking geprüft werden kann.
        fitness = (
            -1.0
            - (index * 0.01)
        )

        fallback_population.set_fitness(
            strategy_id=individual.strategy_id,
            fitness=fitness,
            survives=False,
            rejection_reason="test_rejection",
        )

    # --------------------------------------------------------
    # Es darf KEINE echten Survivors geben.
    # --------------------------------------------------------

    assert (
        len(
            fallback_population.survivors()
        )
        == 0
    )

    print(
        "PASS: Zero real survivors"
    )

    # --------------------------------------------------------
    # Fallback-Eltern müssen verfügbar sein.
    # --------------------------------------------------------

    fallback_parents = (
        fallback_population.evolution_parents()
    )

    assert len(
        fallback_parents
    ) > 0

    print(
        f"PASS: "
        f"{len(fallback_parents)} fallback parents"
    )

    # --------------------------------------------------------
    # Nächste Generation muss trotzdem entstehen.
    # --------------------------------------------------------

    fallback_next = (
        fallback_population.create_next_generation()
    )

    assert len(
        fallback_next.individuals
    ) == config.POPULATION_SIZE

    print(
        "PASS: Next generation created "
        "with fallback"
    )

    # --------------------------------------------------------
    # Generation muss 2 sein.
    # --------------------------------------------------------

    assert (
        fallback_next.generation
        == 2
    )

    print(
        "PASS: Fallback generation increment"
    )

    # --------------------------------------------------------
    # Alle neuen Genome müssen valide sein.
    # --------------------------------------------------------

    for individual in (
        fallback_next.individuals
    ):

        individual.genome.validate()

    print(
        "PASS: Fallback genomes valid"
    )

    print()
    print(
        "POPULATION SELF TEST PASSED"
    )
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    self_test()
