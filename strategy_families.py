"""
Evolution Trader
================
Strategy Families

Implementiert die vier Strategie-Familien:

1. Momentum
2. Mean Reversion
3. Volatility Breakout
4. Range Grid

Dieses Modul erzeugt Handelssignale aus OHLCV-Kerzen.

Noch KEIN echtes Trading.
Noch KEINE Orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from strategy_genome import StrategyGenome


# ============================================================
# CANDLE
# ============================================================

@dataclass
class Candle:
    """
    Eine einzelne OHLCV-Kerze.
    """

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


# ============================================================
# SIGNAL
# ============================================================

@dataclass
class Signal:
    """
    Ergebnis einer Strategieentscheidung.

    direction:
        1  = Long
        0  = kein Trade
       -1  = Short

    Für V1 verwenden wir nur Long.
    """

    direction: int
    strength: float
    reason: str


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def closes(candles: List[Candle]) -> List[float]:
    """Gibt alle Schlusskurse zurück."""

    return [c.close for c in candles]


def mean(values: List[float]) -> float:
    """Arithmetisches Mittel."""

    if not values:
        return 0.0

    return sum(values) / len(values)


def stddev(values: List[float]) -> float:
    """Standardabweichung."""

    if len(values) < 2:
        return 0.0

    avg = mean(values)

    variance = sum(
        (value - avg) ** 2
        for value in values
    ) / len(values)

    return variance ** 0.5


def highest(values: List[float]) -> float:
    """Höchster Wert."""

    return max(values) if values else 0.0


def lowest(values: List[float]) -> float:
    """Niedrigster Wert."""

    return min(values) if values else 0.0


# ============================================================
# MOMENTUM
# ============================================================

def momentum_signal(
    candles: List[Candle],
    genome: StrategyGenome,
) -> Signal:
    """
    Momentum-Strategie.

    Idee:

    Wenn der aktuelle Kurs deutlich über dem Kurs
    des Lookback-Zeitraums liegt, entsteht ein Long-Signal.

    entry_threshold wird als prozentuale Mindestbewegung
    interpretiert.

    Beispiel:

        Lookback = 20
        Threshold = 0.5 %

    Kurs muss mindestens 0.5 % über dem Lookback-Kurs liegen.
    """

    lookback = genome.lookback

    if len(candles) <= lookback:
        return Signal(
            direction=0,
            strength=0.0,
            reason="not_enough_data",
        )

    current_price = candles[-1].close
    previous_price = candles[-lookback - 1].close

    if previous_price <= 0:
        return Signal(
            direction=0,
            strength=0.0,
            reason="invalid_price",
        )

    momentum = (
        current_price / previous_price
    ) - 1.0

    threshold = genome.entry_threshold / 100.0

    if momentum >= threshold:

        strength = min(
            abs(momentum) / max(threshold, 1e-9),
            3.0,
        )

        return Signal(
            direction=1,
            strength=strength,
            reason=f"momentum_up_{momentum:.4%}",
        )

    return Signal(
        direction=0,
        strength=0.0,
        reason=f"momentum_below_threshold_{momentum:.4%}",
    )


# ============================================================
# MEAN REVERSION
# ============================================================

def mean_reversion_signal(
    candles: List[Candle],
    genome: StrategyGenome,
) -> Signal:
    """
    Mean-Reversion-Strategie.

    Idee:

    Wenn der Kurs deutlich unter seinem historischen Mittelwert
    liegt, erwarten wir eine Rückkehr Richtung Mittelwert.

    Für V1 wird nur Long gehandelt.
    """

    lookback = genome.lookback

    if len(candles) < lookback:
        return Signal(
            direction=0,
            strength=0.0,
            reason="not_enough_data",
        )

    prices = closes(
        candles[-lookback:]
    )

    current_price = prices[-1]

    average = mean(prices)
    deviation = stddev(prices)

    if deviation <= 0:
        return Signal(
            direction=0,
            strength=0.0,
            reason="no_volatility",
        )

    z_score = (
        current_price - average
    ) / deviation

    threshold = genome.entry_threshold

    # Kurs liegt deutlich unter dem Mittelwert
    if z_score <= -threshold:

        strength = min(
            abs(z_score) / max(threshold, 1e-9),
            3.0,
        )

        return Signal(
            direction=1,
            strength=strength,
            reason=f"mean_reversion_z_{z_score:.2f}",
        )

    return Signal(
        direction=0,
        strength=0.0,
        reason=f"z_score_{z_score:.2f}",
    )


# ============================================================
# VOLATILITY BREAKOUT
# ============================================================

def volatility_breakout_signal(
    candles: List[Candle],
    genome: StrategyGenome,
) -> Signal:
    """
    Volatility-Breakout-Strategie.

    Idee:

    Ein Ausbruch über das bisherige Hoch kombiniert
    mit ausreichender Volatilität erzeugt ein Long-Signal.
    """

    lookback = genome.lookback

    if len(candles) <= lookback:
        return Signal(
            direction=0,
            strength=0.0,
            reason="not_enough_data",
        )

    historical = candles[-lookback - 1:-1]

    historical_high = max(
        candle.high
        for candle in historical
    )

    current = candles[-1]

    prices = [
        candle.close
        for candle in historical
    ]

    volatility = stddev(prices)

    if volatility <= 0:
        return Signal(
            direction=0,
            strength=0.0,
            reason="no_volatility",
        )

    breakout_distance = (
        current.close / historical_high
    ) - 1.0

    threshold = genome.entry_threshold / 100.0

    # Breakout muss positiv und ausreichend groß sein
    if (
        current.close > historical_high
        and breakout_distance >= threshold
    ):

        strength = min(
            breakout_distance / max(threshold, 1e-9),
            3.0,
        )

        return Signal(
            direction=1,
            strength=strength,
            reason=(
                f"volatility_breakout_"
                f"{breakout_distance:.4%}"
            ),
        )

    return Signal(
        direction=0,
        strength=0.0,
        reason="no_breakout",
    )


# ============================================================
# RANGE GRID
# ============================================================

def range_grid_signal(
    candles: List[Candle],
    genome: StrategyGenome,
) -> Signal:
    """
    Range/Grid-Strategie.

    Idee:

    Der aktuelle Preis wird innerhalb einer historischen
    Handelsspanne betrachtet.

    Je näher der Preis am unteren Ende der Range liegt,
    desto eher entsteht ein Long-Signal.

    Die tatsächlichen Grid-Orders werden später im
    Backtester umgesetzt.
    """

    lookback = genome.lookback

    if len(candles) < lookback:
        return Signal(
            direction=0,
            strength=0.0,
            reason="not_enough_data",
        )

    recent = candles[-lookback:]

    high = max(
        candle.high
        for candle in recent
    )

    low = min(
        candle.low
        for candle in recent
    )

    current_price = candles[-1].close

    range_size = high - low

    if range_size <= 0:
        return Signal(
            direction=0,
            strength=0.0,
            reason="no_range",
        )

    position_in_range = (
        current_price - low
    ) / range_size

    spacing = genome.spacing

    # Wir interessieren uns für den unteren Bereich
    # der Range.

    lower_zone = min(
        spacing * genome.grid_levels,
        0.45,
    )

    if position_in_range <= lower_zone:

        strength = (
            1.0
            - position_in_range
        )

        return Signal(
            direction=1,
            strength=max(strength, 0.1),
            reason=(
                f"range_lower_zone_"
                f"{position_in_range:.2%}"
            ),
        )

    return Signal(
        direction=0,
        strength=0.0,
        reason=(
            f"range_position_"
            f"{position_in_range:.2%}"
        ),
    )


# ============================================================
# UNIVERSAL DISPATCHER
# ============================================================

def generate_signal(
    candles: List[Candle],
    genome: StrategyGenome,
) -> Signal:
    """
    Ruft automatisch die richtige Strategie-Familie auf.
    """

    family = genome.family

    if family == "momentum":
        return momentum_signal(
            candles,
            genome,
        )

    if family == "mean_reversion":
        return mean_reversion_signal(
            candles,
            genome,
        )

    if family == "volatility_breakout":
        return volatility_breakout_signal(
            candles,
            genome,
        )

    if family == "range_grid":
        return range_grid_signal(
            candles,
            genome,
        )

    raise ValueError(
        f"Unbekannte Strategie-Familie: {family}"
    )


# ============================================================
# SELF TEST
# ============================================================

def create_test_candles(
    count: int = 300,
) -> List[Candle]:
    """
    Erzeugt deterministische Testdaten.

    Keine echten Marktdaten.
    Nur für Unit-Tests.
    """

    candles = []

    price = 100.0

    for i in range(count):

        # leichte künstliche Bewegung
        cycle = (i % 40) / 40.0

        if cycle < 0.5:
            price *= 1.001
        else:
            price *= 0.999

        candles.append(
            Candle(
                timestamp=i,
                open=price * 0.999,
                high=price * 1.002,
                low=price * 0.998,
                close=price,
                volume=1000.0,
            )
        )

    return candles


def self_test() -> None:
    """
    Selbsttest aller vier Strategie-Familien.
    """

    from random import Random

    print("=" * 70)
    print("EVOLUTION TRADER - STRATEGY FAMILY TEST")
    print("=" * 70)

    rng = Random(42)

    candles = create_test_candles()

    for family in [
        "momentum",
        "mean_reversion",
        "volatility_breakout",
        "range_grid",
    ]:

        genome = StrategyGenome.random(
            family=family,
            rng=rng,
        )

        signal = generate_signal(
            candles,
            genome,
        )

        print()
        print(f"FAMILY: {family}")
        print(f"GENOME: {genome.short_description()}")
        print(f"SIGNAL: {signal}")

        assert signal.direction in (-1, 0, 1)
        assert signal.strength >= 0
        assert isinstance(signal.reason, str)

    print()
    print("PASS: Momentum")
    print("PASS: Mean Reversion")
    print("PASS: Volatility Breakout")
    print("PASS: Range Grid")
    print("PASS: Signal Dispatcher")

    print()
    print("STRATEGY FAMILY TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    self_test()
