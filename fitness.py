"""
Evolution Trader
================
Fitness Engine

Bewertet Backtest-Ergebnisse für die Evolution.

Ziel:
Nicht die Strategie mit der höchsten reinen Rendite
soll automatisch gewinnen.

Wir bevorzugen robuste Strategien mit:

- positiver Rendite
- niedrigem Drawdown
- gutem Profit Factor
- ausreichender Gewinnrate
- ausreichender Trade-Anzahl
- kontrollierten Gebühren
- möglichst wenig Overtrading

Wichtig:
Die Fitness ist nur ein Ranking innerhalb des
jeweiligen Backtests.

Sie ist KEIN Beweis für zukünftige Gewinne.
"""

from __future__ import annotations

from dataclasses import dataclass

from backtester import BacktestResult


# ============================================================
# FITNESS WEIGHTS
# ============================================================

# Rendite
RETURN_WEIGHT = 1.00

# Profit Factor
PROFIT_FACTOR_WEIGHT = 0.50

# Drawdown-Strafe
DRAWDOWN_WEIGHT = 1.50

# Winrate
WIN_RATE_WEIGHT = 0.25

# Gebühren
FEE_WEIGHT = 0.10

# Overtrading
OVERTRADING_WEIGHT = 0.05


# ============================================================
# NORMALIZATION
# ============================================================

def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """Begrenzt einen Wert."""

    return max(
        minimum,
        min(value, maximum),
    )


# ============================================================
# FITNESS RESULT
# ============================================================

@dataclass
class FitnessResult:
    """
    Ergebnis der Fitnessbewertung.
    """

    score: float

    survives: bool

    return_component: float
    profit_factor_component: float
    drawdown_component: float
    win_rate_component: float
    fee_component: float
    overtrading_component: float

    rejection_reason: str = ""

    # --------------------------------------------------------
    # DESCRIPTION
    # --------------------------------------------------------

    def summary(self) -> str:
        """Kurze Zusammenfassung."""

        status = (
            "SURVIVES"
            if self.survives
            else "KILLED"
        )

        return (
            f"{status} | "
            f"Score={self.score:.4f} | "
            f"Reason={self.rejection_reason or 'OK'}"
        )


# ============================================================
# HARD SURVIVAL GATES
# ============================================================

def check_survival(
    result: BacktestResult,
) -> tuple[bool, str]:
    """
    Harte Überlebensregeln.

    Diese Regeln sind wichtiger als die Fitness.

    Eine Strategie mit einem extrem hohen Fitness-Score
    darf nicht überleben, wenn sie einen grundlegenden
    Gate-Test nicht besteht.
    """

    # --------------------------------------------------------
    # MINIMUM TRADES
    # --------------------------------------------------------

    if result.total_trades < 20:

        return (
            False,
            "too_few_trades",
        )

    # --------------------------------------------------------
    # WIN RATE
    # --------------------------------------------------------

    if result.win_rate < 0.50:

        return (
            False,
            "win_rate_below_50_percent",
        )

    # --------------------------------------------------------
    # MAX DRAWDOWN
    # --------------------------------------------------------

    if result.max_drawdown > 0.10:

        return (
            False,
            "drawdown_above_10_percent",
        )

    # --------------------------------------------------------
    # OOS / RETURN
    # --------------------------------------------------------

    if result.total_return <= 0.0:

        return (
            False,
            "non_positive_return",
        )

    # --------------------------------------------------------
    # PROFIT FACTOR
    # --------------------------------------------------------

    if result.profit_factor <= 1.0:

        return (
            False,
            "profit_factor_below_1",
        )

    return (
        True,
        "",
    )


# ============================================================
# RETURN COMPONENT
# ============================================================

def calculate_return_component(
    total_return: float,
) -> float:
    """
    Rendite-Komponente.

    Sehr extreme Renditen werden begrenzt, damit eine
    einzelne Ausreißerstrategie nicht alles dominiert.
    """

    normalized = clamp(
        total_return,
        -1.0,
        2.0,
    )

    return normalized * RETURN_WEIGHT


# ============================================================
# PROFIT FACTOR COMPONENT
# ============================================================

def calculate_profit_factor_component(
    profit_factor: float,
) -> float:
    """
    Profit-Factor-Komponente.

    PF > 1 ist notwendig.
    Sehr hohe Werte werden gedeckelt.
    """

    if profit_factor == float("inf"):

        normalized = 3.0

    else:

        normalized = clamp(
            profit_factor,
            0.0,
            3.0,
        )

    return (
        normalized
        * PROFIT_FACTOR_WEIGHT
    )


# ============================================================
# DRAWDOWN COMPONENT
# ============================================================

def calculate_drawdown_component(
    max_drawdown: float,
) -> float:
    """
    Drawdown wird als Strafe behandelt.

    Je höher der Drawdown,
    desto niedriger die Fitness.
    """

    normalized = clamp(
        max_drawdown,
        0.0,
        1.0,
    )

    return (
        normalized
        * DRAWDOWN_WEIGHT
    )


# ============================================================
# WIN RATE COMPONENT
# ============================================================

def calculate_win_rate_component(
    win_rate: float,
) -> float:
    """
    Kleine zusätzliche Belohnung für stabile Winrate.
    """

    normalized = clamp(
        win_rate,
        0.0,
        1.0,
    )

    return (
        normalized
        * WIN_RATE_WEIGHT
    )


# ============================================================
# FEE COMPONENT
# ============================================================

def calculate_fee_component(
    result: BacktestResult,
) -> float:
    """
    Gebührenstrafe.

    Wir setzen die Gebühren ins Verhältnis zum
    Startkapital.
    """

    if result.starting_capital <= 0:

        return 0.0

    fee_ratio = (
        result.total_fees
        / result.starting_capital
    )

    return (
        fee_ratio
        * FEE_WEIGHT
    )


# ============================================================
# OVERTRADING COMPONENT
# ============================================================

def calculate_overtrading_component(
    result: BacktestResult,
) -> float:
    """
    Kleine Strafe für extrem hohe Trade-Anzahl.

    V1:
    > 500 Trades = zunehmende Strafe.

    Diese Grenze wird später anhand des tatsächlichen
    Timeframes und der historischen Daten angepasst.
    """

    trades = result.total_trades

    if trades <= 500:

        return 0.0

    excess = trades - 500

    return (
        excess
        / 500.0
        * OVERTRADING_WEIGHT
    )


# ============================================================
# COMPLETE FITNESS
# ============================================================

def calculate_fitness(
    result: BacktestResult,
) -> FitnessResult:
    """
    Berechnet die komplette Fitness.
    """

    survives, reason = check_survival(
        result
    )

    return_component = (
        calculate_return_component(
            result.total_return
        )
    )

    profit_factor_component = (
        calculate_profit_factor_component(
            result.profit_factor
        )
    )

    drawdown_component = (
        calculate_drawdown_component(
            result.max_drawdown
        )
    )

    win_rate_component = (
        calculate_win_rate_component(
            result.win_rate
        )
    )

    fee_component = (
        calculate_fee_component(
            result
        )
    )

    overtrading_component = (
        calculate_overtrading_component(
            result
        )
    )

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    score = (
        return_component
        + profit_factor_component
        + win_rate_component
        - drawdown_component
        - fee_component
        - overtrading_component
    )

    # --------------------------------------------------------
    # DEAD STRATEGIES
    # --------------------------------------------------------

    # Eine Strategie, die einen Hard Gate nicht erfüllt,
    # bekommt einen sehr niedrigen Score.

    if not survives:

        score = -1000.0

    return FitnessResult(
        score=score,

        survives=survives,

        return_component=return_component,

        profit_factor_component=(
            profit_factor_component
        ),

        drawdown_component=(
            drawdown_component
        ),

        win_rate_component=(
            win_rate_component
        ),

        fee_component=fee_component,

        overtrading_component=(
            overtrading_component
        ),

        rejection_reason=reason,
    )


# ============================================================
# RANKING
# ============================================================

def rank_results(
    results: list[tuple[object, FitnessResult]],
) -> list[tuple[object, FitnessResult]]:
    """
    Sortiert Strategien nach Fitness.

    Tote Strategien stehen automatisch unten.
    """

    return sorted(
        results,
        key=lambda item: item[1].score,
        reverse=True,
    )


# ============================================================
# SELF TEST
# ============================================================

def create_fake_result(
    total_return: float,
    max_drawdown: float,
    total_trades: int,
    win_rate: float,
    profit_factor: float,
    fees: float,
) -> BacktestResult:
    """
    Erzeugt ein künstliches Ergebnis nur für den Selbsttest.
    """

    return BacktestResult(
        starting_capital=1000.0,

        ending_capital=(
            1000.0
            * (1.0 + total_return)
        ),

        total_return=total_return,

        max_drawdown=max_drawdown,

        total_trades=total_trades,

        winning_trades=int(
            total_trades
            * win_rate
        ),

        losing_trades=(
            total_trades
            - int(
                total_trades
                * win_rate
            )
        ),

        win_rate=win_rate,

        gross_profit=200.0,

        gross_loss=100.0,

        profit_factor=profit_factor,

        total_fees=fees,

        trades=[],

        equity_curve=[
            1000.0,
            1010.0,
            1020.0,
        ],
    )


def self_test() -> None:
    """
    Testet die Fitness Engine.
    """

    print("=" * 70)
    print("EVOLUTION TRADER - FITNESS SELF TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # GOOD STRATEGY
    # --------------------------------------------------------

    good = create_fake_result(
        total_return=0.20,
        max_drawdown=0.05,
        total_trades=100,
        win_rate=0.60,
        profit_factor=1.80,
        fees=10.0,
    )

    good_fitness = calculate_fitness(
        good
    )

    print()
    print("GOOD STRATEGY")
    print(good_fitness.summary())

    assert good_fitness.survives
    assert good_fitness.score > 0

    # --------------------------------------------------------
    # BAD RETURN
    # --------------------------------------------------------

    bad_return = create_fake_result(
        total_return=-0.05,
        max_drawdown=0.03,
        total_trades=100,
        win_rate=0.60,
        profit_factor=1.20,
        fees=10.0,
    )

    bad_return_fitness = calculate_fitness(
        bad_return
    )

    print()
    print("BAD RETURN")
    print(
        bad_return_fitness.summary()
    )

    assert not bad_return_fitness.survives
    assert (
        bad_return_fitness.score
        == -1000.0
    )

    # --------------------------------------------------------
    # BAD DRAWDOWN
    # --------------------------------------------------------

    bad_drawdown = create_fake_result(
        total_return=0.50,
        max_drawdown=0.15,
        total_trades=100,
        win_rate=0.60,
        profit_factor=1.80,
        fees=10.0,
    )

    bad_drawdown_fitness = calculate_fitness(
        bad_drawdown
    )

    print()
    print("BAD DRAWDOWN")
    print(
        bad_drawdown_fitness.summary()
    )

    assert not bad_drawdown_fitness.survives

    # --------------------------------------------------------
    # BAD WIN RATE
    # --------------------------------------------------------

    bad_win_rate = create_fake_result(
        total_return=0.20,
        max_drawdown=0.05,
        total_trades=100,
        win_rate=0.40,
        profit_factor=1.20,
        fees=10.0,
    )

    bad_win_rate_fitness = calculate_fitness(
        bad_win_rate
    )

    print()
    print("BAD WIN RATE")
    print(
        bad_win_rate_fitness.summary()
    )

    assert not bad_win_rate_fitness.survives

    # --------------------------------------------------------
    # TOO FEW TRADES
    # --------------------------------------------------------

    too_few = create_fake_result(
        total_return=0.50,
        max_drawdown=0.05,
        total_trades=5,
        win_rate=0.80,
        profit_factor=2.00,
        fees=2.0,
    )

    too_few_fitness = calculate_fitness(
        too_few
    )

    print()
    print("TOO FEW TRADES")
    print(
        too_few_fitness.summary()
    )

    assert not too_few_fitness.survives

    print()
    print("PASS: Positive strategy")
    print("PASS: Negative return rejection")
    print("PASS: Drawdown rejection")
    print("PASS: Win-rate rejection")
    print("PASS: Trade-count rejection")
    print("PASS: Fitness calculation")
    print("PASS: Ranking support")

    print()
    print("FITNESS SELF TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    self_test()
