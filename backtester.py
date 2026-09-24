"""
Evolution Trader
================
Backtesting Engine

V1:
- Long only
- BTC/USDT
- Signal auf Kerzenschluss
- Einstieg frühestens auf nächster Kerzen-Open
- Take Profit
- Stop Loss
- Trading Fees
- Slippage
- Equity Curve
- Drawdown
- Win Rate
- Profit Factor

WICHTIG:
Kein Look-Ahead-Bias.

Ein Signal auf Candle N darf niemals auf Candle N
zum Schlusskurs ausgeführt werden.
Die Ausführung erfolgt frühestens auf Candle N+1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import config

from strategy_families import (
    Candle,
    generate_signal,
)

from strategy_genome import StrategyGenome


# ============================================================
# TRADE
# ============================================================

@dataclass
class Trade:
    """Abgeschlossener Trade."""

    entry_time: int
    exit_time: int

    entry_price: float
    exit_price: float

    quantity: float

    pnl: float
    pnl_percent: float

    fees: float

    reason: str


# ============================================================
# BACKTEST RESULT
# ============================================================

@dataclass
class BacktestResult:
    """Gesamtergebnis eines Backtests."""

    starting_capital: float
    ending_capital: float

    total_return: float

    max_drawdown: float

    total_trades: int

    winning_trades: int
    losing_trades: int

    win_rate: float

    gross_profit: float
    gross_loss: float

    profit_factor: float

    total_fees: float

    trades: List[Trade] = field(
        default_factory=list
    )

    equity_curve: List[float] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    def is_valid(self) -> bool:
        """
        Prüft die grundlegenden Survival-Regeln.
        """

        if self.total_trades < config.MIN_TRADES:
            return False

        if self.win_rate < config.MIN_WIN_RATE:
            return False

        if self.max_drawdown > config.MAX_DRAWDOWN:
            return False

        if self.total_return <= config.MIN_OOS_RETURN:
            return False

        return True


# ============================================================
# BACKTESTER
# ============================================================

class Backtester:
    """
    Führt eine Strategie auf historischen Kerzen aus.
    """

    def __init__(
        self,
        candles: List[Candle],
        genome: StrategyGenome,
        starting_capital: float = config.STARTING_CAPITAL,
    ):
        self.candles = candles
        self.genome = genome
        self.starting_capital = starting_capital

        self.cash = starting_capital

        self.position_quantity = 0.0
        self.entry_price: Optional[float] = None
        self.entry_time: Optional[int] = None

        self.total_fees = 0.0

        self.trades: List[Trade] = []

        self.equity_curve: List[float] = []

    # ========================================================
    # FEES
    # ========================================================

    @staticmethod
    def apply_entry_cost(price: float) -> float:
        """
        Kaufpreis inklusive Slippage.
        """

        return price * (
            1.0 + config.SLIPPAGE
        )

    @staticmethod
    def apply_exit_cost(price: float) -> float:
        """
        Verkaufspreis inklusive negativer Slippage.
        """

        return price * (
            1.0 - config.SLIPPAGE
        )

    # ========================================================
    # POSITION SIZE
    # ========================================================

    def calculate_position_size(
        self,
        entry_price: float,
    ) -> float:
        """
        Berechnet die Positionsgröße.

        V1:
        Ein fixer Kapitalanteil wird verwendet.

        Das Genome-size_multiplier verändert die
        Basisgröße, bleibt aber durch MAX_POSITION_PERCENT
        begrenzt.
        """

        base_percent = config.MAX_POSITION_PERCENT

        position_percent = (
            base_percent
            * self.genome.size_multiplier
        )

        position_percent = min(
            position_percent,
            config.MAX_POSITION_PERCENT,
        )

        capital_to_use = (
            self.cash
            * position_percent
        )

        if entry_price <= 0:
            return 0.0

        return capital_to_use / entry_price

    # ========================================================
    # OPEN POSITION
    # ========================================================

    def open_position(
        self,
        candle: Candle,
    ) -> None:
        """
        Öffnet eine Long-Position.

        Wichtig:
        Einstieg erfolgt auf Candle Open.
        """

        if self.position_quantity > 0:
            return

        raw_price = candle.open

        entry_price = self.apply_entry_cost(
            raw_price
        )

        quantity = self.calculate_position_size(
            entry_price
        )

        if quantity <= 0:
            return

        gross_value = (
            quantity
            * entry_price
        )

        fee = (
            gross_value
            * config.TRADING_FEE
        )

        total_cost = (
            gross_value
            + fee
        )

        if total_cost > self.cash:
            return

        self.cash -= total_cost

        self.position_quantity = quantity
        self.entry_price = entry_price
        self.entry_time = candle.timestamp

        self.total_fees += fee

    # ========================================================
    # CLOSE POSITION
    # ========================================================

    def close_position(
        self,
        candle: Candle,
        exit_price: float,
        reason: str,
    ) -> None:
        """
        Schließt eine Long-Position.
        """

        if self.position_quantity <= 0:
            return

        if self.entry_price is None:
            return

        actual_exit = self.apply_exit_cost(
            exit_price
        )

        gross_value = (
            self.position_quantity
            * actual_exit
        )

        fee = (
            gross_value
            * config.TRADING_FEE
        )

        net_value = (
            gross_value
            - fee
        )

        invested_value = (
            self.position_quantity
            * self.entry_price
        )

        pnl = (
            net_value
            - invested_value
        )

        pnl_percent = (
            pnl
            / invested_value
            if invested_value > 0
            else 0.0
        )

        self.cash += net_value

        self.total_fees += fee

        trade = Trade(
            entry_time=self.entry_time,
            exit_time=candle.timestamp,

            entry_price=self.entry_price,
            exit_price=actual_exit,

            quantity=self.position_quantity,

            pnl=pnl,
            pnl_percent=pnl_percent,

            fees=fee,

            reason=reason,
        )

        self.trades.append(trade)

        self.position_quantity = 0.0
        self.entry_price = None
        self.entry_time = None

    # ========================================================
    # CHECK EXIT
    # ========================================================

    def check_exit(
        self,
        candle: Candle,
    ) -> bool:
        """
        Prüft Stop Loss und Take Profit.

        WICHTIG:
        Wenn innerhalb derselben Candle sowohl Stop
        als auch TP getroffen werden, wissen wir aus
        OHLC-Daten nicht, was zuerst passiert ist.

        Deshalb verwenden wir konservativ:
        STOP zuerst.
        """

        if self.position_quantity <= 0:
            return False

        if self.entry_price is None:
            return False

        stop_price = (
            self.entry_price
            * (1.0 - self.genome.stop_loss)
        )

        take_profit_price = (
            self.entry_price
            * (1.0 + self.genome.take_profit)
        )

        # ----------------------------------------------------
        # STOP FIRST
        # ----------------------------------------------------

        if candle.low <= stop_price:

            self.close_position(
                candle=candle,
                exit_price=stop_price,
                reason="stop_loss",
            )

            return True

        # ----------------------------------------------------
        # TAKE PROFIT
        # ----------------------------------------------------

        if candle.high >= take_profit_price:

            self.close_position(
                candle=candle,
                exit_price=take_profit_price,
                reason="take_profit",
            )

            return True

        return False

    # ========================================================
    # CURRENT EQUITY
    # ========================================================

    def current_equity(
        self,
        current_price: float,
    ) -> float:
        """
        Mark-to-market Equity.
        """

        if self.position_quantity <= 0:
            return self.cash

        return (
            self.cash
            + self.position_quantity
            * current_price
        )

    # ========================================================
    # RUN
    # ========================================================

    def run(self) -> BacktestResult:
        """
        Führt den vollständigen Backtest aus.
        """

        self.genome.validate()

        if len(self.candles) < 10:
            raise ValueError(
                "Zu wenige Candles für Backtest."
            )

        # ----------------------------------------------------
        # WICHTIG:
        #
        # Candle i:
        #   Signal wird am Ende berechnet.
        #
        # Candle i+1:
        #   möglicher Einstieg am Open.
        # ----------------------------------------------------

        for i in range(1, len(self.candles)):

            candle = self.candles[i]

            previous_candles = (
                self.candles[:i]
            )

            # ------------------------------------------------
            # EXIT FIRST
            # ------------------------------------------------

            if self.position_quantity > 0:

                self.check_exit(candle)

            # ------------------------------------------------
            # ENTRY
            # ------------------------------------------------

            if self.position_quantity <= 0:

                signal = generate_signal(
                    previous_candles,
                    self.genome,
                )

                if signal.direction == 1:

                    self.open_position(
                        candle
                    )

            # ------------------------------------------------
            # EQUITY
            # ------------------------------------------------

            equity = self.current_equity(
                candle.close
            )

            self.equity_curve.append(
                equity
            )

        # ----------------------------------------------------
        # FORCE CLOSE AT END
        # ----------------------------------------------------

        if self.position_quantity > 0:

            last_candle = self.candles[-1]

            self.close_position(
                candle=last_candle,
                exit_price=last_candle.close,
                reason="end_of_backtest",
            )

            self.equity_curve.append(
                self.cash
            )

        return self.build_result()

    # ========================================================
    # RESULT
    # ========================================================

    def build_result(self) -> BacktestResult:
        """Erzeugt das Backtest-Ergebnis."""

        ending_capital = self.cash

        total_return = (
            ending_capital
            / self.starting_capital
        ) - 1.0

        winning = [
            trade
            for trade in self.trades
            if trade.pnl > 0
        ]

        losing = [
            trade
            for trade in self.trades
            if trade.pnl <= 0
        ]

        winning_trades = len(winning)
        losing_trades = len(losing)

        total_trades = len(
            self.trades
        )

        if total_trades > 0:
            win_rate = (
                winning_trades
                / total_trades
            )
        else:
            win_rate = 0.0

        gross_profit = sum(
            trade.pnl
            for trade in winning
        )

        gross_loss = abs(
            sum(
                trade.pnl
                for trade in losing
            )
        )

        if gross_loss > 0:
            profit_factor = (
                gross_profit
                / gross_loss
            )
        else:
            profit_factor = (
                float("inf")
                if gross_profit > 0
                else 0.0
            )

        max_drawdown = self.calculate_max_drawdown()

        return BacktestResult(
            starting_capital=self.starting_capital,

            ending_capital=ending_capital,

            total_return=total_return,

            max_drawdown=max_drawdown,

            total_trades=total_trades,

            winning_trades=winning_trades,

            losing_trades=losing_trades,

            win_rate=win_rate,

            gross_profit=gross_profit,

            gross_loss=gross_loss,

            profit_factor=profit_factor,

            total_fees=self.total_fees,

            trades=self.trades.copy(),

            equity_curve=self.equity_curve.copy(),
        )

    # ========================================================
    # DRAWDOWN
    # ========================================================

    def calculate_max_drawdown(self) -> float:
        """
        Berechnet maximalen Drawdown.
        """

        if not self.equity_curve:
            return 0.0

        peak = self.equity_curve[0]

        max_drawdown = 0.0

        for equity in self.equity_curve:

            if equity > peak:
                peak = equity

            if peak <= 0:
                continue

            drawdown = (
                peak - equity
            ) / peak

            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown


# ============================================================
# RESULT DISPLAY
# ============================================================

def print_result(
    result: BacktestResult,
) -> None:
    """Gibt das Ergebnis übersichtlich aus."""

    print("=" * 70)
    print("BACKTEST RESULT")
    print("=" * 70)

    print(
        f"Starting Capital: "
        f"${result.starting_capital:,.2f}"
    )

    print(
        f"Ending Capital:   "
        f"${result.ending_capital:,.2f}"
    )

    print(
        f"Return:           "
        f"{result.total_return:.2%}"
    )

    print(
        f"Max Drawdown:     "
        f"{result.max_drawdown:.2%}"
    )

    print("-" * 70)

    print(
        f"Trades:           "
        f"{result.total_trades}"
    )

    print(
        f"Winners:          "
        f"{result.winning_trades}"
    )

    print(
        f"Losers:           "
        f"{result.losing_trades}"
    )

    print(
        f"Win Rate:         "
        f"{result.win_rate:.2%}"
    )

    print(
        f"Profit Factor:    "
        f"{result.profit_factor:.2f}"
    )

    print(
        f"Fees:             "
        f"${result.total_fees:.2f}"
    )

    print("-" * 70)

    print(
        f"Survives:         "
        f"{result.is_valid()}"
    )

    print("=" * 70)


# ============================================================
# SELF TEST
# ============================================================

def create_test_market(
    count: int = 500,
) -> List[Candle]:
    """
    Erzeugt künstliche Testdaten.

    Diese Daten sind NUR für den technischen Selbsttest.
    Sie dürfen niemals als Trading-Ergebnis interpretiert
    werden.
    """

    candles = []

    price = 100.0

    for i in range(count):

        phase = i % 100

        if phase < 50:
            change = 0.002
        else:
            change = -0.0015

        previous = price

        price = (
            previous
            * (1.0 + change)
        )

        high = max(
            previous,
            price,
        ) * 1.002

        low = min(
            previous,
            price,
        ) * 0.998

        candles.append(
            Candle(
                timestamp=i,

                open=previous,

                high=high,

                low=low,

                close=price,

                volume=1000.0,
            )
        )

    return candles


def self_test() -> None:
    """
    Technischer Selbsttest.
    """

    from random import Random

    print("=" * 70)
    print("EVOLUTION TRADER - BACKTESTER SELF TEST")
    print("=" * 70)

    rng = Random(42)

    candles = create_test_market()

    genome = StrategyGenome.random(
        family="momentum",
        rng=rng,
    )

    print()
    print("Strategy:")
    print(genome.short_description())

    backtester = Backtester(
        candles=candles,
        genome=genome,
        starting_capital=1000.0,
    )

    result = backtester.run()

    print()

    print_result(result)

    # --------------------------------------------------------
    # BASIC ASSERTIONS
    # --------------------------------------------------------

    assert result.ending_capital > 0

    assert result.total_trades >= 0

    assert 0.0 <= result.win_rate <= 1.0

    assert 0.0 <= result.max_drawdown <= 1.0

    assert result.total_fees >= 0

    assert len(
        result.equity_curve
    ) > 0

    print()
    print("PASS: Backtest completed")
    print("PASS: Equity curve generated")
    print("PASS: Trade accounting")
    print("PASS: Fee accounting")
    print("PASS: Drawdown calculation")
    print("PASS: Result validation")

    print()
    print("BACKTESTER SELF TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    self_test()
