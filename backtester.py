"""
Evolution Trader
Backtester

V1:
- Long only
- No look-ahead bias
- Entry at next candle open
- Stop-loss checked before take-profit
- Fees and slippage included
- Position sizing based on strategy genome
- Paper/backtest only
"""

from dataclasses import dataclass
from typing import List, Optional

import config
from strategy_genome import StrategyGenome
from strategy_families import Candle, Signal, generate_signal


# ============================================================
# TRADE
# ============================================================

@dataclass
class Trade:
    entry_time: object
    exit_time: object

    entry_price: float
    exit_price: float

    position_size: float

    pnl: float
    pnl_percent: float

    fee: float

    exit_reason: str


# ============================================================
# BACKTEST RESULT
# ============================================================

@dataclass
class BacktestResult:
    starting_capital: float
    ending_balance: float

    total_return: float
    max_drawdown: float

    total_trades: int
    winning_trades: int
    losing_trades: int

    win_rate: float
    profit_factor: float

    total_fees: float

    equity_curve: List[float]
    trades: List[Trade]

    # --------------------------------------------------------
    # Survival gate
    # --------------------------------------------------------

    def is_valid(self) -> bool:
        if self.total_trades < config.MIN_TRADES:
            return False

        if self.win_rate < config.MIN_WIN_RATE:
            return False

        if self.max_drawdown > config.MAX_DRAWDOWN:
            return False

        if self.total_return <= 0:
            return False

        return True


# ============================================================
# BACKTESTER
# ============================================================

class Backtester:

    def __init__(
        self,
        candles: List[Candle],
        genome: StrategyGenome,
        starting_capital: Optional[float] = None,
    ):
        self.candles = candles
        self.genome = genome

        self.starting_capital = (
            starting_capital
            if starting_capital is not None
            else config.STARTING_CAPITAL
        )

        self.balance = self.starting_capital

        self.position = None

        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []

        self.total_fees = 0.0

    # ========================================================
    # FEES
    # ========================================================

    def calculate_fee(self, value: float) -> float:
        return value * config.TRADING_FEE

    # ========================================================
    # SLIPPAGE
    # ========================================================

    def apply_entry_slippage(self, price: float) -> float:
        return price * (1.0 + config.SLIPPAGE)

    def apply_exit_slippage(self, price: float) -> float:
        return price * (1.0 - config.SLIPPAGE)

    # ========================================================
    # POSITION SIZE
    # ========================================================

    def calculate_position_size(self) -> float:
        """
        Position size as percentage of current balance.

        V1 keeps the maximum exposure at MAX_POSITION_PERCENT.
        """

        base_percent = config.MAX_POSITION_PERCENT

        position_percent = base_percent * self.genome.size_multiplier

        position_percent = min(
            position_percent,
            config.MAX_POSITION_PERCENT,
        )

        position_percent = max(
            position_percent,
            0.0,
        )

        return self.balance * position_percent

    # ========================================================
    # OPEN POSITION
    # ========================================================

    def open_position(
        self,
        candle: Candle,
        signal: Signal,
    ) -> None:

        if self.position is not None:
            return

        if signal.direction != 1:
            return

        raw_entry_price = candle.open

        entry_price = self.apply_entry_slippage(
            raw_entry_price
        )

        position_size = self.calculate_position_size()

        if position_size <= 0:
            return

        entry_fee = self.calculate_fee(position_size)

        self.balance -= entry_fee
        self.total_fees += entry_fee

        self.position = {
            "entry_time": candle.timestamp,
            "entry_price": entry_price,
            "position_size": position_size,
            "entry_fee": entry_fee,
            "stop_loss": entry_price * (
                1.0 - self.genome.stop_loss
            ),
            "take_profit": entry_price * (
                1.0 + self.genome.take_profit
            ),
        }

    # ========================================================
    # CLOSE POSITION
    # ========================================================

    def close_position(
        self,
        candle: Candle,
        exit_price: float,
        reason: str,
    ) -> None:

        if self.position is None:
            return

        position = self.position

        exit_price = self.apply_exit_slippage(
            exit_price
        )

        entry_price = position["entry_price"]
        position_size = position["position_size"]

        price_change = (
            exit_price - entry_price
        ) / entry_price

        pnl = position_size * price_change

        exit_value = position_size + pnl

        exit_fee = self.calculate_fee(
            exit_value
        )

        net_pnl = pnl - exit_fee

        self.balance += net_pnl

        self.total_fees += exit_fee

        pnl_percent = (
            net_pnl / position_size
            if position_size > 0
            else 0.0
        )

        trade = Trade(
            entry_time=position["entry_time"],
            exit_time=candle.timestamp,

            entry_price=entry_price,
            exit_price=exit_price,

            position_size=position_size,

            pnl=net_pnl,
            pnl_percent=pnl_percent,

            fee=position["entry_fee"] + exit_fee,

            exit_reason=reason,
        )

        self.trades.append(trade)

        self.position = None

    # ========================================================
    # CHECK EXIT
    # ========================================================

    def check_exit(
        self,
        candle: Candle,
    ) -> None:

        if self.position is None:
            return

        stop_loss = self.position["stop_loss"]
        take_profit = self.position["take_profit"]

        # ----------------------------------------------------
        # IMPORTANT:
        # If both SL and TP are touched during the same candle,
        # assume STOP LOSS happened first.
        # This is conservative and avoids optimistic results.
        # ----------------------------------------------------

        if candle.low <= stop_loss:

            self.close_position(
                candle=candle,
                exit_price=stop_loss,
                reason="stop_loss",
            )

            return

        if candle.high >= take_profit:

            self.close_position(
                candle=candle,
                exit_price=take_profit,
                reason="take_profit",
            )

            return

    # ========================================================
    # CURRENT EQUITY
    # ========================================================

    def current_equity(
        self,
        candle: Candle,
    ) -> float:

        equity = self.balance

        if self.position is not None:

            position = self.position

            current_price = candle.close

            entry_price = position["entry_price"]
            position_size = position["position_size"]

            price_change = (
                current_price - entry_price
            ) / entry_price

            unrealized_pnl = (
                position_size * price_change
            )

            equity += unrealized_pnl

        return equity

    # ========================================================
    # RUN BACKTEST
    # ========================================================

    def run(self) -> BacktestResult:

        candles = self.candles

        if len(candles) < 10:
            raise ValueError(
                "Not enough candles for backtest."
            )

        # ----------------------------------------------------
        # Main candle loop
        # ----------------------------------------------------

        for i in range(1, len(candles)):

            candle = candles[i]

            # ------------------------------------------------
            # First check an existing position.
            # ------------------------------------------------

            self.check_exit(candle)

            # ------------------------------------------------
            # Only generate a new signal if we are flat.
            # ------------------------------------------------

            if self.position is None:

                # ====================================================
                # PERFORMANCE OPTIMIZATION
                #
                # Previously:
                #
                # previous_candles = candles[:i]
                #
                # This copied the entire history on every iteration.
                #
                # We only need enough candles for the genome's
                # lookback period.
                # ====================================================

                window_size = max(
                    self.genome.lookback + 2,
                    10,
                )

                start_index = max(
                    0,
                    i - window_size,
                )

                previous_candles = candles[
                    start_index:i
                ]

                if len(previous_candles) > 0:

                    signal = generate_signal(
                        previous_candles,
                        self.genome,
                    )

                    # ------------------------------------------------
                    # IMPORTANT:
                    #
                    # Signal is generated from candles BEFORE the
                    # current candle.
                    #
                    # Entry therefore happens at current candle OPEN.
                    #
                    # This prevents look-ahead bias.
                    # ------------------------------------------------

                    if signal.direction == 1:

                        self.open_position(
                            candle,
                            signal,
                        )

            # ------------------------------------------------
            # Track equity after processing this candle.
            # ------------------------------------------------

            equity = self.current_equity(
                candle
            )

            self.equity_curve.append(
                equity
            )

        # ====================================================
        # FORCE CLOSE AT END
        # ====================================================

        if self.position is not None:

            final_candle = candles[-1]

            self.close_position(
                candle=final_candle,
                exit_price=final_candle.close,
                reason="end_of_backtest",
            )

            self.equity_curve.append(
                self.balance
            )

        # ====================================================
        # BUILD RESULT
        # ====================================================

        return self.build_result()

    # ========================================================
    # BUILD RESULT
    # ========================================================

    def build_result(self) -> BacktestResult:

        ending_balance = self.balance

        total_return = (
            ending_balance
            / self.starting_capital
        ) - 1.0

        total_trades = len(self.trades)

        winning_trades = sum(
            1
            for trade in self.trades
            if trade.pnl > 0
        )

        losing_trades = sum(
            1
            for trade in self.trades
            if trade.pnl < 0
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
            for trade in self.trades
            if trade.pnl > 0
        )

        gross_loss = sum(
            trade.pnl
            for trade in self.trades
            if trade.pnl < 0
        )

        if gross_loss < 0:

            profit_factor = (
                gross_profit
                / abs(gross_loss)
            )

        elif gross_profit > 0:

            profit_factor = float("inf")

        else:

            profit_factor = 0.0

        max_drawdown = (
            self.calculate_max_drawdown()
        )

        return BacktestResult(

            starting_capital=self.starting_capital,

            ending_balance=ending_balance,

            total_return=total_return,

            max_drawdown=max_drawdown,

            total_trades=total_trades,

            winning_trades=winning_trades,

            losing_trades=losing_trades,

            win_rate=win_rate,

            profit_factor=profit_factor,

            total_fees=self.total_fees,

            equity_curve=self.equity_curve,

            trades=self.trades,
        )

    # ========================================================
    # MAX DRAWDOWN
    # ========================================================

    def calculate_max_drawdown(self) -> float:

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

    # ========================================================
    # PRINT RESULT
    # ========================================================

    def print_result(
        self,
        result: BacktestResult,
    ) -> None:

        print()
        print("=" * 60)
        print("BACKTEST RESULT")
        print("=" * 60)

        print(
            f"Starting capital: "
            f"${result.starting_capital:,.2f}"
        )

        print(
            f"Ending balance:   "
            f"${result.ending_balance:,.2f}"
        )

        print(
            f"Return:           "
            f"{result.total_return * 100:.2f}%"
        )

        print(
            f"Max drawdown:     "
            f"{result.max_drawdown * 100:.2f}%"
        )

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
            f"Win rate:         "
            f"{result.win_rate * 100:.2f}%"
        )

        if result.profit_factor == float("inf"):

            pf_text = "INF"

        else:

            pf_text = (
                f"{result.profit_factor:.2f}"
            )

        print(
            f"Profit factor:    "
            f"{pf_text}"
        )

        print(
            f"Total fees:       "
            f"${result.total_fees:,.2f}"
        )

        print(
            f"Survived:         "
            f"{result.is_valid()}"
        )

        print("=" * 60)


# ============================================================
# SYNTHETIC TEST DATA
# ============================================================

def create_test_candles(
    count: int = 200,
) -> List[Candle]:

    candles = []

    price = 100.0

    for i in range(count):

        # Create deterministic price movement.
        if i % 20 < 10:
            price *= 1.001
        else:
            price *= 0.999

        open_price = price

        high_price = (
            open_price * 1.005
        )

        low_price = (
            open_price * 0.995
        )

        close_price = price

        volume = 1000.0

        candles.append(
            Candle(
                timestamp=i,

                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,

                volume=volume,
            )
        )

    return candles


# ============================================================
# SELF TEST
# ============================================================

def self_test() -> None:

    print()
    print("=" * 60)
    print("BACKTESTER SELF TEST")
    print("=" * 60)

    candles = create_test_candles(
        250
    )

    genome = StrategyGenome.random(
        family="momentum"
    )

    print()
    print("Test genome:")
    print(
        genome.short_description()
    )

    backtester = Backtester(
        candles=candles,
        genome=genome,
        starting_capital=1000.0,
    )

    result = backtester.run()

    backtester.print_result(
        result
    )

    # --------------------------------------------------------
    # Basic assertions
    # --------------------------------------------------------

    assert result.starting_capital == 1000.0

    assert result.ending_balance >= 0

    assert result.total_trades >= 0

    assert 0.0 <= result.win_rate <= 1.0

    assert result.max_drawdown >= 0.0

    assert result.max_drawdown <= 1.0

    assert result.total_fees >= 0.0

    assert len(
        result.equity_curve
    ) > 0

    print()
    print("ALL BACKTESTER TESTS PASSED")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    self_test()
