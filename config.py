"""
Evolution Trader
================
Zentrale Konfiguration für die Evolution Engine.

V1:
- BTC/USDT
- 5-Minuten-Daten
- 96 Strategien pro Generation
- 4 Strategie-Familien
- 70 % Training / 30 % Out-of-Sample
- Paper Trading only
- KEINE Live-Orders
"""

# ============================================================
# PROJECT
# ============================================================

PROJECT_NAME = "Evolution Trader"
VERSION = "0.1.0"

# ============================================================
# MARKET
# ============================================================

SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"

# ============================================================
# EVOLUTION
# ============================================================

POPULATION_SIZE = 96

STRATEGY_FAMILIES = [
    "momentum",
    "mean_reversion",
    "volatility_breakout",
    "range_grid",
]

FAMILIES_COUNT = len(STRATEGY_FAMILIES)

# 96 / 4 = 24 Strategien pro Familie
STRATEGIES_PER_FAMILY = POPULATION_SIZE // FAMILIES_COUNT

# Neue zufällige Strategien pro Generation
RANDOM_INJECTIONS = 8

# Anteil der besten Strategien, die als Eltern verwendet werden
ELITE_PERCENT = 0.20

# Mutationswahrscheinlichkeit
MUTATION_RATE = 0.15

# Crossover-Wahrscheinlichkeit
CROSSOVER_RATE = 0.80

# ============================================================
# DATA SPLIT
# ============================================================

TRAIN_PERCENT = 0.70
OOS_PERCENT = 0.30

assert abs(TRAIN_PERCENT + OOS_PERCENT - 1.0) < 1e-9

# ============================================================
# SURVIVAL RULES
# ============================================================

# Strategie darf im OOS-Test maximal diesen Drawdown haben
MAX_DRAWDOWN = 0.10

# Mindest-Winrate
MIN_WIN_RATE = 0.50

# Mindestanzahl Trades
MIN_TRADES = 20

# OOS darf nicht negativ sein
MIN_OOS_RETURN = 0.0

# ============================================================
# CAPITAL
# ============================================================

STARTING_CAPITAL = 1000.00

# Noch KEIN echtes Geld
PAPER_TRADING = True
LIVE_TRADING = False

# ============================================================
# FEES / SLIPPAGE
# ============================================================

# Konservative Platzhalter für V1.
# Werden später anhand der tatsächlichen Börse angepasst.

TRADING_FEE = 0.001       # 0.10 %
SLIPPAGE = 0.0005         # 0.05 %

# ============================================================
# RISK
# ============================================================

# Maximales Risiko pro Trade
MAX_RISK_PER_TRADE = 0.01

# Maximale Position relativ zum verfügbaren Kapital
MAX_POSITION_PERCENT = 0.10

# Maximaler Tagesverlust
MAX_DAILY_LOSS = 0.05

# ============================================================
# BACKTEST
# ============================================================

ALLOW_SHORT = False

INITIAL_EQUITY = STARTING_CAPITAL

# ============================================================
# EVOLUTION CONTROL
# ============================================================

# Maximale Anzahl Generationen für einen Lauf
MAX_GENERATIONS = 100

# Anzahl der besten Strategien, die direkt übernommen werden
ELITE_COUNT = 10

# ============================================================
# RANDOMNESS
# ============================================================

# Reproduzierbare Tests
RANDOM_SEED = 42

# ============================================================
# SAFETY
# ============================================================

# Diese beiden Flags müssen für V1 so bleiben.
# Kein Code darf echte Orders senden.

assert PAPER_TRADING is True
assert LIVE_TRADING is False

# ============================================================
# INFORMATION
# ============================================================

def print_config():
    """Gibt die aktuelle Konfiguration aus."""

    print("=" * 60)
    print("EVOLUTION TRADER")
    print("=" * 60)

    print(f"Version:              {VERSION}")
    print(f"Symbol:               {SYMBOL}")
    print(f"Timeframe:            {TIMEFRAME}")

    print("-" * 60)

    print(f"Population:           {POPULATION_SIZE}")
    print(f"Families:             {FAMILIES_COUNT}")
    print(f"Per Family:           {STRATEGIES_PER_FAMILY}")
    print(f"Random Injection:     {RANDOM_INJECTIONS}")

    print("-" * 60)

    print(f"Training:             {TRAIN_PERCENT:.0%}")
    print(f"OOS:                  {OOS_PERCENT:.0%}")

    print("-" * 60)

    print(f"Max Drawdown:         {MAX_DRAWDOWN:.1%}")
    print(f"Min Win Rate:         {MIN_WIN_RATE:.1%}")
    print(f"Min Trades:           {MIN_TRADES}")
    print(f"Min OOS Return:       {MIN_OOS_RETURN:.1%}")

    print("-" * 60)

    print(f"Starting Capital:     ${STARTING_CAPITAL:,.2f}")
    print(f"Paper Trading:        {PAPER_TRADING}")
    print(f"Live Trading:         {LIVE_TRADING}")

    print("=" * 60)


if __name__ == "__main__":
    print_config()
