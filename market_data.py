"""
Evolution Trader
================
Market Data Engine

V1:
- BTC/USD
- 5-Minuten-Candles
- öffentliche Coinbase-Daten
- KEINE API-Keys
- Speicherung als CSV

Die Market-Data-Schicht ist bewusst vom Backtester getrennt.

Datenfluss:

Internet
   ↓
Coinbase public candles
   ↓
Candle objects
   ↓
CSV
   ↓
Backtester
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from strategy_families import Candle


# ============================================================
# CONFIGURATION
# ============================================================

PRODUCT_ID = "BTC-USD"

GRANULARITY_SECONDS = 300

BASE_URL = (
    "https://api.exchange.coinbase.com/products/"
    f"{PRODUCT_ID}/candles"
)

DATA_DIR = Path("data")

CSV_FILE = DATA_DIR / "BTC_USD_5m.csv"

# Coinbase liefert pro Anfrage nur eine begrenzte Anzahl
# Candles. Deshalb holen wir die Historie in Blöcken.
MAX_CANDLES_PER_REQUEST = 300

REQUEST_DELAY_SECONDS = 0.25


# ============================================================
# HTTP
# ============================================================

def http_get_json(
    url: str,
) -> object:
    """
    Öffentliche HTTP-GET-Anfrage.

    Keine API-Keys.
    """

    request = Request(
        url,
        headers={
            "User-Agent": (
                "Evolution-Trader/0.1 "
                "(public market data)"
            ),
            "Accept": "application/json",
        },
    )

    with urlopen(
        request,
        timeout=30,
    ) as response:

        raw = response.read().decode(
            "utf-8"
        )

    import json

    return json.loads(raw)


# ============================================================
# TIMESTAMP
# ============================================================

def timestamp_to_iso(
    timestamp: int,
) -> str:
    """Unix timestamp → UTC ISO."""

    return datetime.fromtimestamp(
        timestamp,
        tz=timezone.utc,
    ).isoformat()


# ============================================================
# DOWNLOAD ONE CHUNK
# ============================================================

def download_chunk(
    start_timestamp: int,
    end_timestamp: int,
) -> List[Candle]:
    """
    Lädt einen Datenblock.

    Coinbase-Candle-Format:

    [
        time,
        low,
        high,
        open,
        close,
        volume
    ]
    """

    params = urlencode(
        {
            "granularity": GRANULARITY_SECONDS,
            "start": datetime.fromtimestamp(
                start_timestamp,
                tz=timezone.utc,
            ).isoformat(),
            "end": datetime.fromtimestamp(
                end_timestamp,
                tz=timezone.utc,
            ).isoformat(),
        }
    )

    url = f"{BASE_URL}?{params}"

    payload = http_get_json(url)

    if not isinstance(payload, list):
        raise RuntimeError(
            f"Unerwartete API-Antwort: {payload}"
        )

    candles: List[Candle] = []

    for row in payload:

        if not isinstance(row, list):
            continue

        if len(row) < 6:
            continue

        timestamp = int(row[0])

        low = float(row[1])
        high = float(row[2])
        open_price = float(row[3])
        close = float(row[4])
        volume = float(row[5])

        # Grundlegende Datenvalidierung
        if open_price <= 0:
            continue

        if close <= 0:
            continue

        if low <= 0:
            continue

        if high <= 0:
            continue

        if high < low:
            continue

        candles.append(
            Candle(
                timestamp=timestamp,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
        )

    return candles


# ============================================================
# DOWNLOAD HISTORY
# ============================================================

def download_history(
    days: int = 30,
) -> List[Candle]:
    """
    Lädt historische BTC/USD-5m-Daten.

    Standard:
        30 Tage

    30 Tage sind für den ersten technischen Test
    ausreichend.

    Später erweitern wir das auf deutlich längere
    Zeiträume.
    """

    if days <= 0:
        raise ValueError(
            "days muss > 0 sein."
        )

    now = int(
        datetime.now(
            timezone.utc
        ).timestamp()
    )

    start = (
        now
        - days * 24 * 60 * 60
    )

    candles: List[Candle] = []

    chunk_seconds = (
        GRANULARITY_SECONDS
        * MAX_CANDLES_PER_REQUEST
    )

    current_end = now

    print("=" * 70)
    print("EVOLUTION TRADER - MARKET DATA")
    print("=" * 70)

    print(
        f"Product:       {PRODUCT_ID}"
    )

    print(
        f"Timeframe:     5m"
    )

    print(
        f"Requested:     {days} days"
    )

    print()

    while current_end > start:

        current_start = max(
            start,
            current_end - chunk_seconds,
        )

        print(
            "Downloading:",
            timestamp_to_iso(current_start),
            "→",
            timestamp_to_iso(current_end),
        )

        try:

            chunk = download_chunk(
                current_start,
                current_end,
            )

        except Exception as exc:

            print(
                "ERROR downloading chunk:",
                exc,
            )

            raise

        candles.extend(chunk)

        # Ein kleines Delay schützt die öffentliche
        # Schnittstelle vor unnötig schnellen Anfragen.
        time.sleep(
            REQUEST_DELAY_SECONDS
        )

        # Einen Tick zurückgehen, damit wir nicht
        # immer dieselbe Kerze anfordern.
        current_end = (
            current_start
            - GRANULARITY_SECONDS
        )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    candles.sort(
        key=lambda candle: candle.timestamp
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    unique = {}

    for candle in candles:

        unique[
            candle.timestamp
        ] = candle

    candles = list(
        unique.values()
    )

    candles.sort(
        key=lambda candle: candle.timestamp
    )

    print()

    print(
        f"Downloaded candles: {len(candles)}"
    )

    if candles:

        print(
            "First:",
            timestamp_to_iso(
                candles[0].timestamp
            ),
        )

        print(
            "Last:",
            timestamp_to_iso(
                candles[-1].timestamp
            ),
        )

    print("=" * 70)

    return candles


# ============================================================
# CSV SAVE
# ============================================================

def save_csv(
    candles: List[Candle],
    path: Path = CSV_FILE,
) -> None:
    """
    Speichert Candles als CSV.
    """

    if not candles:
        raise ValueError(
            "Keine Candles zum Speichern."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "timestamp",
                "datetime_utc",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )

        for candle in candles:

            writer.writerow(
                [
                    candle.timestamp,
                    timestamp_to_iso(
                        candle.timestamp
                    ),
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume,
                ]
            )

    print(
        f"Saved: {path}"
    )


# ============================================================
# CSV LOAD
# ============================================================

def load_csv(
    path: Path = CSV_FILE,
) -> List[Candle]:
    """
    Lädt Candles aus einer CSV.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Datei nicht gefunden: {path}"
        )

    candles: List[Candle] = []

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            candles.append(
                Candle(
                    timestamp=int(
                        row["timestamp"]
                    ),

                    open=float(
                        row["open"]
                    ),

                    high=float(
                        row["high"]
                    ),

                    low=float(
                        row["low"]
                    ),

                    close=float(
                        row["close"]
                    ),

                    volume=float(
                        row["volume"]
                    ),
                )
            )

    candles.sort(
        key=lambda candle: candle.timestamp
    )

    return candles


# ============================================================
# DATA QUALITY
# ============================================================

def validate_data(
    candles: List[Candle],
) -> None:
    """
    Prüft die Datenqualität.

    Ein Backtest darf nicht auf offensichtlich
    beschädigten Daten laufen.
    """

    if not candles:
        raise ValueError(
            "Keine Marktdaten."
        )

    previous_timestamp: Optional[int] = None

    for index, candle in enumerate(candles):

        # ----------------------------------------------------
        # POSITIVE PRICES
        # ----------------------------------------------------

        if candle.open <= 0:
            raise ValueError(
                f"Invalid open at {index}"
            )

        if candle.high <= 0:
            raise ValueError(
                f"Invalid high at {index}"
            )

        if candle.low <= 0:
            raise ValueError(
                f"Invalid low at {index}"
            )

        if candle.close <= 0:
            raise ValueError(
                f"Invalid close at {index}"
            )

        # ----------------------------------------------------
        # OHLC CONSISTENCY
        # ----------------------------------------------------

        if candle.high < candle.low:
            raise ValueError(
                f"High < Low at {index}"
            )

        if candle.high < candle.open:
            raise ValueError(
                f"High < Open at {index}"
            )

        if candle.high < candle.close:
            raise ValueError(
                f"High < Close at {index}"
            )

        if candle.low > candle.open:
            raise ValueError(
                f"Low > Open at {index}"
            )

        if candle.low > candle.close:
            raise ValueError(
                f"Low > Close at {index}"
            )

        # ----------------------------------------------------
        # TIMESTAMP ORDER
        # ----------------------------------------------------

        if (
            previous_timestamp is not None
            and candle.timestamp
            <= previous_timestamp
        ):
            raise ValueError(
                "Timestamps are not strictly increasing."
            )

        previous_timestamp = (
            candle.timestamp
        )

    print(
        f"Data validation PASSED "
        f"({len(candles)} candles)"
    )


# ============================================================
# GAP DETECTION
# ============================================================

def find_gaps(
    candles: List[Candle],
) -> List[tuple]:
    """
    Findet fehlende 5-Minuten-Kerzen.

    Gibt Paare zurück:

        (previous_timestamp, current_timestamp)
    """

    gaps = []

    expected = (
        GRANULARITY_SECONDS
    )

    for previous, current in zip(
        candles,
        candles[1:],
    ):

        difference = (
            current.timestamp
            - previous.timestamp
        )

        if difference != expected:

            gaps.append(
                (
                    previous.timestamp,
                    current.timestamp,
                )
            )

    return gaps


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    candles: List[Candle],
) -> None:
    """Zeigt eine Zusammenfassung der Daten."""

    print("=" * 70)
    print("MARKET DATA SUMMARY")
    print("=" * 70)

    print(
        f"Candles:       {len(candles)}"
    )

    if candles:

        print(
            "First candle:",
            timestamp_to_iso(
                candles[0].timestamp
            ),
        )

        print(
            "Last candle:",
            timestamp_to_iso(
                candles[-1].timestamp
            ),
        )

        print(
            f"First close:   "
            f"${candles[0].close:,.2f}"
        )

        print(
            f"Last close:    "
            f"${candles[-1].close:,.2f}"
        )

    gaps = find_gaps(candles)

    print(
        f"Gaps:          {len(gaps)}"
    )

    if gaps:

        print()
        print(
            "WARNING:"
        )

        print(
            "Missing candle intervals detected."
        )

    print("=" * 70)


# ============================================================
# SELF TEST
# ============================================================

def self_test() -> None:
    """
    Lädt 1 Tag echte Daten und prüft die Pipeline.

    Noch kein Backtest.
    """

    print("=" * 70)
    print("EVOLUTION TRADER - MARKET DATA SELF TEST")
    print("=" * 70)

    candles = download_history(
        days=1
    )

    validate_data(
        candles
    )

    print_summary(
        candles
    )

    save_csv(
        candles
    )

    reloaded = load_csv(
        CSV_FILE
    )

    validate_data(
        reloaded
    )

    assert len(reloaded) > 0

    print()
    print(
        "PASS: Download"
    )

    print(
        "PASS: Validation"
    )

    print(
        "PASS: CSV save"
    )

    print(
        "PASS: CSV reload"
    )

    print()
    print(
        "MARKET DATA SELF TEST PASSED"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    self_test()
