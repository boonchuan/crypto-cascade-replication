"""
fetch_data.py — Appendix A, Step 1: Data acquisition.

Downloads 1-minute klines from Binance public REST APIs and caches them
locally as parquet. Subsequent runs (including the full sensitivity grid)
read from cache and never re-hit the API.

Series fetched:
  BTCUSDT spot        https://api.binance.com/api/v3/klines
  BTCUSDT perpetual   https://fapi.binance.com/fapi/v1/klines
  BTCUSDT mark price  https://fapi.binance.com/fapi/v1/markPriceKlines
  SOLUSDT spot, perpetual (same endpoints)

Usage:
  python -m src.fetch_data --start 2024-01-01 --end 2026-04-30
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

ENDPOINTS = {
    "spot": "https://api.binance.com/api/v3/klines",
    "perp": "https://fapi.binance.com/fapi/v1/klines",
    "mark": "https://fapi.binance.com/fapi/v1/markPriceKlines",
}

SERIES = [
    ("BTCUSDT", "spot"),
    ("BTCUSDT", "perp"),
    ("BTCUSDT", "mark"),
    ("SOLUSDT", "spot"),
    ("SOLUSDT", "perp"),
]

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "n_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]

REQUEST_LIMIT = 1000          # candles per request (Appendix A)
SLEEP_BETWEEN_REQUESTS = 0.25  # stay well inside Binance weight limits


def _ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def fetch_series(symbol: str, kind: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Paginated 1m kline download for one (symbol, kind)."""
    url = ENDPOINTS[kind]
    rows = []
    cursor = _ms(start)
    end_ms = _ms(end)
    session = requests.Session()

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": "1m",
            "startTime": cursor,
            "endTime": end_ms,
            "limit": REQUEST_LIMIT,
        }
        resp = session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + 60_000  # next minute after last open_time
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    if not rows:
        raise RuntimeError(f"No data returned for {symbol} {kind}")

    # markPriceKlines returns the same 12-field layout as klines
    df = pd.DataFrame(rows, columns=KLINE_COLS[: len(rows[0])])
    df = df[["open_time", "open", "high", "low", "close"] +
            (["volume"] if kind != "mark" else [])]
    for c in df.columns:
        if c != "open_time":
            df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.rename(columns={"open_time": "ts"})

    # Appendix A: remove duplicate minute candles, keep first occurrence
    df = df.drop_duplicates(subset="ts", keep="first").sort_values("ts")

    # Appendix A: forward-fill rare missing minutes on a complete grid
    full_index = pd.date_range(df["ts"].iloc[0], df["ts"].iloc[-1],
                               freq="1min", tz="UTC")
    df = df.set_index("ts").reindex(full_index).ffill()
    df.index.name = "ts"
    return df.reset_index()


def cache_path(symbol: str, kind: str) -> Path:
    return DATA_DIR / f"{symbol}_{kind}_1m.parquet"


def load_or_fetch(symbol: str, kind: str, start: datetime, end: datetime,
                  force: bool = False) -> pd.DataFrame:
    path = cache_path(symbol, kind)
    if path.exists() and not force:
        return pd.read_parquet(path)
    print(f"Fetching {symbol} {kind} 1m klines "
          f"{start.date()} -> {end.date()} ...")
    df = fetch_series(symbol, kind, start, end)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"  saved {len(df):,} rows -> {path.name}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2026-04-30")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if cached")
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end).replace(hour=23, minute=59)

    for symbol, kind in SERIES:
        load_or_fetch(symbol, kind, start, end, force=args.force)
    print("All series cached.")


if __name__ == "__main__":
    main()
