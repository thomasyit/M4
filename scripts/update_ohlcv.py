import os
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
import requests

BASE_DIR = Path(os.getenv("DATA_LOADER_BASE", "/opt/M4"))
DATA_DIR = BASE_DIR / "binance_historical"
SYMBOL = os.getenv("DATA_LOADER_SYMBOL", "BTCUSDT")
TIMEFRAMES = ["1h", "4h", "1d"]
LOCK_PATH = BASE_DIR / "state" / "update_ohlcv.lock"
RECENT_ONLY = os.getenv("OHLCV_RECENT_ONLY", "true").lower() == "true"

BINANCE_URL = "https://api.binance.com/api/v3/klines"


def _acquire_lock(max_age_sec: int = 1800) -> bool:
    if LOCK_PATH.exists():
        age = time.time() - LOCK_PATH.stat().st_mtime
        if age < max_age_sec:
            return False
    LOCK_PATH.write_text(str(int(time.time())))
    return True


def _release_lock() -> None:
    try:
        LOCK_PATH.unlink()
    except FileNotFoundError:
        pass


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _fetch_klines(symbol: str, interval: str, start_ms: Optional[int]) -> List[List]:
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": 1000,
    }
    if start_ms is not None:
        params["startTime"] = start_ms
    resp = requests.get(BINANCE_URL, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _update_timeframe(symbol: str, interval: str) -> None:
    folder = DATA_DIR / f"ohlcv_{interval}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{symbol}_{interval}.csv"

    df = _load_csv(path)
    if df is None or df.empty:
        last_ms = None
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    else:
        last_ts = df["timestamp"].max()
        last_ms = int(last_ts.timestamp() * 1000) + 1

    start_ms = None if RECENT_ONLY else last_ms
    klines = _fetch_klines(symbol, interval, start_ms)
    if not klines:
        return

    new_rows = []
    for k in klines:
        ts = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc)
        new_rows.append(
            {
                "timestamp": ts,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            }
        )

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([df, new_df], ignore_index=True)
    combined.drop_duplicates(subset=["timestamp"], inplace=True)
    combined.sort_values("timestamp", inplace=True)
    combined.to_csv(path, index=False)


def main() -> int:
    if not _acquire_lock():
        return 0
    try:
        for interval in TIMEFRAMES:
            _update_timeframe(SYMBOL, interval)
    finally:
        _release_lock()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
