from __future__ import annotations

import calendar
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import requests

BINANCE_BASE = "https://api.binance.com"
KLINE_COLS = ["open", "high", "low", "close", "volume", "amount"]

_INTERVAL_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


def download_binance_klines_for_symbols(
    *,
    symbols: Iterable[str],
    out_dir: str | Path,
    from_date: str,
    till_date: str,
    interval: str = "1h",
    timestamp_offset_hours: int = 3,
    sleep_seconds: float = 0.1,
) -> dict[str, int]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for symbol in symbols:
        secid = str(symbol).upper()
        df = fetch_binance_klines(
            symbol=secid,
            from_date=from_date,
            till_date=till_date,
            interval=interval,
            timestamp_offset_hours=timestamp_offset_hours,
        )
        if not df.empty:
            df.to_csv(out_path / f"candles_{secid}.csv", index=False)
        counts[secid] = len(df)
        time.sleep(max(float(sleep_seconds), 0.0))
    return counts


def fetch_binance_klines(
    *,
    symbol: str,
    from_date: str,
    till_date: str,
    interval: str = "1h",
    timestamp_offset_hours: int = 3,
    limit: int = 1000,
) -> pd.DataFrame:
    interval = str(interval)
    if interval not in _INTERVAL_MS:
        raise ValueError(f"unsupported Binance interval: {interval}")
    start_ms, end_ms = _date_range_to_utc_ms(from_date, till_date, timestamp_offset_hours=timestamp_offset_hours)
    url = f"{BINANCE_BASE}/api/v3/klines"
    rows: list[Sequence[Any]] = []
    cursor = start_ms
    while cursor < end_ms:
        response = requests.get(
            url,
            params={
                "symbol": str(symbol).upper(),
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms - 1,
                "limit": max(min(int(limit), 1000), 1),
            },
            timeout=30,
        )
        response.raise_for_status()
        chunk = response.json()
        if not chunk:
            break
        rows.extend(chunk)
        last_open = int(chunk[-1][0])
        next_cursor = last_open + _INTERVAL_MS[interval]
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(chunk) < max(min(int(limit), 1000), 1):
            break
    return binance_klines_to_candles(rows, timestamp_offset_hours=timestamp_offset_hours)


def binance_klines_to_candles(rows: Sequence[Sequence[Any]], *, timestamp_offset_hours: int = 3) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["timestamps", "end", *KLINE_COLS])
    df = pd.DataFrame(rows)
    if df.shape[1] < 8:
        raise ValueError("Binance kline rows must contain at least 8 columns")
    offset = pd.Timedelta(hours=int(timestamp_offset_hours))
    out = pd.DataFrame(
        {
            "timestamps": pd.to_datetime(pd.to_numeric(df[0], errors="coerce"), unit="ms", utc=True).dt.tz_localize(None) + offset,
            "end": pd.to_datetime(pd.to_numeric(df[6], errors="coerce"), unit="ms", utc=True).dt.tz_localize(None) + offset,
            "open": pd.to_numeric(df[1], errors="coerce"),
            "high": pd.to_numeric(df[2], errors="coerce"),
            "low": pd.to_numeric(df[3], errors="coerce"),
            "close": pd.to_numeric(df[4], errors="coerce"),
            "volume": pd.to_numeric(df[5], errors="coerce"),
            "amount": pd.to_numeric(df[7], errors="coerce"),
        }
    )
    out = out.dropna(subset=["timestamps", "end", "open", "high", "low", "close"])
    out["volume"] = out["volume"].fillna(0.0)
    out["amount"] = out["amount"].fillna(0.0)
    return out.sort_values("timestamps").drop_duplicates(subset=["timestamps"]).reset_index(drop=True)


def _date_range_to_utc_ms(from_date: str, till_date: str, *, timestamp_offset_hours: int) -> tuple[int, int]:
    start_local = pd.Timestamp(from_date)
    end_local = pd.Timestamp(till_date)
    if _date_only(till_date):
        end_local += pd.Timedelta(days=1)
    start_utc = start_local.to_pydatetime() - timedelta(hours=int(timestamp_offset_hours))
    end_utc = end_local.to_pydatetime() - timedelta(hours=int(timestamp_offset_hours))
    return _naive_utc_to_ms(start_utc), _naive_utc_to_ms(end_utc)


def _date_only(value: str) -> bool:
    stripped = str(value).strip()
    return len(stripped) == 10 and stripped[4] == "-" and stripped[7] == "-"


def _naive_utc_to_ms(value) -> int:
    return int(calendar.timegm(value.timetuple()) * 1000 + value.microsecond / 1000)
