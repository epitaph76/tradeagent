from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import requests

MOEX_BASE = "https://iss.moex.com/iss"
KLINE_COLS = ["open", "high", "low", "close", "volume", "amount"]


def download_candles_for_instruments(
    *,
    secids: Iterable[str],
    out_dir: str | Path,
    from_date: str,
    till_date: str,
    interval: int = 60,
    source_interval: int | None = None,
    engine: str = "stock",
    market: str = "shares",
    sleep_seconds: float = 0.1,
) -> dict[str, int]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    source = int(source_interval or interval)
    counts = {}
    for secid in secids:
        df = fetch_moex_candles(
            secid=secid,
            from_date=from_date,
            till_date=till_date,
            interval=int(interval),
            source_interval=source,
            engine=engine,
            market=market,
        )
        if not df.empty:
            df.to_csv(out_path / f"candles_{secid}.csv", index=False)
        counts[secid] = len(df)
        time.sleep(max(float(sleep_seconds), 0.0))
    return counts


def download_futures_candles_for_roots(
    *,
    roots: Iterable[str],
    out_dir: str | Path,
    from_date: str,
    till_date: str,
    interval: int = 60,
    source_interval: int | None = None,
    sleep_seconds: float = 0.1,
) -> tuple[dict[str, int], dict[str, str]]:
    resolved = resolve_futures_roots(roots=roots, till_date=till_date)
    counts = download_candles_for_instruments(
        secids=resolved.values(),
        out_dir=out_dir,
        from_date=from_date,
        till_date=till_date,
        interval=interval,
        source_interval=source_interval,
        engine="futures",
        market="forts",
        sleep_seconds=sleep_seconds,
    )
    return counts, resolved


def fetch_moex_candles(
    *,
    secid: str,
    from_date: str,
    till_date: str,
    interval: int,
    source_interval: int,
    engine: str = "stock",
    market: str = "shares",
) -> pd.DataFrame:
    url = f"{MOEX_BASE}/engines/{engine}/markets/{market}/securities/{secid}/candles.json"
    chunks = []
    start = 0
    while True:
        js = _iss_get_json(
            url,
            params={
                "interval": int(source_interval),
                "from": from_date,
                "till": till_date,
                "start": start,
                "iss.meta": "off",
            },
        )
        part = _iss_block_to_df(js, "candles")
        if part.empty:
            break
        chunks.append(part)
        start += len(part)
        cursor = _iss_block_to_df(js, "candles.cursor")
        if not cursor.empty and {"TOTAL", "PAGESIZE"}.issubset(cursor.columns):
            total = int(cursor["TOTAL"].iloc[0])
            if start >= total:
                break
        elif len(part) < 500:
            break

    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True).rename(columns={"begin": "timestamps", "value": "amount"})
    needed = ["timestamps", "end", *KLINE_COLS]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"{secid}: missing MOEX candle columns {missing}; got {list(df.columns)}")
    df = df[needed].copy()
    df["timestamps"] = pd.to_datetime(df["timestamps"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    for col in KLINE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamps", "end", "open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0.0)
    df["amount"] = df["amount"].fillna(0.0)
    df = df.sort_values("timestamps").drop_duplicates(subset=["timestamps"]).reset_index(drop=True)
    if int(source_interval) != int(interval):
        df = _resample_candles(df, int(interval))
    return df


def resolve_futures_roots(*, roots: Iterable[str], till_date: str) -> dict[str, str]:
    securities = fetch_moex_futures_securities()
    if securities.empty:
        raise RuntimeError("MOEX futures securities list is empty")
    required = {"SECID", "ASSETCODE", "LASTTRADEDATE"}
    missing = required - set(securities.columns)
    if missing:
        raise ValueError(f"MOEX futures securities response is missing columns: {sorted(missing)}")

    df = securities.copy()
    df["ASSETCODE"] = df["ASSETCODE"].astype(str)
    df["LASTTRADEDATE"] = pd.to_datetime(df["LASTTRADEDATE"], errors="coerce")
    cutoff = pd.Timestamp(till_date)
    resolved: dict[str, str] = {}
    for root in roots:
        asset_code = _futures_asset_code(root)
        candidates = df[(df["ASSETCODE"] == asset_code) & (df["LASTTRADEDATE"] >= cutoff)].copy()
        if candidates.empty:
            raise RuntimeError(f"no active MOEX futures contract for root={root!r}, asset_code={asset_code!r}, till={till_date}")
        candidates = candidates.sort_values(["LASTTRADEDATE", "SECID"])
        resolved[str(root)] = str(candidates["SECID"].iloc[0])
    return resolved


def fetch_moex_futures_securities() -> pd.DataFrame:
    url = f"{MOEX_BASE}/engines/futures/markets/forts/securities.json"
    js = _iss_get_json(url, params={"iss.meta": "off"})
    return _iss_block_to_df(js, "securities")


def _resample_candles(df: pd.DataFrame, interval: int) -> pd.DataFrame:
    tmp = df.copy()
    tmp["bucket"] = tmp["timestamps"].dt.floor(f"{interval}min")
    out = (
        tmp.groupby("bucket", sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            amount=("amount", "sum"),
            end=("end", "max"),
        )
        .reset_index()
        .rename(columns={"bucket": "timestamps"})
    )
    return out[["timestamps", "end", *KLINE_COLS]]


def _iss_get_json(url: str, params: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _iss_block_to_df(js: Mapping[str, Any], block_name: str) -> pd.DataFrame:
    block = js.get(block_name)
    if isinstance(block, dict) and "columns" in block and "data" in block:
        return pd.DataFrame(block["data"], columns=block["columns"])
    return pd.DataFrame()


def _futures_asset_code(root: str) -> str:
    normalized = str(root).strip()
    aliases = {
        "GD": "GOLD",
        "GOLD": "GOLD",
        "MXI": "MXI",
        "IMOEX": "MXI",
    }
    upper = normalized.upper()
    if upper in aliases:
        return aliases[upper]
    if upper == "SI":
        return "Si"
    return upper
