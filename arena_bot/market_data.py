from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
import requests

from .types import Instrument, MarketMetrics, MarketSnapshot

EQUITY_SAVED_CANDLE_SPREAD_PCT = 0.0005
FUTURE_SAVED_CANDLE_SPREAD_PCT = 0.0002
CRYPTO_SAVED_CANDLE_SPREAD_PCT = 0.0002


class MarketDataProvider(Protocol):
    def snapshots(self, as_of: datetime, instruments: Sequence[Instrument]) -> Mapping[str, MarketSnapshot]:
        ...

    def candles(self, as_of: datetime, instruments: Sequence[Instrument]) -> Mapping[str, pd.DataFrame]:
        ...

    def metrics(self, as_of: datetime, instruments: Sequence[Instrument]) -> Mapping[str, MarketMetrics]:
        ...


@dataclass
class StaticMarketDataProvider:
    snapshot_rows: Mapping[str, MarketSnapshot] = field(default_factory=dict)
    candle_rows: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    metric_rows: Mapping[str, MarketMetrics] = field(default_factory=dict)

    def snapshots(self, as_of: datetime, instruments: Sequence[Instrument]) -> Mapping[str, MarketSnapshot]:
        return {
            instrument.secid: self.snapshot_rows.get(instrument.secid, MarketSnapshot(secid=instrument.secid))
            for instrument in instruments
        }

    def candles(self, as_of: datetime, instruments: Sequence[Instrument]) -> Mapping[str, pd.DataFrame]:
        return {
            instrument.secid: self.candle_rows.get(instrument.secid, pd.DataFrame()).copy()
            for instrument in instruments
        }

    def metrics(self, as_of: datetime, instruments: Sequence[Instrument]) -> Mapping[str, MarketMetrics]:
        snapshots = self.snapshots(as_of, instruments)
        candles = self.candles(as_of, instruments)
        out = {}
        for instrument in instruments:
            secid = instrument.secid
            if secid in self.metric_rows:
                out[secid] = self.metric_rows[secid]
            else:
                out[secid] = compute_market_metrics(secid, candles.get(secid, pd.DataFrame()), snapshots.get(secid))
        return out


class EmptyMarketDataProvider(StaticMarketDataProvider):
    pass


class AlgoPackRealtimeMarketDataProvider:
    def __init__(
        self,
        *,
        fallback: MarketDataProvider,
        token: str,
        base_url: str = "https://apim.moex.com",
        timeout: float = 10.0,
        retries: int = 2,
        session: requests.Session | None = None,
    ):
        self.fallback = fallback
        self.token = str(token or "").strip()
        self.base_url = str(base_url or "https://apim.moex.com").rstrip("/")
        self.timeout = float(timeout)
        self.retries = max(int(retries), 0)
        self.session = session or requests.Session()
        self._snapshot_cache_key: tuple[str, tuple[str, ...]] | None = None
        self._snapshot_cache: dict[str, MarketSnapshot] = {}

    def snapshots(self, as_of: datetime, instruments: Sequence[Instrument]) -> Mapping[str, MarketSnapshot]:
        secids = tuple(instrument.secid for instrument in instruments)
        key = (as_of.isoformat(timespec="seconds"), secids)
        if self._snapshot_cache_key == key:
            return dict(self._snapshot_cache)

        fallback_rows = dict(self.fallback.snapshots(as_of, instruments))
        out = dict(fallback_rows)
        if not self.token:
            self._snapshot_cache_key = key
            self._snapshot_cache = out
            return dict(out)

        for instrument in instruments:
            if instrument.asset_class not in {"equity", "future"}:
                continue
            snapshot = self._fetch_snapshot(instrument, fallback_rows.get(instrument.secid))
            if snapshot is not None:
                out[instrument.secid] = snapshot

        self._snapshot_cache_key = key
        self._snapshot_cache = out
        return dict(out)

    def candles(self, as_of: datetime, instruments: Sequence[Instrument]) -> Mapping[str, pd.DataFrame]:
        return self.fallback.candles(as_of, instruments)

    def metrics(self, as_of: datetime, instruments: Sequence[Instrument]) -> Mapping[str, MarketMetrics]:
        snapshots = self.snapshots(as_of, instruments)
        candles = self.candles(as_of, instruments)
        return {
            instrument.secid: compute_market_metrics(instrument.secid, candles.get(instrument.secid, pd.DataFrame()), snapshots.get(instrument.secid))
            for instrument in instruments
        }

    @property
    def _headers(self) -> dict[str, str]:
        value = self.token if self.token.lower().startswith("bearer ") else f"Bearer {self.token}"
        return {"Authorization": value}

    def _fetch_snapshot(self, instrument: Instrument, fallback: MarketSnapshot | None) -> MarketSnapshot | None:
        try:
            orderbook = self._fetch_orderbook(instrument)
        except Exception:
            return fallback
        try:
            marketdata = self._fetch_marketdata(instrument)
        except Exception:
            marketdata = {}

        bid, ask = _best_orderbook_bid_ask(orderbook)
        source = "algopack_orderbook"
        if bid <= 0 or ask <= 0 or ask < bid:
            bid = _finite_row_float(marketdata, "BID")
            ask = _finite_row_float(marketdata, "OFFER")
            source = "algopack_marketdata"
        if bid <= 0 or ask <= 0 or ask < bid:
            return fallback

        last_price = _finite_row_float(marketdata, "LAST")
        if last_price <= 0 and fallback is not None:
            last_price = float(fallback.last_price or 0.0)
        if last_price <= 0:
            last_price = (bid + ask) / 2.0

        volume_value = _finite_row_float(marketdata, "VALTODAY")
        if volume_value <= 0:
            volume_value = _finite_row_float(marketdata, "VALUE")
        if volume_value <= 0 and fallback is not None:
            volume_value = float(fallback.volume_value or 0.0)

        return MarketSnapshot(
            secid=instrument.secid,
            last_price=last_price,
            bid=bid,
            ask=ask,
            volume_value=volume_value,
            source=source,
        )

    def _fetch_marketdata(self, instrument: Instrument) -> Mapping[str, Any]:
        rows = self._get_block(f"{self._security_url(instrument)}.json", block_name="marketdata", params={"iss.only": "marketdata"})
        return rows[0] if rows else {}

    def _fetch_orderbook(self, instrument: Instrument) -> list[Mapping[str, Any]]:
        return self._get_block(f"{self._security_url(instrument)}/orderbook.json", block_name="orderbook", params={})

    def _security_url(self, instrument: Instrument) -> str:
        if instrument.asset_class == "future":
            engine, market, board = "futures", "forts", instrument.boardid or "RFUD"
        else:
            engine, market, board = "stock", "shares", instrument.boardid or "TQBR"
        return (
            f"{self.base_url}/iss/engines/{engine}/markets/{market}/boards/"
            f"{str(board).lower()}/securities/{instrument.secid.lower()}"
        )

    def _get_block(self, url: str, *, block_name: str, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        last_error: Exception | None = None
        response = None
        for attempt in range(self.retries + 1):
            try:
                response = self.session.get(url, params=dict(params), headers=self._headers, timeout=self.timeout)
                response.raise_for_status()
                break
            except Exception as exc:
                last_error = exc
                response = None
                if attempt < self.retries:
                    time.sleep(0.25 * (attempt + 1))
        if response is None:
            if last_error is not None:
                raise last_error
            return []
        payload = response.json()
        block = payload.get(block_name) if isinstance(payload, Mapping) else None
        if not isinstance(block, Mapping):
            return []
        columns = block.get("columns")
        data = block.get("data")
        if not isinstance(columns, list) or not isinstance(data, list):
            return []
        return [dict(zip(columns, row)) for row in data if isinstance(row, list)]


class BinanceRealtimeMarketDataProvider:
    def __init__(
        self,
        *,
        fallback: MarketDataProvider,
        base_url: str = "https://api.binance.com",
        timeout: float = 10.0,
        retries: int = 2,
        session: requests.Session | None = None,
    ):
        self.fallback = fallback
        self.base_url = str(base_url or "https://api.binance.com").rstrip("/")
        self.timeout = float(timeout)
        self.retries = max(int(retries), 0)
        self.session = session or requests.Session()
        self._snapshot_cache_key: tuple[str, tuple[str, ...]] | None = None
        self._snapshot_cache: dict[str, MarketSnapshot] = {}

    def snapshots(self, as_of: datetime, instruments: Sequence[Instrument]) -> Mapping[str, MarketSnapshot]:
        secids = tuple(instrument.secid for instrument in instruments)
        key = (as_of.isoformat(timespec="seconds"), secids)
        if self._snapshot_cache_key == key:
            return dict(self._snapshot_cache)

        fallback_rows = dict(self.fallback.snapshots(as_of, instruments))
        out = dict(fallback_rows)
        for instrument in instruments:
            if instrument.asset_class != "crypto":
                continue
            snapshot = self._fetch_snapshot(instrument, fallback_rows.get(instrument.secid))
            if snapshot is not None:
                out[instrument.secid] = snapshot

        self._snapshot_cache_key = key
        self._snapshot_cache = out
        return dict(out)

    def candles(self, as_of: datetime, instruments: Sequence[Instrument]) -> Mapping[str, pd.DataFrame]:
        return self.fallback.candles(as_of, instruments)

    def metrics(self, as_of: datetime, instruments: Sequence[Instrument]) -> Mapping[str, MarketMetrics]:
        snapshots = self.snapshots(as_of, instruments)
        candles = self.candles(as_of, instruments)
        return {
            instrument.secid: compute_market_metrics(instrument.secid, candles.get(instrument.secid, pd.DataFrame()), snapshots.get(instrument.secid))
            for instrument in instruments
        }

    def _fetch_snapshot(self, instrument: Instrument, fallback: MarketSnapshot | None) -> MarketSnapshot | None:
        try:
            book = self._get_json(
                f"{self.base_url}/api/v3/ticker/bookTicker",
                params={"symbol": instrument.secid.upper()},
            )
        except Exception:
            return fallback

        bid = _finite_row_float(book, "bidPrice")
        ask = _finite_row_float(book, "askPrice")
        if bid <= 0 or ask <= 0 or ask < bid:
            return fallback

        last_price = _finite_row_float(book, "lastPrice")
        volume_value = 0.0
        if last_price <= 0 or volume_value <= 0:
            try:
                ticker = self._get_json(
                    f"{self.base_url}/api/v3/ticker/24hr",
                    params={"symbol": instrument.secid.upper()},
                )
            except Exception:
                ticker = {}
            if last_price <= 0:
                last_price = _finite_row_float(ticker, "lastPrice")
            volume_value = _finite_row_float(ticker, "quoteVolume")

        if last_price <= 0 and fallback is not None:
            last_price = float(fallback.last_price or 0.0)
        if last_price <= 0:
            last_price = (bid + ask) / 2.0
        if volume_value <= 0 and fallback is not None:
            volume_value = float(fallback.volume_value or 0.0)

        return MarketSnapshot(
            secid=instrument.secid,
            last_price=last_price,
            bid=bid,
            ask=ask,
            volume_value=volume_value,
            source="binance_book_ticker",
        )

    def _get_json(self, url: str, *, params: Mapping[str, Any]) -> Mapping[str, Any]:
        last_error: Exception | None = None
        response = None
        for attempt in range(self.retries + 1):
            try:
                response = self.session.get(url, params=dict(params), timeout=self.timeout)
                response.raise_for_status()
                break
            except Exception as exc:
                last_error = exc
                response = None
                if attempt < self.retries:
                    time.sleep(0.25 * (attempt + 1))
        if response is None:
            if last_error is not None:
                raise last_error
            return {}
        payload = response.json()
        return payload if isinstance(payload, Mapping) else {}


class SavedCandleMarketDataProvider:
    def __init__(
        self,
        *,
        directories: Sequence[str | Path],
        filename_patterns: Sequence[str] = ("candles_{secid}.csv", "candles_1m_{secid}.csv"),
        history_rows: int = 512,
    ):
        self.directories = tuple(Path(path) for path in directories)
        self.filename_patterns = tuple(filename_patterns)
        self.history_rows = int(history_rows)
        self._cache: dict[str, pd.DataFrame] = {}

    def snapshots(self, as_of: datetime, instruments: Sequence[Instrument]) -> Mapping[str, MarketSnapshot]:
        candles = self.candles(as_of, instruments)
        out = {}
        for instrument in instruments:
            df = candles.get(instrument.secid, pd.DataFrame())
            if df.empty:
                out[instrument.secid] = MarketSnapshot(secid=instrument.secid, source="saved_candles_missing")
                continue
            row = df.iloc[-1]
            close = float(row.get("close", 0.0) or 0.0)
            amount = float(row.get("amount", 0.0) or 0.0)
            bid, ask = _saved_candle_bid_ask(close, instrument)
            out[instrument.secid] = MarketSnapshot(
                secid=instrument.secid,
                last_price=close,
                bid=bid,
                ask=ask,
                volume_value=amount,
                source="saved_candles",
            )
        return out

    def candles(self, as_of: datetime, instruments: Sequence[Instrument]) -> Mapping[str, pd.DataFrame]:
        out = {}
        cutoff = pd.Timestamp(as_of)
        for instrument in instruments:
            raw = self._load(instrument.secid)
            if raw.empty:
                out[instrument.secid] = pd.DataFrame()
                continue
            df = raw[raw["timestamps"] <= cutoff].tail(max(self.history_rows, 2)).copy()
            out[instrument.secid] = df
        return out

    def metrics(self, as_of: datetime, instruments: Sequence[Instrument]) -> Mapping[str, MarketMetrics]:
        snapshots = self.snapshots(as_of, instruments)
        candles = self.candles(as_of, instruments)
        return {
            instrument.secid: compute_market_metrics(instrument.secid, candles.get(instrument.secid, pd.DataFrame()), snapshots.get(instrument.secid))
            for instrument in instruments
        }

    def available_secids(self) -> list[str]:
        secids: set[str] = set()
        for directory in self.directories:
            if not directory.exists():
                continue
            for path in directory.glob("candles_*.csv"):
                name = path.stem
                if name.startswith("candles_1m_"):
                    secids.add(name.removeprefix("candles_1m_"))
                elif name.startswith("candles_"):
                    secids.add(name.removeprefix("candles_"))
        return sorted(secids)

    def _load(self, secid: str) -> pd.DataFrame:
        if secid in self._cache:
            return self._cache[secid]
        path = self._find_file(secid)
        if path is None:
            self._cache[secid] = pd.DataFrame()
            return self._cache[secid]
        df = pd.read_csv(path)
        if "timestamps" not in df.columns:
            self._cache[secid] = pd.DataFrame()
            return self._cache[secid]
        df["timestamps"] = pd.to_datetime(df["timestamps"], errors="coerce")
        df = df.dropna(subset=["timestamps"]).sort_values("timestamps").reset_index(drop=True)
        if "high" not in df.columns:
            df["high"] = df[["open", "close"]].max(axis=1) if {"open", "close"}.issubset(df.columns) else df.get("close", 0.0)
        if "low" not in df.columns:
            df["low"] = df[["open", "close"]].min(axis=1) if {"open", "close"}.issubset(df.columns) else df.get("close", 0.0)
        if "amount" not in df.columns:
            close = pd.to_numeric(df.get("close", 0.0), errors="coerce").fillna(0.0)
            volume = pd.to_numeric(df.get("volume", 0.0), errors="coerce").fillna(0.0)
            df["amount"] = close * volume
        self._cache[secid] = df
        return df

    def _find_file(self, secid: str) -> Path | None:
        for directory in self.directories:
            for pattern in self.filename_patterns:
                path = directory / pattern.format(secid=secid)
                if path.exists():
                    return path
        return None


def compute_market_metrics(secid: str, candles: pd.DataFrame, snapshot: MarketSnapshot | None = None) -> MarketMetrics:
    snapshot = snapshot or MarketSnapshot(secid=secid)
    if candles is None or candles.empty or "close" not in candles:
        return MarketMetrics(
            secid=secid,
            realized_volatility=0.0,
            atr_pct=0.0,
            volume_value=float(snapshot.volume_value or 0.0),
            spread_pct=float(snapshot.spread_pct),
            missing_candles=1,
            candle_count=0,
        )
    df = candles.copy()
    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    returns = close.pct_change().dropna()
    realized_vol = float(returns.std(ddof=0) or 0.0)
    atr_pct = _atr_pct(df)
    volume_value = _volume_value(df, snapshot)
    missing = int(df[["open", "high", "low", "close"]].isna().any(axis=1).sum()) if set(["open", "high", "low", "close"]).issubset(df.columns) else 0
    return MarketMetrics(
        secid=secid,
        realized_volatility=_finite(realized_vol),
        atr_pct=_finite(atr_pct),
        volume_value=_finite(volume_value),
        spread_pct=_finite(snapshot.spread_pct, default=1.0),
        missing_candles=missing,
        candle_count=int(len(close)),
    )


def build_market_features(
    *,
    selected_secids: Sequence[str],
    snapshots: Mapping[str, MarketSnapshot],
    metrics: Mapping[str, MarketMetrics],
    signal_scores: Mapping[str, float],
) -> dict[str, float]:
    selected = list(selected_secids)
    spreads = [float(metrics[s].spread_pct) for s in selected if s in metrics]
    vols = [float(metrics[s].realized_volatility) for s in selected if s in metrics]
    prices = [float(snapshots[s].last_price) for s in selected if s in snapshots and snapshots[s].last_price > 0]
    scores = [float(signal_scores[s]) for s in selected if s in signal_scores]
    return {
        "selected_count": float(len(selected)),
        "avg_spread_pct": float(np.mean(spreads)) if spreads else 0.0,
        "avg_realized_volatility": float(np.mean(vols)) if vols else 0.0,
        "priced_count": float(len(prices)),
        "signal_spread": (max(scores) - min(scores)) if scores else 0.0,
        "signal_mean": float(np.mean(scores)) if scores else 0.5,
        "missing_candles_total": float(sum(metrics[s].missing_candles for s in selected if s in metrics)),
    }


def _atr_pct(df: pd.DataFrame) -> float:
    required = {"high", "low", "close"}
    if not required.issubset(df.columns):
        return 0.0
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1)
    true_range = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    last_close = float(close.dropna().iloc[-1]) if not close.dropna().empty else 0.0
    if last_close <= 0:
        return 0.0
    return float(true_range.tail(20).mean() / last_close)


def _volume_value(df: pd.DataFrame, snapshot: MarketSnapshot) -> float:
    if snapshot.volume_value > 0:
        return float(snapshot.volume_value)
    if {"volume", "close"}.issubset(df.columns):
        volume = pd.to_numeric(df["volume"], errors="coerce")
        close = pd.to_numeric(df["close"], errors="coerce")
        return float((volume * close).tail(20).mean() or 0.0)
    return 0.0


def _finite(value: float, *, default: float = 0.0) -> float:
    return float(value) if math.isfinite(float(value)) else default


def _finite_row_float(row: Mapping[str, Any], key: str) -> float:
    try:
        value = float(row.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0
    return value if math.isfinite(value) else 0.0


def _best_orderbook_bid_ask(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    bids = []
    asks = []
    for row in rows:
        price = _finite_row_float(row, "PRICE")
        if price <= 0:
            continue
        side = str(row.get("BUYSELL", "")).strip().upper()
        if side == "B":
            bids.append(price)
        elif side == "S":
            asks.append(price)
    bid = max(bids) if bids else 0.0
    ask = min(asks) if asks else 0.0
    return bid, ask


def _saved_candle_bid_ask(close: float, instrument: Instrument) -> tuple[float, float]:
    if float(close) <= 0:
        return 0.0, 0.0
    if instrument.asset_class == "future":
        spread = FUTURE_SAVED_CANDLE_SPREAD_PCT
    elif instrument.asset_class == "crypto":
        spread = CRYPTO_SAVED_CANDLE_SPREAD_PCT
    else:
        spread = EQUITY_SAVED_CANDLE_SPREAD_PCT
    half_spread = spread / 2.0
    return float(close) * (1.0 - half_spread), float(close) * (1.0 + half_spread)
