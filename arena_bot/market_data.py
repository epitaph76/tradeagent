from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

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
