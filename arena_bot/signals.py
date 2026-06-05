from __future__ import annotations

from datetime import datetime
from typing import Mapping, Protocol, Sequence

import pandas as pd

from .types import Instrument, SignalRow, clamp


class SignalProvider(Protocol):
    name: str

    def score(
        self,
        as_of: datetime,
        instruments: Sequence[Instrument],
        candles: Mapping[str, pd.DataFrame],
    ) -> Sequence[SignalRow]:
        ...


class MomentumSignalProvider:
    name = "momentum"

    def __init__(self, *, lookback_rows: int = 20, score_scale: float = 0.10):
        self.lookback_rows = lookback_rows
        self.score_scale = score_scale

    def score(
        self,
        as_of: datetime,
        instruments: Sequence[Instrument],
        candles: Mapping[str, pd.DataFrame],
    ) -> Sequence[SignalRow]:
        rows = []
        for instrument in instruments:
            df = candles.get(instrument.secid, pd.DataFrame())
            value = _momentum_score(df, lookback_rows=self.lookback_rows, score_scale=self.score_scale)
            confidence = min(abs(value - 0.5) * 4.0, 1.0)
            rows.append(
                SignalRow(
                    as_of=as_of,
                    secid=instrument.secid,
                    signal_name=self.name,
                    bullish_score=value,
                    confidence=confidence,
                    reason="close_to_close_momentum",
                )
            )
        return rows


class StaticSignalProvider:
    def __init__(self, name: str, scores: Mapping[str, float], *, confidence: float = 1.0):
        self.name = name
        self.scores = dict(scores)
        self.confidence = confidence

    def score(
        self,
        as_of: datetime,
        instruments: Sequence[Instrument],
        candles: Mapping[str, pd.DataFrame],
    ) -> Sequence[SignalRow]:
        return [
            SignalRow(as_of=as_of, secid=instrument.secid, signal_name=self.name, bullish_score=self.scores[instrument.secid], confidence=self.confidence)
            for instrument in instruments
            if instrument.secid in self.scores
        ]


class EmptyKronosSignalProvider(StaticSignalProvider):
    def __init__(self):
        super().__init__("kronos", {}, confidence=0.0)


def with_equity_kronos_fallback(
    *,
    as_of: datetime,
    instruments: Sequence[Instrument],
    kronos_rows: Sequence[SignalRow],
    momentum_rows: Sequence[SignalRow],
) -> tuple[SignalRow, ...]:
    out = list(kronos_rows) + list(momentum_rows)
    kronos_by_secid = {row.secid: row for row in kronos_rows if row.signal_name == "kronos" and row.confidence > 0}
    momentum_by_secid = {row.secid: row for row in momentum_rows if row.signal_name == "momentum"}
    for instrument in instruments:
        if instrument.asset_class != "equity" or instrument.secid in kronos_by_secid:
            continue
        momentum = momentum_by_secid.get(instrument.secid)
        if momentum is None:
            continue
        out.append(
            SignalRow(
                as_of=as_of,
                secid=instrument.secid,
                signal_name="kronos",
                bullish_score=momentum.bullish_score,
                confidence=min(momentum.confidence, 0.5),
                reason="momentum_fallback_for_missing_kronos",
            )
        )
    return tuple(out)


def latest_signal_scores(rows: Sequence[SignalRow], signal_name: str = "kronos") -> dict[str, float]:
    return {row.secid: float(row.bullish_score) for row in rows if row.signal_name == signal_name}


def _momentum_score(df: pd.DataFrame, *, lookback_rows: int, score_scale: float) -> float:
    if df is None or df.empty or "close" not in df:
        return 0.5
    close = pd.to_numeric(df["close"], errors="coerce").dropna().tail(max(lookback_rows, 2))
    if len(close) < 2:
        return 0.5
    first = float(close.iloc[0])
    last = float(close.iloc[-1])
    if first <= 0:
        return 0.5
    ret = last / first - 1.0
    return clamp(0.5 + ret / max(score_scale, 1e-9))

