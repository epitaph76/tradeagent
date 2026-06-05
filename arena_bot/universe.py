from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .types import Instrument, MarketMetrics, MarketSnapshot


@dataclass(frozen=True)
class UniverseSelection:
    instruments: tuple[Instrument, ...]
    diagnostics: Mapping[str, Any]

    @property
    def secids(self) -> tuple[str, ...]:
        return tuple(instrument.secid for instrument in self.instruments)


def select_universe(
    instruments: Sequence[Instrument],
    *,
    snapshots: Mapping[str, MarketSnapshot],
    metrics: Mapping[str, MarketMetrics],
    max_equities: int = 10,
) -> UniverseSelection:
    eligible_equities: list[tuple[float, Instrument]] = []
    selected_non_equities: list[Instrument] = []
    rejected: dict[str, str] = {}

    for instrument in instruments:
        reason = _untradable_reason(instrument, snapshots.get(instrument.secid), metrics.get(instrument.secid))
        if reason:
            rejected[instrument.secid] = reason
            continue
        if instrument.asset_class == "equity":
            eligible_equities.append((_universe_score(metrics[instrument.secid]), instrument))
        else:
            selected_non_equities.append(instrument)

    eligible_equities.sort(key=lambda item: (item[0], item[1].secid), reverse=True)
    selected_equities = [instrument for _, instrument in eligible_equities[: max(max_equities, 0)]]
    selected = tuple(selected_equities + selected_non_equities)
    return UniverseSelection(
        instruments=selected,
        diagnostics={
            "selected_count": len(selected),
            "selected_equities": [instrument.secid for instrument in selected_equities],
            "selected_non_equities": [instrument.secid for instrument in selected_non_equities],
            "eligible_equities_count": len(eligible_equities),
            "rejected": rejected,
        },
    )


def _untradable_reason(
    instrument: Instrument,
    snapshot: MarketSnapshot | None,
    metrics: MarketMetrics | None,
) -> str:
    if not instrument.enabled:
        return "disabled"
    if instrument.lot_size <= 0:
        return "missing_lot_size"
    if snapshot is None or snapshot.last_price <= 0:
        return "missing_price"
    if metrics is None or metrics.candle_count <= 0:
        return "missing_candles"
    return ""


def _universe_score(metrics: MarketMetrics) -> float:
    liquidity = metrics.volume_value / 1_000_000.0
    return (
        10.0 * metrics.realized_volatility
        + 5.0 * metrics.atr_pct
        + min(liquidity, 100.0) / 100.0
        - 2.0 * metrics.spread_pct
        - 0.05 * metrics.missing_candles
    )

