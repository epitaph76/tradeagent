from __future__ import annotations

from typing import Any, Mapping, Sequence

from .types import BaseSelectorConfig, Instrument, SelectorDecision, SignalRow


def build_selector_portfolio(
    config: BaseSelectorConfig,
    *,
    instruments: Sequence[Instrument],
    signals: Sequence[SignalRow],
) -> SelectorDecision:
    allowed = {
        instrument.secid: instrument
        for instrument in instruments
        if config.asset_filter is None or instrument.asset_class in set(config.asset_filter)
    }
    score_by_secid = _combined_bull_scores(signals, config.signal_weights)
    candidates = []
    for secid, bull in score_by_secid.items():
        if secid not in allowed:
            continue
        bear = 1.0 - bull
        if bull >= config.threshold and (not config.allow_short or bull >= bear):
            candidates.append({"secid": secid, "side": "long", "score": bull})
        elif config.allow_short and bear >= config.threshold and bear > bull:
            candidates.append({"secid": secid, "side": "short", "score": bear})

    long_candidates = sorted((row for row in candidates if row["side"] == "long"), key=lambda row: (-row["score"], row["secid"]))
    short_candidates = sorted((row for row in candidates if row["side"] == "short"), key=lambda row: (-row["score"], row["secid"]))
    if config.max_long_positions is not None:
        long_candidates = long_candidates[: max(config.max_long_positions, 0)]
    if config.max_short_positions is not None:
        short_candidates = short_candidates[: max(config.max_short_positions, 0)]
    selected = sorted(long_candidates + short_candidates, key=lambda row: (-row["score"], row["secid"]))
    if config.max_positions is not None:
        selected = selected[: max(config.max_positions, 0)]

    abs_weights = _hyperbolic_weights(len(selected), config.rank_power)
    signed_weights = {}
    for row, weight in zip(selected, abs_weights):
        signed = weight * float(config.max_gross)
        signed_weights[str(row["secid"])] = signed if row["side"] == "long" else -signed

    diagnostics = {
        "selector": config.name,
        "selected_tickers_count": len(signed_weights),
        "long_count": sum(1 for value in signed_weights.values() if value > 0),
        "short_count": sum(1 for value in signed_weights.values() if value < 0),
        "threshold": config.threshold,
        "rank_power": config.rank_power,
        "max_positions": config.max_positions,
        "max_long_positions": config.max_long_positions,
        "max_short_positions": config.max_short_positions,
        "asset_filter": list(config.asset_filter or []),
        "selected_tickers": list(signed_weights),
        "weights": signed_weights,
        "candidate_count_before_caps": len(candidates),
        "candidate_scores": {row["secid"]: row["score"] for row in selected},
    }
    return SelectorDecision(name=config.name, weights=signed_weights, diagnostics=diagnostics)


def _combined_bull_scores(signals: Sequence[SignalRow], signal_weights: Mapping[str, float]) -> dict[str, float]:
    grouped: dict[str, dict[str, SignalRow]] = {}
    for row in signals:
        grouped.setdefault(row.secid, {})[row.signal_name] = row

    out = {}
    for secid, rows in grouped.items():
        numerator = 0.0
        denominator = 0.0
        for signal_name, weight in signal_weights.items():
            weight = float(weight)
            if weight <= 0 or signal_name not in rows:
                continue
            row = rows[signal_name]
            effective_weight = weight * max(float(row.confidence), 0.0)
            if effective_weight <= 0:
                continue
            numerator += effective_weight * float(row.bullish_score)
            denominator += effective_weight
        if denominator > 0:
            out[secid] = numerator / denominator
    return out


def _hyperbolic_weights(n: int, rank_power: float) -> list[float]:
    if n <= 0:
        return []
    raw = [1.0 if rank_power == 0 else (rank + 1) ** (-float(rank_power)) for rank in range(n)]
    total = sum(raw)
    return [value / total for value in raw]

