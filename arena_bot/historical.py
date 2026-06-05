from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import pandas as pd

from .market_data import MarketDataProvider, SavedCandleMarketDataProvider, build_market_features
from .meta_selector import train_daily_lightgbm
from .selectors import build_selector_portfolio
from .signals import MomentumSignalProvider, SignalProvider, latest_signal_scores, with_equity_kronos_fallback
from .storage import StateStore
from .types import Instrument, RuntimeConfig
from .universe import select_universe


@dataclass(frozen=True)
class HistoricalBatchResult:
    requested_intervals: int
    replayed_intervals: int
    first_as_of: str
    last_as_of: str
    market_feature_rows: int
    selector_return_rows: int
    training_rows: int
    cumulative_returns: Mapping[str, float]
    training_metadata: Mapping[str, Any] | None = None


def run_historical_batch(
    *,
    config: RuntimeConfig,
    market_data: MarketDataProvider,
    state: StateStore,
    kronos_provider: SignalProvider,
    intervals: int,
    from_dt: datetime | None = None,
    till_dt: datetime | None = None,
    reset_history: bool = False,
    train_lightgbm: bool = False,
    progress_every: int = 0,
) -> HistoricalBatchResult:
    if reset_history:
        state.clear_training_history()

    timestamps = _replay_timestamps(market_data, config.instruments, from_dt=from_dt, till_dt=till_dt)
    if len(timestamps) < 2:
        raise RuntimeError("not enough historical timestamps to replay")
    requested = max(int(intervals), 1)
    selected = timestamps[-(requested + 1) :]
    pairs = list(zip(selected[:-1], selected[1:]))
    if not pairs:
        raise RuntimeError("not enough historical intervals to replay")

    momentum_provider = MomentumSignalProvider()
    selector_states: dict[str, dict[str, float]] = {selector.name: {} for selector in config.base_selectors}
    market_rows = 0
    return_cells = 0
    selector_nav: dict[str, float] = {}
    for idx, (as_of, next_as_of) in enumerate(pairs, start=1):
        features, returns = _build_interval_training_row(
            as_of=as_of,
            next_as_of=next_as_of,
            config=config,
            market_data=market_data,
            kronos_provider=kronos_provider,
            momentum_provider=momentum_provider,
            selector_states=selector_states,
        )
        as_of_s = as_of.isoformat(timespec="seconds")
        state.save_market_features(as_of_s, features)
        state.append_selector_returns(as_of_s, returns)
        market_rows += 1
        return_cells += len(returns)
        for selector, value in returns.items():
            selector_nav[selector] = selector_nav.get(selector, 1.0) * (1.0 + float(value or 0.0))
        if progress_every > 0 and (idx % progress_every == 0 or idx == len(pairs)):
            cumulative = {selector: nav - 1.0 for selector, nav in selector_nav.items()}
            leader = max(cumulative, key=cumulative.get) if cumulative else ""
            print(
                json.dumps(
                    {
                        "event": "historical_batch_progress",
                        "completed_intervals": idx,
                        "total_intervals": len(pairs),
                        "as_of": as_of_s,
                        "next_as_of": next_as_of.isoformat(timespec="seconds"),
                        "best_selector": leader,
                        "best_cumulative_return": cumulative.get(leader, 0.0) if leader else 0.0,
                    },
                    ensure_ascii=False,
                    default=str,
                ),
                flush=True,
            )

    rows = state.load_lightgbm_training_rows(limit=config.lightgbm.train_lookback_intervals)
    metadata = None
    if train_lightgbm:
        metadata = train_daily_lightgbm(
            rows=rows,
            model_dir=_model_dir(config),
            base_selectors=[selector.name for selector in config.base_selectors],
            min_train_intervals=config.lightgbm.min_train_intervals,
            train_lookback_intervals=config.lightgbm.train_lookback_intervals,
            rank_power=config.lightgbm.rank_power,
            n_estimators=config.lightgbm.n_estimators,
        )

    return HistoricalBatchResult(
        requested_intervals=requested,
        replayed_intervals=len(pairs),
        first_as_of=pairs[0][0].isoformat(timespec="seconds"),
        last_as_of=pairs[-1][0].isoformat(timespec="seconds"),
        market_feature_rows=market_rows,
        selector_return_rows=return_cells,
        training_rows=len(rows),
        cumulative_returns={selector: nav - 1.0 for selector, nav in selector_nav.items()},
        training_metadata=metadata,
    )


def _build_interval_training_row(
    *,
    as_of: datetime,
    next_as_of: datetime,
    config: RuntimeConfig,
    market_data: MarketDataProvider,
    kronos_provider: SignalProvider,
    momentum_provider: SignalProvider,
    selector_states: MutableMapping[str, dict[str, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    snapshots = market_data.snapshots(as_of, config.instruments)
    candles = market_data.candles(as_of, config.instruments)
    metrics = market_data.metrics(as_of, config.instruments)
    universe = select_universe(
        config.instruments,
        snapshots=snapshots,
        metrics=metrics,
        max_equities=config.max_equities,
    )
    selected = universe.instruments
    kronos_rows = tuple(kronos_provider.score(as_of, selected, candles))
    momentum_rows = tuple(momentum_provider.score(as_of, selected, candles))
    signals = with_equity_kronos_fallback(
        as_of=as_of,
        instruments=selected,
        kronos_rows=kronos_rows,
        momentum_rows=momentum_rows,
    )
    signal_scores = latest_signal_scores(signals, "kronos")
    signal_scores.update({k: v for k, v in latest_signal_scores(signals, "momentum").items() if k not in signal_scores})
    features = build_market_features(
        selected_secids=universe.secids,
        snapshots=snapshots,
        metrics=metrics,
        signal_scores=signal_scores,
    )
    selector_decisions = {
        selector.name: build_selector_portfolio(selector, instruments=selected, signals=signals)
        for selector in config.base_selectors
    }
    next_snapshots = market_data.snapshots(next_as_of, config.instruments)
    returns = {}
    for selector_name, decision in selector_decisions.items():
        target_weights = dict(decision.weights)
        returns[selector_name] = _selector_turnover_return(
            selector_states.get(selector_name, {}),
            target_weights,
            snapshots,
            next_snapshots,
            commission_rate=float(config.risk.commission_rate),
        )
        selector_states[selector_name] = target_weights
    return features, returns


def _selector_turnover_return(
    previous_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
    current_snapshots: Mapping[str, Any],
    next_snapshots: Mapping[str, Any],
    *,
    commission_rate: float,
) -> float:
    mark_to_market = 0.0
    turnover_cost = 0.0
    secids = set(previous_weights) | set(target_weights)
    for secid in secids:
        current = current_snapshots.get(secid)
        nxt = next_snapshots.get(secid)
        if current is None:
            continue
        previous_weight = float(previous_weights.get(secid, 0.0) or 0.0)
        target_weight = float(target_weights.get(secid, 0.0) or 0.0)
        delta = target_weight - previous_weight
        turnover_cost += abs(delta) * _one_way_trade_cost(delta, current, commission_rate=commission_rate)

        if target_weight == 0 or nxt is None:
            continue
        current_mid = _snapshot_mid(current)
        next_mid = _snapshot_mid(nxt)
        if current_mid <= 0 or next_mid <= 0:
            continue
        if target_weight > 0:
            mark_to_market += abs(target_weight) * (next_mid / current_mid - 1.0)
        else:
            mark_to_market += abs(target_weight) * (current_mid / next_mid - 1.0)
    return mark_to_market - turnover_cost


def _one_way_trade_cost(delta_weight: float, snapshot: Any, *, commission_rate: float) -> float:
    if delta_weight == 0:
        return 0.0
    mid = _snapshot_mid(snapshot)
    if mid <= 0:
        return float(commission_rate)
    if delta_weight > 0:
        execution = _snapshot_ask(snapshot)
        spread_cost = max(execution / mid - 1.0, 0.0) if execution > 0 else 0.0
    else:
        execution = _snapshot_bid(snapshot)
        spread_cost = max(1.0 - execution / mid, 0.0) if execution > 0 else 0.0
    return spread_cost + float(commission_rate)


def _snapshot_mid(snapshot: Any) -> float:
    last = float(getattr(snapshot, "last_price", 0.0) or 0.0)
    if last > 0:
        return last
    bid = float(getattr(snapshot, "bid", 0.0) or 0.0)
    ask = float(getattr(snapshot, "ask", 0.0) or 0.0)
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return max(bid, ask, 0.0)


def _snapshot_bid(snapshot: Any) -> float:
    bid = float(getattr(snapshot, "bid", 0.0) or 0.0)
    if bid > 0:
        return bid
    last = float(getattr(snapshot, "last_price", 0.0) or 0.0)
    spread = float(getattr(snapshot, "spread_pct", 0.0) or 0.0)
    return last * (1.0 - spread / 2.0) if last > 0 else 0.0


def _snapshot_ask(snapshot: Any) -> float:
    ask = float(getattr(snapshot, "ask", 0.0) or 0.0)
    if ask > 0:
        return ask
    last = float(getattr(snapshot, "last_price", 0.0) or 0.0)
    spread = float(getattr(snapshot, "spread_pct", 0.0) or 0.0)
    return last * (1.0 + spread / 2.0) if last > 0 else 0.0


def _replay_timestamps(
    market_data: MarketDataProvider,
    instruments: Sequence[Instrument],
    *,
    from_dt: datetime | None,
    till_dt: datetime | None,
) -> list[datetime]:
    if isinstance(market_data, SavedCandleMarketDataProvider):
        per_instrument = []
        for instrument in instruments:
            df = market_data._load(instrument.secid)  # Uses the provider's normalized CSV cache.
            if df.empty or "timestamps" not in df.columns:
                continue
            per_instrument.append(set(pd.to_datetime(df["timestamps"]).dropna().dt.floor("s")))
        if not per_instrument:
            return []
        common = set.intersection(*per_instrument)
        out = [pd.Timestamp(ts).to_pydatetime() for ts in sorted(common)]
    else:
        raise RuntimeError("historical-batch currently requires saved_candles market data")

    if from_dt is not None:
        out = [ts for ts in out if ts >= from_dt]
    if till_dt is not None:
        out = [ts for ts in out if ts <= till_dt]
    return out


def _model_dir(config: RuntimeConfig) -> Path:
    model_path = Path(config.lightgbm.model_dir)
    return model_path if model_path.is_absolute() else Path(config.data_dir) / model_path
