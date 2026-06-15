from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from ..config import load_config
from ..kronos_provider import RealKronosSignalProvider
from ..market_data import SavedCandleMarketDataProvider
from ..ranking.scoring import compute_candidate_vectors, instrument_scales_from_history
from ..storage import StateStore
from ..trading_calendar import CacheFirstSessionCalendarProvider
from ..types import Instrument, RuntimeConfig, SignalRow
from .positive_weights import (
    _bid_ask,
    _safe_float,
    _signal_pred_ohlcv,
    _spread_rate,
    execution_net_return,
    load_candidate_cache,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stats = build_session_positive_cache_from_args(args)
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build-positive-cache",
        description="Build or extend positive candidate cache on per-instrument trading-session hours.",
    )
    parser.add_argument("--config", default="configs/universe_v1_may1_14.yaml")
    parser.add_argument("--candles-dir", required=True)
    parser.add_argument("--from", dest="from_date", required=True)
    parser.add_argument("--till", dest="till_date", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--base-cache",
        action="append",
        default=[],
        help="Existing candidate cache to seed the output with. Can be passed multiple times.",
    )
    parser.add_argument(
        "--state-db",
        default="",
        help="SQLite state DB used for Kronos forecast caching. Defaults to <output>.sqlite3.",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate output before copying base caches and computing missing rows.",
    )
    parser.add_argument(
        "--include-crypto-24h",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow crypto instruments on every candle timestamp when their session state is trading.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def build_session_positive_cache_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return build_session_positive_cache(
        config_path=Path(str(args.config)),
        candles_dir=Path(str(args.candles_dir)),
        from_dt=datetime.fromisoformat(str(args.from_date)),
        till_dt=datetime.fromisoformat(str(args.till_date)),
        output_path=Path(str(args.output)),
        base_caches=[Path(str(path)) for path in getattr(args, "base_cache", [])],
        state_db=Path(str(args.state_db)) if str(getattr(args, "state_db", "") or "").strip() else None,
        progress_every=max(int(getattr(args, "progress_every", 0) or 0), 0),
        overwrite=bool(getattr(args, "overwrite", False)),
        include_crypto_24h=bool(getattr(args, "include_crypto_24h", True)),
    )


def build_session_positive_cache(
    *,
    config_path: Path,
    candles_dir: Path,
    from_dt: datetime,
    till_dt: datetime,
    output_path: Path,
    base_caches: Sequence[Path] = (),
    state_db: Path | None = None,
    progress_every: int = 10,
    overwrite: bool = False,
    include_crypto_24h: bool = True,
) -> dict[str, Any]:
    config = load_config(config_path)
    instruments = tuple(instrument for instrument in config.instruments if instrument.enabled)
    if not instruments:
        raise RuntimeError("config has no enabled instruments")

    provider = SavedCandleMarketDataProvider(
        directories=[candles_dir],
        history_rows=int(getattr(config.kronos, "context_rows", 512) or 512),
    )
    state_path = state_db or output_path.with_suffix(".sqlite3")
    state = StateStore(state_path)
    kronos = RealKronosSignalProvider(config=config.kronos, state=state)
    session_calendar = CacheFirstSessionCalendarProvider(config=config.trading_session, state=state)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and output_path.exists():
        output_path.unlink()

    existing_keys = _seed_output_from_base_caches(output_path=output_path, base_caches=base_caches)
    raw_by_secid = {instrument.secid: provider._load(instrument.secid) for instrument in instruments}
    timestamps = _candidate_timestamps(
        raw_by_secid=raw_by_secid,
        from_dt=from_dt,
        till_dt=till_dt,
        interval_minutes=max(int(config.rebalance.decision_interval_minutes), 1),
    )

    rows_written = 0
    skipped_existing = 0
    skipped_no_eligible = 0
    skipped_no_signal = 0
    per_hour: dict[str, int] = {}
    per_asset_class: dict[str, int] = {}
    per_secid: dict[str, int] = {}

    with output_path.open("a", encoding="utf-8") as fh:
        for idx, as_of in enumerate(timestamps, start=1):
            eligible, next_by_secid = _eligible_instruments_for_timestamp(
                as_of=as_of,
                till_dt=till_dt,
                config=config,
                instruments=instruments,
                raw_by_secid=raw_by_secid,
                provider=provider,
                session_calendar=session_calendar,
                existing_keys=existing_keys,
                include_crypto_24h=include_crypto_24h,
            )
            if not eligible:
                skipped_no_eligible += 1
                _print_progress(progress_every, idx, len(timestamps), as_of, rows_written)
                continue

            candles = provider.candles(as_of, eligible)
            signals = {
                row.secid: row
                for row in kronos.score(as_of, eligible, candles)
                if row.signal_name == "kronos"
            }
            if not signals:
                skipped_no_signal += 1
                _print_progress(progress_every, idx, len(timestamps), as_of, rows_written)
                continue

            snapshots = provider.snapshots(as_of, eligible)
            metrics = provider.metrics(as_of, eligible)
            for instrument in eligible:
                if (as_of.isoformat(timespec="seconds"), instrument.secid, "long") in existing_keys:
                    skipped_existing += 2
                    continue
                signal = signals.get(instrument.secid)
                pred = _signal_pred_ohlcv(signal)
                if pred is None:
                    continue
                rows = _candidate_rows_for_instrument(
                    config=config,
                    provider=provider,
                    instrument=instrument,
                    as_of=as_of,
                    next_ts=next_by_secid[instrument.secid],
                    pred=pred,
                    candles=candles,
                    snapshots=snapshots,
                    metrics=metrics,
                )
                for row in rows:
                    key = (str(row["as_of"]), str(row["secid"]), str(row["side"]))
                    if key in existing_keys:
                        skipped_existing += 1
                        continue
                    fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    existing_keys.add(key)
                    rows_written += 1
                    per_hour[str(row["as_of"])[11:13]] = per_hour.get(str(row["as_of"])[11:13], 0) + 1
                    per_asset_class[str(row["asset_class"])] = per_asset_class.get(str(row["asset_class"]), 0) + 1
                    per_secid[str(row["secid"])] = per_secid.get(str(row["secid"]), 0) + 1

            _print_progress(progress_every, idx, len(timestamps), as_of, rows_written)

    total_rows = len(load_candidate_cache(output_path))
    return {
        "output": str(output_path),
        "state_db": str(state_path),
        "timestamps_considered": len(timestamps),
        "rows_written": rows_written,
        "total_rows": total_rows,
        "skipped_existing": skipped_existing,
        "skipped_no_eligible_timestamps": skipped_no_eligible,
        "skipped_no_signal_timestamps": skipped_no_signal,
        "per_hour_written": dict(sorted(per_hour.items())),
        "per_asset_class_written": dict(sorted(per_asset_class.items())),
        "per_secid_written": dict(sorted(per_secid.items())),
    }


def _seed_output_from_base_caches(*, output_path: Path, base_caches: Sequence[Path]) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    if output_path.exists():
        for row in load_candidate_cache(output_path):
            keys.add(_row_key(row))
        return keys

    with output_path.open("w", encoding="utf-8") as out:
        for cache in base_caches:
            if not cache.exists():
                continue
            for row in load_candidate_cache(cache):
                key = _row_key(row)
                if key in keys:
                    continue
                out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                keys.add(key)
    return keys


def _candidate_timestamps(
    *,
    raw_by_secid: Mapping[str, pd.DataFrame],
    from_dt: datetime,
    till_dt: datetime,
    interval_minutes: int,
) -> list[datetime]:
    values: set[pd.Timestamp] = set()
    for raw in raw_by_secid.values():
        if raw.empty or "timestamps" not in raw:
            continue
        ts = pd.to_datetime(raw["timestamps"], errors="coerce").dropna().dt.floor("s")
        values.update(value for value in ts if pd.Timestamp(from_dt) <= value <= pd.Timestamp(till_dt))
    out = [pd.Timestamp(value).to_pydatetime() for value in sorted(values)]
    return [value for value in out if _is_rebalance_tick(value, interval_minutes=interval_minutes)]


def _eligible_instruments_for_timestamp(
    *,
    as_of: datetime,
    till_dt: datetime,
    config: RuntimeConfig,
    instruments: Sequence[Instrument],
    raw_by_secid: Mapping[str, pd.DataFrame],
    provider: SavedCandleMarketDataProvider,
    session_calendar: CacheFirstSessionCalendarProvider,
    existing_keys: set[tuple[str, str, str]],
    include_crypto_24h: bool,
) -> tuple[tuple[Instrument, ...], dict[str, datetime]]:
    candles = provider.candles(as_of, instruments)
    session = session_calendar.session_state(as_of, instruments, candles=candles)
    session_by_secid = session.get("_session_by_secid") or session.get("session_by_secid") or {}
    eligible: list[Instrument] = []
    next_by_secid: dict[str, datetime] = {}
    as_of_s = as_of.isoformat(timespec="seconds")
    pred_len = max(int(config.kronos.pred_len), 1)
    interval = timedelta(minutes=max(int(config.rebalance.decision_interval_minutes), 1))
    for instrument in instruments:
        raw = raw_by_secid.get(instrument.secid, pd.DataFrame())
        if raw.empty or not _has_exact_timestamp(raw, as_of):
            continue
        if (as_of_s, instrument.secid, "long") in existing_keys and (as_of_s, instrument.secid, "short") in existing_keys:
            continue
        row = session_by_secid.get(instrument.secid, {}) if isinstance(session_by_secid, Mapping) else {}
        if not bool(row.get("entry_allowed", True)):
            continue
        if instrument.asset_class == "crypto" and not include_crypto_24h:
            continue
        next_ts = _next_instrument_timestamp(raw, as_of)
        if next_ts is None or next_ts > till_dt:
            continue
        max_target_time = row.get("_kronos_cutoff_dt") if isinstance(row, Mapping) else None
        if isinstance(max_target_time, datetime):
            forecast_target = _forecast_target_time(as_of=as_of, raw=raw, pred_len=pred_len, tzinfo=max_target_time.tzinfo)
            recheck_target = _as_tz(as_of, max_target_time.tzinfo) + interval
            if forecast_target > max_target_time or recheck_target > max_target_time:
                continue
        eligible.append(instrument)
        next_by_secid[instrument.secid] = next_ts
    return tuple(eligible), next_by_secid


def _candidate_rows_for_instrument(
    *,
    config: RuntimeConfig,
    provider: SavedCandleMarketDataProvider,
    instrument: Instrument,
    as_of: datetime,
    next_ts: datetime,
    pred: Mapping[str, float],
    candles: Mapping[str, pd.DataFrame],
    snapshots: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    snapshot = snapshots.get(instrument.secid)
    metric = metrics.get(instrument.secid)
    bid, ask = _bid_ask(snapshot)
    if bid <= 0.0 or ask <= 0.0 or ask < bid:
        return []
    future_snapshot = provider.snapshots(next_ts, [instrument]).get(instrument.secid)
    future_bid, future_ask = _bid_ask(future_snapshot)
    if future_bid <= 0.0 or future_ask <= 0.0 or future_ask < future_bid:
        return []

    spread_rate = _spread_rate(snapshot=snapshot, metric=metric)
    commission_rate = max(float(config.risk.commission_rate), 0.0)
    slippage_rate = max(spread_rate, 0.0) * max(float(config.risk.slippage_spread_multiplier), 0.0)
    scales = instrument_scales_from_history(
        candles.get(instrument.secid),
        snapshot=snapshot,
        metric=metric,
    )
    current_mid = (bid + ask) / 2.0
    future_mid = (future_bid + future_ask) / 2.0
    out: list[dict[str, Any]] = []
    for side in ("long", "short"):
        candidate = {
            "secid": instrument.secid,
            "direction": side,
            "side": side,
            "bid": bid,
            "ask": ask,
            "pred_open": pred["open"],
            "pred_high": pred["high"],
            "pred_low": pred["low"],
            "pred_close": pred["close"],
            "commission_rate": commission_rate,
            "slippage_rate": slippage_rate,
        }
        try:
            vectors = compute_candidate_vectors(candidate, scales)
        except Exception:
            continue
        out.append(
            {
                "as_of": as_of.isoformat(timespec="seconds"),
                "trade_date": as_of.date().isoformat(),
                "secid": instrument.secid,
                "asset_class": instrument.asset_class,
                "side": side,
                "positive_vector": vectors["positive_vector"],
                "raw_vector_metrics": vectors["raw_metrics"],
                "current_bid": bid,
                "current_ask": ask,
                "current_mid": current_mid,
                "pred_open": pred["open"],
                "pred_high": pred["high"],
                "pred_low": pred["low"],
                "pred_close": pred["close"],
                "commission_rate": commission_rate,
                "slippage_rate": slippage_rate,
                "future_as_of": next_ts.isoformat(timespec="seconds"),
                "future_bid": future_bid,
                "future_ask": future_ask,
                "future_mid": future_mid,
                "realized_net_return": execution_net_return(
                    side=side,
                    entry_bid=bid,
                    entry_ask=ask,
                    exit_bid=future_bid,
                    exit_ask=future_ask,
                    commission_rate=commission_rate,
                    slippage_rate=slippage_rate,
                ),
            }
        )
    return out


def _forecast_target_time(*, as_of: datetime, raw: pd.DataFrame, pred_len: int, tzinfo: Any) -> datetime:
    step = timedelta(hours=1)
    current = _as_tz(as_of, tzinfo)
    if not raw.empty and "timestamps" in raw:
        ts = pd.to_datetime(raw["timestamps"], errors="coerce").dropna()
        ts = ts[ts <= pd.Timestamp(as_of)]
        if not ts.empty:
            current = _timestamp_to_datetime(ts.iloc[-1], tzinfo)
            if len(ts) >= 2:
                prev = _timestamp_to_datetime(ts.iloc[-2], tzinfo)
                delta = current - prev
                if delta.total_seconds() > 0:
                    step = delta
    return current + step * max(int(pred_len), 1)


def _is_rebalance_tick(value: datetime, *, interval_minutes: int) -> bool:
    if value.second != 0 or value.microsecond != 0:
        return False
    minute_of_day = value.hour * 60 + value.minute
    return minute_of_day % max(int(interval_minutes), 1) == 0


def _has_exact_timestamp(raw: pd.DataFrame, as_of: datetime) -> bool:
    if raw.empty or "timestamps" not in raw:
        return False
    return bool((raw["timestamps"] == pd.Timestamp(as_of)).any())


def _next_instrument_timestamp(raw: pd.DataFrame, as_of: datetime) -> datetime | None:
    if raw.empty or "timestamps" not in raw:
        return None
    later = raw["timestamps"][raw["timestamps"] > pd.Timestamp(as_of)]
    if later.empty:
        return None
    return pd.Timestamp(later.iloc[0]).to_pydatetime()


def _timestamp_to_datetime(value: Any, tzinfo: Any) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.to_pydatetime().replace(tzinfo=tzinfo)
    return ts.to_pydatetime().astimezone(tzinfo)


def _as_tz(value: datetime, tzinfo: Any) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=tzinfo)
    return value.astimezone(tzinfo)


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("as_of") or ""), str(row.get("secid") or ""), str(row.get("side") or ""))


def _print_progress(progress_every: int, idx: int, total: int, as_of: datetime, rows_written: int) -> None:
    if progress_every <= 0:
        return
    if idx % progress_every != 0 and idx != total:
        return
    print(
        json.dumps(
            {
                "event": "build_positive_cache_progress",
                "completed_timestamps": idx,
                "total_timestamps": total,
                "as_of": as_of.isoformat(timespec="seconds"),
                "rows_written": rows_written,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
