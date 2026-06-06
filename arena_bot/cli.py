from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .arenago import ArenaGoClient
from .config import load_config
from .crypto_download import download_binance_klines_for_symbols
from .historical import run_historical_batch
from .kronos_provider import RealKronosSignalProvider
from .logging import JsonlLogger
from .market_data import SavedCandleMarketDataProvider, StaticMarketDataProvider
from .meta_selector import train_daily_lightgbm
from .moex_download import download_candles_for_instruments, download_futures_candles_for_roots
from .runtime import RuntimeEngine
from .storage import StateStore
from .types import MarketMetrics, MarketSnapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arena-bot")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("run-once", "diagnose", "run-live", "train-lightgbm"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/default.yaml")
        p.add_argument("--as-of", default="")

    p = sub.add_parser("historical-batch")
    p.add_argument("--config", default="configs/saved_candles_paper.yaml")
    p.add_argument("--intervals", type=int, default=96)
    p.add_argument("--from", dest="from_date", default="")
    p.add_argument("--till", dest="till_date", default="")
    p.add_argument("--train-lightgbm", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--reset-history", action="store_true")
    p.add_argument("--kronos-refresh-cache", action="store_true")
    p.add_argument("--progress-every", type=int, default=0)

    p = sub.add_parser("download-candles")
    p.add_argument("--config", default="configs/saved_candles_paper.yaml")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--from", dest="from_date", required=True)
    p.add_argument("--till", dest="till_date", required=True)
    p.add_argument("--interval", type=int, default=60)
    p.add_argument("--source-interval", type=int, default=60)
    p.add_argument("--asset-class", choices=("equity", "future", "crypto", "all"), default="equity")
    p.add_argument("--futures-roots", nargs="*", default=None)
    p.add_argument("--crypto-symbols", nargs="*", default=None)
    p.add_argument("--binance-interval", default="1h")
    p.add_argument("--timestamp-offset-hours", type=int, default=3)

    args = parser.parse_args(argv)
    if args.command == "run-once":
        return _run_once(args)
    if args.command == "diagnose":
        return _diagnose(args)
    if args.command == "run-live":
        return _run_live(args)
    if args.command == "train-lightgbm":
        return _train_lightgbm(args)
    if args.command == "historical-batch":
        return _historical_batch(args)
    if args.command == "download-candles":
        return _download_candles(args)
    raise AssertionError(args.command)


def run_once_main() -> None:
    raise SystemExit(main(["run-once", *_script_args()]))


def diagnose_main() -> None:
    raise SystemExit(main(["diagnose", *_script_args()]))


def run_live_main() -> None:
    raise SystemExit(main(["run-live", *_script_args()]))


def train_lightgbm_main() -> None:
    raise SystemExit(main(["train-lightgbm", *_script_args()]))


def _run_once(args: argparse.Namespace) -> int:
    engine = _build_engine(args.config)
    result = engine.run_once(_parse_as_of(args.as_of))
    print(json.dumps({"decision_id": result.decision_id, "target_weights": result.target_weights}, ensure_ascii=False, indent=2, default=str))
    return 0


def _diagnose(args: argparse.Namespace) -> int:
    engine = _build_engine(args.config)
    result = engine.run_once(_parse_as_of(args.as_of))
    print(
        json.dumps(
            {
                "decision_id": result.decision_id,
                "selector_diagnostics": result.selector_diagnostics,
                "blend_diagnostics": result.blend_diagnostics,
                "orders": [order.__dict__ for order in result.orders],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    return 0


def _run_live(args: argparse.Namespace) -> int:
    engine = _build_engine(args.config)
    interval = max(int(getattr(engine.config.rebalance, "exit_interval_minutes", 1)), 1)
    last_run: datetime | None = None
    while True:
        now = _floor_to_interval_minute(datetime.now(), interval)
        if last_run is None or now > last_run:
            engine.run_once(now)
            last_run = now
        next_run = now + timedelta(minutes=interval)
        sleep_seconds = max((next_run - datetime.now()).total_seconds(), 0.5)
        time.sleep(sleep_seconds)


def _train_lightgbm(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    state = StateStore(Path(config.data_dir) / "arena_state.sqlite3")
    rows = state.load_lightgbm_training_rows(limit=config.lightgbm.train_lookback_intervals)
    metadata = train_daily_lightgbm(
        rows=rows,
        model_dir=_model_dir(config),
        base_selectors=[selector.name for selector in config.base_selectors],
        min_train_intervals=config.lightgbm.min_train_intervals,
        train_lookback_intervals=config.lightgbm.train_lookback_intervals,
        rank_power=config.lightgbm.rank_power,
        n_estimators=config.lightgbm.n_estimators,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, default=str))
    return 0


def _floor_to_interval_minute(value: datetime, interval_minutes: int) -> datetime:
    interval = max(int(interval_minutes), 1)
    minute_of_day = value.hour * 60 + value.minute
    floored = minute_of_day - (minute_of_day % interval)
    return value.replace(hour=floored // 60, minute=floored % 60, second=0, microsecond=0)


def _historical_batch(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if config.live_orders:
        config = replace(config, mode="paper")
    data_dir = Path(config.data_dir)
    state = StateStore(data_dir / "arena_state.sqlite3")
    market_data = _static_market_data_from_config(args.config)
    kronos = RealKronosSignalProvider(
        config=config.kronos,
        state=state,
        refresh_cache=bool(args.kronos_refresh_cache),
    )
    result = run_historical_batch(
        config=config,
        market_data=market_data,
        state=state,
        kronos_provider=kronos,
        intervals=int(args.intervals),
        from_dt=_parse_as_of(args.from_date),
        till_dt=_parse_as_of(args.till_date),
        reset_history=bool(args.reset_history),
        train_lightgbm=bool(args.train_lightgbm),
        progress_every=int(args.progress_every),
    )
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2, default=str))
    return 0


def _download_candles(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    asset_classes = {"equity", "future", "crypto"} if args.asset_class == "all" else {str(args.asset_class)}
    payload: dict[str, object] = {"out_dir": args.out_dir, "rows": {}}

    if "equity" in asset_classes:
        equity_counts = download_candles_for_instruments(
            secids=[instrument.secid for instrument in config.instruments if instrument.asset_class == "equity"],
            out_dir=args.out_dir,
            from_date=args.from_date,
            till_date=args.till_date,
            interval=int(args.interval),
            source_interval=int(args.source_interval),
            engine="stock",
            market="shares",
        )
        payload["rows"] = {**dict(payload["rows"]), "equity": equity_counts}

    if "future" in asset_classes:
        roots = args.futures_roots
        if roots:
            future_counts, resolved = download_futures_candles_for_roots(
                roots=roots,
                out_dir=args.out_dir,
                from_date=args.from_date,
                till_date=args.till_date,
                interval=int(args.interval),
                source_interval=int(args.source_interval),
            )
            payload["futures_resolved"] = resolved
        else:
            future_counts = download_candles_for_instruments(
                secids=[instrument.secid for instrument in config.instruments if instrument.asset_class == "future"],
                out_dir=args.out_dir,
                from_date=args.from_date,
                till_date=args.till_date,
                interval=int(args.interval),
                source_interval=int(args.source_interval),
                engine="futures",
                market="forts",
            )
        payload["rows"] = {**dict(payload["rows"]), "future": future_counts}

    if "crypto" in asset_classes:
        symbols = args.crypto_symbols or [instrument.secid for instrument in config.instruments if instrument.asset_class == "crypto"]
        crypto_counts = download_binance_klines_for_symbols(
            symbols=symbols,
            out_dir=args.out_dir,
            from_date=args.from_date,
            till_date=args.till_date,
            interval=str(args.binance_interval),
            timestamp_offset_hours=int(args.timestamp_offset_hours),
        )
        payload["rows"] = {**dict(payload["rows"]), "crypto": crypto_counts}

    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


def _build_engine(config_path: str) -> RuntimeEngine:
    config = load_config(config_path)
    data_dir = Path(config.data_dir)
    state = StateStore(data_dir / "arena_state.sqlite3")
    logger = JsonlLogger(data_dir / "logs")
    market_data = _static_market_data_from_config(config_path)
    kronos_provider = None
    if config.kronos.enabled:
        kronos_provider = RealKronosSignalProvider(config=config.kronos, state=state)
    client = None
    if config.live_orders:
        client = ArenaGoClient(
            os.environ.get("SANDBOX_API_KEY", ""),
            base_url=os.environ.get("ARENA_BASE_URL", "https://arenago.ru"),
        )
    return RuntimeEngine(
        config=config,
        market_data=market_data,
        kronos_provider=kronos_provider,
        state=state,
        logger=logger,
        arenago_client=client,
    )


def _static_market_data_from_config(config_path: str) -> StaticMarketDataProvider:
    import yaml

    path = Path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    raw_market = raw.get("market_data", {}) if isinstance(raw, dict) else {}
    saved = raw_market.get("saved_candles") or {}
    if saved.get("directories"):
        return SavedCandleMarketDataProvider(
            directories=[str(path) for path in saved.get("directories")],
            filename_patterns=tuple(saved.get("filename_patterns") or ("candles_{secid}.csv", "candles_1m_{secid}.csv")),
            history_rows=int(saved.get("history_rows", 512)),
        )
    snapshots = {
        secid: MarketSnapshot(
            secid=secid,
            last_price=float(row.get("last_price", 0.0)),
            bid=float(row.get("bid", 0.0)),
            ask=float(row.get("ask", 0.0)),
            volume_value=float(row.get("volume_value", 0.0)),
            source="config",
        )
        for secid, row in (raw_market.get("snapshots") or {}).items()
    }
    metrics = {
        secid: MarketMetrics(
            secid=secid,
            realized_volatility=float(row.get("realized_volatility", 0.0)),
            atr_pct=float(row.get("atr_pct", 0.0)),
            volume_value=float(row.get("volume_value", 0.0)),
            spread_pct=float(row.get("spread_pct", 1.0)),
            missing_candles=int(row.get("missing_candles", 0)),
            candle_count=int(row.get("candle_count", 0)),
        )
        for secid, row in (raw_market.get("metrics") or {}).items()
    }
    candles = {}
    for secid, rows in (raw_market.get("candles") or {}).items():
        candles[secid] = pd.DataFrame(rows)
    return StaticMarketDataProvider(snapshot_rows=snapshots, metric_rows=metrics, candle_rows=candles)


def _parse_as_of(value: str) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _script_args() -> list[str]:
    import sys

    return sys.argv[1:]


def _model_dir(config) -> Path:
    model_path = Path(config.lightgbm.model_dir)
    return model_path if model_path.is_absolute() else Path(config.data_dir) / model_path


if __name__ == "__main__":
    raise SystemExit(main())
