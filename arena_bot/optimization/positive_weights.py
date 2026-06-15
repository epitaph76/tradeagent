from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, time
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from ..cli import _static_market_data_from_config
from ..config import load_config
from ..kronos_provider import RealKronosSignalProvider
from ..market_data import SavedCandleMarketDataProvider
from ..ranking.scoring import (
    BASELINE_RISK_WEIGHTS,
    POSITIVE_METRICS,
    compute_candidate_vectors,
    cosine_strength_score,
    instrument_scales_from_history,
    save_instrument_weights_yaml,
)
from ..runtime_backtest import _runtime_replay_timestamps
from ..storage import StateStore
from ..types import Instrument, RuntimeConfig, SignalRow

STAGE = "global_positive_v1"
DEFAULT_CACHE_NAME = "positive_candidates.jsonl"
ECONOMIC_POSITIVE_PRIOR = {
    "net_edge_score": 0.22,
    "close_score": 0.16,
    "rr_score": 0.14,
    "edge_risk_quality": 0.13,
    "mae_score": 0.12,
    "edge_z_score": 0.08,
    "candle_quality": 0.07,
    "wick_score": 0.05,
    "body_score": 0.03,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="optimize-positive-weights",
        description="Optimize global positive vector weights and a positive threshold without using risk filtering.",
    )
    parser.add_argument("--config", default="configs/universe_v1_may1_14.yaml")
    parser.add_argument("--from", dest="from_date", required=True)
    parser.add_argument("--till", dest="till_date", required=True)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--study-db", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--candidate-cache", default="")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--positive-threshold-min", type=float, default=0.35)
    parser.add_argument("--positive-threshold-max", type=float, default=0.80)
    return parser.parse_args(list(argv) if argv is not None else None)


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is required for optimization. Install with: python -m pip install -e .[optimize]") from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    config_path = Path(str(args.config))
    from_dt = datetime.fromisoformat(str(args.from_date))
    till_dt = datetime.fromisoformat(str(args.till_date))
    trials = max(int(args.trials), 1)
    fixed_top_k = _resolve_fixed_top_k(args.top_k)
    positive_threshold_min, positive_threshold_max = validate_positive_threshold_bounds(
        args.positive_threshold_min,
        args.positive_threshold_max,
    )
    cache_path = _candidate_cache_path(args)

    rebuilt_cache = False
    if bool(args.rebuild_cache) or not cache_path.exists():
        build_candidate_cache(
            config_path=config_path,
            from_dt=from_dt,
            till_dt=till_dt,
            cache_path=cache_path,
        )
        rebuilt_cache = True
    rows = load_candidate_cache(cache_path)
    if not rows and not rebuilt_cache:
        build_candidate_cache(
            config_path=config_path,
            from_dt=from_dt,
            till_dt=till_dt,
            cache_path=cache_path,
        )
        rows = load_candidate_cache(cache_path)
    if not rows:
        raise RuntimeError(f"candidate cache is empty: {cache_path}")

    study_db = Path(str(args.study_db))
    study_db.parent.mkdir(parents=True, exist_ok=True)
    storage_url = "sqlite:///" + study_db.resolve().as_posix()
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="maximize",
        study_name=STAGE,
        storage=storage_url,
        load_if_exists=True,
        sampler=sampler,
    )

    def objective(trial: Any) -> float:
        raw = {metric: trial.suggest_float(f"pos_raw_{metric}", -3.0, 3.0) for metric in POSITIVE_METRICS}
        weights = softmax_weights(raw, POSITIVE_METRICS)
        threshold = float(trial.suggest_float("positive_threshold", positive_threshold_min, positive_threshold_max))
        top_k = fixed_top_k if fixed_top_k is not None else int(trial.suggest_int("top_k", 1, 2))
        return float(evaluate_candidate_rows(rows, weights, positive_threshold=threshold, top_k=top_k)["objective"])

    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best_params = dict(study.best_params)
    positive_weights = weights_from_best_params(best_params)
    positive_threshold = float(best_params.get("positive_threshold", 0.0))
    top_k = fixed_top_k if fixed_top_k is not None else int(best_params.get("top_k", 1))
    evaluation = evaluate_candidate_rows(
        rows,
        positive_weights,
        positive_threshold=positive_threshold,
        top_k=top_k,
        include_bucket_report=True,
    )

    config = load_config(config_path)
    instrument_weights = global_instrument_weights(
        config=config,
        positive_weights=positive_weights,
        positive_threshold=positive_threshold,
    )
    output_path = Path(str(args.output))
    save_instrument_weights_yaml(instrument_weights, output_path)

    report = {
        "stage": STAGE,
        "best_value": float(study.best_value),
        "best_params": best_params,
        "economic_positive_prior": normalized_positive_prior(),
        "positive_weights": positive_weights,
        "positive_threshold": positive_threshold,
        "positive_threshold_min": positive_threshold_min,
        "positive_threshold_max": positive_threshold_max,
        "top_k": top_k,
        "fold_metrics": evaluation["fold_metrics"],
        "bucket_report": evaluation["bucket_report"],
        "trade_count": evaluation["trade_count"],
        "avg_trade_return": evaluation["avg_trade_return"],
        "winrate": evaluation["winrate"],
        "profit_factor": evaluation["profit_factor"],
        "max_drawdown": evaluation["max_drawdown"],
        "daily_return_mean": evaluation["daily_return_mean"],
        "daily_return_std": evaluation["daily_return_std"],
        "notes": {
            "risk_vector_used": False,
            "risk_threshold_written_for_compatibility": 1.0,
        },
    }
    report_path = Path(str(args.report))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"stage": STAGE, "best_value": float(study.best_value), "output": str(output_path), "report": str(report_path)}, ensure_ascii=False))
    return report


def normalize_named_weights(raw: Mapping[str, Any], metrics: Sequence[str]) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        raise ValueError("named weights must be a mapping")
    missing = [metric for metric in metrics if metric not in raw]
    if missing:
        raise ValueError(f"missing named weights: {', '.join(missing)}")
    values = {metric: _finite_float(raw.get(metric), name=metric) for metric in metrics}
    invalid = [metric for metric, value in values.items() if value < 0.0]
    if invalid:
        raise ValueError(f"named weights must be non-negative: {', '.join(invalid)}")
    total = sum(values.values())
    if total <= 0.0:
        raise ValueError("named weights total must be positive")
    return {metric: values[metric] / total for metric in metrics}


def normalized_positive_prior() -> dict[str, float]:
    return normalize_named_weights(ECONOMIC_POSITIVE_PRIOR, POSITIVE_METRICS)


def softmax_weights(
    raw_values: Mapping[str, float] | Sequence[float],
    metrics: Sequence[str] = POSITIVE_METRICS,
    *,
    prior: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    values: list[float] = []
    if isinstance(raw_values, Mapping):
        for metric in metrics:
            if metric not in raw_values:
                raise ValueError(f"missing raw weight: {metric}")
            values.append(_finite_float(raw_values[metric], name=metric))
    else:
        if len(raw_values) != len(metrics):
            raise ValueError("raw weight sequence length must match metrics")
        values = [_finite_float(value, name=str(metric)) for metric, value in zip(metrics, raw_values)]
    if not values:
        raise ValueError("metrics must not be empty")
    prior_weights = _softmax_prior_weights(metrics, prior=prior)
    logits = [math.log(prior_weights[metric]) + values[idx] for idx, metric in enumerate(metrics)]
    shifted = [value - max(logits) for value in logits]
    exps = [math.exp(value) for value in shifted]
    total = sum(exps)
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("softmax total must be positive and finite")
    return {metric: exps[idx] / total for idx, metric in enumerate(metrics)}


def weights_from_best_params(best_params: Mapping[str, Any]) -> dict[str, float]:
    raw = {metric: float(best_params[f"pos_raw_{metric}"]) for metric in POSITIVE_METRICS}
    return softmax_weights(raw, POSITIVE_METRICS)


def validate_positive_threshold_bounds(min_value: Any, max_value: Any) -> tuple[float, float]:
    lower = _safe_float(min_value, default=float("nan"))
    upper = _safe_float(max_value, default=float("nan"))
    if not math.isfinite(lower) or not math.isfinite(upper) or lower < 0.0 or upper > 1.0 or lower > upper:
        raise ValueError("positive threshold bounds must satisfy 0.0 <= min <= max <= 1.0")
    return lower, upper


def _softmax_prior_weights(metrics: Sequence[str], *, prior: Mapping[str, Any] | None) -> dict[str, float]:
    if prior is not None:
        weights = normalize_named_weights(prior, metrics)
    elif tuple(metrics) == tuple(POSITIVE_METRICS):
        weights = normalized_positive_prior()
    else:
        weights = {metric: 1.0 / len(metrics) for metric in metrics}
    non_positive = [metric for metric, value in weights.items() if value <= 0.0]
    if non_positive:
        raise ValueError(f"softmax prior weights must be positive: {', '.join(non_positive)}")
    return weights


def global_instrument_weights(
    *,
    config: RuntimeConfig,
    positive_weights: Mapping[str, float],
    positive_threshold: float,
) -> dict[str, dict[str, Any]]:
    threshold = _validate_unit_interval(positive_threshold, name="positive_threshold")
    return {
        instrument.secid: {
            "positive_weights": {metric: float(positive_weights[metric]) for metric in POSITIVE_METRICS},
            "positive_threshold": threshold,
            "risk_weights": dict(BASELINE_RISK_WEIGHTS),
            "risk_threshold": 1.0,
        }
        for instrument in config.instruments
        if instrument.enabled
    }


def build_candidate_cache(
    *,
    config_path: Path,
    from_dt: datetime,
    till_dt: datetime,
    cache_path: Path,
) -> int:
    config = load_config(config_path)
    market_data = _static_market_data_from_config(str(config_path))
    if not isinstance(market_data, SavedCandleMarketDataProvider):
        raise RuntimeError("positive optimizer requires market_data.saved_candles")

    instruments = tuple(instrument for instrument in config.instruments if instrument.enabled)
    if not instruments:
        raise RuntimeError("config has no enabled instruments")
    timestamps = _runtime_replay_timestamps(market_data, instruments, from_dt=from_dt, till_dt=till_dt)
    decision_times = [ts for ts in timestamps[:-1] if _is_decision_timestamp(ts, config=config)]
    state = StateStore(Path(config.data_dir) / "arena_state.sqlite3")
    kronos = RealKronosSignalProvider(config=config.kronos, state=state)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with cache_path.open("w", encoding="utf-8") as fh:
        for as_of in decision_times:
            eligible = []
            next_by_secid: dict[str, datetime] = {}
            for instrument in instruments:
                raw = market_data._load(instrument.secid)
                if raw.empty or not _has_exact_timestamp(raw, as_of):
                    continue
                next_ts = _next_instrument_timestamp(raw, as_of)
                if next_ts is None or next_ts > till_dt:
                    continue
                eligible.append(instrument)
                next_by_secid[instrument.secid] = next_ts
            if not eligible:
                continue
            eligible_instruments = tuple(eligible)
            candles = market_data.candles(as_of, eligible_instruments)
            signals = {
                row.secid: row
                for row in kronos.score(as_of, eligible_instruments, candles)
                if row.signal_name == "kronos"
            }
            if not signals:
                continue
            snapshots = market_data.snapshots(as_of, eligible_instruments)
            metrics = market_data.metrics(as_of, eligible_instruments)
            for instrument in eligible_instruments:
                next_ts = next_by_secid[instrument.secid]
                signal = signals.get(instrument.secid)
                pred = _signal_pred_ohlcv(signal)
                if pred is None:
                    continue
                snapshot = snapshots.get(instrument.secid)
                metric = metrics.get(instrument.secid)
                bid, ask = _bid_ask(snapshot)
                if bid <= 0.0 or ask <= 0.0 or ask < bid:
                    continue
                future_snapshot = market_data.snapshots(next_ts, [instrument]).get(instrument.secid)
                future_bid, future_ask = _bid_ask(future_snapshot)
                if future_bid <= 0.0 or future_ask <= 0.0 or future_ask < future_bid:
                    continue
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
                    row = {
                        "as_of": as_of.isoformat(timespec="seconds"),
                        "trade_date": _trade_date(as_of, config=config),
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
                    fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    rows_written += 1
    return rows_written


def load_candidate_cache(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, Mapping) and isinstance(row.get("positive_vector"), Mapping):
                out.append(dict(row))
    return out


def evaluate_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
    positive_weights: Mapping[str, float],
    *,
    positive_threshold: float,
    top_k: int,
    include_bucket_report: bool = False,
) -> dict[str, Any]:
    threshold = _validate_unit_interval(positive_threshold, name="positive_threshold")
    top_k = max(int(top_k), 1)
    grouped: dict[str, list[tuple[int, Mapping[str, Any], float]]] = defaultdict(list)
    scored_rows: list[tuple[int, Mapping[str, Any], float]] = []
    trade_dates = sorted({str(row.get("trade_date") or str(row.get("as_of", ""))[:10]) for row in rows})

    for idx, row in enumerate(rows):
        vector = row.get("positive_vector")
        if not isinstance(vector, Mapping):
            continue
        score = cosine_strength_score(vector, positive_weights)
        as_of = str(row.get("as_of") or "")
        scored = (idx, row, score)
        grouped[as_of].append(scored)
        scored_rows.append(scored)

    daily_factors = {day: 1.0 for day in trade_dates}
    daily_trade_returns: dict[str, list[float]] = {day: [] for day in trade_dates}
    selected_indices: set[int] = set()
    trade_returns: list[float] = []

    for as_of in sorted(grouped):
        by_secid: dict[str, tuple[int, Mapping[str, Any], float]] = {}
        for idx, row, score in grouped[as_of]:
            if score < threshold:
                continue
            secid = str(row.get("secid") or "")
            current = by_secid.get(secid)
            if current is None or _candidate_sort_key(idx, row, score) < _candidate_sort_key(*current):
                by_secid[secid] = (idx, row, score)
        selected = sorted(by_secid.values(), key=lambda item: _candidate_sort_key(*item))[:top_k]
        if not selected:
            continue
        selected_returns = [_finite_row_return(row) for _, row, _ in selected]
        timestamp_return = statistics.fmean(selected_returns)
        day = str(selected[0][1].get("trade_date") or str(as_of)[:10])
        daily_factors.setdefault(day, 1.0)
        daily_trade_returns.setdefault(day, [])
        daily_factors[day] *= 1.0 + timestamp_return
        daily_trade_returns[day].extend(selected_returns)
        trade_returns.extend(selected_returns)
        selected_indices.update(idx for idx, _, _ in selected)

    fold_metrics = []
    daily_returns = []
    for fold_idx, day in enumerate(sorted(daily_factors), start=1):
        values = daily_trade_returns.get(day, [])
        daily_return = daily_factors[day] - 1.0
        daily_returns.append(daily_return)
        fold_metrics.append(
            {
                "fold": fold_idx,
                "train_days": fold_idx - 1,
                "validate_date": day,
                "daily_return": daily_return,
                "trade_count": len(values),
                "avg_trade_return": statistics.fmean(values) if values else 0.0,
                "winrate": _winrate(values),
            }
        )

    daily_mean = statistics.fmean(daily_returns) if daily_returns else 0.0
    daily_std = statistics.pstdev(daily_returns) if len(daily_returns) > 1 else 0.0
    max_drawdown = _max_drawdown(daily_returns)
    trading_days = max(len(trade_dates), 1)
    trades_per_day = len(trade_returns) / trading_days
    min_trades_per_day = max(0.5, 0.5 * float(top_k))
    min_total_trades = max(1, math.ceil(trading_days * min_trades_per_day))
    overtrading_penalty = max(0.0, trades_per_day - max(2.0, top_k * 8.0)) * 0.0005
    undertrading_penalty = max(0.0, min_total_trades - len(trade_returns)) / min_total_trades
    objective = daily_mean - 0.5 * daily_std - max_drawdown - overtrading_penalty - undertrading_penalty

    out = {
        "objective": objective,
        "fold_metrics": fold_metrics,
        "trade_count": len(trade_returns),
        "avg_trade_return": statistics.fmean(trade_returns) if trade_returns else 0.0,
        "winrate": _winrate(trade_returns),
        "profit_factor": _profit_factor(trade_returns),
        "max_drawdown": max_drawdown,
        "daily_return_mean": daily_mean,
        "daily_return_std": daily_std,
        "overtrading_penalty": overtrading_penalty,
        "undertrading_penalty": undertrading_penalty,
    }
    if include_bucket_report:
        out["bucket_report"] = bucket_report(scored_rows, selected_indices)
    else:
        out["bucket_report"] = []
    return out


def bucket_report(scored_rows: Sequence[tuple[int, Mapping[str, Any], float]], selected_indices: set[int]) -> list[dict[str, Any]]:
    buckets = [
        {
            "bucket_range": f"[{idx / 10:.1f}, {(idx + 1) / 10:.1f}{']' if idx == 9 else ')'}",
            "candidate_count": 0,
            "selected_count": 0,
            "_return_sum": 0.0,
            "_wins": 0,
            "_score_sum": 0.0,
        }
        for idx in range(10)
    ]
    for idx, row, score in scored_rows:
        bucket_idx = min(max(int(score * 10), 0), 9)
        bucket = buckets[bucket_idx]
        ret = _finite_row_return(row)
        bucket["candidate_count"] += 1
        bucket["selected_count"] += 1 if idx in selected_indices else 0
        bucket["_return_sum"] += ret
        bucket["_wins"] += 1 if ret > 0.0 else 0
        bucket["_score_sum"] += score
    out = []
    for bucket in buckets:
        count = int(bucket["candidate_count"])
        out.append(
            {
                "bucket_range": bucket["bucket_range"],
                "candidate_count": count,
                "selected_count": int(bucket["selected_count"]),
                "avg_realized_net_return": float(bucket["_return_sum"]) / count if count else 0.0,
                "winrate": float(bucket["_wins"]) / count if count else 0.0,
                "avg_positive_score": float(bucket["_score_sum"]) / count if count else 0.0,
            }
        )
    return out


def execution_net_return(
    *,
    side: str,
    entry_bid: float,
    entry_ask: float,
    exit_bid: float,
    exit_ask: float,
    commission_rate: float,
    slippage_rate: float,
) -> float:
    cost = 2.0 * max(float(commission_rate), 0.0) + 2.0 * max(float(slippage_rate), 0.0)
    if side == "long":
        return ((float(exit_bid) - float(entry_ask)) / float(entry_ask)) - cost if entry_ask > 0 else 0.0
    if side == "short":
        return ((float(entry_bid) - float(exit_ask)) / float(entry_bid)) - cost if entry_bid > 0 else 0.0
    raise ValueError(f"unsupported side: {side}")


def _candidate_cache_path(args: argparse.Namespace) -> Path:
    explicit = str(getattr(args, "candidate_cache", "") or "").strip()
    if explicit:
        return Path(explicit)
    report = Path(str(args.report))
    return report.parent / DEFAULT_CACHE_NAME


def _resolve_fixed_top_k(value: Any) -> int | None:
    if value is None:
        return None
    top_k = int(value)
    if top_k < 1 or top_k > 2:
        raise ValueError("--top-k must be 1 or 2")
    return top_k


def _signal_pred_ohlcv(signal: SignalRow | None) -> dict[str, float] | None:
    if signal is None:
        return None
    pred = signal.metadata.get("pred_ohlcv") if isinstance(signal.metadata, Mapping) else None
    if not isinstance(pred, Mapping):
        return None
    out = {}
    for key in ("open", "high", "low", "close"):
        value = _safe_float(pred.get(key))
        if value <= 0.0:
            return None
        out[key] = value
    if out["high"] < out["low"]:
        return None
    return out


def _is_decision_timestamp(as_of: datetime, *, config: RuntimeConfig) -> bool:
    interval = max(int(config.rebalance.decision_interval_minutes), 1)
    local = _local_datetime(as_of, str(config.trading_session.timezone))
    if bool(config.trading_session.enabled):
        entry_start = _parse_time(str(config.trading_session.entry_start))
        cutoff = _parse_time(str(config.trading_session.new_entry_cutoff))
        if local.time() < entry_start or local.time() > cutoff:
            return False
        anchor = local.replace(hour=entry_start.hour, minute=entry_start.minute, second=0, microsecond=0)
        elapsed_seconds = (local - anchor).total_seconds()
    else:
        elapsed_seconds = float(local.hour * 60 + local.minute) * 60.0
    elapsed_minutes = int(elapsed_seconds // 60)
    return elapsed_seconds >= 0.0 and local.second == 0 and local.microsecond == 0 and elapsed_minutes % interval == 0


def _trade_date(as_of: datetime, *, config: RuntimeConfig) -> str:
    return _local_datetime(as_of, str(config.trading_session.timezone)).date().isoformat()


def _local_datetime(value: datetime, timezone: str) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(ZoneInfo(timezone)).replace(tzinfo=None)


def _parse_time(value: str) -> time:
    parts = str(value).split(":")
    return time(hour=int(parts[0]), minute=int(parts[1]), second=int(parts[2]) if len(parts) > 2 else 0)


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


def _bid_ask(snapshot: Any) -> tuple[float, float]:
    if snapshot is None:
        return 0.0, 0.0
    return _safe_float(getattr(snapshot, "bid", 0.0)), _safe_float(getattr(snapshot, "ask", 0.0))


def _spread_rate(*, snapshot: Any, metric: Any) -> float:
    spread = getattr(metric, "spread_pct", None) if metric is not None else None
    if spread is None and snapshot is not None:
        spread = getattr(snapshot, "spread_pct", None)
    return max(_safe_float(spread, default=1.0), 0.0)


def _candidate_sort_key(idx: int, row: Mapping[str, Any], score: float) -> tuple[float, str, str, int]:
    return (-float(score), str(row.get("secid") or ""), str(row.get("side") or ""), int(idx))


def _finite_row_return(row: Mapping[str, Any]) -> float:
    return _safe_float(row.get("realized_net_return"), default=0.0)


def _winrate(values: Sequence[float]) -> float:
    return sum(1 for value in values if float(value) > 0.0) / len(values) if values else 0.0


def _profit_factor(values: Sequence[float]) -> float | None:
    gains = sum(float(value) for value in values if float(value) > 0.0)
    losses = sum(float(value) for value in values if float(value) < 0.0)
    if losses == 0.0:
        return None if gains > 0.0 else 0.0
    return gains / abs(losses)


def _max_drawdown(daily_returns: Sequence[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for value in daily_returns:
        equity *= 1.0 + float(value)
        peak = max(peak, equity)
        if peak > 0.0:
            max_dd = max(max_dd, (peak - equity) / peak)
    return max_dd


def _validate_unit_interval(value: Any, *, name: str) -> float:
    out = _safe_float(value, default=float("nan"))
    if not math.isfinite(out) or out < 0.0 or out > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return out


def _finite_float(value: Any, *, name: str) -> float:
    out = _safe_float(value, default=float("nan"))
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


if __name__ == "__main__":
    raise SystemExit(main())
