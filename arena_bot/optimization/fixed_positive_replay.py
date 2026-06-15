from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..config import load_config
from ..market_data import SavedCandleMarketDataProvider
from ..ranking.scoring import (
    POSITIVE_METRICS,
    RISK_METRICS,
    compute_candidate_vectors,
    cosine_strength_score,
    instrument_scales_from_history,
    weighted_sum,
)
from ..types import Instrument
from .positive_weights import ECONOMIC_POSITIVE_PRIOR, load_candidate_cache, normalize_named_weights

FIXED_RISK_WEIGHTS = {
    "false_breakout_risk": 0.16,
    "wide_spread_risk": 0.30,
    "late_entry_risk": 0.20,
    "high_mae_risk": 0.24,
    "direction_conflict_risk": 0.10,
}


@dataclass
class OpenPosition:
    secid: str
    side: str
    entry_as_of: str
    entry_bid: float
    entry_ask: float
    entry_price: float
    entry_commission_rate: float
    entry_slippage_rate: float
    entry_positive_score: float
    entry_risk_score: float
    entry_net_edge: float
    entry_net_edge_score: float


@dataclass
class AllocatedOpenPosition:
    position: OpenPosition
    allocation: float
    entry_rank: int
    entry_reason: str


@dataclass(frozen=True)
class SingleTopRuntimeLikeParams:
    min_net_edge: float
    min_rank_gap: float
    max_gross_pred_return: float
    target_abs_weight: float
    max_positive_score: float | None = None
    min_raw_mae_pct: float | None = None
    max_raw_rr: float | None = None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(json.dumps({"summary": str(Path(args.output_dir) / "summary.json"), **_summary_console_payload(summary)}, ensure_ascii=False))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fixed-positive-replay",
        description="Replay fixed economic-positive top-1 entries while logging risk scores without using risk as a filter.",
    )
    parser.add_argument("--config", default="configs/universe_v1_may1_14.yaml")
    parser.add_argument("--candles-dir", required=True)
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--from", dest="from_date", required=True)
    parser.add_argument("--till", dest="till_date", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n-log", type=int, default=10)
    parser.add_argument(
        "--max-positive-score",
        type=float,
        default=None,
        help="Optional upper filter for single-top runtime-like modes; top-1 with positive_score above this value is blocked without backfill.",
    )
    parser.add_argument(
        "--min-raw-mae-pct",
        type=float,
        default=None,
        help="Optional strict lower filter for single-top runtime-like modes; top-1 with raw_vector_metrics.mae_pct <= this value is blocked without backfill.",
    )
    parser.add_argument(
        "--max-raw-rr",
        type=float,
        default=None,
        help="Optional upper filter for single-top runtime-like modes; top-1 with raw_vector_metrics.rr >= this value is blocked without backfill.",
    )
    parser.add_argument(
        "--strategy",
        choices=("top1-hold-same", "top2-rotate-on-loss", "single-top-vector-runtime-like", "single-top-net-edge-runtime-like"),
        default="top1-hold-same",
        help="Replay strategy. top1-hold-same keeps the original fixed-positive logic; top2-rotate-on-loss splits rank 1/2 and rotates after hourly losses; single-top-vector-runtime-like mimics kronos_single_top filters while ranking by vector score; single-top-net-edge-runtime-like uses the same filters but ranks by raw net_edge.",
    )
    parser.add_argument(
        "--close-before-session-gap",
        action="store_true",
        help="For single-top runtime-like modes, close any open position at the current row future price when the next decision timestamp is not the next hour.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(str(args.config))
    from_dt = datetime.fromisoformat(str(args.from_date))
    till_dt = datetime.fromisoformat(str(args.till_date))
    rows = [
        row
        for row in load_candidate_cache(Path(str(args.candidate_cache)))
        if _timestamp_in_range(str(row.get("as_of") or ""), from_dt=from_dt, till_dt=till_dt)
    ]
    instruments = {instrument.secid: instrument for instrument in config.instruments if instrument.enabled}
    provider = SavedCandleMarketDataProvider(
        directories=[Path(str(args.candles_dir))],
        history_rows=int(getattr(config, "kronos", object()).context_rows or 512),
    )
    risk_enricher = _provider_risk_enricher(provider=provider, instruments=instruments)
    single_top_params = single_top_runtime_like_params_from_config(
        config,
        max_positive_score=_optional_unit_interval(getattr(args, "max_positive_score", None), name="max_positive_score"),
        min_raw_mae_pct=_optional_non_negative_float(getattr(args, "min_raw_mae_pct", None), name="min_raw_mae_pct"),
        max_raw_rr=_optional_non_negative_float(getattr(args, "max_raw_rr", None), name="max_raw_rr"),
    )
    if str(args.strategy) == "top2-rotate-on-loss":
        result = replay_candidate_rows_top2_rotate_on_loss(
            rows,
            risk_enricher=risk_enricher,
            top_n_log=max(int(args.top_n_log), 1),
        )
    elif str(args.strategy) == "single-top-vector-runtime-like":
        result = replay_candidate_rows_single_top_runtime_like(
            rows,
            risk_enricher=risk_enricher,
            top_n_log=max(int(args.top_n_log), 1),
            params=single_top_params,
            ranking_metric="positive_score",
            close_before_session_gap=bool(getattr(args, "close_before_session_gap", False)),
        )
    elif str(args.strategy) == "single-top-net-edge-runtime-like":
        result = replay_candidate_rows_single_top_runtime_like(
            rows,
            risk_enricher=risk_enricher,
            top_n_log=max(int(args.top_n_log), 1),
            params=single_top_params,
            ranking_metric="net_edge",
            close_before_session_gap=bool(getattr(args, "close_before_session_gap", False)),
        )
    else:
        result = replay_candidate_rows(
            rows,
            risk_enricher=risk_enricher,
            top_n_log=max(int(args.top_n_log), 1),
        )
    output_dir = Path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "ranked_decisions.jsonl", result["decision_rows"])
    _write_jsonl(output_dir / "position_events.jsonl", result["events"])
    _write_trades_csv(output_dir / "trades.csv", result["trades"])
    summary = {
        **result["summary"],
        "config": str(args.config),
        "candles_dir": str(args.candles_dir),
        "candidate_cache": str(args.candidate_cache),
        "from": str(args.from_date),
        "till": str(args.till_date),
        "top_n_log": max(int(args.top_n_log), 1),
        "strategy": str(args.strategy),
        "max_positive_score": single_top_params.max_positive_score,
        "min_raw_mae_pct": single_top_params.min_raw_mae_pct,
        "max_raw_rr": single_top_params.max_raw_rr,
        "close_before_session_gap": bool(getattr(args, "close_before_session_gap", False)),
        "fixed_positive_weights": fixed_positive_weights(),
        "fixed_risk_weights": normalized_fixed_risk_weights(),
        "notes": {
            "risk_used_for_filtering": False,
            "risk_used_for_ranking": False,
            "entry_filter": (
                "rank1/rank2 raw_vector_metrics.net_edge > 0 and positive_vector.net_edge_score > 0"
                if str(args.strategy) == "top2-rotate-on-loss"
                else "single top runtime-like filters ranked by positive_score"
                if str(args.strategy) == "single-top-vector-runtime-like"
                else "single top runtime-like filters ranked by raw net_edge"
                if str(args.strategy) == "single-top-net-edge-runtime-like"
                else "top1 raw_vector_metrics.net_edge > 0 and positive_vector.net_edge_score > 0"
            ),
            "top2_rotation_rule": (
                "Hourly check: split rank1/rank2 50/50; if one prior leg is negative, next hour goes 100% to the positive leg; if all prior legs are negative, stay in cash for one hour."
                if str(args.strategy) == "top2-rotate-on-loss"
                else ""
            ),
            "single_top_runtime_like_rule": (
                "Cache replay of kronos_single_top mechanics: one active position, hold same top, close to cash on failed top1 filters, no backfill, rank by positive_score instead of gross_pred_return."
                if str(args.strategy) == "single-top-vector-runtime-like"
                else "Cache replay of kronos_single_top mechanics: one active position, hold same top, close to cash on failed top1 filters, no backfill, rank by raw net_edge."
                if str(args.strategy) == "single-top-net-edge-runtime-like"
                else ""
            ),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def fixed_positive_weights() -> dict[str, float]:
    return normalize_named_weights(ECONOMIC_POSITIVE_PRIOR, POSITIVE_METRICS)


def normalized_fixed_risk_weights() -> dict[str, float]:
    return normalize_named_weights(FIXED_RISK_WEIGHTS, RISK_METRICS)


def single_top_runtime_like_params_from_config(
    config: Any,
    *,
    max_positive_score: float | None = None,
    min_raw_mae_pct: float | None = None,
    max_raw_rr: float | None = None,
) -> SingleTopRuntimeLikeParams:
    entry = config.trade_lifecycle.entry
    configured = min(max(_safe_float(getattr(entry, "single_top_target_weight", 1.0)), 0.0), 10.0)
    max_gross = max(_safe_float(getattr(config.portfolio, "max_gross", 1.0)), 0.0)
    cash_buffer = min(max(_safe_float(getattr(config.risk, "cash_buffer_pct", 0.0)), 0.0), 0.95)
    safety = min(max(_safe_float(getattr(config.risk, "sizing_safety_pct", 0.0)), 0.0), 0.95)
    buffered_full_capital = max(1.0 - cash_buffer, 0.0) * (1.0 - safety)
    return SingleTopRuntimeLikeParams(
        min_net_edge=_safe_float(getattr(entry, "single_top_min_net_edge", 0.0)),
        min_rank_gap=max(_safe_float(getattr(entry, "single_top_min_rank_gap", 0.0)), 0.0),
        max_gross_pred_return=_safe_float(getattr(entry, "single_top_max_gross_pred_return", 1.0)),
        target_abs_weight=max(min(configured, max_gross, buffered_full_capital), 0.0),
        max_positive_score=max_positive_score,
        min_raw_mae_pct=min_raw_mae_pct,
        max_raw_rr=max_raw_rr,
    )


def replay_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    positive_weights: Mapping[str, float] | None = None,
    risk_weights: Mapping[str, float] | None = None,
    risk_enricher: Callable[[Mapping[str, Any]], Mapping[str, float]] | None = None,
    top_n_log: int = 10,
    close_before_session_gap: bool = False,
) -> dict[str, Any]:
    positive_weights = dict(positive_weights or fixed_positive_weights())
    risk_weights = dict(risk_weights or normalized_fixed_risk_weights())
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("as_of") or ""), []).append(row)

    position: OpenPosition | None = None
    events: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    equity = 1.0
    equity_curve = [equity]
    last_scored_by_secid: dict[str, dict[str, Any]] = {}

    sorted_as_of = sorted(grouped)
    for index, as_of in enumerate(sorted_as_of):
        next_as_of = sorted_as_of[index + 1] if index + 1 < len(sorted_as_of) else None
        scored = [
            score_replay_candidate(
                row,
                positive_weights=positive_weights,
                risk_weights=risk_weights,
                risk_enricher=risk_enricher,
            )
            for row in grouped[as_of]
        ]
        scored.sort(key=lambda item: (-float(item["positive_score"]), str(item["secid"]), str(item["side"])))
        for rank, item in enumerate(scored, start=1):
            item["rank"] = rank
        last_scored_by_secid = {str(item["secid"]): item for item in scored}
        top1 = scored[0] if scored else None
        top1_passes = bool(top1 and candidate_has_positive_edge(top1))
        action = "no_candidates" if top1 is None else ("top1_edge_pass" if top1_passes else "skip_no_edge")
        reason = "" if top1_passes else ("no_candidates" if top1 is None else "top1_net_edge_not_positive")
        position_before = _position_payload(position)

        if position is not None:
            same_top1 = bool(top1 and position.secid == str(top1["secid"]) and position.side == str(top1["side"]))
            if same_top1 and top1_passes:
                action = "hold_same_top1"
                reason = "same_secid_side_top1"
                events.append({"event": "hold_same_top1", "as_of": as_of, "secid": position.secid, "side": position.side, **_top_event_payload(top1)})
            else:
                exit_reason = "switch" if top1_passes else reason
                if top1_passes:
                    events.append(
                        {
                            "event": "switch",
                            "as_of": as_of,
                            "from_secid": position.secid,
                            "from_side": position.side,
                            "to_secid": str(top1["secid"]),
                            "to_side": str(top1["side"]),
                            **_top_event_payload(top1),
                        }
                    )
                trade = close_position(position, exit_row=last_scored_by_secid.get(position.secid), exit_as_of=as_of, reason=exit_reason)
                if trade is not None:
                    equity *= 1.0 + float(trade["net_return"])
                    equity_curve.append(equity)
                    trades.append(trade)
                    events.append({"event": "exit", "as_of": as_of, **_trade_event_payload(trade)})
                    position = None
                else:
                    events.append({"event": "exit_skipped_missing_price", "as_of": as_of, "secid": position.secid, "side": position.side})
                if top1_passes:
                    position = open_position(top1, as_of=as_of)
                    action = "switch"
                    reason = "switched_to_top1"
                    events.append({"event": "entry", "as_of": as_of, **_position_payload(position), **_top_event_payload(top1)})
                elif top1 is not None:
                    action = "exit_skip_no_edge"
                    events.append({"event": "skip_no_edge", "as_of": as_of, **_top_event_payload(top1)})
        elif top1_passes and top1 is not None:
            position = open_position(top1, as_of=as_of)
            action = "entry"
            reason = "top1_edge_pass"
            events.append({"event": "entry", "as_of": as_of, **_position_payload(position), **_top_event_payload(top1)})
        elif top1 is not None:
            events.append({"event": "skip_no_edge", "as_of": as_of, **_top_event_payload(top1)})

        position_after = _position_payload(position)
        for item in scored[: max(int(top_n_log), 1)]:
            decision_rows.append(
                {
                    "as_of": as_of,
                    "rank": int(item["rank"]),
                    "secid": item["secid"],
                    "side": item["side"],
                    "asset_class": item.get("asset_class", ""),
                    "positive_score": item["positive_score"],
                    "risk_score": item["risk_score"],
                    "risk_vector": item["risk_vector"],
                    "net_edge": item["net_edge"],
                    "net_edge_score": item["net_edge_score"],
                    "realized_net_return": item.get("realized_net_return", 0.0),
                    "current_bid": item.get("current_bid", 0.0),
                    "current_ask": item.get("current_ask", 0.0),
                    "future_bid": item.get("future_bid", 0.0),
                    "future_ask": item.get("future_ask", 0.0),
                    "action": action if int(item["rank"]) == 1 else "not_top1",
                    "reason": reason if int(item["rank"]) == 1 else "",
                    "position_before": position_before,
                    "position_after": position_after,
                    "positive_vector": item.get("positive_vector", {}),
                    "raw_vector_metrics": item.get("raw_vector_metrics", {}),
                }
            )

    if position is not None:
        exit_row = last_scored_by_secid.get(position.secid)
        final_as_of = str(exit_row.get("future_as_of") or exit_row.get("as_of") or "") if exit_row else ""
        trade = close_position(position, exit_row=exit_row, exit_as_of=final_as_of, reason="final_close", use_future=True)
        if trade is not None:
            equity *= 1.0 + float(trade["net_return"])
            equity_curve.append(equity)
            trades.append(trade)
            events.append({"event": "exit", "as_of": final_as_of, **_trade_event_payload(trade)})

    return {
        "summary": build_summary(trades, equity_curve, decision_rows),
        "decision_rows": decision_rows,
        "events": events,
        "trades": trades,
    }


def replay_candidate_rows_top2_rotate_on_loss(
    rows: Sequence[Mapping[str, Any]],
    *,
    positive_weights: Mapping[str, float] | None = None,
    risk_weights: Mapping[str, float] | None = None,
    risk_enricher: Callable[[Mapping[str, Any]], Mapping[str, float]] | None = None,
    top_n_log: int = 10,
) -> dict[str, Any]:
    positive_weights = dict(positive_weights or fixed_positive_weights())
    risk_weights = dict(risk_weights or normalized_fixed_risk_weights())
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("as_of") or ""), []).append(row)

    positions: list[AllocatedOpenPosition] = []
    events: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    equity = 1.0
    equity_curve = [equity]
    period_returns: list[float] = []
    cash_period_count = 0
    last_scored_by_secid: dict[str, dict[str, Any]] = {}

    for as_of in sorted(grouped):
        scored = [
            score_replay_candidate(
                row,
                positive_weights=positive_weights,
                risk_weights=risk_weights,
                risk_enricher=risk_enricher,
            )
            for row in grouped[as_of]
        ]
        scored.sort(key=lambda item: (-float(item["positive_score"]), str(item["secid"]), str(item["side"])))
        for rank, item in enumerate(scored, start=1):
            item["rank"] = rank
        last_scored_by_secid = {str(item["secid"]): item for item in scored}
        current_by_key = {_candidate_key(item): item for item in scored}
        top2 = scored[:2]
        eligible_top2 = [item for item in top2 if candidate_has_positive_edge(item)]

        closed_this_period: list[dict[str, Any]] = []
        period_return = 0.0
        prior_positions = positions
        positions = []
        action = "no_candidates" if not scored else "cash"
        reason = "no_candidates" if not scored else "awaiting_signal"

        if prior_positions:
            for allocated in prior_positions:
                exit_row = last_scored_by_secid.get(allocated.position.secid)
                trade = close_position(
                    allocated.position,
                    exit_row=exit_row,
                    exit_as_of=as_of,
                    reason="hourly_check",
                )
                if trade is None:
                    events.append(
                        {
                            "event": "exit_skipped_missing_price",
                            "as_of": as_of,
                            "secid": allocated.position.secid,
                            "side": allocated.position.side,
                            "capital_fraction": allocated.allocation,
                        }
                    )
                    continue
                trade["capital_fraction"] = float(allocated.allocation)
                trade["entry_rank"] = int(allocated.entry_rank)
                trade["entry_reason"] = allocated.entry_reason
                trade["portfolio_return"] = float(allocated.allocation) * float(trade["net_return"])
                closed_this_period.append(trade)

            if closed_this_period:
                positives = [trade for trade in closed_this_period if _safe_float(trade.get("net_return")) > 0.0]
                all_negative = not positives
                has_negative = any(_safe_float(trade.get("net_return")) <= 0.0 for trade in closed_this_period)
                if all_negative:
                    exit_reason = "all_legs_loss_to_cash"
                elif has_negative:
                    exit_reason = "loss_rotate_to_winner"
                else:
                    exit_reason = "rebalance_top2_after_profit"

                for trade in closed_this_period:
                    trade["exit_reason"] = exit_reason
                    period_return += _safe_float(trade.get("portfolio_return"))
                    trades.append(trade)
                    events.append({"event": "exit", "as_of": as_of, **_trade_event_payload(trade)})

                equity *= 1.0 + period_return
                equity_curve.append(equity)
                period_returns.append(period_return)

                if all_negative:
                    cash_period_count += 1
                    action = "cash_after_all_legs_loss"
                    reason = "all_legs_loss_to_cash"
                    events.append(
                        {
                            "event": "cash_after_all_legs_loss",
                            "as_of": as_of,
                            "period_return": period_return,
                            "equity": equity,
                        }
                    )
                elif has_negative:
                    winner = max(positives, key=lambda trade: _safe_float(trade.get("net_return")))
                    winner_key = (str(winner.get("secid") or ""), str(winner.get("side") or ""))
                    winner_candidate = current_by_key.get(winner_key)
                    if winner_candidate is not None and candidate_has_positive_edge(winner_candidate):
                        positions = [
                            AllocatedOpenPosition(
                                position=open_position(winner_candidate, as_of=as_of),
                                allocation=1.0,
                                entry_rank=int(winner_candidate.get("rank", 0) or 0),
                                entry_reason="rotate_full_to_winner",
                            )
                        ]
                        action = "rotate_full_to_winner"
                        reason = "one_leg_loss_other_leg_positive"
                        events.append(
                            {
                                "event": "rotate_full_to_winner",
                                "as_of": as_of,
                                "secid": winner_key[0],
                                "side": winner_key[1],
                                "capital_fraction": 1.0,
                                "winner_net_return": _safe_float(winner.get("net_return")),
                                **_ranked_event_payload(winner_candidate, prefix="winner"),
                            }
                        )
                    else:
                        cash_period_count += 1
                        action = "cash_after_winner_not_eligible"
                        reason = "winner_not_available_or_edge_failed"
                        events.append({"event": "cash_after_winner_not_eligible", "as_of": as_of, "winner_key": list(winner_key)})
                else:
                    positions = _open_top2_allocations(eligible_top2, as_of=as_of, reason="rebalance_top2_after_profit")
                    action = "rebalance_top2_after_profit" if positions else "cash_no_eligible_top2"
                    reason = "all_prior_legs_positive" if positions else "top2_edge_failed"
                    _append_entry_events(events, as_of=as_of, positions=positions, candidates=current_by_key)

        elif eligible_top2:
            positions = _open_top2_allocations(eligible_top2, as_of=as_of, reason="initial_or_cash_reentry_top2")
            action = "entry_top2"
            reason = "rank1_rank2_edge_pass"
            _append_entry_events(events, as_of=as_of, positions=positions, candidates=current_by_key)
        else:
            cash_period_count += 1
            action = "skip_no_edge"
            reason = "top2_net_edge_not_positive"
            events.append({"event": "skip_no_edge", "as_of": as_of})

        position_after = [_allocated_position_payload(position) for position in positions]
        target_allocations = {_candidate_key_from_position(position.position): position.allocation for position in positions}
        for item in scored[: max(int(top_n_log), 1)]:
            key = _candidate_key(item)
            row_action = action if key in target_allocations else ("top2_not_allocated" if int(item["rank"]) <= 2 else "not_top2")
            row_reason = reason if key in target_allocations or int(item["rank"]) <= 2 else ""
            decision_rows.append(
                {
                    "as_of": as_of,
                    "rank": int(item["rank"]),
                    "secid": item["secid"],
                    "side": item["side"],
                    "asset_class": item.get("asset_class", ""),
                    "positive_score": item["positive_score"],
                    "risk_score": item["risk_score"],
                    "risk_vector": item["risk_vector"],
                    "net_edge": item["net_edge"],
                    "net_edge_score": item["net_edge_score"],
                    "realized_net_return": item.get("realized_net_return", 0.0),
                    "current_bid": item.get("current_bid", 0.0),
                    "current_ask": item.get("current_ask", 0.0),
                    "future_bid": item.get("future_bid", 0.0),
                    "future_ask": item.get("future_ask", 0.0),
                    "target_allocation": target_allocations.get(key, 0.0),
                    "period_return": period_return,
                    "portfolio_equity": equity,
                    "action": row_action,
                    "reason": row_reason,
                    "position_after": position_after,
                    "positive_vector": item.get("positive_vector", {}),
                    "raw_vector_metrics": item.get("raw_vector_metrics", {}),
                }
            )

    if positions:
        final_period_return = 0.0
        final_as_of = ""
        for allocated in positions:
            exit_row = last_scored_by_secid.get(allocated.position.secid)
            final_as_of = str(exit_row.get("future_as_of") or exit_row.get("as_of") or "") if exit_row else final_as_of
            trade = close_position(
                allocated.position,
                exit_row=exit_row,
                exit_as_of=final_as_of,
                reason="final_close",
                use_future=True,
            )
            if trade is None:
                continue
            trade["capital_fraction"] = float(allocated.allocation)
            trade["entry_rank"] = int(allocated.entry_rank)
            trade["entry_reason"] = allocated.entry_reason
            trade["portfolio_return"] = float(allocated.allocation) * float(trade["net_return"])
            final_period_return += _safe_float(trade.get("portfolio_return"))
            trades.append(trade)
            events.append({"event": "exit", "as_of": final_as_of, **_trade_event_payload(trade)})
        if final_period_return:
            equity *= 1.0 + final_period_return
            equity_curve.append(equity)
            period_returns.append(final_period_return)

    summary = build_summary(trades, equity_curve, decision_rows)
    summary.update(
        {
            "mode": "fixed_positive_top2_rotate_on_loss",
            "portfolio_period_count": len(period_returns),
            "cash_period_count": cash_period_count,
            "avg_period_return": statistics.fmean(period_returns) if period_returns else 0.0,
            "period_winrate": sum(1 for value in period_returns if value > 0.0) / len(period_returns) if period_returns else 0.0,
        }
    )
    return {
        "summary": summary,
        "decision_rows": decision_rows,
        "events": events,
        "trades": trades,
    }


def replay_candidate_rows_single_top_runtime_like(
    rows: Sequence[Mapping[str, Any]],
    *,
    params: SingleTopRuntimeLikeParams,
    ranking_metric: str = "positive_score",
    positive_weights: Mapping[str, float] | None = None,
    risk_weights: Mapping[str, float] | None = None,
    risk_enricher: Callable[[Mapping[str, Any]], Mapping[str, float]] | None = None,
    top_n_log: int = 10,
    close_before_session_gap: bool = False,
) -> dict[str, Any]:
    if ranking_metric not in {"positive_score", "net_edge"}:
        raise ValueError("ranking_metric must be one of: positive_score, net_edge")
    positive_weights = dict(positive_weights or fixed_positive_weights())
    risk_weights = dict(risk_weights or normalized_fixed_risk_weights())
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("as_of") or ""), []).append(row)

    position: AllocatedOpenPosition | None = None
    events: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    equity = 1.0
    equity_curve = [equity]
    last_scored_by_secid: dict[str, dict[str, Any]] = {}

    sorted_as_of = sorted(grouped)
    for index, as_of in enumerate(sorted_as_of):
        next_as_of = sorted_as_of[index + 1] if index + 1 < len(sorted_as_of) else None
        scored = [
            score_replay_candidate(
                row,
                positive_weights=positive_weights,
                risk_weights=risk_weights,
                risk_enricher=risk_enricher,
            )
            for row in grouped[as_of]
        ]
        scored.sort(key=lambda item: (-_ranking_value(item, ranking_metric), str(item["secid"]), str(item["side"])))
        for rank, item in enumerate(scored, start=1):
            item["rank"] = rank
            item["ranking_metric"] = ranking_metric
            item["ranking_score"] = _ranking_value(item, ranking_metric)
            item["gross_pred_return"] = _candidate_gross_pred_return(item)
            item["filter_reason"] = _single_top_runtime_like_filter_reason(
                gross_pred_return=float(item["gross_pred_return"]),
                net_edge=_safe_float(item.get("net_edge")),
                positive_score=_safe_float(item.get("positive_score")),
                raw_mae_pct=_candidate_raw_mae_pct(item),
                raw_rr=_candidate_raw_rr(item),
                params=params,
            )
            item["single_top_passed_filters"] = item["filter_reason"] == ""
            next_item = scored[rank] if rank < len(scored) else None
            rank_gap_to_next = (
                _ranking_value(item, ranking_metric) - _ranking_value(next_item, ranking_metric)
                if next_item is not None
                else None
            )
            item["rank_gap_to_next"] = rank_gap_to_next
            if rank == 1:
                item["top1_top2_score_gap"] = rank_gap_to_next
            elif not item.get("reason"):
                item["reason"] = "not_top_1"

        last_scored_by_secid = {str(item["secid"]): item for item in scored}
        top = scored[0] if scored else None
        position_before = _allocated_position_payload(position) if position is not None else {}
        action = "skip_flat"
        reason = "no_candidates"
        if top is not None:
            top_filter_reason = _single_top_runtime_like_filter_reason(
                gross_pred_return=_candidate_gross_pred_return(top),
                net_edge=_safe_float(top.get("net_edge")),
                positive_score=_safe_float(top.get("positive_score")),
                raw_mae_pct=_candidate_raw_mae_pct(top),
                raw_rr=_candidate_raw_rr(top),
                params=params,
                rank_gap=top.get("top1_top2_score_gap"),
            )
            top["filter_reason"] = top_filter_reason
            top["single_top_passed_filters"] = top_filter_reason == ""
            top["reason"] = ""
            top_key = _candidate_key(top)
            same_current = bool(position and _candidate_key_from_position(position.position) == top_key)

            if top_filter_reason:
                top["reason"] = top_filter_reason
                if position is not None:
                    trade = close_position(
                        position.position,
                        exit_row=last_scored_by_secid.get(position.position.secid),
                        exit_as_of=as_of,
                        reason="single_top_close_to_cash",
                    )
                    if trade is not None:
                        _attach_allocation_to_trade(trade, position)
                        equity *= 1.0 + _safe_float(trade.get("portfolio_return"))
                        equity_curve.append(equity)
                        trades.append(trade)
                        events.append({"event": "exit", "as_of": as_of, **_trade_event_payload(trade)})
                    events.append({"event": "single_top_close_to_cash", "as_of": as_of, "filter_reason": top_filter_reason, **_ranked_event_payload(top, prefix="top1")})
                    position = None
                    action = "close_to_cash"
                    reason = "single_top_close_to_cash"
                else:
                    events.append({"event": "skip_no_entry", "as_of": as_of, "filter_reason": top_filter_reason, **_ranked_event_payload(top, prefix="top1")})
                    action = "skip_flat"
                    reason = top_filter_reason
            elif params.target_abs_weight <= 0.0:
                top["reason"] = "single_top_no_target_weight"
                action = "skip_flat"
                reason = "single_top_no_target_weight"
            elif same_current and position is not None:
                top["selected"] = True
                top["allocated_weight"] = position.allocation
                top["target_weight"] = position.allocation if str(top.get("side")) == "long" else -position.allocation
                top["reason"] = "same_top_hold"
                action = "hold_same"
                reason = "same_top_hold"
                events.append({"event": "hold_same", "as_of": as_of, **_allocated_position_payload(position), **_ranked_event_payload(top, prefix="top1")})
            else:
                if position is not None:
                    trade = close_position(
                        position.position,
                        exit_row=last_scored_by_secid.get(position.position.secid),
                        exit_as_of=as_of,
                        reason="single_top_rebalance",
                    )
                    if trade is not None:
                        _attach_allocation_to_trade(trade, position)
                        equity *= 1.0 + _safe_float(trade.get("portfolio_return"))
                        equity_curve.append(equity)
                        trades.append(trade)
                        events.append({"event": "exit", "as_of": as_of, **_trade_event_payload(trade)})
                    events.append({"event": "switch", "as_of": as_of, "to_secid": str(top["secid"]), "to_side": str(top["side"]), **_ranked_event_payload(top, prefix="top1")})
                top["selected"] = True
                top["allocated_weight"] = params.target_abs_weight
                top["target_weight"] = params.target_abs_weight if str(top.get("side")) == "long" else -params.target_abs_weight
                top["reason"] = "single_top_entry"
                position = AllocatedOpenPosition(
                    position=open_position(top, as_of=as_of),
                    allocation=params.target_abs_weight,
                    entry_rank=int(top.get("rank", 0) or 0),
                    entry_reason="single_top_vector_entry",
                )
                events.append({"event": "entry", "as_of": as_of, **_allocated_position_payload(position), **_ranked_event_payload(top, prefix="top1")})
                action = "open" if not position_before else "switch"
                reason = "single_top_entry"

        session_gap_after = close_before_session_gap and _has_session_gap_after(as_of, next_as_of)
        session_gap_exit_as_of = ""
        session_gap_exit_trade: dict[str, Any] | None = None
        if session_gap_after and position is not None:
            exit_row = last_scored_by_secid.get(position.position.secid)
            session_gap_exit_as_of = _future_as_of(exit_row) or as_of
            session_gap_exit_trade = close_position(
                position.position,
                exit_row=exit_row,
                exit_as_of=session_gap_exit_as_of,
                reason="session_gap_close",
                use_future=True,
            )
            if session_gap_exit_trade is not None:
                _attach_allocation_to_trade(session_gap_exit_trade, position)
                equity *= 1.0 + _safe_float(session_gap_exit_trade.get("portfolio_return"))
                equity_curve.append(equity)
                trades.append(session_gap_exit_trade)
                events.append({"event": "exit", "as_of": session_gap_exit_as_of, **_trade_event_payload(session_gap_exit_trade)})
            events.append(
                {
                    "event": "session_gap_close",
                    "as_of": as_of,
                    "exit_as_of": session_gap_exit_as_of,
                    "next_as_of": next_as_of or "",
                    **_allocated_position_payload(position),
                }
            )
            position = None
            action = f"{action}_then_session_gap_close" if action else "session_gap_close"
            reason = "session_gap_close"

        position_after = _allocated_position_payload(position) if position is not None else {}
        for item in scored[: max(int(top_n_log), 1)]:
            is_top = int(item["rank"]) == 1
            decision_rows.append(
                {
                    "as_of": as_of,
                    "rank": int(item["rank"]),
                    "secid": item["secid"],
                    "side": item["side"],
                    "asset_class": item.get("asset_class", ""),
                    "selected": bool(item.get("selected", False)),
                    "ranking_metric": ranking_metric,
                    "ranking_score": item.get("ranking_score", 0.0),
                    "positive_score": item["positive_score"],
                    "risk_score": item["risk_score"],
                    "risk_vector": item["risk_vector"],
                    "gross_pred_return": item.get("gross_pred_return", 0.0),
                    "net_edge": item["net_edge"],
                    "net_edge_score": item["net_edge_score"],
                    "rank_gap_to_next": item.get("rank_gap_to_next"),
                    "top1_top2_score_gap": item.get("top1_top2_score_gap"),
                    "filter_reason": item.get("filter_reason", ""),
                    "single_top_passed_filters": bool(item.get("single_top_passed_filters", False)),
                    "realized_net_return": item.get("realized_net_return", 0.0),
                    "current_bid": item.get("current_bid", 0.0),
                    "current_ask": item.get("current_ask", 0.0),
                    "future_bid": item.get("future_bid", 0.0),
                    "future_ask": item.get("future_ask", 0.0),
                    "target_allocation": params.target_abs_weight if bool(item.get("selected", False)) else 0.0,
                    "portfolio_equity": equity,
                    "action": action if is_top else "not_top_1",
                    "reason": reason if is_top else str(item.get("reason") or "not_top_1"),
                    "session_gap_after": bool(session_gap_after),
                    "next_as_of": next_as_of or "",
                    "session_gap_exit_as_of": session_gap_exit_as_of,
                    "position_before": position_before,
                    "position_after": position_after,
                    "positive_vector": item.get("positive_vector", {}),
                    "raw_vector_metrics": item.get("raw_vector_metrics", {}),
                }
            )

    if position is not None:
        exit_row = last_scored_by_secid.get(position.position.secid)
        final_as_of = str(exit_row.get("future_as_of") or exit_row.get("as_of") or "") if exit_row else ""
        trade = close_position(position.position, exit_row=exit_row, exit_as_of=final_as_of, reason="final_close", use_future=True)
        if trade is not None:
            _attach_allocation_to_trade(trade, position)
            equity *= 1.0 + _safe_float(trade.get("portfolio_return"))
            equity_curve.append(equity)
            trades.append(trade)
            events.append({"event": "exit", "as_of": final_as_of, **_trade_event_payload(trade)})

    summary = build_summary(trades, equity_curve, decision_rows)
    weighted_commission = sum(
        (_safe_float(trade.get("entry_commission_rate")) + _safe_float(trade.get("exit_commission_rate")))
        * _safe_float(trade.get("capital_fraction"))
        for trade in trades
    )
    weighted_slippage = sum(
        (_safe_float(trade.get("entry_slippage_rate")) + _safe_float(trade.get("exit_slippage_rate")))
        * _safe_float(trade.get("capital_fraction"))
        for trade in trades
    )
    portfolio_returns = [_safe_float(trade.get("portfolio_return")) for trade in trades]
    summary.update(
        {
            "mode": "fixed_positive_single_top_runtime_like",
            "ranking_metric": ranking_metric,
            "single_top_min_net_edge": params.min_net_edge,
            "single_top_min_rank_gap": params.min_rank_gap,
            "single_top_max_gross_pred_return": params.max_gross_pred_return,
            "max_positive_score": params.max_positive_score,
            "min_raw_mae_pct": params.min_raw_mae_pct,
            "max_raw_rr": params.max_raw_rr,
            "target_abs_weight": params.target_abs_weight,
            "avg_portfolio_trade_return": statistics.fmean(portfolio_returns) if portfolio_returns else 0.0,
            "weighted_commission": weighted_commission,
            "weighted_slippage": weighted_slippage,
            "close_before_session_gap": close_before_session_gap,
            "session_gap_close_count": sum(1 for trade in trades if str(trade.get("exit_reason")) == "session_gap_close"),
        }
    )
    return {
        "summary": summary,
        "decision_rows": decision_rows,
        "events": events,
        "trades": trades,
    }


def score_replay_candidate(
    row: Mapping[str, Any],
    *,
    positive_weights: Mapping[str, float],
    risk_weights: Mapping[str, float],
    risk_enricher: Callable[[Mapping[str, Any]], Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    positive_vector = dict(row.get("positive_vector") or {})
    if "risk_vector" in row and isinstance(row.get("risk_vector"), Mapping):
        risk_vector = {metric: _clip01(row["risk_vector"].get(metric, 0.0)) for metric in RISK_METRICS}  # type: ignore[index]
    elif risk_enricher is not None:
        risk_vector = {metric: _clip01(risk_enricher(row).get(metric, 0.0)) for metric in RISK_METRICS}
    else:
        risk_vector = _risk_vector_from_row(row)
    raw_metrics = dict(row.get("raw_vector_metrics") or {})
    return {
        **dict(row),
        "positive_vector": positive_vector,
        "risk_vector": risk_vector,
        "positive_score": cosine_strength_score(positive_vector, positive_weights),
        "risk_score": weighted_sum(risk_vector, risk_weights),
        "net_edge": _safe_float(raw_metrics.get("net_edge")),
        "net_edge_score": _safe_float(positive_vector.get("net_edge_score")),
    }


def candidate_has_positive_edge(candidate: Mapping[str, Any]) -> bool:
    return _safe_float(candidate.get("net_edge")) > 0.0 and _safe_float(candidate.get("net_edge_score")) > 0.0


def open_position(candidate: Mapping[str, Any], *, as_of: str) -> OpenPosition:
    side = str(candidate["side"])
    entry_bid = _safe_float(candidate.get("current_bid"))
    entry_ask = _safe_float(candidate.get("current_ask"))
    entry_price = entry_ask if side == "long" else entry_bid
    return OpenPosition(
        secid=str(candidate["secid"]),
        side=side,
        entry_as_of=as_of,
        entry_bid=entry_bid,
        entry_ask=entry_ask,
        entry_price=entry_price,
        entry_commission_rate=_safe_float(candidate.get("commission_rate")),
        entry_slippage_rate=_safe_float(candidate.get("slippage_rate")),
        entry_positive_score=_safe_float(candidate.get("positive_score")),
        entry_risk_score=_safe_float(candidate.get("risk_score")),
        entry_net_edge=_safe_float(candidate.get("net_edge")),
        entry_net_edge_score=_safe_float(candidate.get("net_edge_score")),
    )


def close_position(
    position: OpenPosition,
    *,
    exit_row: Mapping[str, Any] | None,
    exit_as_of: str,
    reason: str,
    use_future: bool = False,
) -> dict[str, Any] | None:
    if exit_row is None:
        return None
    if use_future:
        exit_bid = _safe_float(exit_row.get("future_bid"))
        exit_ask = _safe_float(exit_row.get("future_ask"))
    else:
        exit_bid = _safe_float(exit_row.get("current_bid"))
        exit_ask = _safe_float(exit_row.get("current_ask"))
    if exit_bid <= 0.0 or exit_ask <= 0.0:
        return None
    exit_price = exit_bid if position.side == "long" else exit_ask
    exit_commission_rate = _safe_float(exit_row.get("commission_rate"))
    exit_slippage_rate = _safe_float(exit_row.get("slippage_rate"))
    net_return = trade_net_return(
        side=position.side,
        entry_bid=position.entry_bid,
        entry_ask=position.entry_ask,
        exit_bid=exit_bid,
        exit_ask=exit_ask,
        entry_commission_rate=position.entry_commission_rate,
        entry_slippage_rate=position.entry_slippage_rate,
        exit_commission_rate=exit_commission_rate,
        exit_slippage_rate=exit_slippage_rate,
    )
    return {
        "entry_as_of": position.entry_as_of,
        "exit_as_of": exit_as_of,
        "secid": position.secid,
        "side": position.side,
        "entry_price": position.entry_price,
        "exit_price": exit_price,
        "entry_bid": position.entry_bid,
        "entry_ask": position.entry_ask,
        "exit_bid": exit_bid,
        "exit_ask": exit_ask,
        "entry_commission_rate": position.entry_commission_rate,
        "entry_slippage_rate": position.entry_slippage_rate,
        "exit_commission_rate": exit_commission_rate,
        "exit_slippage_rate": exit_slippage_rate,
        "net_return": net_return,
        "hold_hours": _hold_hours(position.entry_as_of, exit_as_of),
        "entry_positive_score": position.entry_positive_score,
        "entry_risk_score": position.entry_risk_score,
        "entry_net_edge": position.entry_net_edge,
        "entry_net_edge_score": position.entry_net_edge_score,
        "exit_reason": reason,
    }


def trade_net_return(
    *,
    side: str,
    entry_bid: float,
    entry_ask: float,
    exit_bid: float,
    exit_ask: float,
    entry_commission_rate: float,
    entry_slippage_rate: float,
    exit_commission_rate: float,
    exit_slippage_rate: float,
) -> float:
    costs = (
        max(float(entry_commission_rate), 0.0)
        + max(float(entry_slippage_rate), 0.0)
        + max(float(exit_commission_rate), 0.0)
        + max(float(exit_slippage_rate), 0.0)
    )
    if side == "long":
        return ((float(exit_bid) - float(entry_ask)) / float(entry_ask)) - costs if entry_ask > 0.0 else 0.0
    if side == "short":
        return ((float(entry_bid) - float(exit_ask)) / float(entry_bid)) - costs if entry_bid > 0.0 else 0.0
    raise ValueError(f"unsupported side: {side}")


def build_summary(
    trades: Sequence[Mapping[str, Any]],
    equity_curve: Sequence[float],
    decision_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    returns = [_safe_float(row.get("net_return")) for row in trades]
    risk_scores = [_safe_float(row.get("entry_risk_score")) for row in trades]
    hold_hours = [_safe_float(row.get("hold_hours")) for row in trades]
    final_equity = float(equity_curve[-1]) if equity_curve else 1.0
    return {
        "mode": "fixed_positive_replay",
        "decision_count": len({str(row.get("as_of") or "") for row in decision_rows}),
        "logged_candidate_rows": len(decision_rows),
        "trade_count": len(trades),
        "final_equity": final_equity,
        "total_return": final_equity - 1.0,
        "avg_trade_return": statistics.fmean(returns) if returns else 0.0,
        "winrate": sum(1 for value in returns if value > 0.0) / len(returns) if returns else 0.0,
        "profit_factor": _profit_factor(returns),
        "max_drawdown": _max_drawdown(equity_curve),
        "avg_hold_hours": statistics.fmean(hold_hours) if hold_hours else 0.0,
        "risk_score_min": min(risk_scores) if risk_scores else 0.0,
        "risk_score_mean": statistics.fmean(risk_scores) if risk_scores else 0.0,
        "risk_score_max": max(risk_scores) if risk_scores else 0.0,
    }


def _provider_risk_enricher(
    *,
    provider: SavedCandleMarketDataProvider,
    instruments: Mapping[str, Instrument],
) -> Callable[[Mapping[str, Any]], Mapping[str, float]]:
    scale_cache: dict[tuple[str, str], Mapping[str, float]] = {}

    def enrich(row: Mapping[str, Any]) -> Mapping[str, float]:
        secid = str(row.get("secid") or "")
        as_of = datetime.fromisoformat(str(row.get("as_of")))
        instrument = instruments.get(secid)
        if instrument is None:
            return _risk_vector_from_row(row)
        key = (secid, as_of.isoformat(timespec="seconds"))
        if key not in scale_cache:
            candles = provider.candles(as_of, [instrument]).get(secid)
            snapshot = provider.snapshots(as_of, [instrument]).get(secid)
            metric = provider.metrics(as_of, [instrument]).get(secid)
            scale_cache[key] = instrument_scales_from_history(candles, snapshot=snapshot, metric=metric)
        return _risk_vector_from_row(row, instrument_scales=scale_cache[key])

    return enrich


def _risk_vector_from_row(row: Mapping[str, Any], instrument_scales: Mapping[str, float] | None = None) -> Mapping[str, float]:
    candidate = {
        "secid": str(row.get("secid") or ""),
        "direction": str(row.get("side") or ""),
        "side": str(row.get("side") or ""),
        "bid": _safe_float(row.get("current_bid")),
        "ask": _safe_float(row.get("current_ask")),
        "pred_open": _safe_float(row.get("pred_open")),
        "pred_high": _safe_float(row.get("pred_high")),
        "pred_low": _safe_float(row.get("pred_low")),
        "pred_close": _safe_float(row.get("pred_close")),
        "commission_rate": _safe_float(row.get("commission_rate")),
        "slippage_rate": _safe_float(row.get("slippage_rate")),
    }
    return compute_candidate_vectors(candidate, instrument_scales)["risk_vector"]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_trades_csv(path: Path, trades: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "entry_as_of",
        "exit_as_of",
        "secid",
        "side",
        "entry_price",
        "exit_price",
        "entry_bid",
        "entry_ask",
        "exit_bid",
        "exit_ask",
        "entry_commission_rate",
        "entry_slippage_rate",
        "exit_commission_rate",
        "exit_slippage_rate",
        "net_return",
        "hold_hours",
        "entry_positive_score",
        "entry_risk_score",
        "entry_net_edge",
        "entry_net_edge_score",
        "capital_fraction",
        "portfolio_return",
        "entry_rank",
        "entry_reason",
        "exit_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in trades:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _candidate_gross_pred_return(candidate: Mapping[str, Any]) -> float:
    raw = candidate.get("raw_vector_metrics")
    if isinstance(raw, Mapping):
        gross = raw.get("gross_edge")
        if gross is not None:
            return abs(_safe_float(gross))
    return abs(_safe_float(candidate.get("gross_pred_return")))


def _candidate_raw_mae_pct(candidate: Mapping[str, Any]) -> float:
    raw = candidate.get("raw_vector_metrics")
    if isinstance(raw, Mapping):
        return _safe_float(raw.get("mae_pct"))
    return _safe_float(candidate.get("raw_mae_pct"))


def _candidate_raw_rr(candidate: Mapping[str, Any]) -> float:
    raw = candidate.get("raw_vector_metrics")
    if isinstance(raw, Mapping):
        return _safe_float(raw.get("rr"))
    return _safe_float(candidate.get("raw_rr"))


def _ranking_value(candidate: Mapping[str, Any], ranking_metric: str) -> float:
    if ranking_metric == "net_edge":
        return _safe_float(candidate.get("net_edge"))
    if ranking_metric == "positive_score":
        return _safe_float(candidate.get("positive_score"))
    raise ValueError(f"unsupported ranking metric: {ranking_metric}")


def _single_top_runtime_like_filter_reason(
    *,
    gross_pred_return: float,
    net_edge: float,
    positive_score: float,
    raw_mae_pct: float,
    raw_rr: float,
    params: SingleTopRuntimeLikeParams,
    rank_gap: Any | None = None,
) -> str:
    if params.max_positive_score is not None and float(positive_score) > float(params.max_positive_score):
        return "positive_score_above_max"
    if params.min_raw_mae_pct is not None and float(raw_mae_pct) <= float(params.min_raw_mae_pct):
        return "single_top_raw_mae_pct_below_min"
    if params.max_raw_rr is not None and float(raw_rr) >= float(params.max_raw_rr):
        return "single_top_raw_rr_above_max"
    if float(gross_pred_return) > float(params.max_gross_pred_return):
        return "single_top_gross_pred_return_cap"
    if float(net_edge) < float(params.min_net_edge):
        return "single_top_net_edge_below_min"
    if rank_gap is not None and _safe_float(rank_gap) < float(params.min_rank_gap):
        return "single_top_rank_gap_below_min"
    return ""


def _has_session_gap_after(as_of: str, next_as_of: str | None) -> bool:
    if not next_as_of:
        return True
    try:
        current_dt = datetime.fromisoformat(as_of)
        next_dt = datetime.fromisoformat(next_as_of)
    except ValueError:
        return True
    return next_dt - current_dt > timedelta(hours=1)


def _future_as_of(row: Mapping[str, Any] | None) -> str:
    if row is None:
        return ""
    return str(row.get("future_as_of") or row.get("as_of") or "")


def _attach_allocation_to_trade(trade: dict[str, Any], position: AllocatedOpenPosition) -> None:
    trade["capital_fraction"] = float(position.allocation)
    trade["entry_rank"] = int(position.entry_rank)
    trade["entry_reason"] = position.entry_reason
    trade["portfolio_return"] = float(position.allocation) * _safe_float(trade.get("net_return"))


def _candidate_key(candidate: Mapping[str, Any]) -> tuple[str, str]:
    return (str(candidate.get("secid") or ""), str(candidate.get("side") or ""))


def _candidate_key_from_position(position: OpenPosition) -> tuple[str, str]:
    return (position.secid, position.side)


def _open_top2_allocations(
    candidates: Sequence[Mapping[str, Any]],
    *,
    as_of: str,
    reason: str,
) -> list[AllocatedOpenPosition]:
    selected = list(candidates[:2])
    if not selected:
        return []
    allocation = 0.5 if len(selected) == 2 else 1.0
    return [
        AllocatedOpenPosition(
            position=open_position(candidate, as_of=as_of),
            allocation=allocation,
            entry_rank=int(candidate.get("rank", 0) or 0),
            entry_reason=reason,
        )
        for candidate in selected
    ]


def _append_entry_events(
    events: list[dict[str, Any]],
    *,
    as_of: str,
    positions: Sequence[AllocatedOpenPosition],
    candidates: Mapping[tuple[str, str], Mapping[str, Any]],
) -> None:
    for allocated in positions:
        key = _candidate_key_from_position(allocated.position)
        candidate = candidates.get(key, {})
        events.append(
            {
                "event": "entry",
                "as_of": as_of,
                **_allocated_position_payload(allocated),
                **_ranked_event_payload(candidate, prefix="entry"),
            }
        )


def _position_payload(position: OpenPosition | None) -> dict[str, Any]:
    if position is None:
        return {}
    return {
        "secid": position.secid,
        "side": position.side,
        "entry_as_of": position.entry_as_of,
        "entry_price": position.entry_price,
        "entry_positive_score": position.entry_positive_score,
        "entry_risk_score": position.entry_risk_score,
    }


def _allocated_position_payload(position: AllocatedOpenPosition) -> dict[str, Any]:
    return {
        **_position_payload(position.position),
        "capital_fraction": float(position.allocation),
        "entry_rank": int(position.entry_rank),
        "entry_reason": position.entry_reason,
    }


def _top_event_payload(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "top1_secid": str(candidate.get("secid") or ""),
        "top1_side": str(candidate.get("side") or ""),
        "top1_positive_score": _safe_float(candidate.get("positive_score")),
        "top1_risk_score": _safe_float(candidate.get("risk_score")),
        "top1_net_edge": _safe_float(candidate.get("net_edge")),
        "top1_net_edge_score": _safe_float(candidate.get("net_edge_score")),
    }


def _ranked_event_payload(candidate: Mapping[str, Any], *, prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_rank": int(candidate.get("rank", 0) or 0),
        f"{prefix}_secid": str(candidate.get("secid") or ""),
        f"{prefix}_side": str(candidate.get("side") or ""),
        f"{prefix}_positive_score": _safe_float(candidate.get("positive_score")),
        f"{prefix}_risk_score": _safe_float(candidate.get("risk_score")),
        f"{prefix}_net_edge": _safe_float(candidate.get("net_edge")),
        f"{prefix}_net_edge_score": _safe_float(candidate.get("net_edge_score")),
    }


def _trade_event_payload(trade: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "secid": trade.get("secid", ""),
        "side": trade.get("side", ""),
        "net_return": trade.get("net_return", 0.0),
        "capital_fraction": trade.get("capital_fraction", ""),
        "portfolio_return": trade.get("portfolio_return", ""),
        "exit_reason": trade.get("exit_reason", ""),
        "hold_hours": trade.get("hold_hours", 0.0),
    }


def _timestamp_in_range(value: str, *, from_dt: datetime, till_dt: datetime) -> bool:
    try:
        parsed = datetime.fromisoformat(value)
    except Exception:
        return False
    return from_dt <= parsed <= till_dt


def _hold_hours(entry_as_of: str, exit_as_of: str) -> float:
    try:
        entry = datetime.fromisoformat(str(entry_as_of))
        exit_ = datetime.fromisoformat(str(exit_as_of))
    except Exception:
        return 0.0
    return max((exit_ - entry).total_seconds() / 3600.0, 0.0)


def _profit_factor(values: Sequence[float]) -> float | None:
    gains = sum(value for value in values if value > 0.0)
    losses = sum(value for value in values if value < 0.0)
    if losses == 0.0:
        return None if gains > 0.0 else 0.0
    return gains / abs(losses)


def _max_drawdown(equity_curve: Sequence[float]) -> float:
    peak = float(equity_curve[0]) if equity_curve else 1.0
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, float(value))
        if peak > 0.0:
            max_dd = max(max_dd, (peak - float(value)) / peak)
    return max_dd


def _summary_console_payload(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "trade_count": int(summary.get("trade_count", 0) or 0),
        "total_return": float(summary.get("total_return", 0.0) or 0.0),
        "winrate": float(summary.get("winrate", 0.0) or 0.0),
        "profit_factor": summary.get("profit_factor"),
        "max_drawdown": float(summary.get("max_drawdown", 0.0) or 0.0),
    }


def _clip01(value: Any) -> float:
    return max(0.0, min(1.0, _safe_float(value)))


def _optional_unit_interval(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except Exception:
        parsed = math.nan
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be a finite float in [0, 1]")
    return parsed


def _optional_non_negative_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except Exception:
        parsed = math.nan
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be a finite non-negative float")
    return parsed


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out if math.isfinite(out) else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
