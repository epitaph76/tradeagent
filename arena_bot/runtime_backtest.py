from __future__ import annotations

import argparse
import csv
import json
import shutil
import traceback
from dataclasses import replace
from datetime import datetime, time
from pathlib import Path
from typing import Mapping

from .accounting import account_to_payload, mark_account
from .cli import _static_market_data_from_config
from .config import load_config
from .historical import _replay_timestamps
from .kronos_provider import RealKronosSignalProvider
from .logging import JsonlLogger
from .runtime import RuntimeEngine
from .storage import StateStore
from .types import Instrument, MarketSnapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="runtime-backtest")
    parser.add_argument("--config", default="configs/universe_v1_may1_14.yaml")
    parser.add_argument("--from", dest="from_date", required=True)
    parser.add_argument("--till", dest="till_date", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        print(
            json.dumps(
                {"event": "runtime_backtest_error", "error": str(exc), "traceback": traceback.format_exc()},
                ensure_ascii=False,
            ),
            flush=True,
        )
        raise
    return 0


def run(args: argparse.Namespace) -> None:
    config_path = str(args.config)
    run_dir = Path(args.run_dir)
    from_dt = datetime.fromisoformat(str(args.from_date))
    till_dt = datetime.fromisoformat(str(args.till_date))
    progress_every = max(int(args.progress_every), 0)

    base_config = load_config(config_path)
    base_state_path = Path(base_config.data_dir) / "arena_state.sqlite3"
    run_state_path = run_dir / "arena_state.sqlite3"
    run_dir.mkdir(parents=True, exist_ok=True)
    if base_state_path.exists():
        shutil.copy2(base_state_path, run_state_path)

    state = StateStore(run_state_path)
    with state.connect() as conn:
        for table in ("paper_positions", "orders", "decisions", "selector_paper_positions", "account_state", "position_giveback_state"):
            conn.execute(f"DELETE FROM {table}")
        conn.commit()

    config = replace(base_config, data_dir=str(run_dir), mode="paper")
    market_data = _static_market_data_from_config(config_path)
    timestamps = _replay_timestamps(market_data, config.instruments, from_dt=from_dt, till_dt=till_dt)
    if len(timestamps) < 2:
        raise RuntimeError("not enough timestamps")
    decision_times = _runtime_decision_times(timestamps, config=config, from_dt=from_dt, till_dt=till_dt)
    final_ts = timestamps[-1]

    engine = RuntimeEngine(
        config=config,
        market_data=market_data,
        kronos_provider=RealKronosSignalProvider(config=config.kronos, state=state),
        state=state,
        logger=JsonlLogger(run_dir / "logs", stdout=False),
    )
    instruments = {instrument.secid: instrument for instrument in config.instruments}
    cash = float(config.risk.starting_cash)
    lots: dict[str, int] = {}
    total_commission = 0.0
    executed_orders = 0
    orders_by_secid: dict[str, int] = {}
    trades_file = (run_dir / "trades.csv").open("w", encoding="utf-8", newline="")
    account_file = (run_dir / "account_curve.csv").open("w", encoding="utf-8", newline="")
    ranked_file = (run_dir / "ranked_top.jsonl").open("w", encoding="utf-8")
    blocked_file = (run_dir / "risk_blocked_orders.jsonl").open("w", encoding="utf-8")
    trade_writer = csv.DictWriter(
        trades_file,
        fieldnames=[
            "completed",
            "as_of",
            "secid",
            "direction",
            "qty",
            "price",
            "notional",
            "commission",
            "kind",
            "reason",
            "cash_after",
            "position_lots_after",
        ],
    )
    account_writer = csv.DictWriter(
        account_file,
        fieldnames=[
            "completed",
            "as_of",
            "equity_liquidation",
            "return_pct",
            "cash",
            "account_equity",
            "account_gross",
            "account_net",
            "account_margin_used",
            "account_available_cash",
            "account_available_gross",
            "positions_count",
            "positions_json",
            "executed_orders_total",
            "commission_paid",
        ],
    )
    trade_writer.writeheader()
    account_writer.writeheader()

    def liquidation_equity(ts: datetime) -> float:
        snapshots = market_data.snapshots(ts, config.instruments)
        equity = cash
        close_commission = 0.0
        for secid, pos_lots in lots.items():
            if pos_lots == 0:
                continue
            instrument = instruments[secid]
            snapshot = snapshots[secid]
            if pos_lots > 0:
                price = float(snapshot.bid or snapshot.last_price or 0.0)
            else:
                price = float(snapshot.ask or snapshot.last_price or 0.0)
            value = pos_lots * instrument.lot_size * price
            equity += value
            close_commission += abs(value) * float(config.risk.commission_rate)
        return equity - close_commission

    def marked_account(ts: datetime):
        snapshots = market_data.snapshots(ts, config.instruments)
        prices = {secid: snapshot.last_price for secid, snapshot in snapshots.items()}
        return mark_account(
            cash=cash,
            positions=lots,
            prices=prices,
            instruments=instruments,
            risk=config.risk,
            max_gross=config.portfolio.max_gross,
        )

    print(
        json.dumps(
            {
                "event": "runtime_backtest_start",
                "run_dir": str(run_dir),
                "entry_mode": config.trade_lifecycle.entry.mode,
                "timestamps": len(timestamps),
                "decisions": len(decision_times),
                "first_as_of": decision_times[0].isoformat(timespec="seconds"),
                "last_decision_as_of": decision_times[-1].isoformat(timespec="seconds"),
                "final_mark_as_of": final_ts.isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    try:
        for idx, as_of in enumerate(decision_times, start=1):
            result = engine.run_once(as_of)
            decision_payload = _load_decision_payload(state, result.decision_id)
            as_of_s = as_of.isoformat(timespec="seconds")
            for row in decision_payload.get("entry_ranked_candidates", []):
                ranked_file.write(json.dumps({"completed": idx, "as_of": as_of_s, **dict(row)}, ensure_ascii=False) + "\n")
            for row in decision_payload.get("risk_blocked_orders", []):
                blocked_file.write(json.dumps({"completed": idx, "as_of": as_of_s, **dict(row)}, ensure_ascii=False) + "\n")

            snapshots = market_data.snapshots(as_of, config.instruments)
            order_rows = []
            for order in result.orders:
                if order.status not in {"dry_run", "submitted"}:
                    continue
                instrument = instruments[order.secid]
                snapshot = snapshots[order.secid]
                price = float(order.request.get("price") or _execution_price(snapshot, order.direction))
                delta = int(order.quantity) if order.direction == "B" else -int(order.quantity)
                notional = float(order.request.get("order_value") or (abs(delta) * instrument.lot_size * price))
                commission = notional * float(config.risk.commission_rate)
                if order.direction == "B":
                    cash -= notional + commission
                else:
                    cash += notional - commission
                lots[order.secid] = int(lots.get(order.secid, 0)) + delta
                if lots[order.secid] == 0:
                    lots.pop(order.secid, None)
                total_commission += commission
                executed_orders += 1
                orders_by_secid[order.secid] = orders_by_secid.get(order.secid, 0) + 1
                order_row = {
                    "secid": order.secid,
                    "direction": order.direction,
                    "qty": int(order.quantity),
                    "price": price,
                    "notional": notional,
                    "commission": commission,
                    "kind": order.request.get("order_kind"),
                    "reason": order.request.get("reason"),
                }
                order_rows.append(order_row)
                trade_writer.writerow(
                    {
                        "completed": idx,
                        "as_of": as_of_s,
                        **order_row,
                        "cash_after": cash,
                        "position_lots_after": int(lots.get(order.secid, 0)),
                    }
                )

            equity = liquidation_equity(as_of)
            account = marked_account(as_of)
            return_pct = (equity / float(config.risk.starting_cash) - 1.0) * 100.0
            account_writer.writerow(
                {
                    "completed": idx,
                    "as_of": as_of_s,
                    "equity_liquidation": equity,
                    "return_pct": return_pct,
                    "cash": cash,
                    "account_equity": account.equity,
                    "account_gross": account.gross,
                    "account_net": account.net,
                    "account_margin_used": account.margin_used,
                    "account_available_cash": account.available_cash,
                    "account_available_gross": account.available_gross,
                    "positions_count": len(lots),
                    "positions_json": json.dumps(dict(sorted(lots.items())), ensure_ascii=False, sort_keys=True),
                    "executed_orders_total": executed_orders,
                    "commission_paid": total_commission,
                }
            )
            trades_file.flush()
            account_file.flush()
            ranked_file.flush()
            blocked_file.flush()
            if progress_every and (idx % progress_every == 0 or idx == 1 or idx == len(decision_times)):
                print(
                    json.dumps(
                        {
                            "event": "runtime_backtest_progress",
                            "completed": idx,
                            "total": len(decision_times),
                            "as_of": as_of_s,
                            "equity_liquidation": equity,
                            "return_pct": return_pct,
                            "cash": cash,
                            "account": account_to_payload(account),
                            "positions_count": len(lots),
                            "positions": dict(sorted(lots.items())),
                            "orders_this_tick": order_rows,
                            "executed_orders_total": executed_orders,
                            "commission_paid": total_commission,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        final_equity = liquidation_equity(final_ts)
        final_account = marked_account(final_ts)
        summary = {
            "event": "runtime_backtest_done",
            "run_dir": str(run_dir),
            "first_as_of": decision_times[0].isoformat(timespec="seconds"),
            "last_decision_as_of": decision_times[-1].isoformat(timespec="seconds"),
            "final_mark_as_of": final_ts.isoformat(timespec="seconds"),
            "decisions": len(decision_times),
            "starting_cash": float(config.risk.starting_cash),
            "final_equity_liquidation": final_equity,
            "return_pct": (final_equity / float(config.risk.starting_cash) - 1.0) * 100.0,
            "cash": cash,
            "account": account_to_payload(final_account),
            "open_positions": dict(sorted(lots.items())),
            "executed_orders_total": executed_orders,
            "commission_paid": total_commission,
            "orders_by_secid": dict(sorted(orders_by_secid.items())),
            "exports": {
                "trades_csv": str(run_dir / "trades.csv"),
                "ranked_top_jsonl": str(run_dir / "ranked_top.jsonl"),
                "risk_blocked_orders_jsonl": str(run_dir / "risk_blocked_orders.jsonl"),
                "account_curve_csv": str(run_dir / "account_curve.csv"),
            },
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        trades_file.close()
        account_file.close()
        ranked_file.close()
        blocked_file.close()


def _execution_price(snapshot: MarketSnapshot, direction: str) -> float:
    if direction == "B":
        return float(snapshot.ask or snapshot.last_price or 0.0)
    return float(snapshot.bid or snapshot.last_price or 0.0)


def _runtime_decision_times(
    timestamps: list[datetime],
    *,
    config: object,
    from_dt: datetime,
    till_dt: datetime,
) -> list[datetime]:
    decision_times = list(timestamps[:-1])
    session = getattr(config, "trading_session", None)
    if session is None or not bool(getattr(session, "enabled", False)):
        return decision_times
    force_flat_raw = str(getattr(session, "force_flat_time", "") or "")
    if not force_flat_raw:
        return decision_times
    force_flat = _parse_hhmm(force_flat_raw)
    dates = sorted({ts.date() for ts in timestamps})
    synthetic = [
        datetime.combine(day, force_flat)
        for day in dates
        if from_dt <= datetime.combine(day, force_flat) <= till_dt
        and timestamps[0] <= datetime.combine(day, force_flat) <= timestamps[-1]
    ]
    return sorted(set(decision_times).union(synthetic))


def _parse_hhmm(value: str) -> time:
    hour, minute = str(value).split(":", 1)
    return time(hour=int(hour), minute=int(minute))


def _load_decision_payload(state: StateStore, decision_id: str) -> dict[str, object]:
    with state.connect() as conn:
        row = conn.execute("SELECT payload_json FROM decisions WHERE decision_id = ?", (decision_id,)).fetchone()
    if row is None:
        return {}
    try:
        payload = json.loads(row["payload_json"])
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
