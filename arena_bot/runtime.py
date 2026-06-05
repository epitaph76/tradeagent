from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, replace
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

from .accounting import account_to_payload, filter_orders_no_leverage, mark_account, project_orders
from .arenago import ArenaGoClient
from .kronos_provider import RealKronosSignalProvider
from .logging import JsonlLogger
from .market_data import EmptyMarketDataProvider, MarketDataProvider, build_market_features
from .meta_selector import LightGBMMetaSelector, RollingRankWeightedMetaSelector
from .order_manager import OrderManager
from .portfolio import blend_selector_portfolios, prune_blended_weights
from .selectors import build_selector_portfolio
from .signals import EmptyKronosSignalProvider, MomentumSignalProvider, SignalProvider, latest_signal_scores, with_equity_kronos_fallback
from .storage import StateStore
from .types import AccountState, DecisionResult, Instrument, RuntimeConfig, SignalRow, TradingSessionConfig
from .universe import select_universe


class RuntimeEngine:
    def __init__(
        self,
        *,
        config: RuntimeConfig,
        market_data: MarketDataProvider | None = None,
        kronos_provider: SignalProvider | None = None,
        exit_kronos_provider: SignalProvider | None = None,
        momentum_provider: SignalProvider | None = None,
        state: StateStore | None = None,
        logger: JsonlLogger | None = None,
        arenago_client: ArenaGoClient | None = None,
    ):
        self.config = config
        self.market_data = market_data or EmptyMarketDataProvider()
        data_dir = Path(config.data_dir)
        self.state = state or StateStore(data_dir / "arena_state.sqlite3")
        self.logger = logger or JsonlLogger(data_dir / "logs")
        self.kronos_provider = kronos_provider or EmptyKronosSignalProvider()
        self.exit_kronos_provider = exit_kronos_provider or self._build_exit_kronos_provider()
        self.momentum_provider = momentum_provider or MomentumSignalProvider()
        self.instruments_by_secid: dict[str, Instrument] = {instrument.secid: instrument for instrument in config.instruments}
        self.lightgbm = LightGBMMetaSelector(
            model_dir=_model_dir(config),
            base_selectors=[selector.name for selector in config.base_selectors],
            rank_power=config.lightgbm.rank_power,
            max_model_age_hours=config.lightgbm.max_model_age_hours,
        )
        self.rolling = RollingRankWeightedMetaSelector(
            base_selectors=[selector.name for selector in config.base_selectors],
            lookback=config.lightgbm.rolling_lookback_intervals,
            rank_power=config.lightgbm.rank_power,
        )
        self.order_manager = OrderManager(
            config=config.risk,
            instruments=self.instruments_by_secid,
            state=self.state,
            bot_name=config.bot_name,
            client=arenago_client,
            live_orders=config.live_orders,
        )

    def _build_exit_kronos_provider(self) -> SignalProvider:
        lifecycle = self.config.trade_lifecycle
        if not self.config.kronos.enabled or not lifecycle.exit.enabled:
            return EmptyKronosSignalProvider()
        return RealKronosSignalProvider(
            config=replace(
                self.config.kronos,
                pred_len=max(int(lifecycle.exit.pred_len), 1),
                sample_count=max(int(lifecycle.exit.sample_count), 1),
            ),
            state=self.state,
        )

    def run_once(self, as_of: datetime | None = None) -> DecisionResult:
        as_of = as_of or datetime.now()
        as_of_s = as_of.isoformat(timespec="seconds")
        session = _trading_session_state(as_of, self.config.trading_session)
        snapshots = self.market_data.snapshots(as_of, self.config.instruments)
        candles = self.market_data.candles(as_of, self.config.instruments)
        metrics = self.market_data.metrics(as_of, self.config.instruments)
        prices = {secid: snapshot.last_price for secid, snapshot in snapshots.items()}
        buy_prices, sell_prices = _execution_price_maps(snapshots)
        positions_before_exit = self.order_manager.current_positions()
        account_before = self._load_account_state(positions_before_exit, prices)
        if bool(session.get("force_flat_required", False)):
            return self._run_force_flat_session(
                as_of=as_of,
                as_of_s=as_of_s,
                session=session,
                snapshots=snapshots,
                metrics=metrics,
                candles=candles,
                prices=prices,
                buy_prices=buy_prices,
                sell_prices=sell_prices,
                positions_before_exit=positions_before_exit,
                account_before=account_before,
            )
        if bool(session.get("enabled", False)) and str(session.get("session_state")) in {"pre_session", "warmup", "closed"}:
            return self._run_session_idle(
                as_of=as_of,
                as_of_s=as_of_s,
                session=session,
                positions=positions_before_exit,
                prices=prices,
                account=account_before,
            )
        exit_diagnostics, exit_scores = self._build_exit_plan(
            as_of=as_of,
            positions=positions_before_exit,
            snapshots=snapshots,
            metrics=metrics,
            candles=candles,
            session=session,
        )
        exit_orders = self.order_manager.plan_exit_orders(
            close_secids=set(exit_diagnostics.get("close_secids", [])),
            target_scores=exit_scores,
            positions=positions_before_exit,
            prices=prices,
            buy_prices=buy_prices,
            sell_prices=sell_prices,
        )
        exit_close_secids = {order.secid for order in exit_orders if order.target_lots == 0}
        exit_diagnostics["planned_close_secids"] = sorted(exit_close_secids)
        positions_after_exit = _positions_after_exit(positions_before_exit, exit_close_secids)
        account_after_exit = project_orders(
            cash=account_before.cash,
            positions=positions_before_exit,
            orders=exit_orders,
            prices=prices,
            instruments=self.instruments_by_secid,
            risk=self.config.risk,
            max_gross=self.config.portfolio.max_gross,
        )
        risk_cap_target_weights, risk_cap_diagnostics = _build_risk_cap_targets(
            account=account_after_exit,
            positions=positions_after_exit,
            prices=prices,
            instruments=self.instruments_by_secid,
            risk=self.config.risk,
            max_gross=self.config.portfolio.max_gross,
        )
        risk_cap_orders = [
            replace(order, reason="risk_cap_pass")
            for order in self.order_manager.plan_orders(
                target_weights=risk_cap_target_weights,
                target_scores={},
                positions=positions_after_exit,
                prices=prices,
                cash=account_after_exit.cash,
                buy_prices=buy_prices,
                sell_prices=sell_prices,
            )
        ]
        positions_after_risk = _positions_after_orders(positions_after_exit, risk_cap_orders)
        account_after_risk = project_orders(
            cash=account_after_exit.cash,
            positions=positions_after_exit,
            orders=risk_cap_orders,
            prices=prices,
            instruments=self.instruments_by_secid,
            risk=self.config.risk,
            max_gross=self.config.portfolio.max_gross,
        )
        risk_cap_diagnostics["planned_orders"] = [asdict(order) for order in risk_cap_orders]
        free_capital_after_exit = account_after_risk.available_gross

        selector_decisions: dict[str, Any] = {}
        selector_diagnostics: dict[str, Any] = {}
        signals: tuple[SignalRow, ...] = tuple()
        universe_diagnostics: dict[str, Any] = {}
        entry_block_reason = ""
        horizon_diagnostics: dict[str, Any] = {}
        selected: tuple[Instrument, ...] = tuple()
        if not bool(session.get("entry_allowed", True)):
            entry_block_reason = str(session.get("action_reason") or "session_entry_cutoff")
            universe_diagnostics = {"status": "session_skipped", "reason": entry_block_reason}
        else:
            universe = select_universe(
                self.config.instruments,
                snapshots=snapshots,
                metrics=metrics,
                max_equities=self.config.max_equities,
            )
            universe_diagnostics = universe.diagnostics
            for secid, reason in universe.diagnostics.get("rejected", {}).items():
                self.logger.write("instrument_untradable", {"as_of": as_of_s, "secid": secid, "reason": reason})
            selected = universe.instruments
            horizon_ok, horizon_diagnostics = _kronos_horizon_check(
                candles=candles,
                instruments=selected,
                as_of=as_of,
                pred_len=max(int(self.config.kronos.pred_len), 1),
                max_target_time=session.get("_kronos_cutoff_dt"),
            )
            if not horizon_ok:
                entry_block_reason = "session_kronos_horizon_exceeds_close"

        if entry_block_reason:
            target_weights = _position_weights(positions_after_risk, prices, self.instruments_by_secid, max(float(account_after_risk.equity), 1.0))
            selector_weights = {"session_guard": 1.0}
            features = build_market_features(
                selected_secids=tuple(instrument.secid for instrument in selected),
                snapshots=snapshots,
                metrics=metrics,
                signal_scores={},
            )
            if features:
                self.state.save_market_features(as_of_s, features)
            meta_payload = {
                "mode": "session_guard",
                "reason": entry_block_reason,
                "features": features,
            }
            if horizon_diagnostics:
                meta_payload["kronos_horizon"] = horizon_diagnostics
            entry_diagnostics = {
                "status": "blocked",
                "reason": entry_block_reason,
                "selected_count": 0,
                "ranked_candidates": [],
            }
            if horizon_diagnostics:
                entry_diagnostics["kronos_horizon"] = horizon_diagnostics
            blend_diagnostics = {
                "mode": "session_guard",
                "reason": entry_block_reason,
                "final_target_positions_count": len([weight for weight in target_weights.values() if abs(float(weight)) > 1e-12]),
            }
            self.logger.write("selector_model_ready", {"as_of": as_of_s, **meta_payload})
        else:
            self._update_selector_returns(as_of_s, {secid: row.last_price for secid, row in snapshots.items()})
            if _entry_mode(self.config) == "kronos_rank":
                kronos_rows = tuple(self.kronos_provider.score(as_of, selected, candles))
                signals = tuple(row for row in kronos_rows if row.signal_name == "kronos")
                signal_scores = latest_signal_scores(signals, "kronos")
                features = build_market_features(
                    selected_secids=tuple(instrument.secid for instrument in selected),
                    snapshots=snapshots,
                    metrics=metrics,
                    signal_scores=signal_scores,
                )
                self.state.save_market_features(as_of_s, features)
                selector_weights = {"kronos_rank": 1.0}
                meta_payload = {
                    "mode": "bypassed_kronos_rank",
                    "reason": "trade_lifecycle.entry.mode=kronos_rank",
                    "features": features,
                }
                selector_diagnostics = {}
                target_weights, entry_diagnostics = self._build_kronos_rank_entry_targets(
                    instruments=selected,
                    positions=positions_after_risk,
                    account=account_after_risk,
                    prices=prices,
                    snapshots=snapshots,
                    metrics=metrics,
                    signals=signals,
                    blocked_secids=exit_close_secids,
                )
                blend_diagnostics = {
                    "mode": "bypassed_kronos_rank",
                    "ranking_mode": "kronos_rank",
                    "final_target_positions_count": len([weight for weight in target_weights.values() if abs(float(weight)) > 1e-12]),
                    "ranked_candidates_count": len(entry_diagnostics.get("ranked_candidates", [])),
                    "selected_count": int(entry_diagnostics.get("selected_count", 0) or 0),
                }
                self.logger.write("selector_model_ready", {"as_of": as_of_s, **meta_payload})
            else:
                kronos_rows = tuple(self.kronos_provider.score(as_of, selected, candles))
                momentum_rows = tuple(self.momentum_provider.score(as_of, selected, candles))
                signals = with_equity_kronos_fallback(as_of=as_of, instruments=selected, kronos_rows=kronos_rows, momentum_rows=momentum_rows)
                signal_scores = latest_signal_scores(signals, "kronos")
                signal_scores.update({k: v for k, v in latest_signal_scores(signals, "momentum").items() if k not in signal_scores})

                selector_decisions = {
                    selector.name: build_selector_portfolio(selector, instruments=selected, signals=signals)
                    for selector in self.config.base_selectors
                }
                selector_diagnostics = {name: dict(decision.diagnostics) for name, decision in selector_decisions.items()}
                self.logger.write("base_selectors_ready", {"as_of": as_of_s, "selectors": selector_diagnostics})

                features = build_market_features(
                    selected_secids=tuple(instrument.secid for instrument in selected),
                    snapshots=snapshots,
                    metrics=metrics,
                    signal_scores=signal_scores,
                )
                self.state.save_market_features(as_of_s, features)

                history = self.state.load_selector_return_history(limit=self.config.lightgbm.train_lookback_intervals)
                meta_result = self.lightgbm.predict_weights(features)
                if meta_result.mode != "lightgbm" or not meta_result.selector_weights:
                    selector_weights = self.rolling.weights(history)
                    meta_payload = {**asdict(meta_result), "fallback_weights": selector_weights}
                else:
                    selector_weights = meta_result.selector_weights
                    meta_payload = asdict(meta_result)
                self.logger.write("selector_model_ready", {"as_of": as_of_s, **meta_payload})

                blended = blend_selector_portfolios(selector_decisions, selector_weights)
                candidate_weights, blend_diagnostics = prune_blended_weights(blended, self.config.portfolio)
                target_weights, entry_diagnostics = self._build_entry_targets(
                    candidate_weights=candidate_weights,
                    positions=positions_after_risk,
                    account=account_after_risk,
                    prices=prices,
                    signals=signals,
                    blocked_secids=exit_close_secids,
                )
        target_scores = _target_scores(target_weights, signals)
        remaining_order_slots = max(int(self.config.risk.max_orders_per_rebalance) - len(exit_orders) - len(risk_cap_orders), 0)
        entry_orders = self.order_manager.plan_entry_orders(
            target_weights=target_weights,
            target_scores=target_scores,
            positions=positions_after_risk,
            prices=prices,
            cash=account_after_risk.cash,
            buy_prices=buy_prices,
            sell_prices=sell_prices,
        )[:remaining_order_slots]
        planned_candidates = list(exit_orders) + list(risk_cap_orders) + list(entry_orders)
        planned, risk_blocked_orders, projected_account_after_orders = filter_orders_no_leverage(
            cash=account_before.cash,
            positions=positions_before_exit,
            orders=planned_candidates,
            prices=prices,
            instruments=self.instruments_by_secid,
            risk=self.config.risk,
            max_gross=self.config.portfolio.max_gross,
        )
        decision_id = _decision_id(as_of_s, selector_weights, target_weights)
        orders = self.order_manager.execute_orders(decision_id=decision_id, as_of=as_of, orders=planned)
        executed_planned = [order for order, result in zip(planned, orders) if result.status in {"dry_run", "submitted"}]
        account_after_orders = project_orders(
            cash=account_before.cash,
            positions=positions_before_exit,
            orders=executed_planned,
            prices=prices,
            instruments=self.instruments_by_secid,
            risk=self.config.risk,
            max_gross=self.config.portfolio.max_gross,
        )
        particle_tracker_sync = self._sync_particle_trackers_after_orders(
            as_of=as_of,
            planned=planned,
            orders=orders,
            snapshots=snapshots,
            metrics=metrics,
            candles=candles,
            session=session,
        )
        self.state.save_account_state(account_to_payload(account_after_orders), account_id=self.config.bot_name, as_of=as_of_s)
        for selector_name, decision in selector_decisions.items():
            self.state.save_selector_positions(selector_name, decision.weights, prices, as_of_s)

        payload = {
            "decision_id": decision_id,
            "as_of": as_of_s,
            "mode": self.config.mode,
            "news_enabled": False,
            "session": _session_public_payload(session),
            "session_state": session.get("session_state"),
            "session_open": session.get("session_open"),
            "entry_start": session.get("entry_start"),
            "new_entry_cutoff": session.get("new_entry_cutoff"),
            "kronos_cutoff": session.get("kronos_cutoff"),
            "force_flat_time": session.get("force_flat_time"),
            "session_close": session.get("session_close"),
            "kronos_allowed": bool(session.get("kronos_allowed", True)),
            "entry_allowed": bool(session.get("entry_allowed", True)),
            "force_flat_required": bool(session.get("force_flat_required", False)),
            "universe": universe_diagnostics,
            "selector_weights": selector_weights,
            "selector_model": meta_payload,
            "target_weights": target_weights,
            "selector_diagnostics": selector_diagnostics,
            "blend_diagnostics": blend_diagnostics,
            "exit_diagnostics": exit_diagnostics,
            "risk_cap_diagnostics": risk_cap_diagnostics,
            "entry_diagnostics": entry_diagnostics,
            "account_before": account_to_payload(account_before),
            "account_after_exit": account_to_payload(account_after_exit),
            "account_after_risk": account_to_payload(account_after_risk),
            "projected_account_after_orders": account_to_payload(projected_account_after_orders),
            "account_after_orders": account_to_payload(account_after_orders),
            "risk_blocked_orders": risk_blocked_orders,
            "free_capital_after_exit": free_capital_after_exit,
            "held_positions_after_exit": positions_after_exit,
            "held_positions_after_risk": positions_after_risk,
            "particle_tracker_sync": particle_tracker_sync,
            "entry_ranked_candidates": entry_diagnostics.get("ranked_candidates", []),
            "orders": [asdict(order) for order in orders],
        }
        self.state.insert_decision(decision_id, as_of_s, payload)
        self.logger.write("decision", payload)
        return DecisionResult(
            decision_id=decision_id,
            as_of=as_of,
            selector_weights=selector_weights,
            target_weights=target_weights,
            selector_diagnostics=selector_diagnostics,
            blend_diagnostics=blend_diagnostics,
            orders=orders,
        )

    def _run_session_idle(
        self,
        *,
        as_of: datetime,
        as_of_s: str,
        session: Mapping[str, Any],
        positions: Mapping[str, int],
        prices: Mapping[str, float],
        account: AccountState,
    ) -> DecisionResult:
        reason = str(session.get("action_reason") or "session_blocked")
        if str(session.get("session_state")) == "closed" and any(int(lots or 0) != 0 for lots in positions.values()):
            self.logger.write(
                "session_position_violation",
                {"as_of": as_of_s, "positions": {secid: lots for secid, lots in positions.items() if int(lots or 0) != 0}},
            )
            reason = "session_position_violation"
        selector_weights = {"session_guard": 1.0}
        target_weights = _position_weights(positions, prices, self.instruments_by_secid, max(float(account.equity), 1.0))
        selector_diagnostics: dict[str, Any] = {}
        blend_diagnostics = {
            "mode": "session_guard",
            "reason": reason,
            "final_target_positions_count": len(target_weights),
        }
        exit_diagnostics = {
            "enabled": bool(self.config.trade_lifecycle.exit.enabled),
            "status": "session_blocked",
            "positions_count": len([lots for lots in positions.values() if int(lots or 0) != 0]),
            "close_secids": [],
            "held": {
                secid: {"action": "hold", "action_reason": reason}
                for secid, lots in positions.items()
                if int(lots or 0) != 0
            },
        }
        risk_cap_diagnostics = {"enabled": False, "status": "session_blocked", "reason": reason}
        entry_diagnostics = {"status": "blocked", "reason": reason}
        decision_id = _decision_id(as_of_s, selector_weights, target_weights)
        self.state.save_account_state(account_to_payload(account), account_id=self.config.bot_name, as_of=as_of_s)
        payload = {
            "decision_id": decision_id,
            "as_of": as_of_s,
            "mode": self.config.mode,
            "news_enabled": False,
            "session": _session_public_payload(session),
            "session_state": session.get("session_state"),
            "session_open": session.get("session_open"),
            "entry_start": session.get("entry_start"),
            "new_entry_cutoff": session.get("new_entry_cutoff"),
            "kronos_cutoff": session.get("kronos_cutoff"),
            "force_flat_time": session.get("force_flat_time"),
            "session_close": session.get("session_close"),
            "kronos_allowed": bool(session.get("kronos_allowed", True)),
            "entry_allowed": bool(session.get("entry_allowed", True)),
            "force_flat_required": bool(session.get("force_flat_required", False)),
            "universe": {"status": "session_skipped", "reason": reason},
            "selector_weights": selector_weights,
            "selector_model": {"mode": "session_guard", "reason": reason},
            "target_weights": target_weights,
            "selector_diagnostics": selector_diagnostics,
            "blend_diagnostics": blend_diagnostics,
            "exit_diagnostics": exit_diagnostics,
            "risk_cap_diagnostics": risk_cap_diagnostics,
            "entry_diagnostics": entry_diagnostics,
            "account_before": account_to_payload(account),
            "account_after_exit": account_to_payload(account),
            "account_after_risk": account_to_payload(account),
            "projected_account_after_orders": account_to_payload(account),
            "account_after_orders": account_to_payload(account),
            "risk_blocked_orders": [],
            "free_capital_after_exit": float(account.available_gross),
            "held_positions_after_exit": dict(positions),
            "held_positions_after_risk": dict(positions),
            "particle_tracker_sync": {},
            "entry_ranked_candidates": [],
            "orders": [],
        }
        self.state.insert_decision(decision_id, as_of_s, payload)
        self.logger.write("decision", payload)
        return DecisionResult(
            decision_id=decision_id,
            as_of=as_of,
            selector_weights=selector_weights,
            target_weights=target_weights,
            selector_diagnostics=selector_diagnostics,
            blend_diagnostics=blend_diagnostics,
            orders=tuple(),
        )

    def _run_force_flat_session(
        self,
        *,
        as_of: datetime,
        as_of_s: str,
        session: Mapping[str, Any],
        snapshots: Mapping[str, Any],
        metrics: Mapping[str, Any],
        candles: Mapping[str, Any],
        prices: Mapping[str, float],
        buy_prices: Mapping[str, float],
        sell_prices: Mapping[str, float],
        positions_before_exit: Mapping[str, int],
        account_before: AccountState,
    ) -> DecisionResult:
        close_secids = sorted(secid for secid, lots in positions_before_exit.items() if int(lots or 0) != 0)
        exit_scores = {secid: 1.0 for secid in close_secids}
        exit_diagnostics = {
            "enabled": bool(self.config.trade_lifecycle.exit.enabled),
            "status": "session_force_flat",
            "positions_count": len(close_secids),
            "close_secids": close_secids,
            "held": {
                secid: {
                    "action": "close",
                    "action_reason": "session_force_flat",
                    "side": "long" if int(positions_before_exit.get(secid, 0) or 0) > 0 else "short",
                }
                for secid in close_secids
            },
        }
        exit_orders = self.order_manager.plan_exit_orders(
            close_secids=set(close_secids),
            target_scores=exit_scores,
            positions=positions_before_exit,
            prices=prices,
            buy_prices=buy_prices,
            sell_prices=sell_prices,
        )
        exit_close_secids = {order.secid for order in exit_orders if order.target_lots == 0}
        exit_diagnostics["planned_close_secids"] = sorted(exit_close_secids)
        positions_after_exit = _positions_after_exit(positions_before_exit, exit_close_secids)
        account_after_exit = project_orders(
            cash=account_before.cash,
            positions=positions_before_exit,
            orders=exit_orders,
            prices=prices,
            instruments=self.instruments_by_secid,
            risk=self.config.risk,
            max_gross=self.config.portfolio.max_gross,
        )
        planned, risk_blocked_orders, projected_account_after_orders = filter_orders_no_leverage(
            cash=account_before.cash,
            positions=positions_before_exit,
            orders=list(exit_orders),
            prices=prices,
            instruments=self.instruments_by_secid,
            risk=self.config.risk,
            max_gross=self.config.portfolio.max_gross,
        )
        selector_weights = {"session_force_flat": 1.0}
        target_weights = _position_weights(positions_after_exit, prices, self.instruments_by_secid, max(float(account_after_exit.equity), 1.0))
        decision_id = _decision_id(as_of_s, selector_weights, target_weights)
        orders = self.order_manager.execute_orders(decision_id=decision_id, as_of=as_of, orders=planned)
        executed_planned = [order for order, result in zip(planned, orders) if result.status in {"dry_run", "submitted"}]
        account_after_orders = project_orders(
            cash=account_before.cash,
            positions=positions_before_exit,
            orders=executed_planned,
            prices=prices,
            instruments=self.instruments_by_secid,
            risk=self.config.risk,
            max_gross=self.config.portfolio.max_gross,
        )
        particle_tracker_sync = self._sync_particle_trackers_after_orders(
            as_of=as_of,
            planned=planned,
            orders=orders,
            snapshots=snapshots,
            metrics=metrics,
            candles=candles,
            session=session,
        )
        force_flat_deleted = []
        for order, result in zip(planned, orders):
            if int(getattr(order, "target_lots", 0) or 0) == 0 and result.status in {"dry_run", "submitted"}:
                self.state.delete_kronos_exit_tracker(order.secid)
                force_flat_deleted.append(order.secid)
        if force_flat_deleted:
            particle_tracker_sync = {
                **dict(particle_tracker_sync),
                "force_flat_deleted": sorted(set(force_flat_deleted)),
            }
        self.state.save_account_state(account_to_payload(account_after_orders), account_id=self.config.bot_name, as_of=as_of_s)
        selector_diagnostics: dict[str, Any] = {}
        blend_diagnostics = {
            "mode": "session_force_flat",
            "reason": "session_force_flat",
            "final_target_positions_count": len(target_weights),
        }
        risk_cap_diagnostics = {"enabled": False, "status": "session_skipped", "reason": "session_force_flat"}
        entry_diagnostics = {"status": "blocked", "reason": "session_force_flat"}
        payload = {
            "decision_id": decision_id,
            "as_of": as_of_s,
            "mode": self.config.mode,
            "news_enabled": False,
            "session": _session_public_payload(session),
            "session_state": session.get("session_state"),
            "session_open": session.get("session_open"),
            "entry_start": session.get("entry_start"),
            "new_entry_cutoff": session.get("new_entry_cutoff"),
            "kronos_cutoff": session.get("kronos_cutoff"),
            "force_flat_time": session.get("force_flat_time"),
            "session_close": session.get("session_close"),
            "kronos_allowed": bool(session.get("kronos_allowed", False)),
            "entry_allowed": bool(session.get("entry_allowed", False)),
            "force_flat_required": bool(session.get("force_flat_required", True)),
            "universe": {"status": "session_skipped", "reason": "session_force_flat"},
            "selector_weights": selector_weights,
            "selector_model": {"mode": "session_force_flat", "reason": "session_force_flat"},
            "target_weights": target_weights,
            "selector_diagnostics": selector_diagnostics,
            "blend_diagnostics": blend_diagnostics,
            "exit_diagnostics": exit_diagnostics,
            "risk_cap_diagnostics": risk_cap_diagnostics,
            "entry_diagnostics": entry_diagnostics,
            "account_before": account_to_payload(account_before),
            "account_after_exit": account_to_payload(account_after_exit),
            "account_after_risk": account_to_payload(account_after_exit),
            "projected_account_after_orders": account_to_payload(projected_account_after_orders),
            "account_after_orders": account_to_payload(account_after_orders),
            "risk_blocked_orders": risk_blocked_orders,
            "free_capital_after_exit": float(account_after_exit.available_gross),
            "held_positions_after_exit": positions_after_exit,
            "held_positions_after_risk": positions_after_exit,
            "particle_tracker_sync": particle_tracker_sync,
            "entry_ranked_candidates": [],
            "orders": [asdict(order) for order in orders],
        }
        self.state.insert_decision(decision_id, as_of_s, payload)
        self.logger.write("decision", payload)
        return DecisionResult(
            decision_id=decision_id,
            as_of=as_of,
            selector_weights=selector_weights,
            target_weights=target_weights,
            selector_diagnostics=selector_diagnostics,
            blend_diagnostics=blend_diagnostics,
            orders=orders,
        )

    def _load_account_state(self, positions: Mapping[str, int], prices: Mapping[str, float]) -> AccountState:
        stored = self.state.load_account_state(self.config.bot_name)
        if stored is None:
            net = _signed_position_value(positions, prices, self.instruments_by_secid)
            cash = float(self.config.risk.starting_cash) - net
        else:
            cash = float(stored.get("cash", self.config.risk.starting_cash) or self.config.risk.starting_cash)
        return mark_account(
            cash=cash,
            positions=positions,
            prices=prices,
            instruments=self.instruments_by_secid,
            risk=self.config.risk,
            max_gross=self.config.portfolio.max_gross,
        )

    def _build_exit_plan(
        self,
        *,
        as_of: datetime,
        positions: Mapping[str, int],
        snapshots: Mapping[str, Any],
        metrics: Mapping[str, Any],
        candles: Mapping[str, Any],
        session: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, float]]:
        lifecycle = self.config.trade_lifecycle
        held = [
            self.instruments_by_secid[secid]
            for secid, lots in positions.items()
            if int(lots or 0) != 0 and secid in self.instruments_by_secid
        ]
        diagnostics: dict[str, Any] = {
            "enabled": bool(lifecycle.exit.enabled),
            "positions_count": len(held),
            "close_secids": [],
            "missing_forecasts": [],
            "held": {},
        }
        if not lifecycle.exit.enabled:
            diagnostics["status"] = "disabled"
            return diagnostics, {}
        if not held:
            diagnostics["status"] = "skipped_no_positions"
            return diagnostics, {}
        if session is not None and not bool(session.get("exit_allowed", True)):
            reason = str(session.get("action_reason") or "session_exit_blocked")
            diagnostics["status"] = "session_blocked"
            diagnostics["held"] = {instrument.secid: {"action": "hold", "action_reason": reason} for instrument in held}
            return diagnostics, {}
        if bool(getattr(lifecycle.exit, "particle_enabled", False)):
            return self._build_particle_exit_plan(
                as_of=as_of,
                held=held,
                positions=positions,
                snapshots=snapshots,
                metrics=metrics,
                candles=candles,
                diagnostics=diagnostics,
                session=session,
            )
        if session is not None and not bool(session.get("kronos_allowed", True)):
            reason = str(session.get("action_reason") or "session_kronos_blocked")
            diagnostics["status"] = "session_kronos_blocked"
            diagnostics["held"] = {instrument.secid: {"action": "hold", "action_reason": reason} for instrument in held}
            return diagnostics, {}

        rows = tuple(self.exit_kronos_provider.score(as_of, held, candles))
        by_secid = {row.secid: row for row in rows if row.signal_name == "kronos"}
        scores: dict[str, float] = {}
        close_secids: list[str] = []
        for instrument in held:
            secid = instrument.secid
            row = by_secid.get(secid)
            pred_return = _signal_pred_return(row)
            if pred_return is None:
                diagnostics["missing_forecasts"].append(secid)
                diagnostics["held"][secid] = {"action": "hold", "reason": "exit_forecast_missing"}
                self.logger.write("exit_forecast_missing", {"as_of": as_of.isoformat(timespec="seconds"), "secid": secid})
                continue
            side = 1.0 if int(positions.get(secid, 0) or 0) > 0 else -1.0
            costs = _trade_cost_breakdown(secid, snapshots, metrics, self.config.risk)
            one_way_cost = costs["one_way_cost"]
            hold_edge = side * pred_return - one_way_cost
            close = hold_edge <= float(lifecycle.exit.edge_threshold)
            if close:
                close_secids.append(secid)
            scores[secid] = abs(float(hold_edge))
            diagnostics["held"][secid] = {
                "action": "close" if close else "hold",
                "side": "long" if side > 0 else "short",
                "pred_return": pred_return,
                "spread_pct": costs["spread_pct"],
                "commission_rate": costs["commission_rate"],
                "slippage_one_way": costs["slippage_one_way"],
                "one_way_cost": one_way_cost,
                "hold_edge": hold_edge,
                "edge_threshold": float(lifecycle.exit.edge_threshold),
            }
        diagnostics["close_secids"] = close_secids
        diagnostics["status"] = "ready"
        return diagnostics, scores

    def _build_particle_exit_plan(
        self,
        *,
        as_of: datetime,
        held: tuple[Instrument, ...] | list[Instrument],
        positions: Mapping[str, int],
        snapshots: Mapping[str, Any],
        metrics: Mapping[str, Any],
        candles: Mapping[str, Any],
        diagnostics: dict[str, Any],
        session: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, float]]:
        lifecycle = self.config.trade_lifecycle
        diagnostics["particle_enabled"] = True
        kronos_allowed = True if session is None else bool(session.get("kronos_allowed", True))
        allow_new_trackers = True if session is None else bool(session.get("allow_new_trackers", True))
        tracker_max_target_time = None if session is None else session.get("_force_flat_dt")
        held_secids = {instrument.secid for instrument in held}
        for secid in set(self.state.load_kronos_exit_trackers()) - held_secids:
            self.state.delete_kronos_exit_tracker(secid)

        scores: dict[str, float] = {}
        close_secids: list[str] = []
        for instrument in held:
            secid = instrument.secid
            lots = int(positions.get(secid, 0) or 0)
            side = _tracker_side_for_lots(lots)
            if side is None:
                continue
            tracker, tracker_diag = self._ensure_particle_tracker(
                as_of=as_of,
                instrument=instrument,
                side=side,
                candles=candles,
                allow_create=allow_new_trackers,
                allow_forecast=kronos_allowed,
                max_target_time=tracker_max_target_time,
            )
            if tracker is None:
                if not kronos_allowed:
                    reason = str(tracker_diag.get("reason") or (session.get("action_reason") if session else "session_kronos_blocked"))
                    diagnostics["held"][secid] = {
                        "action": "hold",
                        "side": side,
                        "particle_enabled": True,
                        "action_reason": reason,
                        **tracker_diag,
                    }
                    continue
                legacy_diag, legacy_score, legacy_close = self._legacy_exit_decision(
                    as_of=as_of,
                    instrument=instrument,
                    positions=positions,
                    snapshots=snapshots,
                    metrics=metrics,
                    candles=candles,
                    action_reason=str(tracker_diag.get("reason", "particle_tracker_missing_fallback")),
                )
                diagnostics["held"][secid] = legacy_diag
                if legacy_score is not None:
                    scores[secid] = legacy_score
                if legacy_close:
                    close_secids.append(secid)
                continue

            tracker, update_diag = self._update_particle_observations(
                as_of=as_of,
                tracker=tracker,
                instrument=instrument,
                metrics=metrics,
                candles=candles,
            )
            state = dict(tracker.get("state") or {})
            refresh_reason = ""
            ess = _effective_sample_size(state.get("weights", []))
            min_ess = float(tracker["sample_count"]) * float(lifecycle.exit.particle_ess_refresh_fraction)
            if int(tracker["current_step"]) >= int(tracker["horizon"]):
                refresh_reason = "horizon_exhausted"
            elif ess < min_ess:
                refresh_reason = "ess_below_threshold"
            if refresh_reason:
                extension_count = int(state.get("extension_count", 0) or 0)
                refreshed, refresh_diag = self._ensure_particle_tracker(
                    as_of=as_of,
                    instrument=instrument,
                    side=side,
                    candles=candles,
                    force=True,
                    extension_count=extension_count,
                    allow_create=allow_new_trackers,
                    allow_forecast=kronos_allowed,
                    max_target_time=tracker_max_target_time,
                )
                if refreshed is None:
                    if not kronos_allowed:
                        update_diag["refresh_reason"] = refresh_reason
                        update_diag["refresh_failed_reason"] = refresh_diag.get("reason", "")
                    else:
                        legacy_diag, legacy_score, legacy_close = self._legacy_exit_decision(
                            as_of=as_of,
                            instrument=instrument,
                            positions=positions,
                            snapshots=snapshots,
                            metrics=metrics,
                            candles=candles,
                            action_reason=f"{refresh_reason}_fallback",
                        )
                        legacy_diag["refresh_reason"] = refresh_reason
                        legacy_diag["refresh_failed_reason"] = refresh_diag.get("reason", "")
                        diagnostics["held"][secid] = legacy_diag
                        if legacy_score is not None:
                            scores[secid] = legacy_score
                        if legacy_close:
                            close_secids.append(secid)
                        continue
                else:
                    tracker = refreshed
                    state = dict(tracker.get("state") or {})
                    update_diag["refresh_reason"] = refresh_reason

            costs = _trade_cost_breakdown(secid, snapshots, metrics, self.config.risk)
            current_close = _current_close(secid, snapshots, candles, state)
            plan = _particle_exit_plan(
                state=state,
                current_step=int(tracker["current_step"]),
                current_close=current_close,
                side_value=_side_value(side),
                one_way_cost=costs["one_way_cost"],
                min_expected_profit=float(lifecycle.exit.particle_min_expected_profit),
                min_plan_probability=float(lifecycle.exit.particle_min_plan_probability),
            )
            best = plan.get("best") or {}
            selected_plan = plan.get("selected") or {}
            planned_exit_at = str(tracker.get("planned_exit_at") or "")
            planned_dt = _parse_dt(planned_exit_at)
            due = planned_dt is not None and as_of >= planned_dt
            action = "hold"
            action_reason = "particle_plan_hold"
            if due:
                action = "close"
                action_reason = "planned_exit_due"
            elif not best:
                action = "close"
                action_reason = "no_future_steps"
            elif float(best.get("expected_net", 0.0) or 0.0) <= 0:
                action = "close"
                action_reason = "expected_net_non_positive"
            elif float(best.get("probability_plus", 0.0) or 0.0) < float(lifecycle.exit.particle_close_probability):
                action = "close"
                action_reason = "probability_below_close_threshold"
            elif not selected_plan and update_diag.get("refresh_reason"):
                action = "close"
                action_reason = "no_viable_exit_after_refresh"
            elif selected_plan:
                planned_exit_at, action_reason = self._apply_particle_plan(
                    tracker=tracker,
                    state=state,
                    selected_plan=selected_plan,
                    as_of=as_of,
                )
            elif not planned_exit_at:
                action = "close"
                action_reason = "no_viable_exit"

            weights = [float(value) for value in state.get("weights", [])]
            ess = _effective_sample_size(weights)
            confidence = _particle_confidence(ess, len(weights))
            tracker["planned_exit_at"] = planned_exit_at
            tracker["confidence"] = confidence
            tracker["state"] = state
            self._save_particle_tracker(tracker)
            if action == "close":
                close_secids.append(secid)
            score = abs(float(best.get("score", 0.0) or 0.0))
            scores[secid] = score
            diagnostics["held"][secid] = {
                "action": action,
                "side": side,
                "particle_enabled": True,
                "current_step": int(tracker["current_step"]),
                "planned_exit_at": planned_exit_at,
                "planned_exit_step": state.get("planned_exit_step"),
                "expected_net": float(best.get("expected_net", 0.0) or 0.0),
                "probability_plus": float(best.get("probability_plus", 0.0) or 0.0),
                "score": float(best.get("score", 0.0) or 0.0),
                "confidence": confidence,
                "ess": ess,
                "refresh_reason": update_diag.get("refresh_reason", ""),
                "extension_count": int(state.get("extension_count", 0) or 0),
                "action_reason": action_reason,
                "weights_summary": _particle_weights_summary(weights),
                "one_way_cost": costs["one_way_cost"],
                "spread_pct": costs["spread_pct"],
                **tracker_diag,
                **update_diag,
            }
        diagnostics["close_secids"] = close_secids
        diagnostics["status"] = "ready"
        return diagnostics, scores

    def _legacy_exit_decision(
        self,
        *,
        as_of: datetime,
        instrument: Instrument,
        positions: Mapping[str, int],
        snapshots: Mapping[str, Any],
        metrics: Mapping[str, Any],
        candles: Mapping[str, Any],
        action_reason: str,
    ) -> tuple[dict[str, Any], float | None, bool]:
        rows = tuple(self.exit_kronos_provider.score(as_of, [instrument], candles))
        row = next((item for item in rows if item.signal_name == "kronos" and item.secid == instrument.secid), None)
        pred_return = _signal_pred_return(row)
        if pred_return is None:
            self.logger.write("exit_forecast_missing", {"as_of": as_of.isoformat(timespec="seconds"), "secid": instrument.secid})
            return {
                "action": "hold",
                "reason": "exit_forecast_missing",
                "particle_enabled": True,
                "action_reason": action_reason,
            }, None, False
        side_value = 1.0 if int(positions.get(instrument.secid, 0) or 0) > 0 else -1.0
        costs = _trade_cost_breakdown(instrument.secid, snapshots, metrics, self.config.risk)
        hold_edge = side_value * pred_return - costs["one_way_cost"]
        close = hold_edge <= float(self.config.trade_lifecycle.exit.edge_threshold)
        return {
            "action": "close" if close else "hold",
            "side": "long" if side_value > 0 else "short",
            "pred_return": pred_return,
            "particle_enabled": True,
            "fallback": "edge_exit",
            "action_reason": action_reason,
            "spread_pct": costs["spread_pct"],
            "commission_rate": costs["commission_rate"],
            "slippage_one_way": costs["slippage_one_way"],
            "one_way_cost": costs["one_way_cost"],
            "hold_edge": hold_edge,
            "edge_threshold": float(self.config.trade_lifecycle.exit.edge_threshold),
        }, abs(float(hold_edge)), close

    def _ensure_particle_tracker(
        self,
        *,
        as_of: datetime,
        instrument: Instrument,
        side: str,
        candles: Mapping[str, Any],
        force: bool = False,
        extension_count: int | None = None,
        allow_create: bool = True,
        allow_forecast: bool = True,
        max_target_time: datetime | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        lifecycle = self.config.trade_lifecycle
        existing = self.state.load_kronos_exit_tracker(instrument.secid)
        configured_horizon = max(int(lifecycle.exit.particle_horizon), 1)
        sample_count = max(int(lifecycle.exit.particle_sample_count), 1)
        if existing is not None and not force:
            if existing["side"] == side and int(existing["sample_count"]) == sample_count:
                return existing, {"tracker_created": False, "configured_horizon": configured_horizon}
            self.state.delete_kronos_exit_tracker(instrument.secid)
        if not allow_create:
            return None, {"reason": "session_entry_cutoff", "tracker_created": False}
        if not allow_forecast:
            return None, {"reason": "session_kronos_blocked", "tracker_created": False}
        horizon, horizon_diag = _kronos_clipped_horizon(
            candles=candles,
            instruments=[instrument],
            as_of=as_of,
            pred_len=configured_horizon,
            max_target_time=max_target_time,
        )
        if horizon <= 0:
            return None, {
                "reason": "session_kronos_horizon_exceeds_close",
                "tracker_created": False,
                "kronos_horizon": horizon_diag,
            }
        forecast_paths = getattr(self.exit_kronos_provider, "forecast_paths", None)
        if not callable(forecast_paths):
            return None, {"reason": "particle_forecast_paths_unavailable"}
        try:
            paths_by_secid = forecast_paths(
                as_of,
                [instrument],
                candles,
                pred_len=horizon,
                sample_count=sample_count,
                max_target_time=max_target_time,
            )
        except TypeError as exc:
            if "max_target_time" not in str(exc):
                raise
            paths_by_secid = forecast_paths(
                as_of,
                [instrument],
                candles,
                pred_len=horizon,
                sample_count=sample_count,
            )
        payload = paths_by_secid.get(instrument.secid) if isinstance(paths_by_secid, Mapping) else None
        if not payload or not payload.get("paths"):
            return None, {"reason": "particle_paths_missing"}
        actual_sample_count = int(payload.get("sample_count") or len(payload.get("paths") or []))
        weights = [1.0 / actual_sample_count for _ in range(actual_sample_count)]
        state = {
            "paths": payload["paths"],
            "timestamps": payload.get("timestamps", []),
            "weights": weights,
            "observed_timestamps": [],
            "extension_count": int(extension_count if extension_count is not None else 0),
            "planned_exit_step": None,
            "planned_expected_net": 0.0,
            "planned_probability_plus": 0.0,
            "planned_score": 0.0,
            "last_close": float(payload.get("last_close", 0.0) or 0.0),
        }
        tracker = {
            "secid": instrument.secid,
            "side": side,
            "created_at": as_of.isoformat(timespec="seconds"),
            "last_updated_at": as_of.isoformat(timespec="seconds"),
            "horizon": int(payload.get("horizon") or horizon),
            "sample_count": actual_sample_count,
            "current_step": 0,
            "planned_exit_at": "",
            "confidence": 1.0,
            "state": state,
        }
        self._save_particle_tracker(tracker)
        loaded = self.state.load_kronos_exit_tracker(instrument.secid)
        return loaded, {
            "tracker_created": True,
            "configured_horizon": configured_horizon,
            "effective_horizon": int(tracker["horizon"]),
            "kronos_horizon": horizon_diag,
        }

    def _update_particle_observations(
        self,
        *,
        as_of: datetime,
        tracker: dict[str, Any],
        instrument: Instrument,
        metrics: Mapping[str, Any],
        candles: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        state = dict(tracker.get("state") or {})
        paths = list(state.get("paths") or [])
        timestamps = list(state.get("timestamps") or [])
        weights = _normalize_weights(state.get("weights", []), len(paths))
        current_step = int(tracker.get("current_step", 0) or 0)
        updated_steps = 0
        vol = _particle_volatility(instrument.secid, metrics)
        while current_step < len(timestamps) and current_step < len(paths[0] if paths else []):
            actual = _candle_for_timestamp(candles.get(instrument.secid), timestamps[current_step])
            if actual is None:
                break
            weights = _update_particle_weights(paths, weights, current_step, actual, vol)
            state.setdefault("observed_timestamps", []).append(timestamps[current_step])
            current_step += 1
            updated_steps += 1
        state["weights"] = weights
        tracker["state"] = state
        tracker["current_step"] = current_step
        tracker["last_updated_at"] = as_of.isoformat(timespec="seconds")
        tracker["confidence"] = _particle_confidence(_effective_sample_size(weights), len(weights))
        return tracker, {"updated_steps": updated_steps}

    def _apply_particle_plan(
        self,
        *,
        tracker: dict[str, Any],
        state: dict[str, Any],
        selected_plan: Mapping[str, Any],
        as_of: datetime,
    ) -> tuple[str, str]:
        lifecycle = self.config.trade_lifecycle
        candidate_at = str(selected_plan.get("timestamp") or "")
        old_at = str(tracker.get("planned_exit_at") or "")
        old_dt = _parse_dt(old_at)
        candidate_dt = _parse_dt(candidate_at)
        old_expected = float(state.get("planned_expected_net", 0.0) or 0.0)
        extension_count = int(state.get("extension_count", 0) or 0)
        if not candidate_at or candidate_dt is None:
            return old_at, "particle_plan_hold"
        action_reason = "planned_exit_updated"
        if not old_at or old_dt is None or candidate_dt <= old_dt:
            planned_at = candidate_at
        elif (
            float(selected_plan.get("expected_net", 0.0) or 0.0) >= old_expected + float(lifecycle.exit.particle_extend_min_improvement)
            and extension_count < int(lifecycle.exit.particle_max_extensions)
        ):
            planned_at = candidate_at
            extension_count += 1
            action_reason = "planned_exit_extended"
        else:
            planned_at = old_at
            action_reason = "planned_exit_kept"
        if planned_at == candidate_at:
            state["planned_exit_step"] = int(selected_plan.get("step", 0) or 0)
            state["planned_expected_net"] = float(selected_plan.get("expected_net", 0.0) or 0.0)
            state["planned_probability_plus"] = float(selected_plan.get("probability_plus", 0.0) or 0.0)
            state["planned_score"] = float(selected_plan.get("score", 0.0) or 0.0)
            state["extension_count"] = extension_count
            state["planned_updated_at"] = as_of.isoformat(timespec="seconds")
        return planned_at, action_reason

    def _save_particle_tracker(self, tracker: Mapping[str, Any]) -> None:
        self.state.save_kronos_exit_tracker(
            secid=str(tracker["secid"]),
            side=str(tracker["side"]),
            created_at=str(tracker["created_at"]),
            last_updated_at=str(tracker["last_updated_at"]),
            horizon=int(tracker["horizon"]),
            sample_count=int(tracker["sample_count"]),
            current_step=int(tracker["current_step"]),
            planned_exit_at=str(tracker.get("planned_exit_at") or ""),
            confidence=float(tracker.get("confidence", 0.0) or 0.0),
            state=dict(tracker.get("state") or {}),
        )

    def _sync_particle_trackers_after_orders(
        self,
        *,
        as_of: datetime,
        planned: list[Any],
        orders: tuple[Any, ...],
        snapshots: Mapping[str, Any],
        metrics: Mapping[str, Any],
        candles: Mapping[str, Any],
        session: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        lifecycle = self.config.trade_lifecycle
        if not bool(getattr(lifecycle.exit, "particle_enabled", False)) or self.config.live_orders:
            return {}
        allow_new_trackers = True if session is None else bool(session.get("allow_new_trackers", True))
        allow_forecast = True if session is None else bool(session.get("kronos_allowed", True))
        max_target_time = None if session is None else session.get("_force_flat_dt")
        created: list[str] = []
        deleted: list[str] = []
        for order, result in zip(planned, orders):
            if result.status not in {"dry_run", "submitted"}:
                continue
            secid = str(order.secid)
            if int(order.target_lots) == 0:
                self.state.delete_kronos_exit_tracker(secid)
                deleted.append(secid)
                continue
            side = _tracker_side_for_lots(int(order.target_lots))
            instrument = self.instruments_by_secid.get(secid)
            if side is None or instrument is None:
                continue
            existing = self.state.load_kronos_exit_tracker(secid)
            if existing is not None and existing.get("side") == side:
                continue
            tracker, _ = self._ensure_particle_tracker(
                as_of=as_of,
                instrument=instrument,
                side=side,
                candles=candles,
                force=existing is not None,
                allow_create=allow_new_trackers,
                allow_forecast=allow_forecast,
                max_target_time=max_target_time,
            )
            if tracker is not None:
                state = dict(tracker.get("state") or {})
                costs = _trade_cost_breakdown(secid, snapshots, metrics, self.config.risk)
                plan = _particle_exit_plan(
                    state=state,
                    current_step=int(tracker["current_step"]),
                    current_close=_current_close(secid, snapshots, candles, state),
                    side_value=_side_value(side),
                    one_way_cost=costs["one_way_cost"],
                    min_expected_profit=float(lifecycle.exit.particle_min_expected_profit),
                    min_plan_probability=float(lifecycle.exit.particle_min_plan_probability),
                )
                selected_plan = plan.get("selected") or {}
                if selected_plan:
                    planned_exit_at, _ = self._apply_particle_plan(
                        tracker=tracker,
                        state=state,
                        selected_plan=selected_plan,
                        as_of=as_of,
                    )
                    tracker["planned_exit_at"] = planned_exit_at
                    tracker["state"] = state
                    self._save_particle_tracker(tracker)
                created.append(secid)
        return {"created": sorted(set(created)), "deleted": sorted(set(deleted))}

    def _build_entry_targets(
        self,
        *,
        candidate_weights: Mapping[str, float],
        positions: Mapping[str, int],
        account: AccountState,
        prices: Mapping[str, float],
        signals: tuple[SignalRow, ...],
        blocked_secids: set[str],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        lifecycle = self.config.trade_lifecycle
        equity = max(float(account.equity), 1.0)
        current_weights = _position_weights(positions, prices, self.instruments_by_secid, equity)
        gross_after_exit = float(account.gross)
        free_value = _entry_free_value(account, self.config.risk)
        free_weight = free_value / equity if equity > 0 else 0.0
        held_count = sum(1 for lots in positions.values() if int(lots or 0) != 0)
        free_slots = max(int(lifecycle.max_total_positions) - held_count, 0)
        ranked = sorted(
            [(secid, float(weight)) for secid, weight in candidate_weights.items() if abs(float(weight)) > 1e-12],
            key=lambda item: (-abs(item[1]), item[0]),
        )
        selected: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        used_new_slots = 0
        for rank, (secid, weight) in enumerate(ranked, start=1):
            current_lots = int(positions.get(secid, 0) or 0)
            candidate_side = 1 if weight > 0 else -1
            current_side = _position_side(current_lots)
            held = current_lots != 0
            selected_flag = False
            reason = ""
            if secid in blocked_secids:
                reason = "closed_in_exit_pass"
            elif held:
                if not lifecycle.entry.include_held_for_topup:
                    reason = "held_topup_disabled"
                elif current_side != candidate_side:
                    reason = "held_opposite_side_entry_skipped"
                elif free_weight <= 0:
                    reason = "no_free_capital"
                else:
                    selected_flag = True
                    reason = "topup"
            else:
                if used_new_slots >= free_slots:
                    reason = "no_free_slot"
                elif free_weight <= 0:
                    reason = "no_free_capital"
                else:
                    selected_flag = True
                    used_new_slots += 1
                    reason = "new_entry"
            item = {
                "rank": rank,
                "secid": secid,
                "candidate_weight": weight,
                "side": "long" if candidate_side > 0 else "short",
                "held": held,
                "selected": selected_flag,
                "reason": reason,
            }
            candidates.append(item)
            if selected_flag:
                selected.append(item)

        target_weights = dict(current_weights)
        if selected and free_weight > 0:
            for item, allocation in zip(selected, _rank_budget_weights(len(selected), self.config.lightgbm.rank_power)):
                secid = str(item["secid"])
                side = 1.0 if item["side"] == "long" else -1.0
                add_weight = free_weight * allocation
                target_weights[secid] = float(target_weights.get(secid, 0.0) or 0.0) + side * add_weight
                item["allocated_weight"] = add_weight
                item["target_weight"] = target_weights[secid]

        return target_weights, {
            "capital_mode": lifecycle.entry.capital_mode,
            "topup_sizing": lifecycle.entry.topup_sizing,
            "max_total_positions": int(lifecycle.max_total_positions),
            "held_count": held_count,
            "free_slots": free_slots,
            "gross_after_exit": gross_after_exit,
            "margin_used_after_exit": float(account.margin_used),
            "free_capital": free_value,
            "free_weight": free_weight,
            "selected_count": len(selected),
            "ranked_candidates": candidates,
        }

    def _build_kronos_rank_entry_targets(
        self,
        *,
        instruments: tuple[Instrument, ...],
        positions: Mapping[str, int],
        account: AccountState,
        prices: Mapping[str, float],
        snapshots: Mapping[str, Any],
        metrics: Mapping[str, Any],
        signals: tuple[SignalRow, ...],
        blocked_secids: set[str],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        lifecycle = self.config.trade_lifecycle
        equity = max(float(account.equity), 1.0)
        current_weights = _position_weights(positions, prices, self.instruments_by_secid, equity)
        gross_after_exit = float(account.gross)
        free_value = _entry_free_value(account, self.config.risk)
        free_weight = free_value / equity if equity > 0 else 0.0
        held_count = sum(1 for lots in positions.values() if int(lots or 0) != 0)
        free_slots = max(int(lifecycle.max_total_positions) - held_count, 0)
        rows_by_secid = {row.secid: row for row in signals if row.signal_name == "kronos"}

        ranked: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for instrument in instruments:
            secid = instrument.secid
            current_lots = int(positions.get(secid, 0) or 0)
            pred_return = _signal_pred_return(rows_by_secid.get(secid))
            held = current_lots != 0
            base_item: dict[str, Any] = {
                "rank": None,
                "secid": secid,
                "asset_class": instrument.asset_class,
                "pred_return": pred_return,
                "edge": None,
                "side": None,
                "held": held,
                "selected": False,
                "reason": "",
                "allocated_weight": 0.0,
            }
            if pred_return is None:
                base_item["reason"] = "missing_kronos_pred_return"
                skipped.append(base_item)
                continue
            costs = _trade_cost_breakdown(secid, snapshots, metrics, self.config.risk)
            if pred_return == 0:
                net_edge = -costs["round_trip_cost"]
                base_item.update(
                    {
                        "edge": net_edge,
                        "gross_pred_return": 0.0,
                        "net_edge": net_edge,
                        **costs,
                    }
                )
                base_item["reason"] = "zero_pred_return"
                skipped.append(base_item)
                continue
            side = 1 if pred_return > 0 else -1
            gross_pred_return = abs(float(pred_return))
            net_edge = gross_pred_return - costs["round_trip_cost"]
            base_item.update(
                {
                    "pred_return": float(pred_return),
                    "edge": net_edge,
                    "gross_pred_return": gross_pred_return,
                    "net_edge": net_edge,
                    "side": "long" if side > 0 else "short",
                    **costs,
                }
            )
            ranked.append(base_item)

        ranked.sort(
            key=lambda item: (
                -float(item["gross_pred_return"]),
                str(item["secid"]),
            )
        )

        selected: list[dict[str, Any]] = []
        used_new_slots = 0
        blocked_by_orderability: dict[str, str] = {}
        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank

        for _ in range(len(ranked) + 1):
            selected = []
            used_new_slots = 0
            for item in ranked:
                item["selected"] = False
                item["allocated_weight"] = 0.0
                item.pop("target_weight", None)
                item["reason"] = ""
            for item in ranked:
                self._mark_kronos_rank_candidate_selection(
                    item=item,
                    positions=positions,
                    blocked_secids=blocked_secids,
                    blocked_by_orderability=blocked_by_orderability,
                    selected=selected,
                    used_new_slots_ref={"value": used_new_slots},
                    free_slots=free_slots,
                    free_weight=free_weight,
                )
                used_new_slots = sum(1 for row in selected if row["reason"] == "new_entry")

            target_weights = dict(current_weights)
            rank_power = float(lifecycle.entry.rank_power)
            if selected and free_weight > 0:
                for item, allocation in zip(selected, _rank_budget_weights(len(selected), rank_power)):
                    secid = str(item["secid"])
                    side = 1.0 if item["side"] == "long" else -1.0
                    add_weight = free_weight * allocation
                    target_weights[secid] = float(target_weights.get(secid, 0.0) or 0.0) + side * add_weight
                    item["allocated_weight"] = add_weight
                    item["target_weight"] = target_weights[secid]

            blocked_now = None
            for item in selected:
                reason = _incremental_entry_block_reason(
                    secid=str(item["secid"]),
                    target_weight=float(item.get("target_weight", 0.0) or 0.0),
                    positions=positions,
                    prices=prices,
                    instruments=self.instruments_by_secid,
                    equity=equity,
                    min_order_value=float(self.config.risk.min_order_value_rub),
                    min_position_change_weight=float(self.config.risk.min_position_change_weight),
                )
                if reason:
                    blocked_now = (str(item["secid"]), reason)
                    break
            if blocked_now is None:
                break
            blocked_by_orderability[blocked_now[0]] = blocked_now[1]
        else:
            target_weights = dict(current_weights)
            rank_power = float(lifecycle.entry.rank_power)

        selected_after_cost: list[dict[str, Any]] = []
        for item in selected:
            if float(item.get("net_edge", 0.0) or 0.0) <= 0:
                secid = str(item["secid"])
                item["selected"] = False
                item["reason"] = "cost_exceeds_pred_return"
                item["allocated_weight"] = 0.0
                item.pop("target_weight", None)
                if secid in current_weights:
                    target_weights[secid] = current_weights[secid]
                else:
                    target_weights.pop(secid, None)
                continue
            selected_after_cost.append(item)
        selected = selected_after_cost
        used_new_slots = sum(1 for row in selected if row["reason"] == "new_entry")

        candidates = ranked + sorted(skipped, key=lambda item: str(item["secid"]))
        return target_weights, {
            "ranking_mode": "kronos_rank",
            "ranking_metric": str(lifecycle.entry.ranking_metric),
            "capital_mode": lifecycle.entry.capital_mode,
            "topup_sizing": lifecycle.entry.topup_sizing,
            "rank_power": rank_power,
            "max_total_positions": int(lifecycle.max_total_positions),
            "held_count": held_count,
            "free_slots": free_slots,
            "gross_after_exit": gross_after_exit,
            "margin_used_after_exit": float(account.margin_used),
            "free_capital": free_value,
            "free_weight": free_weight,
            "selected_count": len(selected),
            "selected_new_entries": used_new_slots,
            "selected_topups": len(selected) - used_new_slots,
            "ranked_candidates": candidates,
        }

    def _mark_kronos_rank_candidate_selection(
        self,
        *,
        item: dict[str, Any],
        positions: Mapping[str, int],
        blocked_secids: set[str],
        blocked_by_orderability: Mapping[str, str],
        selected: list[dict[str, Any]],
        used_new_slots_ref: dict[str, int],
        free_slots: int,
        free_weight: float,
    ) -> None:
        lifecycle = self.config.trade_lifecycle
        secid = str(item["secid"])
        current_lots = int(positions.get(secid, 0) or 0)
        candidate_side = 1 if item["side"] == "long" else -1
        current_side = _position_side(current_lots)
        if secid in blocked_secids:
            item["reason"] = "closed_in_exit_pass"
        elif secid in blocked_by_orderability:
            item["reason"] = blocked_by_orderability[secid]
        elif current_lots != 0:
            if not lifecycle.entry.include_held_for_topup:
                item["reason"] = "held_topup_disabled"
            elif current_side != candidate_side:
                item["reason"] = "held_opposite_side_entry_skipped"
            elif free_weight <= 0:
                item["reason"] = "no_free_capital"
            else:
                item["selected"] = True
                item["reason"] = "topup"
                selected.append(item)
        else:
            if used_new_slots_ref["value"] >= free_slots:
                item["reason"] = "no_free_slot"
            elif free_weight <= 0:
                item["reason"] = "no_free_capital"
            else:
                item["selected"] = True
                item["reason"] = "new_entry"
                selected.append(item)
                used_new_slots_ref["value"] += 1

    def _free_capital_after_exit(self, positions: Mapping[str, int], prices: Mapping[str, float]) -> float:
        account = self._load_account_state(positions, prices)
        return float(account.available_gross)

    def _update_selector_returns(self, as_of_s: str, prices: Mapping[str, float]) -> None:
        previous = self.state.load_selector_positions()
        if not previous:
            return
        previous_as_of = _latest_selector_position_timestamp(previous)
        if previous_as_of is None or previous_as_of >= as_of_s:
            return
        returns = {}
        for selector, positions in previous.items():
            total = 0.0
            for secid, row in positions.items():
                entry = float(row.get("entry_price", 0.0) or 0.0)
                current = float(prices.get(secid, 0.0) or 0.0)
                weight = float(row.get("weight", 0.0) or 0.0)
                if entry > 0 and current > 0:
                    total += weight * (current / entry - 1.0)
            returns[selector] = total
        self.state.append_selector_returns(previous_as_of, returns)

    def _apply_min_hold_guard(
        self,
        *,
        as_of: datetime,
        target_weights: Mapping[str, float],
        target_scores: Mapping[str, float],
        positions: Mapping[str, int],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        stored = self.state.load_paper_positions()
        guarded = dict(target_weights)
        blocked = []
        for secid, current_lots in positions.items():
            target_weight = float(guarded.get(secid, 0.0) or 0.0)
            if current_lots == 0 or target_weight == 0 or (current_lots > 0) == (target_weight > 0):
                continue
            opened_at = _parse_dt(str(stored.get(secid, {}).get("opened_at", "")))
            if opened_at is None:
                continue
            held_minutes = (as_of - opened_at).total_seconds() / 60.0
            strength = float(target_scores.get(secid, abs(target_weight)) or 0.0)
            if held_minutes < self.config.portfolio.min_hold_minutes and strength < self.config.portfolio.strong_flip_threshold:
                guarded.pop(secid, None)
                blocked.append({"secid": secid, "held_minutes": held_minutes, "target_score": strength})
        return guarded, {"blocked_flips": blocked}


def _target_scores(target_weights: Mapping[str, float], signals: tuple[SignalRow, ...]) -> dict[str, float]:
    by_secid: dict[str, list[SignalRow]] = {}
    for row in signals:
        by_secid.setdefault(row.secid, []).append(row)
    out = {}
    for secid, weight in target_weights.items():
        rows = by_secid.get(secid, [])
        if weight >= 0:
            out[secid] = max((row.bullish_score for row in rows), default=abs(weight))
        else:
            out[secid] = max((1.0 - row.bullish_score for row in rows), default=abs(weight))
    return out


def _decision_id(as_of: str, selector_weights: Mapping[str, float], target_weights: Mapping[str, float]) -> str:
    raw = json.dumps({"as_of": as_of, "selector_weights": selector_weights, "target_weights": target_weights}, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _parse_dt(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _latest_selector_position_timestamp(previous: Mapping[str, Mapping[str, Mapping[str, float]]]) -> str | None:
    values = [
        str(row.get("updated_at") or "")
        for positions in previous.values()
        for row in positions.values()
        if row.get("updated_at")
    ]
    return max(values) if values else None


def _signal_pred_return(row: SignalRow | None) -> float | None:
    if row is None:
        return None
    value = row.metadata.get("pred_return") if row.metadata else None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _tracker_side_for_lots(lots: int) -> str | None:
    if int(lots or 0) > 0:
        return "long"
    if int(lots or 0) < 0:
        return "short"
    return None


def _side_value(side: str) -> float:
    return 1.0 if str(side) == "long" else -1.0


def _current_close(secid: str, snapshots: Mapping[str, Any], candles: Mapping[str, Any], state: Mapping[str, Any]) -> float:
    snapshot = snapshots.get(secid)
    if snapshot is not None:
        price = _finite_float(getattr(snapshot, "last_price", 0.0), default=0.0)
        if price > 0:
            return price
    df = candles.get(secid)
    if df is not None and not getattr(df, "empty", True) and "close" in df:
        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        if not close.empty:
            value = _finite_float(close.iloc[-1], default=0.0)
            if value > 0:
                return value
    return _finite_float(state.get("last_close", 0.0), default=0.0)


def _candle_for_timestamp(candles: Any, timestamp: Any) -> dict[str, float] | None:
    if candles is None or getattr(candles, "empty", True) or "timestamps" not in candles.columns:
        return None
    try:
        target = pd.Timestamp(timestamp)
    except Exception:
        return None
    ts = pd.to_datetime(candles["timestamps"], errors="coerce")
    matches = candles[ts == target]
    if matches.empty:
        return None
    row = matches.iloc[-1]
    out = {col: _finite_float(row.get(col), default=0.0) for col in ("open", "high", "low", "close")}
    if not all(value > 0 for value in out.values()):
        return None
    return out


def _particle_volatility(secid: str, metrics: Mapping[str, Any]) -> float:
    metric = metrics.get(secid)
    atr = _finite_float(getattr(metric, "atr_pct", 0.0), default=0.0) if metric is not None else 0.0
    realized = _finite_float(getattr(metric, "realized_volatility", 0.0), default=0.0) if metric is not None else 0.0
    return max(atr, realized, 1e-4)


def _normalize_weights(weights: Any, n: int) -> list[float]:
    if n <= 0:
        return []
    clean = [_non_negative_float(value, default=0.0) for value in list(weights or [])[:n]]
    if len(clean) < n:
        clean.extend([0.0] * (n - len(clean)))
    total = sum(clean)
    if total <= 0 or not math.isfinite(total):
        return [1.0 / n for _ in range(n)]
    return [value / total for value in clean]


def _update_particle_weights(paths: list[Any], weights: list[float], step: int, actual: Mapping[str, float], vol: float) -> list[float]:
    actual_features = _candle_features(actual)
    if actual_features is None:
        return weights
    next_weights = []
    denom = max(float(vol), 1e-4)
    for idx, path in enumerate(paths):
        old_weight = weights[idx] if idx < len(weights) else 0.0
        if step >= len(path):
            next_weights.append(0.0)
            continue
        pred_features = _candle_features(path[step])
        if pred_features is None:
            next_weights.append(0.0)
            continue
        direction_penalty = 0.5 if pred_features["direction"] != actual_features["direction"] else 0.0
        error = (
            abs(pred_features["close_return"] - actual_features["close_return"]) / denom
            + 0.5 * abs(pred_features["range"] - actual_features["range"]) / denom
            + 0.25 * abs(pred_features["upper_wick"] - actual_features["upper_wick"]) / denom
            + 0.25 * abs(pred_features["lower_wick"] - actual_features["lower_wick"]) / denom
            + direction_penalty
        )
        next_weights.append(old_weight * math.exp(-min(max(error, 0.0), 700.0)))
    return _normalize_weights(next_weights, len(paths))


def _candle_features(candle: Mapping[str, Any]) -> dict[str, float] | None:
    open_ = _finite_float(candle.get("open"), default=0.0)
    high = _finite_float(candle.get("high"), default=0.0)
    low = _finite_float(candle.get("low"), default=0.0)
    close = _finite_float(candle.get("close"), default=0.0)
    base = open_ if open_ > 0 else close
    if base <= 0 or high <= 0 or low <= 0 or close <= 0:
        return None
    upper = max(high - max(open_, close), 0.0) / base
    lower = max(min(open_, close) - low, 0.0) / base
    direction = 1.0 if close > open_ else (-1.0 if close < open_ else 0.0)
    return {
        "close_return": close / base - 1.0,
        "range": max(high - low, 0.0) / base,
        "upper_wick": upper,
        "lower_wick": lower,
        "direction": direction,
    }


def _effective_sample_size(weights: Any) -> float:
    clean = [_non_negative_float(value, default=0.0) for value in list(weights or [])]
    total = sum(clean)
    if total <= 0:
        return 0.0
    normalized = [value / total for value in clean]
    denom = sum(value * value for value in normalized)
    return 1.0 / denom if denom > 0 else 0.0


def _particle_confidence(ess: float, sample_count: int) -> float:
    if sample_count <= 0:
        return 0.0
    return min(max(float(ess) / float(sample_count), 0.0), 1.0)


def _particle_exit_plan(
    *,
    state: Mapping[str, Any],
    current_step: int,
    current_close: float,
    side_value: float,
    one_way_cost: float,
    min_expected_profit: float,
    min_plan_probability: float,
) -> dict[str, Any]:
    paths = list(state.get("paths") or [])
    if not paths or current_close <= 0:
        return {"best": None, "selected": None}
    weights = _normalize_weights(state.get("weights", []), len(paths))
    timestamps = list(state.get("timestamps") or [])
    horizon = min([len(path) for path in paths] + [len(timestamps)])
    best: dict[str, Any] | None = None
    selected: dict[str, Any] | None = None
    for step in range(max(int(current_step), 0), horizon):
        weighted_returns = []
        probability_plus = 0.0
        for path, weight in zip(paths, weights):
            close = _finite_float(path[step].get("close") if step < len(path) else 0.0, default=0.0)
            if close <= 0:
                continue
            net_return = float(side_value) * (close / current_close - 1.0) - float(one_way_cost)
            weighted_returns.append(weight * net_return)
            if net_return > 0:
                probability_plus += weight
        expected_net = sum(weighted_returns)
        score = expected_net * probability_plus
        option = {
            "step": step,
            "timestamp": str(timestamps[step]) if step < len(timestamps) else "",
            "expected_net": expected_net,
            "probability_plus": probability_plus,
            "score": score,
        }
        if best is None or score > float(best["score"]) or (score == float(best["score"]) and step < int(best["step"])):
            best = option
        if expected_net >= float(min_expected_profit) and probability_plus >= float(min_plan_probability):
            if selected is None or score > float(selected["score"]) or (score == float(selected["score"]) and step < int(selected["step"])):
                selected = option
    return {"best": best, "selected": selected}


def _particle_weights_summary(weights: list[float]) -> dict[str, Any]:
    if not weights:
        return {"ess": 0.0, "min": 0.0, "max": 0.0, "top": []}
    top = sorted(enumerate(weights), key=lambda item: item[1], reverse=True)[:3]
    return {
        "ess": _effective_sample_size(weights),
        "min": min(weights),
        "max": max(weights),
        "top": [{"path": int(idx), "weight": float(weight)} for idx, weight in top],
    }


def _finite_float(value: Any, *, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _positions_after_exit(positions: Mapping[str, int], closed_secids: set[str]) -> dict[str, int]:
    return {
        str(secid): int(lots)
        for secid, lots in positions.items()
        if int(lots or 0) != 0 and secid not in closed_secids
    }


def _positions_after_orders(positions: Mapping[str, int], orders: list[Any] | tuple[Any, ...]) -> dict[str, int]:
    out = {str(secid): int(lots or 0) for secid, lots in positions.items() if int(lots or 0) != 0}
    for order in orders:
        delta = int(order.quantity) if order.direction == "B" else -int(order.quantity)
        next_lots = int(out.get(order.secid, 0)) + delta
        if next_lots == 0:
            out.pop(order.secid, None)
        else:
            out[order.secid] = next_lots
    return out


def _build_risk_cap_targets(
    *,
    account: AccountState,
    positions: Mapping[str, int],
    prices: Mapping[str, float],
    instruments: Mapping[str, Instrument],
    risk: Any,
    max_gross: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    current_weights = _position_weights(positions, prices, instruments, max(float(account.equity), 1.0))
    exposure = max(float(account.gross), float(account.margin_used))
    equity = float(account.equity)
    safety = min(max(float(getattr(risk, "sizing_safety_pct", 0.0) or 0.0), 0.0), 0.95)
    target_exposure = max(float(max_gross), 0.0) * max(equity, 0.0) * (1.0 - safety)
    diagnostics = {
        "enabled": True,
        "equity": equity,
        "gross": float(account.gross),
        "margin_used": float(account.margin_used),
        "target_exposure": target_exposure,
        "breached": exposure > target_exposure + 1e-9,
        "scale": 1.0,
    }
    if not current_weights:
        return {}, diagnostics
    if equity <= 0:
        diagnostics["scale"] = 0.0
        return {}, diagnostics
    if exposure <= target_exposure + 1e-9:
        return current_weights, diagnostics
    scale = max(target_exposure / exposure, 0.0) if exposure > 0 else 0.0
    diagnostics["scale"] = scale
    return {secid: float(weight) * scale for secid, weight in current_weights.items()}, diagnostics


def _signed_position_value(
    positions: Mapping[str, int],
    prices: Mapping[str, float],
    instruments: Mapping[str, Instrument],
) -> float:
    value = 0.0
    for secid, lots_raw in positions.items():
        lots = int(lots_raw or 0)
        if lots == 0:
            continue
        instrument = instruments.get(secid)
        if instrument is None:
            continue
        value += lots * instrument.lot_size * float(prices.get(secid, 0.0) or 0.0)
    return value


def _execution_price_maps(snapshots: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    buy_prices: dict[str, float] = {}
    sell_prices: dict[str, float] = {}
    for secid, snapshot in snapshots.items():
        last = float(getattr(snapshot, "last_price", 0.0) or 0.0)
        ask = float(getattr(snapshot, "ask", 0.0) or 0.0)
        bid = float(getattr(snapshot, "bid", 0.0) or 0.0)
        buy_prices[str(secid)] = ask if ask > 0 else last
        sell_prices[str(secid)] = bid if bid > 0 else last
    return buy_prices, sell_prices


def _position_weights(
    positions: Mapping[str, int],
    prices: Mapping[str, float],
    instruments: Mapping[str, Instrument],
    equity: float,
) -> dict[str, float]:
    if equity <= 0:
        return {}
    out = {}
    for secid, lots in positions.items():
        lots = int(lots or 0)
        if lots == 0:
            continue
        instrument = instruments.get(secid)
        price = float(prices.get(secid, 0.0) or 0.0)
        if instrument is None or price <= 0 or instrument.lot_size <= 0:
            continue
        out[secid] = lots * instrument.lot_size * price / equity
    return out


def _entry_free_value(account: AccountState, risk: Any) -> float:
    safety = min(max(float(getattr(risk, "sizing_safety_pct", 0.0) or 0.0), 0.0), 0.95)
    return max(min(float(account.available_gross), float(account.available_cash)) * (1.0 - safety), 0.0)


def _position_side(lots: int) -> int:
    if lots > 0:
        return 1
    if lots < 0:
        return -1
    return 0


def _incremental_entry_block_reason(
    *,
    secid: str,
    target_weight: float,
    positions: Mapping[str, int],
    prices: Mapping[str, float],
    instruments: Mapping[str, Instrument],
    equity: float,
    min_order_value: float,
    min_position_change_weight: float,
) -> str:
    instrument = instruments.get(secid)
    if instrument is None or instrument.lot_size <= 0:
        return "unorderable_missing_instrument"
    price = float(prices.get(secid, 0.0) or 0.0)
    if price <= 0:
        return "unorderable_missing_price"
    current_lots = int(positions.get(secid, 0) or 0)
    target_lots = _target_lots_for_weight(equity, target_weight, price, instrument.lot_size)
    delta = target_lots - current_lots
    if delta == 0:
        return "unorderable_zero_delta"
    if not _is_incremental_entry_lots(current_lots, target_lots):
        return "unorderable_not_incremental_entry"
    order_value = abs(delta) * instrument.lot_size * price
    if order_value < min_order_value:
        return "unorderable_below_min_order_value"
    if equity > 0 and order_value / equity < min_position_change_weight:
        return "unorderable_below_min_position_change"
    return ""


def _target_lots_for_weight(equity: float, weight: float, price: float, lot_size: int) -> int:
    if price <= 0 or lot_size <= 0:
        return 0
    return int(round(float(equity) * float(weight) / (float(price) * int(lot_size))))


def _is_incremental_entry_lots(current_lots: int, target_lots: int) -> bool:
    if current_lots == 0:
        return target_lots != 0
    if current_lots > 0:
        return target_lots > current_lots
    return target_lots < current_lots


def _entry_mode(config: RuntimeConfig) -> str:
    return str(config.trade_lifecycle.entry.mode or "selectors").strip().lower()


def _trading_session_state(as_of: datetime, config: TradingSessionConfig) -> dict[str, Any]:
    if not bool(config.enabled):
        return {
            "enabled": False,
            "session_state": "disabled",
            "session_open": str(config.session_open),
            "entry_start": str(config.entry_start),
            "new_entry_cutoff": str(config.new_entry_cutoff),
            "kronos_cutoff": str(config.kronos_cutoff),
            "force_flat_time": str(config.force_flat_time),
            "session_close": str(config.session_close),
            "kronos_allowed": True,
            "entry_allowed": True,
            "exit_allowed": True,
            "allow_new_trackers": True,
            "force_flat_required": False,
            "action_reason": "",
            "_kronos_cutoff_dt": None,
            "_force_flat_dt": None,
        }
    tz = ZoneInfo(str(config.timezone or "Europe/Moscow"))
    local_as_of = _as_session_datetime(as_of, tz)
    session_open = _session_datetime(local_as_of, config.session_open, tz)
    entry_start = _session_datetime(local_as_of, config.entry_start, tz)
    new_entry_cutoff = _session_datetime(local_as_of, config.new_entry_cutoff, tz)
    kronos_cutoff = _session_datetime(local_as_of, config.kronos_cutoff, tz)
    force_flat_time = _session_datetime(local_as_of, config.force_flat_time, tz)
    session_close = _session_datetime(local_as_of, config.session_close, tz)

    if local_as_of < session_open:
        state = "pre_session"
        reason = "session_pre_open"
    elif local_as_of < entry_start:
        state = "warmup"
        reason = "session_warmup_no_entry"
    elif local_as_of < new_entry_cutoff:
        state = "trade"
        reason = ""
    elif local_as_of < force_flat_time:
        state = "no_new_entries"
        reason = "session_entry_cutoff"
    elif local_as_of < session_close:
        state = "force_flat"
        reason = "session_force_flat"
    else:
        state = "closed"
        reason = "session_closed"

    return {
        "enabled": True,
        "timezone": str(config.timezone),
        "session_state": state,
        "local_as_of": local_as_of.isoformat(timespec="seconds"),
        "session_open": str(config.session_open),
        "entry_start": str(config.entry_start),
        "new_entry_cutoff": str(config.new_entry_cutoff),
        "kronos_cutoff": str(config.kronos_cutoff),
        "force_flat_time": str(config.force_flat_time),
        "session_close": str(config.session_close),
        "session_open_at": session_open.isoformat(timespec="seconds"),
        "entry_start_at": entry_start.isoformat(timespec="seconds"),
        "new_entry_cutoff_at": new_entry_cutoff.isoformat(timespec="seconds"),
        "kronos_cutoff_at": kronos_cutoff.isoformat(timespec="seconds"),
        "force_flat_at": force_flat_time.isoformat(timespec="seconds"),
        "session_close_at": session_close.isoformat(timespec="seconds"),
        "kronos_allowed": state == "trade",
        "entry_allowed": state == "trade",
        "exit_allowed": state in {"trade", "no_new_entries"},
        "allow_new_trackers": state == "trade",
        "force_flat_required": state == "force_flat",
        "flat_all_asset_classes": bool(config.flat_all_asset_classes),
        "action_reason": reason,
        "_kronos_cutoff_dt": kronos_cutoff,
        "_force_flat_dt": force_flat_time,
    }


def _session_public_payload(session: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in session.items() if not str(key).startswith("_")}


def _as_session_datetime(value: datetime, tz: ZoneInfo) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=tz)
    return value.astimezone(tz)


def _session_datetime(as_of: datetime, hhmm: str, tz: ZoneInfo) -> datetime:
    return datetime.combine(as_of.date(), _parse_hhmm(hhmm), tzinfo=tz)


def _parse_hhmm(value: str) -> time:
    hour_raw, minute_raw = str(value).split(":", 1)
    return time(hour=int(hour_raw), minute=int(minute_raw))


def _kronos_horizon_check(
    *,
    candles: Mapping[str, Any],
    instruments: tuple[Instrument, ...] | list[Instrument],
    as_of: datetime,
    pred_len: int,
    max_target_time: datetime | None,
) -> tuple[bool, dict[str, Any]]:
    if max_target_time is None:
        return True, {"enabled": False}
    targets: dict[str, str] = {}
    violations: list[dict[str, str]] = []
    for instrument in instruments:
        target = _forecast_target_time(candles.get(instrument.secid), as_of, pred_len, max_target_time.tzinfo)
        if target is None:
            continue
        targets[instrument.secid] = target.isoformat(timespec="seconds")
        if target > max_target_time:
            violations.append(
                {
                    "secid": instrument.secid,
                    "target_timestamp": target.isoformat(timespec="seconds"),
                    "max_target_time": max_target_time.isoformat(timespec="seconds"),
                }
            )
    return not violations, {
        "enabled": True,
        "pred_len": int(pred_len),
        "max_target_time": max_target_time.isoformat(timespec="seconds"),
        "targets": targets,
        "violations": violations,
    }


def _kronos_clipped_horizon(
    *,
    candles: Mapping[str, Any],
    instruments: tuple[Instrument, ...] | list[Instrument],
    as_of: datetime,
    pred_len: int,
    max_target_time: datetime | None,
) -> tuple[int, dict[str, Any]]:
    requested = max(int(pred_len), 1)
    if max_target_time is None:
        return requested, {
            "enabled": False,
            "requested_pred_len": requested,
            "available_pred_len": requested,
            "effective_pred_len": requested,
            "clipped": False,
            "short_horizon_next_candle": False,
        }
    last_diag: dict[str, Any] = {}
    for candidate in range(requested, 0, -1):
        ok, diag = _kronos_horizon_check(
            candles=candles,
            instruments=instruments,
            as_of=as_of,
            pred_len=candidate,
            max_target_time=max_target_time,
        )
        diag = {
            **diag,
            "requested_pred_len": requested,
            "available_pred_len": candidate,
            "effective_pred_len": candidate,
            "clipped": candidate != requested,
            "short_horizon_next_candle": False,
        }
        if ok:
            if candidate < 3:
                next_ok, next_diag = _kronos_horizon_check(
                    candles=candles,
                    instruments=instruments,
                    as_of=as_of,
                    pred_len=1,
                    max_target_time=max_target_time,
                )
                if next_ok:
                    return 1, {
                        **next_diag,
                        "requested_pred_len": requested,
                        "available_pred_len": candidate,
                        "effective_pred_len": 1,
                        "clipped": requested != 1,
                        "short_horizon_next_candle": True,
                    }
            return candidate, diag
        last_diag = diag
    return 0, {
        **last_diag,
        "requested_pred_len": requested,
        "available_pred_len": 0,
        "effective_pred_len": 0,
        "clipped": True,
        "short_horizon_next_candle": False,
    }


def _forecast_target_time(candles: Any, as_of: datetime, pred_len: int, tzinfo: Any) -> datetime | None:
    step = timedelta(hours=1)
    last = _as_forecast_datetime(as_of, tzinfo)
    if candles is not None and not getattr(candles, "empty", True) and "timestamps" in getattr(candles, "columns", []):
        ts = pd.to_datetime(candles["timestamps"], errors="coerce").dropna()
        if not ts.empty:
            last = _timestamp_to_datetime(ts.iloc[-1], tzinfo)
            if len(ts) >= 2:
                prev = _timestamp_to_datetime(ts.iloc[-2], tzinfo)
                delta = last - prev
                if delta.total_seconds() > 0:
                    step = delta
    return last + step * max(int(pred_len), 1)


def _timestamp_to_datetime(value: Any, tzinfo: Any) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        out = ts.to_pydatetime().replace(tzinfo=tzinfo)
    else:
        out = ts.to_pydatetime().astimezone(tzinfo)
    return out


def _as_forecast_datetime(value: datetime, tzinfo: Any) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=tzinfo)
    return value.astimezone(tzinfo)


def _trade_cost_breakdown(
    secid: str,
    snapshots: Mapping[str, Any],
    metrics: Mapping[str, Any],
    risk: Any,
) -> dict[str, float]:
    spread = _market_spread_pct(secid, snapshots, metrics)
    commission = _non_negative_float(getattr(risk, "commission_rate", 0.0), default=0.0)
    slippage_multiplier = _non_negative_float(getattr(risk, "slippage_spread_multiplier", 0.0), default=0.0)
    slippage_one_way = spread * slippage_multiplier
    one_way_cost = spread / 2.0 + commission + slippage_one_way
    return {
        "spread_pct": spread,
        "commission_rate": commission,
        "slippage_spread_multiplier": slippage_multiplier,
        "slippage_one_way": slippage_one_way,
        "one_way_cost": one_way_cost,
        "round_trip_cost": 2.0 * one_way_cost,
    }


def _market_spread_pct(secid: str, snapshots: Mapping[str, Any], metrics: Mapping[str, Any]) -> float:
    spread_pct = None
    metric = metrics.get(secid)
    if metric is not None:
        spread_pct = getattr(metric, "spread_pct", None)
    if spread_pct is None:
        snapshot = snapshots.get(secid)
        spread_pct = getattr(snapshot, "spread_pct", None) if snapshot is not None else None
    return _non_negative_float(spread_pct, default=1.0)


def _non_negative_float(value: Any, *, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out) or out < 0:
        return default
    return out


def _rank_budget_weights(n: int, rank_power: float) -> list[float]:
    if n <= 0:
        return []
    raw = [1.0 if rank_power == 0 else (rank + 1) ** (-float(rank_power)) for rank in range(n)]
    total = sum(raw)
    return [value / total for value in raw]


def _model_dir(config: RuntimeConfig) -> Path:
    model_path = Path(config.lightgbm.model_dir)
    return model_path if model_path.is_absolute() else Path(config.data_dir) / model_path
