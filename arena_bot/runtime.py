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
from .trading_calendar import CacheFirstSessionCalendarProvider, SessionCalendarProvider
from .types import AccountState, DecisionResult, Instrument, RuntimeConfig, SignalRow, TradeLifecycleEntryMetricsConfig, TradingSessionConfig
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
        session_calendar: SessionCalendarProvider | None = None,
        state: StateStore | None = None,
        logger: JsonlLogger | None = None,
        arenago_client: ArenaGoClient | None = None,
    ):
        self.config = config
        self.market_data = market_data or EmptyMarketDataProvider()
        data_dir = Path(config.data_dir)
        self.state = state or StateStore(data_dir / "arena_state.sqlite3")
        self.logger = logger or JsonlLogger(data_dir / "logs")
        self.session_calendar = session_calendar or CacheFirstSessionCalendarProvider(config=config.trading_session, state=self.state)
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
        if not bool(getattr(lifecycle.exit, "edge_enabled", True)) and not bool(getattr(lifecycle.exit, "particle_enabled", False)):
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
        snapshots = self.market_data.snapshots(as_of, self.config.instruments)
        candles = self.market_data.candles(as_of, self.config.instruments)
        metrics = self.market_data.metrics(as_of, self.config.instruments)
        session = self.session_calendar.session_state(as_of, self.config.instruments, candles=candles)
        prices = {secid: snapshot.last_price for secid, snapshot in snapshots.items()}
        buy_prices, sell_prices = _execution_price_maps(snapshots)
        positions_before_exit = self.order_manager.current_positions()
        account_before = self._load_account_state(positions_before_exit, prices)
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
        entry_rebalance_diagnostics: dict[str, Any] = {}
        horizon_diagnostics: dict[str, Any] = {}
        entry_recheck_diagnostics: dict[str, Any] = {}
        session_filter_diagnostics: dict[str, Any] = {}
        selected: tuple[Instrument, ...] = tuple()
        entry_candidates = _entry_allowed_instruments(self.config.instruments, session)
        if not entry_candidates:
            entry_block_reason = _session_entry_block_reason(session)
            universe_diagnostics = {"status": "session_skipped", "reason": entry_block_reason}
        else:
            universe = select_universe(
                entry_candidates,
                snapshots=snapshots,
                metrics=metrics,
                max_equities=self.config.max_equities,
            )
            universe_diagnostics = universe.diagnostics
            for secid, reason in universe.diagnostics.get("rejected", {}).items():
                self.logger.write("instrument_untradable", {"as_of": as_of_s, "secid": secid, "reason": reason})
            selected = universe.instruments
            selected, session_filter_diagnostics = _filter_entry_instruments_for_session(
                candles=candles,
                instruments=selected,
                as_of=as_of,
                pred_len=max(int(self.config.kronos.pred_len), 1),
                decision_interval_minutes=int(self.config.rebalance.decision_interval_minutes),
                session=session,
            )
            horizon_diagnostics = dict(session_filter_diagnostics.get("kronos_horizon") or {})
            entry_recheck_diagnostics = dict(session_filter_diagnostics.get("entry_recheck_window") or {})
            entry_rebalance_diagnostics = dict(session_filter_diagnostics.get("entry_rebalance_tick") or {})
            if not selected:
                entry_block_reason = str(session_filter_diagnostics.get("primary_reason") or "session_no_entry_candidates")

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
            if entry_recheck_diagnostics:
                meta_payload["entry_recheck_window"] = entry_recheck_diagnostics
            if entry_rebalance_diagnostics:
                meta_payload["entry_rebalance_tick"] = entry_rebalance_diagnostics
            if session_filter_diagnostics:
                meta_payload["session_filter"] = session_filter_diagnostics
            entry_diagnostics = {
                "status": "blocked",
                "reason": entry_block_reason,
                "selected_count": 0,
                "ranked_candidates": [],
            }
            if horizon_diagnostics:
                entry_diagnostics["kronos_horizon"] = horizon_diagnostics
            if entry_recheck_diagnostics:
                entry_diagnostics["entry_recheck_window"] = entry_recheck_diagnostics
            if entry_rebalance_diagnostics:
                entry_diagnostics["entry_rebalance_tick"] = entry_rebalance_diagnostics
            if session_filter_diagnostics:
                entry_diagnostics["session_filter"] = session_filter_diagnostics
            blend_diagnostics = {
                "mode": "session_guard",
                "reason": entry_block_reason,
                "final_target_positions_count": len([weight for weight in target_weights.values() if abs(float(weight)) > 1e-12]),
            }
            self.logger.write("selector_model_ready", {"as_of": as_of_s, **meta_payload})
        else:
            self._update_selector_returns(as_of_s, {secid: row.last_price for secid, row in snapshots.items()})
            entry_mode = _entry_mode(self.config)
            if entry_mode == "kronos_rank":
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
                    session_filter_diagnostics=session_filter_diagnostics,
                )
                blend_diagnostics = {
                    "mode": "bypassed_kronos_rank",
                    "ranking_mode": "kronos_rank",
                    "final_target_positions_count": len([weight for weight in target_weights.values() if abs(float(weight)) > 1e-12]),
                    "ranked_candidates_count": len(entry_diagnostics.get("ranked_candidates", [])),
                    "selected_count": int(entry_diagnostics.get("selected_count", 0) or 0),
                }
                self.logger.write("selector_model_ready", {"as_of": as_of_s, **meta_payload})
            elif entry_mode == "kronos_single_top":
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
                selector_weights = {"kronos_single_top": 1.0}
                meta_payload = {
                    "mode": "bypassed_kronos_single_top",
                    "reason": "trade_lifecycle.entry.mode=kronos_single_top",
                    "features": features,
                }
                selector_diagnostics = {}
                target_weights, entry_diagnostics = self._build_kronos_single_top_targets(
                    instruments=selected,
                    positions=positions_after_risk,
                    account=account_after_risk,
                    prices=prices,
                    snapshots=snapshots,
                    metrics=metrics,
                    signals=signals,
                    blocked_secids=exit_close_secids,
                    session_filter_diagnostics=session_filter_diagnostics,
                )
                blend_diagnostics = {
                    "mode": "bypassed_kronos_single_top",
                    "ranking_mode": "kronos_single_top",
                    "final_target_positions_count": len([weight for weight in target_weights.values() if abs(float(weight)) > 1e-12]),
                    "ranked_candidates_count": len(entry_diagnostics.get("ranked_candidates", [])),
                    "selected_count": int(entry_diagnostics.get("selected_count", 0) or 0),
                    "target_action": entry_diagnostics.get("target_action", ""),
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
        if _entry_mode(self.config) == "kronos_single_top":
            entry_orders = [
                replace(order, reason="single_top_rebalance")
                for order in self.order_manager.plan_orders(
                    target_weights=target_weights,
                    target_scores=target_scores,
                    positions=positions_after_risk,
                    prices=prices,
                    cash=account_after_risk.cash,
                    buy_prices=buy_prices,
                    sell_prices=sell_prices,
                )
            ][:remaining_order_slots]
        else:
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
        giveback_state_sync = self._sync_giveback_state_after_orders(
            as_of=as_of,
            planned=planned,
            orders=orders,
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
            "giveback_state_sync": giveback_state_sync,
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
        giveback_state_sync = self._sync_giveback_state_after_orders(
            as_of=as_of,
            planned=planned,
            orders=orders,
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
            "giveback_state_sync": giveback_state_sync,
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

        scores: dict[str, float] = {}
        close_secids: list[str] = []
        session_by_secid = _session_by_secid(session)
        force_flat_secids = {
            instrument.secid
            for instrument in held
            if bool(_instrument_session(session_by_secid, instrument.secid).get("force_flat_required", False))
        }
        for instrument in held:
            if instrument.secid not in force_flat_secids:
                continue
            side = "long" if int(positions.get(instrument.secid, 0) or 0) > 0 else "short"
            diagnostics["held"][instrument.secid] = {
                "action": "close",
                "action_reason": "session_force_flat",
                "side": side,
                "session": _session_public_payload(_instrument_session(session_by_secid, instrument.secid)),
            }
            scores[instrument.secid] = 1.0
            close_secids.append(instrument.secid)
        remaining_held = [instrument for instrument in held if instrument.secid not in force_flat_secids]
        if not remaining_held:
            diagnostics["status"] = "ready"
            diagnostics["close_secids"] = close_secids
            return diagnostics, scores

        exitable: list[Instrument] = []
        for instrument in remaining_held:
            instrument_session = _instrument_session(session_by_secid, instrument.secid)
            if instrument_session and not bool(instrument_session.get("exit_allowed", True)):
                reason = str(instrument_session.get("action_reason") or "session_exit_blocked")
                diagnostics["held"][instrument.secid] = {
                    "action": "hold",
                    "action_reason": reason,
                    "session": _session_public_payload(instrument_session),
                }
                continue
            exitable.append(instrument)
        remaining_held = exitable
        if not remaining_held:
            diagnostics["status"] = "ready"
            diagnostics["close_secids"] = close_secids
            return diagnostics, scores
        if bool(getattr(lifecycle.exit, "giveback_enabled", False)):
            giveback_scores, giveback_close_secids = self._build_giveback_exit_plan(
                as_of=as_of,
                held=remaining_held,
                positions=positions,
                snapshots=snapshots,
                candles=candles,
                diagnostics=diagnostics,
            )
            scores.update(giveback_scores)
            close_secids.extend(giveback_close_secids)
            closed = set(giveback_close_secids)
            remaining_held = [instrument for instrument in remaining_held if instrument.secid not in closed]
            diagnostics["close_secids"] = list(close_secids)
        if not remaining_held:
            diagnostics["status"] = "ready"
            return diagnostics, scores
        if bool(getattr(lifecycle.exit, "particle_enabled", False)):
            particle_diagnostics, particle_scores = self._build_particle_exit_plan(
                as_of=as_of,
                held=remaining_held,
                positions=positions,
                snapshots=snapshots,
                metrics=metrics,
                candles=candles,
                diagnostics=diagnostics,
                session=session,
            )
            return particle_diagnostics, {**scores, **particle_scores}
        if not bool(getattr(lifecycle.exit, "edge_enabled", True)):
            for instrument in remaining_held:
                diagnostics["held"].setdefault(
                    instrument.secid,
                    {
                        "action": "hold",
                        "action_reason": "exit_rules_disabled",
                        "giveback_enabled": bool(getattr(lifecycle.exit, "giveback_enabled", False)),
                    },
                )
            diagnostics["close_secids"] = close_secids
            diagnostics["status"] = "ready"
            return diagnostics, scores
        kronos_held: list[Instrument] = []
        for instrument in remaining_held:
            instrument_session = _instrument_session(session_by_secid, instrument.secid)
            if instrument_session and not bool(instrument_session.get("kronos_allowed", True)):
                reason = str(instrument_session.get("action_reason") or "session_kronos_blocked")
                diagnostics["held"].setdefault(
                    instrument.secid,
                    {
                        "action": "hold",
                        "action_reason": reason,
                        "session": _session_public_payload(instrument_session),
                    },
                )
                continue
            kronos_held.append(instrument)
        remaining_held = kronos_held
        if not remaining_held:
            diagnostics["status"] = "ready"
            diagnostics["close_secids"] = close_secids
            return diagnostics, scores

        rows = tuple(self.exit_kronos_provider.score(as_of, remaining_held, candles))
        by_secid = {row.secid: row for row in rows if row.signal_name == "kronos"}
        for instrument in remaining_held:
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

    def _build_giveback_exit_plan(
        self,
        *,
        as_of: datetime,
        held: tuple[Instrument, ...] | list[Instrument],
        positions: Mapping[str, int],
        snapshots: Mapping[str, Any],
        candles: Mapping[str, Any],
        diagnostics: dict[str, Any],
    ) -> tuple[dict[str, float], list[str]]:
        lifecycle = self.config.trade_lifecycle
        min_arm_profit = _non_negative_float(getattr(lifecycle.exit, "giveback_min_arm_profit", 0.012), default=0.012)
        ratio_threshold = _non_negative_float(getattr(lifecycle.exit, "giveback_ratio", 0.60), default=0.60)
        held_secids = {instrument.secid for instrument in held}
        for secid in set(self.state.load_position_giveback_states()) - held_secids:
            self.state.delete_position_giveback_state(secid)

        stored_positions = self.state.load_paper_positions()
        scores: dict[str, float] = {}
        close_secids: list[str] = []
        as_of_s = as_of.isoformat(timespec="seconds")
        for instrument in held:
            secid = instrument.secid
            lots = int(positions.get(secid, 0) or 0)
            side = _tracker_side_for_lots(lots)
            if side is None:
                self.state.delete_position_giveback_state(secid)
                continue

            current_price = _current_close(secid, snapshots, candles, {})
            if current_price <= 0:
                diagnostics["held"][secid] = {
                    "action": "hold",
                    "action_reason": "giveback_current_price_missing",
                    "side": side,
                    "giveback_enabled": True,
                }
                continue

            state = self.state.load_position_giveback_state(secid)
            entry_price = _finite_float((state or {}).get("entry_price", 0.0), default=0.0)
            state_side = str((state or {}).get("side") or "")
            state_bootstrapped = state is None or state_side != side or entry_price <= 0
            if state_bootstrapped:
                entry_price = current_price
                previous_mfe = 0.0
                current_pnl = 0.0
                opened_at = str(stored_positions.get(secid, {}).get("opened_at") or as_of_s)
            else:
                previous_mfe = max(_finite_float(state.get("mfe_pct", 0.0), default=0.0), 0.0)
                current_pnl = _side_value(side) * (current_price / entry_price - 1.0)
                opened_at = str(state.get("opened_at") or stored_positions.get(secid, {}).get("opened_at") or as_of_s)

            mfe_pct = max(previous_mfe, current_pnl, 0.0)
            giveback_pct = max(mfe_pct - current_pnl, 0.0) if mfe_pct > 0 else 0.0
            giveback_ratio = giveback_pct / mfe_pct if mfe_pct > 0 else 0.0
            armed = mfe_pct >= min_arm_profit
            close = bool(armed and giveback_ratio >= ratio_threshold)
            action_reason = "giveback_trailing" if close else ("giveback_armed_hold" if armed else "giveback_not_armed")

            self.state.save_position_giveback_state(
                secid=secid,
                side=side,
                entry_price=entry_price,
                mfe_pct=mfe_pct,
                last_pnl_pct=current_pnl,
                opened_at=opened_at,
                updated_at=as_of_s,
            )
            if close:
                close_secids.append(secid)
            scores[secid] = giveback_ratio if close else max(mfe_pct, giveback_ratio)
            diagnostics["held"][secid] = {
                "action": "close" if close else "hold",
                "action_reason": action_reason,
                "side": side,
                "giveback_enabled": True,
                "state_bootstrapped": state_bootstrapped,
                "entry_price": entry_price,
                "current_price": current_price,
                "current_pnl_pct": current_pnl,
                "previous_mfe_pct": previous_mfe,
                "mfe_pct": mfe_pct,
                "giveback_pct": giveback_pct,
                "giveback_ratio": giveback_ratio,
                "giveback_min_arm_profit": min_arm_profit,
                "giveback_ratio_threshold": ratio_threshold,
                "giveback_armed": armed,
                "opened_at": opened_at,
                "updated_at": as_of_s,
            }
        return scores, close_secids

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
        session_by_secid = _session_by_secid(session)
        held_secids = {instrument.secid for instrument in held}
        for secid in set(self.state.load_kronos_exit_trackers()) - held_secids:
            self.state.delete_kronos_exit_tracker(secid)

        scores: dict[str, float] = {}
        close_secids: list[str] = list(diagnostics.get("close_secids", []))
        for instrument in held:
            secid = instrument.secid
            lots = int(positions.get(secid, 0) or 0)
            side = _tracker_side_for_lots(lots)
            if side is None:
                continue
            instrument_session = _instrument_session(session_by_secid, secid)
            kronos_allowed = bool(instrument_session.get("kronos_allowed", True)) if instrument_session else True
            allow_new_trackers = bool(instrument_session.get("allow_new_trackers", True)) if instrument_session else True
            tracker_max_target_time = instrument_session.get("_force_flat_dt") if instrument_session else None
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

    def _sync_giveback_state_after_orders(
        self,
        *,
        as_of: datetime,
        planned: list[Any],
        orders: tuple[Any, ...],
    ) -> dict[str, Any]:
        if self.config.live_orders:
            return {"enabled": bool(getattr(self.config.trade_lifecycle.exit, "giveback_enabled", False)), "status": "live_skipped"}
        as_of_s = as_of.isoformat(timespec="seconds")
        opened_or_reset: list[str] = []
        deleted: list[str] = []
        kept: list[str] = []
        giveback_enabled = bool(getattr(self.config.trade_lifecycle.exit, "giveback_enabled", False))
        for order, result in zip(planned, orders):
            if result.status not in {"dry_run", "submitted"}:
                continue
            side_before = _tracker_side_for_lots(int(getattr(order, "current_lots", 0) or 0))
            side_after = _tracker_side_for_lots(int(getattr(order, "target_lots", 0) or 0))
            if side_after is None:
                self.state.delete_position_giveback_state(order.secid)
                deleted.append(order.secid)
                continue
            if not giveback_enabled:
                continue
            existing = self.state.load_position_giveback_state(order.secid)
            if side_before != side_after or existing is None:
                self.state.save_position_giveback_state(
                    secid=order.secid,
                    side=side_after,
                    entry_price=float(getattr(order, "price", 0.0) or 0.0),
                    mfe_pct=0.0,
                    last_pnl_pct=0.0,
                    opened_at=as_of_s,
                    updated_at=as_of_s,
                )
                opened_or_reset.append(order.secid)
            else:
                kept.append(order.secid)
        return {
            "enabled": giveback_enabled,
            "status": "ready",
            "opened_or_reset": sorted(set(opened_or_reset)),
            "deleted": sorted(set(deleted)),
            "kept": sorted(set(kept)),
        }

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
        session_by_secid = _session_by_secid(session)
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
            instrument_session = _instrument_session(session_by_secid, secid)
            allow_new_trackers = bool(instrument_session.get("allow_new_trackers", True)) if instrument_session else True
            allow_forecast = bool(instrument_session.get("kronos_allowed", True)) if instrument_session else True
            max_target_time = instrument_session.get("_force_flat_dt") if instrument_session else None
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
        session_filter_diagnostics: Mapping[str, Any] | None = None,
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
            signal_row = rows_by_secid.get(secid)
            pred_return = _signal_pred_return(signal_row)
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
            base_item.update(
                _kronos_entry_metrics_payload(
                    secid=secid,
                    side=str(base_item["side"]),
                    row=signal_row,
                    snapshot=snapshots.get(secid),
                    metric=metrics.get(secid),
                    price=float(prices.get(secid, 0.0) or 0.0),
                    costs=costs,
                    config=lifecycle.entry.metrics,
                    session_filter_diagnostics=session_filter_diagnostics,
                )
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

    def _build_kronos_single_top_targets(
        self,
        *,
        instruments: tuple[Instrument, ...],
        positions: Mapping[str, int],
        account: AccountState,
        prices: Mapping[str, float],
        snapshots: Mapping[str, Any],
        metrics: Mapping[str, Any],
        signals: tuple[SignalRow, ...],
        blocked_secids: set[str] | None = None,
        session_filter_diagnostics: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        lifecycle = self.config.trade_lifecycle
        entry = lifecycle.entry
        blocked_secids = set(blocked_secids or set())
        equity = max(float(account.equity), 1.0)
        current_weights = _position_weights(positions, prices, self.instruments_by_secid, equity)
        active_positions = {secid: int(lots or 0) for secid, lots in positions.items() if int(lots or 0) != 0}
        rows_by_secid = {row.secid: row for row in signals if row.signal_name == "kronos"}
        min_net_edge = float(entry.single_top_min_net_edge)
        min_rank_gap = max(float(entry.single_top_min_rank_gap), 0.0)
        max_gross_pred_return = float(entry.single_top_max_gross_pred_return)
        target_abs_weight = _single_top_target_abs_weight(self.config, account)

        ranked: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for instrument in instruments:
            secid = instrument.secid
            current_lots = int(positions.get(secid, 0) or 0)
            current_side = _position_side(current_lots)
            signal_row = rows_by_secid.get(secid)
            pred_return = _signal_pred_return(signal_row)
            item: dict[str, Any] = {
                "rank": None,
                "secid": secid,
                "asset_class": instrument.asset_class,
                "pred_return": pred_return,
                "edge": None,
                "side": None,
                "held": current_lots != 0,
                "selected": False,
                "reason": "",
                "allocated_weight": 0.0,
                "target_weight": 0.0,
                "same_as_current_position": False,
                "single_top_passed_filters": False,
                "blocked_by_exit_pass": secid in blocked_secids,
            }
            if pred_return is None:
                item["reason"] = "missing_kronos_pred_return"
                skipped.append(item)
                continue

            costs = _trade_cost_breakdown(secid, snapshots, metrics, self.config.risk)
            gross_pred_return = abs(float(pred_return))
            net_edge = gross_pred_return - costs["round_trip_cost"]
            side = 1 if float(pred_return) > 0 else (-1 if float(pred_return) < 0 else 0)
            item.update(
                {
                    "pred_return": float(pred_return),
                    "edge": net_edge,
                    "gross_pred_return": gross_pred_return,
                    "net_edge": net_edge,
                    "side": "long" if side > 0 else ("short" if side < 0 else None),
                    **costs,
                }
            )
            if side == 0:
                item["reason"] = "zero_pred_return"
                skipped.append(item)
                continue
            item.update(
                _kronos_entry_metrics_payload(
                    secid=secid,
                    side=str(item["side"]),
                    row=signal_row,
                    snapshot=snapshots.get(secid),
                    metric=metrics.get(secid),
                    price=float(prices.get(secid, 0.0) or 0.0),
                    costs=costs,
                    config=entry.metrics,
                    session_filter_diagnostics=session_filter_diagnostics,
                )
            )
            item["same_as_current_position"] = current_side == side and current_lots != 0
            filter_reason = _single_top_filter_reason(
                gross_pred_return=gross_pred_return,
                net_edge=net_edge,
                min_net_edge=min_net_edge,
                max_gross_pred_return=max_gross_pred_return,
            )
            item["single_top_passed_filters"] = filter_reason == ""
            item["filter_reason"] = filter_reason
            ranked.append(item)

        ranked.sort(key=lambda item: (-float(item["gross_pred_return"]), str(item["secid"])))
        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank
            next_item = ranked[rank] if rank < len(ranked) else None
            rank_gap_to_next = (
                float(item["gross_pred_return"]) - float(next_item["gross_pred_return"])
                if next_item is not None
                else None
            )
            item["rank_gap_to_next"] = rank_gap_to_next
            if rank == 1:
                item["top1_top2_gross_gap"] = rank_gap_to_next
            if rank > 1:
                item["reason"] = "closed_in_exit_pass" if str(item["secid"]) in blocked_secids else "not_top_1"

        target_weights: dict[str, float] = {}
        top = ranked[0] if ranked else None
        top1_secid = str(top["secid"]) if top else ""
        top1_side = str(top["side"]) if top and top.get("side") else ""
        top1_top2_gross_gap = top.get("top1_top2_gross_gap") if top else None
        if top is not None:
            filter_reason = _single_top_filter_reason(
                gross_pred_return=float(top["gross_pred_return"]),
                net_edge=float(top["net_edge"]),
                min_net_edge=min_net_edge,
                max_gross_pred_return=max_gross_pred_return,
                rank_gap=top1_top2_gross_gap,
                min_rank_gap=min_rank_gap,
            )
            top["single_top_passed_filters"] = filter_reason == ""
            top["filter_reason"] = filter_reason
        target_action = "close_to_cash" if active_positions else "skip_flat"
        action_reason = "missing_kronos_pred_return"
        selected_count = 0

        if top is not None:
            filter_reason = str(top.get("filter_reason") or "")
            if top1_secid in blocked_secids:
                top["single_top_passed_filters"] = False
                top["reason"] = "closed_in_exit_pass"
                action_reason = "closed_in_exit_pass"
            elif filter_reason:
                top["reason"] = filter_reason
                action_reason = "single_top_close_to_cash" if active_positions else filter_reason
            elif target_abs_weight <= 0:
                top["reason"] = "single_top_no_target_weight"
                action_reason = "single_top_no_target_weight"
            else:
                top["selected"] = True
                selected_count = 1
                side_value = 1.0 if top["side"] == "long" else -1.0
                same_current = bool(top.get("same_as_current_position")) and len(active_positions) == 1 and top1_secid in active_positions
                if same_current:
                    target_weights = dict(current_weights)
                    target_action = "hold_same"
                    top["reason"] = "same_top_hold"
                    action_reason = "same_top_hold"
                    top["allocated_weight"] = abs(float(current_weights.get(top1_secid, 0.0) or 0.0))
                    top["target_weight"] = float(current_weights.get(top1_secid, 0.0) or 0.0)
                else:
                    target_weight = side_value * target_abs_weight
                    target_weights = {top1_secid: target_weight}
                    target_action = "switch" if active_positions else "open"
                    top["reason"] = "single_top_entry"
                    action_reason = "single_top_entry"
                    top["allocated_weight"] = target_abs_weight
                    top["target_weight"] = target_weight

        top1_passed = bool(top and top.get("single_top_passed_filters"))
        candidates = ranked + sorted(skipped, key=lambda item: str(item["secid"]))
        return target_weights, {
            "ranking_mode": "kronos_single_top",
            "ranking_metric": "gross_pred_return",
            "max_total_positions": int(lifecycle.max_total_positions),
            "held_count": len(active_positions),
            "gross_after_exit": float(account.gross),
            "margin_used_after_exit": float(account.margin_used),
            "free_capital": _entry_free_value(account, self.config.risk),
            "free_weight": _entry_free_value(account, self.config.risk) / equity if equity > 0 else 0.0,
            "single_top_min_net_edge": min_net_edge,
            "single_top_min_rank_gap": min_rank_gap,
            "single_top_max_gross_pred_return": max_gross_pred_return,
            "single_top_target_weight": float(entry.single_top_target_weight),
            "target_abs_weight": target_abs_weight,
            "selected_count": selected_count,
            "top1_secid": top1_secid,
            "top1_side": top1_side,
            "top1_top2_gross_gap": top1_top2_gross_gap,
            "top1_passed_filters": top1_passed,
            "target_action": target_action,
            "action_reason": action_reason,
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


def _kronos_entry_metrics_payload(
    *,
    secid: str,
    side: str | None,
    row: SignalRow | None,
    snapshot: Any,
    metric: Any,
    price: float,
    costs: Mapping[str, Any],
    config: TradeLifecycleEntryMetricsConfig,
    session_filter_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not bool(getattr(config, "enabled", True)):
        return {"kronos_metrics_status": "disabled"}

    pred = _signal_pred_ohlcv(row)
    if pred is None:
        return {"kronos_metrics_status": "missing_pred_ohlcv"}

    payload: dict[str, Any] = {
        "kronos_metrics_status": "ok",
        "pred_open": pred["open"],
        "pred_high": pred["high"],
        "pred_low": pred["low"],
        "pred_close": pred["close"],
    }
    if side not in {"long", "short"}:
        payload["kronos_metrics_status"] = "missing_side"
        return payload

    market = _kronos_entry_market_prices(snapshot=snapshot, price=price, costs=costs)
    if market is None:
        payload["kronos_metrics_status"] = "missing_market_price"
        return payload

    o = pred["open"]
    h = pred["high"]
    l = pred["low"]
    c = pred["close"]
    bid = market["bid"]
    ask = market["ask"]
    mid = market["mid"]
    eps = max(_metrics_config_float(config, "eps", 1e-12), 1e-12)

    pred_range = max(h - l, eps)
    spread_bps = ((ask - bid) / mid) * 10000.0 if mid > 0 else 0.0
    roundtrip_cost_bps = _non_negative_float(costs.get("round_trip_cost"), default=0.0) * 10000.0
    realized_vol_1h_bps = _non_negative_float(getattr(metric, "realized_volatility", 0.0), default=0.0) * 10000.0
    minutes_to_kronos_cutoff = _kronos_metrics_remaining_minutes(secid, session_filter_diagnostics)

    if side == "long":
        raw_edge_bps = ((c - ask) / ask) * 10000.0
        pred_mfe_bps = ((h - ask) / ask) * 10000.0
        pred_mae_bps = ((ask - l) / ask) * 10000.0
        close_quality = (c - l) / pred_range
    else:
        raw_edge_bps = ((bid - c) / bid) * 10000.0
        pred_mfe_bps = ((bid - l) / bid) * 10000.0
        pred_mae_bps = ((h - bid) / bid) * 10000.0
        close_quality = 1.0 - ((c - l) / pred_range)

    net_edge_bps = raw_edge_bps - roundtrip_cost_bps
    pred_rr = pred_mfe_bps / max(pred_mae_bps, eps)
    net_edge_score = _clip01((net_edge_bps - _metrics_config_float(config, "min_edge_bps", 10.0)) / max(_metrics_config_float(config, "edge_scale_bps", 70.0), eps))
    edge_z = net_edge_bps / max(realized_vol_1h_bps, _metrics_config_float(config, "vol_floor_bps", 10.0), eps)
    edge_z_score = _clip01((edge_z - 0.3) / (1.5 - 0.3))
    rr_score = _clip01((pred_rr - 1.0) / (2.5 - 1.0))
    max_allowed_mae_bps = max(50.0, 1.2 * realized_vol_1h_bps)
    mae_score = 1.0 - _clip01(pred_mae_bps / max(max_allowed_mae_bps, eps))
    close_score = _clip01((close_quality - 0.5) / 0.5)

    body_ratio = abs(c - o) / pred_range
    body_score = _clip01((body_ratio - 0.10) / (0.60 - 0.10))
    if side == "long":
        if c <= o:
            body_score *= 0.5
        if c <= mid:
            body_score = 0.0
    else:
        if c >= o:
            body_score *= 0.5
        if c >= mid:
            body_score = 0.0

    upper_wick = max(h - max(o, c), 0.0)
    lower_wick = max(min(o, c) - l, 0.0)
    upper_wick_ratio = upper_wick / pred_range
    lower_wick_ratio = lower_wick / pred_range
    bad_wick_ratio = upper_wick_ratio if side == "long" else lower_wick_ratio
    wick_score = 1.0 - _clip01(bad_wick_ratio / 0.70)
    candle_quality = close_score * body_score * wick_score
    edge_risk_quality = net_edge_score * rr_score * mae_score

    false_breakout_risk = _clip01(bad_wick_ratio * (1.0 - close_score))
    wide_spread_risk = _clip01(spread_bps / max(_metrics_config_float(config, "max_allowed_spread_bps", 20.0), eps))
    required_recheck_minutes = max(_metrics_config_float(config, "required_recheck_minutes", 120.0), eps)
    late_entry_risk = 0.0 if minutes_to_kronos_cutoff is None else _clip01((required_recheck_minutes - minutes_to_kronos_cutoff) / required_recheck_minutes)
    high_mae_risk = 1.0 - mae_score
    if side == "long":
        if c > o and c > mid:
            direction_conflict_risk = 0.0
        elif c > mid and c <= o:
            direction_conflict_risk = 0.5
        else:
            direction_conflict_risk = 1.0
    else:
        if c < o and c < mid:
            direction_conflict_risk = 0.0
        elif c < mid and c >= o:
            direction_conflict_risk = 0.5
        else:
            direction_conflict_risk = 1.0

    positive_metrics = {
        "net_edge_score": net_edge_score,
        "edge_z_score": edge_z_score,
        "rr_score": rr_score,
        "mae_score": mae_score,
        "close_score": close_score,
        "body_score": body_score,
        "wick_score": wick_score,
        "candle_quality": candle_quality,
        "edge_risk_quality": edge_risk_quality,
    }
    risk_metrics = {
        "false_breakout_risk": false_breakout_risk,
        "wide_spread_risk": wide_spread_risk,
        "late_entry_risk": late_entry_risk,
        "high_mae_risk": high_mae_risk,
        "direction_conflict_risk": direction_conflict_risk,
    }
    payload.update(
        {
            "market_price_source": market["source"],
            "current_bid": bid,
            "current_ask": ask,
            "current_mid": mid,
            "pred_range": pred_range,
            "spread_bps": spread_bps,
            "roundtrip_cost_bps": roundtrip_cost_bps,
            "raw_edge_bps": raw_edge_bps,
            "net_edge_bps": net_edge_bps,
            "pred_mfe_bps": pred_mfe_bps,
            "pred_mae_bps": pred_mae_bps,
            "pred_rr": pred_rr,
            "realized_vol_1h_bps": realized_vol_1h_bps,
            "minutes_to_kronos_cutoff": minutes_to_kronos_cutoff,
            "edge_z": edge_z,
            "max_allowed_mae_bps": max_allowed_mae_bps,
            "close_quality": close_quality,
            "body_ratio": body_ratio,
            "upper_wick_ratio": upper_wick_ratio,
            "lower_wick_ratio": lower_wick_ratio,
            "bad_wick_ratio": bad_wick_ratio,
            "positive_metrics": positive_metrics,
            "risk_metrics": risk_metrics,
        }
    )
    return _json_safe_floats(payload)


def _signal_pred_ohlcv(row: SignalRow | None) -> dict[str, float] | None:
    if row is None or not row.metadata:
        return None
    raw = row.metadata.get("pred_ohlcv")
    if not isinstance(raw, Mapping):
        return None
    pred = {col: _finite_float(raw.get(col), default=0.0) for col in ("open", "high", "low", "close")}
    if any(value <= 0 for value in pred.values()):
        return None
    if pred["high"] < pred["low"]:
        return None
    return pred


def _kronos_entry_market_prices(*, snapshot: Any, price: float, costs: Mapping[str, Any]) -> dict[str, float | str] | None:
    bid = _finite_float(getattr(snapshot, "bid", 0.0), default=0.0) if snapshot is not None else 0.0
    ask = _finite_float(getattr(snapshot, "ask", 0.0), default=0.0) if snapshot is not None else 0.0
    if bid > 0 and ask > 0 and ask >= bid:
        mid = (bid + ask) / 2.0
        if mid > 0:
            return {"bid": bid, "ask": ask, "mid": mid, "source": "snapshot_bid_ask"}

    mid = _finite_float(getattr(snapshot, "last_price", 0.0), default=0.0) if snapshot is not None else 0.0
    source = "last_price_plus_spread"
    if mid <= 0:
        mid = _finite_float(price, default=0.0)
        source = "price_map_plus_spread"
    if mid <= 0:
        return None

    spread_pct = _non_negative_float(costs.get("spread_pct"), default=0.0)
    half_spread = min(spread_pct / 2.0, 0.99)
    return {
        "bid": mid * (1.0 - half_spread),
        "ask": mid * (1.0 + half_spread),
        "mid": mid,
        "source": source,
    }


def _kronos_metrics_remaining_minutes(secid: str, session_filter_diagnostics: Mapping[str, Any] | None) -> float | None:
    if not session_filter_diagnostics:
        return None
    by_secid = session_filter_diagnostics.get("by_secid")
    if isinstance(by_secid, Mapping):
        row = by_secid.get(secid)
        if isinstance(row, Mapping):
            recheck = row.get("entry_recheck_window")
            if isinstance(recheck, Mapping):
                value = _finite_float(recheck.get("remaining_minutes"), default=-1.0)
                if value >= 0:
                    return value
    recheck = session_filter_diagnostics.get("entry_recheck_window")
    if isinstance(recheck, Mapping):
        value = _finite_float(recheck.get("remaining_minutes"), default=-1.0)
        if value >= 0:
            return value
    return None


def _metrics_config_float(config: TradeLifecycleEntryMetricsConfig, name: str, default: float) -> float:
    return _finite_float(getattr(config, name, default), default=default)


def _clip01(value: float) -> float:
    return min(max(_finite_float(value, default=0.0), 0.0), 1.0)


def _json_safe_floats(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_floats(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    return value


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


def _single_top_target_abs_weight(config: RuntimeConfig, account: AccountState) -> float:
    if float(account.equity) <= 0:
        return 0.0
    entry = config.trade_lifecycle.entry
    configured = min(max(float(entry.single_top_target_weight), 0.0), 10.0)
    max_gross = max(float(config.portfolio.max_gross), 0.0)
    cash_buffer = min(max(float(config.risk.cash_buffer_pct), 0.0), 0.95)
    safety = min(max(float(config.risk.sizing_safety_pct), 0.0), 0.95)
    buffered_full_capital = max(1.0 - cash_buffer, 0.0) * (1.0 - safety)
    return max(min(configured, max_gross, buffered_full_capital), 0.0)


def _single_top_filter_reason(
    *,
    gross_pred_return: float,
    net_edge: float,
    min_net_edge: float,
    max_gross_pred_return: float,
    rank_gap: float | None = None,
    min_rank_gap: float = 0.0,
) -> str:
    if float(gross_pred_return) > float(max_gross_pred_return):
        return "single_top_gross_pred_return_cap"
    if float(net_edge) < float(min_net_edge):
        return "single_top_net_edge_below_min"
    if rank_gap is not None and float(rank_gap) < float(min_rank_gap):
        return "single_top_rank_gap_below_min"
    return ""


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


def _session_by_secid(session: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    if session is None:
        return {}
    raw = session.get("_session_by_secid") or session.get("session_by_secid") or {}
    if not isinstance(raw, Mapping):
        return {}
    return {str(secid): dict(value) for secid, value in raw.items() if isinstance(value, Mapping)}


def _instrument_session(session_by_secid: Mapping[str, Mapping[str, Any]], secid: str) -> Mapping[str, Any]:
    return session_by_secid.get(str(secid), {})


def _entry_allowed_instruments(instruments: tuple[Instrument, ...], session: Mapping[str, Any] | None) -> tuple[Instrument, ...]:
    session_by_secid = _session_by_secid(session)
    if not session_by_secid:
        return instruments
    return tuple(
        instrument
        for instrument in instruments
        if bool(_instrument_session(session_by_secid, instrument.secid).get("entry_allowed", True))
    )


def _session_entry_block_reason(session: Mapping[str, Any] | None) -> str:
    session_by_secid = _session_by_secid(session)
    reasons = [
        str(row.get("action_reason") or "")
        for row in session_by_secid.values()
        if not bool(row.get("entry_allowed", True)) and str(row.get("action_reason") or "")
    ]
    if reasons:
        return sorted(set(reasons))[0]
    return str((session or {}).get("action_reason") or "session_entry_cutoff")


def _filter_entry_instruments_for_session(
    *,
    candles: Mapping[str, Any],
    instruments: tuple[Instrument, ...] | list[Instrument],
    as_of: datetime,
    pred_len: int,
    decision_interval_minutes: int,
    session: Mapping[str, Any] | None,
) -> tuple[tuple[Instrument, ...], dict[str, Any]]:
    session_by_secid = _session_by_secid(session)
    if not session_by_secid:
        return tuple(instruments), {"enabled": False}
    kept: list[Instrument] = []
    rejected: dict[str, str] = {}
    by_secid: dict[str, Any] = {}
    first_horizon: dict[str, Any] = {}
    first_recheck: dict[str, Any] = {}
    first_rebalance: dict[str, Any] = {}
    for instrument in instruments:
        secid = instrument.secid
        instrument_session = _instrument_session(session_by_secid, secid)
        if instrument_session and not bool(instrument_session.get("entry_allowed", True)):
            reason = str(instrument_session.get("action_reason") or "session_entry_cutoff")
            rejected[secid] = reason
            by_secid[secid] = {"reason": reason, "session": _session_public_payload(instrument_session)}
            continue
        horizon_ok, horizon_diag = _kronos_horizon_check(
            candles=candles,
            instruments=[instrument],
            as_of=as_of,
            pred_len=max(int(pred_len), 1),
            max_target_time=instrument_session.get("_kronos_cutoff_dt"),
        )
        if not horizon_ok:
            reason = "session_kronos_horizon_exceeds_close"
            rejected[secid] = reason
            by_secid[secid] = {"reason": reason, "kronos_horizon": horizon_diag}
            if not first_horizon:
                first_horizon = horizon_diag
            continue
        recheck_ok, recheck_diag = _entry_recheck_window_check(
            as_of=as_of,
            decision_interval_minutes=int(decision_interval_minutes),
            max_target_time=_entry_recheck_deadline(instrument_session),
        )
        if not recheck_ok:
            reason = "session_kronos_recheck_window_too_short"
            rejected[secid] = reason
            by_secid[secid] = {"reason": reason, "entry_recheck_window": recheck_diag}
            if not first_recheck:
                first_recheck = recheck_diag
            continue
        rebalance_ok, rebalance_diag = _entry_rebalance_tick_check(
            session=instrument_session,
            decision_interval_minutes=int(decision_interval_minutes),
        )
        if not rebalance_ok:
            reason = "entry_rebalance_wait"
            rejected[secid] = reason
            by_secid[secid] = {"reason": reason, "entry_rebalance_tick": rebalance_diag}
            if not first_rebalance:
                first_rebalance = rebalance_diag
            continue
        kept.append(instrument)
        by_secid[secid] = {
            "reason": "",
            "kronos_horizon": horizon_diag,
            "entry_recheck_window": recheck_diag,
            "entry_rebalance_tick": rebalance_diag,
        }
    primary_reason = ""
    if rejected and not kept:
        primary_reason = next(iter(rejected.values()))
    diagnostics: dict[str, Any] = {
        "enabled": True,
        "input_count": len(instruments),
        "kept_secids": [instrument.secid for instrument in kept],
        "rejected": rejected,
        "by_secid": by_secid,
        "primary_reason": primary_reason,
    }
    if first_horizon:
        diagnostics["kronos_horizon"] = first_horizon
    if first_recheck:
        diagnostics["entry_recheck_window"] = first_recheck
    if first_rebalance:
        diagnostics["entry_rebalance_tick"] = first_rebalance
    return tuple(kept), diagnostics


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
        "_entry_start_dt": entry_start,
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


def _entry_recheck_deadline(session: Mapping[str, Any]) -> datetime | None:
    values = [
        value
        for value in (session.get("_kronos_cutoff_dt"), session.get("_force_flat_dt"))
        if isinstance(value, datetime)
    ]
    if not values:
        return None
    return min(values)


def _entry_rebalance_tick_check(
    *,
    session: Mapping[str, Any],
    decision_interval_minutes: int,
) -> tuple[bool, dict[str, Any]]:
    interval = max(int(decision_interval_minutes), 1)
    if not bool(session.get("enabled", False)):
        return True, {"enabled": False, "decision_interval_minutes": interval}
    entry_start = session.get("_entry_start_dt")
    local_as_of_raw = session.get("local_as_of")
    if not isinstance(entry_start, datetime) or not local_as_of_raw:
        return True, {"enabled": False, "decision_interval_minutes": interval, "reason": "session_anchor_missing"}
    local_as_of = datetime.fromisoformat(str(local_as_of_raw))
    elapsed_seconds = (local_as_of - entry_start).total_seconds()
    elapsed_minutes = int(elapsed_seconds // 60)
    aligned = (
        elapsed_seconds >= 0
        and local_as_of.second == 0
        and local_as_of.microsecond == 0
        and elapsed_minutes % interval == 0
    )
    next_rebalance_at = local_as_of if aligned else entry_start + timedelta(minutes=((elapsed_minutes // interval) + 1) * interval)
    return aligned, {
        "enabled": True,
        "decision_interval_minutes": interval,
        "entry_start_at": entry_start.isoformat(timespec="seconds"),
        "as_of": local_as_of.isoformat(timespec="seconds"),
        "next_rebalance_at": next_rebalance_at.isoformat(timespec="seconds"),
        "minutes_since_entry_start": max(elapsed_minutes, 0),
    }


def _entry_recheck_window_check(
    *,
    as_of: datetime,
    decision_interval_minutes: int,
    max_target_time: datetime | None,
) -> tuple[bool, dict[str, Any]]:
    interval = max(int(decision_interval_minutes), 1)
    if max_target_time is None:
        return True, {
            "enabled": False,
            "decision_interval_minutes": interval,
        }
    local_as_of = _as_forecast_datetime(as_of, max_target_time.tzinfo)
    next_recheck_at = local_as_of + timedelta(minutes=interval)
    ok = next_recheck_at <= max_target_time
    return ok, {
        "enabled": True,
        "decision_interval_minutes": interval,
        "as_of": local_as_of.isoformat(timespec="seconds"),
        "next_recheck_at": next_recheck_at.isoformat(timespec="seconds"),
        "max_recheck_time": max_target_time.isoformat(timespec="seconds"),
        "remaining_minutes": max((max_target_time - local_as_of).total_seconds() / 60.0, 0.0),
        "violations": []
        if ok
        else [
            {
                "next_recheck_at": next_recheck_at.isoformat(timespec="seconds"),
                "max_recheck_time": max_target_time.isoformat(timespec="seconds"),
            }
        ],
    }


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
