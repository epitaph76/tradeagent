from __future__ import annotations

from datetime import datetime
from typing import Mapping

from .arenago import ArenaGoClient
from .storage import StateStore
from .types import Instrument, OrderResult, PlannedOrder, RiskConfig


class OrderManager:
    def __init__(
        self,
        *,
        config: RiskConfig,
        instruments: Mapping[str, Instrument],
        state: StateStore,
        bot_name: str,
        client: ArenaGoClient | None = None,
        live_orders: bool = False,
    ):
        self.config = config
        self.instruments = dict(instruments)
        self.state = state
        self.bot_name = bot_name
        self.client = client
        self.live_orders = bool(live_orders)

    def current_positions(self) -> dict[str, int]:
        if self.live_orders and self.client is not None:
            response = self.client.positions(self.bot_name)
            if response.ok:
                rows = response.payload if isinstance(response.payload, list) else response.payload.get("positions", [])
                out = {}
                for row in rows or []:
                    secid = str(row.get("secid") or row.get("ticker") or "")
                    if secid:
                        out[secid] = int(float(row.get("position", row.get("lots", 0)) or 0))
                return out
        return {secid: int(row["lots"]) for secid, row in self.state.load_paper_positions().items()}

    def plan_orders(
        self,
        *,
        target_weights: Mapping[str, float],
        target_scores: Mapping[str, float],
        positions: Mapping[str, int],
        prices: Mapping[str, float],
        cash: float,
        buy_prices: Mapping[str, float] | None = None,
        sell_prices: Mapping[str, float] | None = None,
    ) -> list[PlannedOrder]:
        equity = self._estimate_equity(positions, prices, cash)
        planned = []
        for secid in sorted(set(positions) | set(target_weights)):
            instrument = self.instruments.get(secid)
            if instrument is None or instrument.lot_size <= 0:
                continue
            price = float(prices.get(secid, 0.0) or 0.0)
            if price <= 0:
                continue
            target_weight = float(target_weights.get(secid, 0.0) or 0.0)
            current_lots = int(positions.get(secid, 0) or 0)
            target_lots = _target_lots(equity, target_weight, price, instrument.lot_size)
            delta = target_lots - current_lots
            if delta == 0:
                continue
            direction = "B" if delta > 0 else "S"
            execution_price = _execution_price(secid, direction, prices, buy_prices, sell_prices)
            order_value = abs(delta) * instrument.lot_size * execution_price
            if order_value < self.config.min_order_value_rub:
                continue
            if equity > 0 and order_value / equity < self.config.min_position_change_weight:
                continue
            planned.append(
                PlannedOrder(
                    secid=secid,
                    direction=direction,  # type: ignore[arg-type]
                    quantity=abs(delta),
                    current_lots=current_lots,
                    target_lots=target_lots,
                    price=execution_price,
                    lot_size=instrument.lot_size,
                    order_value=order_value,
                    target_weight=target_weight,
                    score=float(target_scores.get(secid, abs(target_weight))),
                    order_kind=_order_kind(current_lots, target_lots),
                )
            )

        planned.sort(key=lambda order: (0 if order.order_kind.endswith("reduce") or order.target_lots == 0 else 1, -abs(order.target_weight), order.secid))
        limited = []
        new_positions = 0
        for order in planned:
            is_new = order.current_lots == 0 and order.target_lots != 0
            if is_new and new_positions >= self.config.max_new_positions_per_rebalance:
                continue
            if len(limited) >= self.config.max_orders_per_rebalance:
                break
            if is_new:
                new_positions += 1
            limited.append(order)
        return limited

    def plan_exit_orders(
        self,
        *,
        close_secids: set[str],
        target_scores: Mapping[str, float],
        positions: Mapping[str, int],
        prices: Mapping[str, float],
        buy_prices: Mapping[str, float] | None = None,
        sell_prices: Mapping[str, float] | None = None,
    ) -> list[PlannedOrder]:
        planned = []
        for secid in sorted(close_secids):
            current_lots = int(positions.get(secid, 0) or 0)
            if current_lots == 0:
                continue
            instrument = self.instruments.get(secid)
            if instrument is None or instrument.lot_size <= 0:
                continue
            price = float(prices.get(secid, 0.0) or 0.0)
            if price <= 0:
                continue
            direction = "S" if current_lots > 0 else "B"
            execution_price = _execution_price(secid, direction, prices, buy_prices, sell_prices)
            order_value = abs(current_lots) * instrument.lot_size * execution_price
            if order_value < self.config.min_order_value_rub:
                continue
            planned.append(
                PlannedOrder(
                    secid=secid,
                    direction=direction,
                    quantity=abs(current_lots),
                    current_lots=current_lots,
                    target_lots=0,
                    price=execution_price,
                    lot_size=instrument.lot_size,
                    order_value=order_value,
                    target_weight=0.0,
                    score=float(target_scores.get(secid, 0.0) or 0.0),
                    order_kind="close_reduce",
                    reason="exit_pass",
                )
            )
        return planned[: max(int(self.config.max_orders_per_rebalance), 0)]

    def plan_entry_orders(
        self,
        *,
        target_weights: Mapping[str, float],
        target_scores: Mapping[str, float],
        positions: Mapping[str, int],
        prices: Mapping[str, float],
        cash: float,
        buy_prices: Mapping[str, float] | None = None,
        sell_prices: Mapping[str, float] | None = None,
    ) -> list[PlannedOrder]:
        equity = self.estimate_equity(positions, prices, cash)
        planned = []
        for secid in sorted(target_weights):
            instrument = self.instruments.get(secid)
            if instrument is None or instrument.lot_size <= 0:
                continue
            price = float(prices.get(secid, 0.0) or 0.0)
            if price <= 0:
                continue
            target_weight = float(target_weights.get(secid, 0.0) or 0.0)
            current_lots = int(positions.get(secid, 0) or 0)
            target_lots = _target_lots(equity, target_weight, price, instrument.lot_size)
            delta = target_lots - current_lots
            if delta == 0 or not _is_incremental_entry(current_lots, target_lots):
                continue
            direction = "B" if delta > 0 else "S"
            execution_price = _execution_price(secid, direction, prices, buy_prices, sell_prices)
            order_value = abs(delta) * instrument.lot_size * execution_price
            if order_value < self.config.min_order_value_rub:
                continue
            if equity > 0 and order_value / equity < self.config.min_position_change_weight:
                continue
            planned.append(
                PlannedOrder(
                    secid=secid,
                    direction=direction,
                    quantity=abs(delta),
                    current_lots=current_lots,
                    target_lots=target_lots,
                    price=execution_price,
                    lot_size=instrument.lot_size,
                    order_value=order_value,
                    target_weight=target_weight,
                    score=float(target_scores.get(secid, abs(target_weight)) or 0.0),
                    order_kind=_order_kind(current_lots, target_lots),
                    reason="entry_pass",
                )
            )

        planned.sort(key=lambda order: (-abs(order.target_weight), order.secid))
        return planned[: max(int(self.config.max_orders_per_rebalance), 0)]

    def execute_orders(self, *, decision_id: str, as_of: datetime, orders: list[PlannedOrder]) -> tuple[OrderResult, ...]:
        results = []
        as_of_s = as_of.isoformat(timespec="seconds")
        for order in orders:
            request = {
                "direction": order.direction,
                "secid": order.secid,
                "quantity": order.quantity,
                "bot": self.bot_name,
                "order_kind": order.order_kind,
                "reason": order.reason,
                "current_lots": order.current_lots,
                "target_lots": order.target_lots,
                "target_weight": order.target_weight,
                "score": order.score,
                "price": order.price,
                "lot_size": order.lot_size,
                "order_value": order.order_value,
            }
            if self.live_orders:
                if self.client is None:
                    result = OrderResult(order.secid, order.direction, order.quantity, "blocked", request, {}, "client_missing")
                else:
                    response = self.client.submit_order(**request)
                    status = "submitted" if response.ok else "failed"
                    result = OrderResult(order.secid, order.direction, order.quantity, status, request, response.payload, response.error)
            else:
                result = OrderResult(order.secid, order.direction, order.quantity, "dry_run", request, {"paper": True}, "")
                self._apply_paper_order(order, as_of_s)
            self.state.insert_order(
                decision_id=decision_id,
                as_of=as_of_s,
                secid=result.secid,
                direction=result.direction,
                quantity=result.quantity,
                status=result.status,
                request=result.request,
                response=result.response,
                error=result.error,
            )
            results.append(result)
        return tuple(results)

    def estimate_equity(self, positions: Mapping[str, int], prices: Mapping[str, float], cash: float) -> float:
        net = 0.0
        for secid, lots in positions.items():
            instrument = self.instruments.get(secid)
            if instrument is None:
                continue
            net += int(lots) * instrument.lot_size * float(prices.get(secid, 0.0) or 0.0)
        return max(float(cash) + net, 1.0)

    def gross_value(self, positions: Mapping[str, int], prices: Mapping[str, float]) -> float:
        gross = 0.0
        for secid, lots in positions.items():
            instrument = self.instruments.get(secid)
            if instrument is None:
                continue
            gross += abs(int(lots)) * instrument.lot_size * float(prices.get(secid, 0.0) or 0.0)
        return gross

    def _estimate_equity(self, positions: Mapping[str, int], prices: Mapping[str, float], cash: float) -> float:
        return self.estimate_equity(positions, prices, cash)

    def _apply_paper_order(self, order: PlannedOrder, as_of_s: str) -> None:
        self.state.upsert_paper_position(order.secid, order.target_lots, order.target_weight, as_of_s)


def _target_lots(equity: float, weight: float, price: float, lot_size: int) -> int:
    if price <= 0 or lot_size <= 0:
        return 0
    raw_lots = equity * weight / (price * lot_size)
    return int(round(raw_lots))


def _order_kind(current_lots: int, target_lots: int) -> str:
    if current_lots == 0 and target_lots != 0:
        return "open"
    if current_lots != 0 and target_lots == 0:
        return "close_reduce"
    if current_lots > 0 and target_lots < current_lots:
        return "sell_reduce"
    if current_lots < 0 and target_lots > current_lots:
        return "buy_reduce"
    return "increase"


def _is_incremental_entry(current_lots: int, target_lots: int) -> bool:
    if current_lots == 0:
        return target_lots != 0
    if current_lots > 0:
        return target_lots > current_lots
    return target_lots < current_lots


def _execution_price(
    secid: str,
    direction: str,
    prices: Mapping[str, float],
    buy_prices: Mapping[str, float] | None,
    sell_prices: Mapping[str, float] | None,
) -> float:
    if direction == "B" and buy_prices is not None:
        return float(buy_prices.get(secid, prices.get(secid, 0.0)) or 0.0)
    if direction == "S" and sell_prices is not None:
        return float(sell_prices.get(secid, prices.get(secid, 0.0)) or 0.0)
    return float(prices.get(secid, 0.0) or 0.0)
