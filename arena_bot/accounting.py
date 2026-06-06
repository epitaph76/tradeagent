from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, Mapping

from .types import AccountState, Instrument, PlannedOrder, RiskConfig


def mark_account(
    *,
    cash: float,
    positions: Mapping[str, int],
    prices: Mapping[str, float],
    instruments: Mapping[str, Instrument],
    risk: RiskConfig,
    max_gross: float,
) -> AccountState:
    net = 0.0
    gross = 0.0
    margin_used = 0.0
    for secid, lots_raw in positions.items():
        lots = int(lots_raw or 0)
        if lots == 0:
            continue
        instrument = instruments.get(secid)
        if instrument is None or instrument.lot_size <= 0:
            continue
        price = float(prices.get(secid, 0.0) or 0.0)
        if price <= 0:
            continue
        value = lots * instrument.lot_size * price
        abs_value = abs(value)
        net += value
        gross += abs_value
        margin_used += abs_value * _margin_rate(instrument, lots, risk)

    equity = float(cash) + net
    cash_buffer = max(equity, 0.0) * max(float(risk.cash_buffer_pct), 0.0)
    gross_limit = max(float(max_gross), 0.0) * max(equity, 0.0)
    return AccountState(
        cash=float(cash),
        equity=equity,
        gross=gross,
        net=net,
        margin_used=margin_used,
        available_cash=max(float(cash) - cash_buffer, 0.0),
        available_gross=max(gross_limit - max(gross, margin_used), 0.0),
        cash_buffer=cash_buffer,
    )


def account_to_payload(account: AccountState) -> dict[str, float]:
    return {key: float(value) for key, value in asdict(account).items()}


def apply_order_cash(cash: float, order: PlannedOrder, risk: RiskConfig) -> float:
    notional = float(order.order_value)
    commission = notional * float(risk.commission_rate)
    if order.direction == "B":
        return float(cash) - notional - commission
    return float(cash) + notional - commission


def positions_after_order(positions: Mapping[str, int], order: PlannedOrder) -> dict[str, int]:
    out = {secid: int(lots or 0) for secid, lots in positions.items() if int(lots or 0) != 0}
    delta = int(order.quantity) if order.direction == "B" else -int(order.quantity)
    next_lots = int(out.get(order.secid, 0)) + delta
    if next_lots == 0:
        out.pop(order.secid, None)
    else:
        out[order.secid] = next_lots
    return out


def project_orders(
    *,
    cash: float,
    positions: Mapping[str, int],
    orders: list[PlannedOrder] | tuple[PlannedOrder, ...],
    prices: Mapping[str, float],
    instruments: Mapping[str, Instrument],
    risk: RiskConfig,
    max_gross: float,
) -> AccountState:
    projected_cash = float(cash)
    projected_positions = {secid: int(lots or 0) for secid, lots in positions.items() if int(lots or 0) != 0}
    for order in orders:
        projected_cash = apply_order_cash(projected_cash, order, risk)
        projected_positions = positions_after_order(projected_positions, order)
    return mark_account(
        cash=projected_cash,
        positions=projected_positions,
        prices=prices,
        instruments=instruments,
        risk=risk,
        max_gross=max_gross,
    )


def filter_orders_no_leverage(
    *,
    cash: float,
    positions: Mapping[str, int],
    orders: list[PlannedOrder],
    prices: Mapping[str, float],
    instruments: Mapping[str, Instrument],
    risk: RiskConfig,
    max_gross: float,
) -> tuple[list[PlannedOrder], list[dict[str, Any]], AccountState]:
    accepted: list[PlannedOrder] = []
    blocked: list[dict[str, Any]] = []
    projected_cash = float(cash)
    projected_positions = {secid: int(lots or 0) for secid, lots in positions.items() if int(lots or 0) != 0}

    for order in orders:
        next_cash = apply_order_cash(projected_cash, order, risk)
        next_positions = positions_after_order(projected_positions, order)
        account = mark_account(
            cash=next_cash,
            positions=next_positions,
            prices=prices,
            instruments=instruments,
            risk=risk,
            max_gross=max_gross,
        )
        reason = _risk_block_reason(
            order=order,
            before_positions=projected_positions,
            after=account,
            max_gross=max_gross,
            risk=risk,
        )
        if reason in _DOWNSIZABLE_RISK_REASONS and not _reduces_exposure(order, projected_positions):
            adjusted = _downsize_order_to_fit(
                order=order,
                cash=projected_cash,
                positions=projected_positions,
                prices=prices,
                instruments=instruments,
                risk=risk,
                max_gross=max_gross,
            )
            if adjusted is not None:
                order = adjusted
                next_cash = apply_order_cash(projected_cash, order, risk)
                next_positions = positions_after_order(projected_positions, order)
                account = mark_account(
                    cash=next_cash,
                    positions=next_positions,
                    prices=prices,
                    instruments=instruments,
                    risk=risk,
                    max_gross=max_gross,
                )
                reason = _risk_block_reason(
                    order=order,
                    before_positions=projected_positions,
                    after=account,
                    max_gross=max_gross,
                    risk=risk,
                )

        if reason:
            blocked.append(
                {
                    "secid": order.secid,
                    "direction": order.direction,
                    "quantity": int(order.quantity),
                    "order_kind": order.order_kind,
                    "reason": reason,
                    "order_value": float(order.order_value),
                    "projected_account": account_to_payload(account),
                }
            )
            continue
        accepted.append(order)
        projected_cash = next_cash
        projected_positions = next_positions

    final_account = mark_account(
        cash=projected_cash,
        positions=projected_positions,
        prices=prices,
        instruments=instruments,
        risk=risk,
        max_gross=max_gross,
    )
    return accepted, blocked, final_account


def _risk_block_reason(
    *,
    order: PlannedOrder,
    before_positions: Mapping[str, int],
    after: AccountState,
    max_gross: float,
    risk: RiskConfig,
) -> str:
    if _reduces_exposure(order, before_positions):
        return ""
    if after.equity <= 0:
        return "risk_equity_non_positive"
    gross_limit = max(float(max_gross), 0.0) * after.equity
    if after.gross > gross_limit + 1e-9:
        return "risk_gross_cap"
    if after.margin_used > gross_limit + 1e-9:
        return "risk_margin_cap"
    if order.direction == "B" and after.cash < after.cash_buffer - 1e-9:
        return "risk_cash_insufficient"
    if after.available_gross <= 0 and max(after.gross, after.margin_used) > gross_limit + 1e-9:
        return "risk_capacity_insufficient"
    return ""


_DOWNSIZABLE_RISK_REASONS = {
    "risk_cash_insufficient",
    "risk_gross_cap",
    "risk_margin_cap",
    "risk_capacity_insufficient",
}


def _downsize_order_to_fit(
    *,
    order: PlannedOrder,
    cash: float,
    positions: Mapping[str, int],
    prices: Mapping[str, float],
    instruments: Mapping[str, Instrument],
    risk: RiskConfig,
    max_gross: float,
) -> PlannedOrder | None:
    instrument = instruments.get(order.secid)
    if instrument is None or instrument.lot_size <= 0 or order.quantity <= 1:
        return None
    mark_price = float(prices.get(order.secid, 0.0) or 0.0)
    if mark_price <= 0 or order.price <= 0:
        return None

    before = mark_account(
        cash=cash,
        positions=positions,
        prices=prices,
        instruments=instruments,
        risk=risk,
        max_gross=max_gross,
    )
    best: PlannedOrder | None = None
    low = 1
    high = int(order.quantity) - 1
    while low <= high:
        quantity = (low + high) // 2
        candidate = _resize_order(order, quantity=quantity, mark_price=mark_price, equity=before.equity)
        if candidate.order_value < float(risk.min_order_value_rub):
            low = quantity + 1
            continue
        next_cash = apply_order_cash(cash, candidate, risk)
        next_positions = positions_after_order(positions, candidate)
        account = mark_account(
            cash=next_cash,
            positions=next_positions,
            prices=prices,
            instruments=instruments,
            risk=risk,
            max_gross=max_gross,
        )
        reason = _risk_block_reason(
            order=candidate,
            before_positions=positions,
            after=account,
            max_gross=max_gross,
            risk=risk,
        )
        if reason:
            high = quantity - 1
        else:
            best = candidate
            low = quantity + 1
    return best


def _resize_order(order: PlannedOrder, *, quantity: int, mark_price: float, equity: float) -> PlannedOrder:
    quantity = max(int(quantity), 0)
    delta = quantity if order.direction == "B" else -quantity
    target_lots = int(order.current_lots) + delta
    target_weight = 0.0
    if equity > 0:
        target_weight = target_lots * int(order.lot_size) * float(mark_price) / float(equity)
    return replace(
        order,
        quantity=quantity,
        target_lots=target_lots,
        order_value=quantity * int(order.lot_size) * float(order.price),
        target_weight=target_weight,
        order_kind=_order_kind_for_lots(int(order.current_lots), target_lots),
    )


def _order_kind_for_lots(current_lots: int, target_lots: int) -> str:
    if current_lots == 0 and target_lots != 0:
        return "open"
    if current_lots != 0 and target_lots == 0:
        return "close_reduce"
    if current_lots > 0 and target_lots < current_lots:
        return "sell_reduce"
    if current_lots < 0 and target_lots > current_lots:
        return "buy_reduce"
    return "increase"


def _reduces_exposure(order: PlannedOrder, positions: Mapping[str, int]) -> bool:
    current = int(positions.get(order.secid, 0) or 0)
    if current == 0:
        return False
    delta = int(order.quantity) if order.direction == "B" else -int(order.quantity)
    target = current + delta
    return abs(target) < abs(current) or target == 0


def _margin_rate(instrument: Instrument, lots: int, risk: RiskConfig) -> float:
    if instrument.asset_class == "future":
        return max(float(risk.future_margin_rate), 0.0)
    if int(lots) < 0:
        return max(float(risk.short_margin_rate), 0.0)
    return max(float(risk.long_margin_rate), 0.0)
