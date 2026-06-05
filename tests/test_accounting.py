from __future__ import annotations

from arena_bot.accounting import filter_orders_no_leverage, mark_account
from arena_bot.order_manager import OrderManager
from arena_bot.storage import StateStore
from arena_bot.types import Instrument, PlannedOrder, RiskConfig


def test_estimate_equity_uses_signed_mark_to_market(tmp_path):
    instruments = {"LONG": Instrument("LONG"), "SHORT": Instrument("SHORT")}
    manager = OrderManager(
        config=RiskConfig(starting_cash=100000),
        instruments=instruments,
        state=StateStore(tmp_path / "state.sqlite3"),
        bot_name="bot",
    )

    equity = manager.estimate_equity(
        positions={"LONG": 10, "SHORT": -5},
        prices={"LONG": 100.0, "SHORT": 200.0},
        cash=100000.0,
    )

    assert equity == 100000.0
    assert manager.gross_value({"LONG": 10, "SHORT": -5}, {"LONG": 100.0, "SHORT": 200.0}) == 2000.0


def test_no_leverage_filter_blocks_unfunded_buy():
    instruments = {"SBER": Instrument("SBER")}
    risk = RiskConfig(starting_cash=100000, commission_rate=0.0005, cash_buffer_pct=0.02)
    order = PlannedOrder(
        secid="SBER",
        direction="B",
        quantity=1000,
        current_lots=0,
        target_lots=1000,
        price=100.1,
        lot_size=1,
        order_value=100100.0,
        target_weight=1.0,
        score=1.0,
        order_kind="open",
        reason="entry_pass",
    )

    accepted, blocked, account = filter_orders_no_leverage(
        cash=100000.0,
        positions={},
        orders=[order],
        prices={"SBER": 100.0},
        instruments=instruments,
        risk=risk,
        max_gross=1.0,
    )

    assert accepted == []
    assert blocked[0]["reason"] in {"risk_cash_insufficient", "risk_gross_cap"}
    assert account.cash == 100000.0


def test_short_uses_full_collateral_without_creating_extra_capacity():
    instruments = {"SBER": Instrument("SBER")}
    risk = RiskConfig(starting_cash=100000, commission_rate=0.0005, cash_buffer_pct=0.02)
    short = PlannedOrder(
        secid="SBER",
        direction="S",
        quantity=800,
        current_lots=0,
        target_lots=-800,
        price=99.9,
        lot_size=1,
        order_value=79920.0,
        target_weight=-0.8,
        score=1.0,
        order_kind="open",
        reason="entry_pass",
    )

    accepted, blocked, account = filter_orders_no_leverage(
        cash=100000.0,
        positions={},
        orders=[short],
        prices={"SBER": 100.0},
        instruments=instruments,
        risk=risk,
        max_gross=1.0,
    )

    assert len(accepted) == 1
    assert blocked == []
    assert account.cash > 100000.0
    assert account.equity < 100000.0
    assert account.margin_used == 80000.0
    assert account.available_gross < 21000.0


def test_mark_account_reports_gross_net_and_margin():
    instruments = {
        "EQ": Instrument("EQ", asset_class="equity"),
        "FUT": Instrument("FUT", asset_class="future"),
    }
    account = mark_account(
        cash=100000.0,
        positions={"EQ": -10, "FUT": 2},
        prices={"EQ": 100.0, "FUT": 1000.0},
        instruments=instruments,
        risk=RiskConfig(short_margin_rate=1.0, future_margin_rate=1.0),
        max_gross=1.0,
    )

    assert account.net == 1000.0
    assert account.gross == 3000.0
    assert account.margin_used == 3000.0
    assert account.equity == 101000.0
