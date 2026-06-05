from __future__ import annotations

from datetime import datetime
from pathlib import Path

from arena_bot.order_manager import OrderManager
from arena_bot.storage import StateStore
from arena_bot.types import Instrument, MarketMetrics, MarketSnapshot, RiskConfig
from arena_bot.universe import select_universe


def test_equities_reduce_to_top_ten_by_metrics():
    instruments = tuple(Instrument(secid=f"T{i:02d}") for i in range(20))
    snapshots = {instrument.secid: MarketSnapshot(instrument.secid, last_price=100.0, bid=99.9, ask=100.1) for instrument in instruments}
    metrics = {
        instrument.secid: MarketMetrics(
            instrument.secid,
            realized_volatility=i / 100.0,
            atr_pct=i / 200.0,
            volume_value=1_000_000 + i,
            spread_pct=0.001,
            candle_count=20,
        )
        for i, instrument in enumerate(instruments)
    }
    selected = select_universe(instruments, snapshots=snapshots, metrics=metrics, max_equities=10)
    assert len(selected.instruments) == 10
    assert selected.secids[0] == "T19"
    assert "T00" not in selected.secids


def test_futures_and_crypto_included_only_when_explicit_and_valid():
    instruments = (
        Instrument("SBER", "equity", 1),
        Instrument("FUT1", "future", 1),
        Instrument("BTC1", "crypto", 1),
        Instrument("BADFUT", "future", 1),
    )
    snapshots = {
        "SBER": MarketSnapshot("SBER", last_price=100),
        "FUT1": MarketSnapshot("FUT1", last_price=1000),
        "BTC1": MarketSnapshot("BTC1", last_price=50000),
        "BADFUT": MarketSnapshot("BADFUT", last_price=0),
    }
    metrics = {
        secid: MarketMetrics(secid, realized_volatility=0.01, atr_pct=0.01, volume_value=1_000_000, spread_pct=0.001, candle_count=10)
        for secid in snapshots
    }
    selected = select_universe(instruments, snapshots=snapshots, metrics=metrics, max_equities=10)
    assert "FUT1" in selected.secids
    assert "BTC1" in selected.secids
    assert "BADFUT" not in selected.secids
    assert selected.diagnostics["rejected"]["BADFUT"] == "missing_price"


def test_order_manager_enforces_order_limits(tmp_path: Path):
    instruments = {f"T{i}": Instrument(f"T{i}", lot_size=1) for i in range(6)}
    manager = OrderManager(
        config=RiskConfig(
            starting_cash=100000,
            min_order_value_rub=100,
            min_position_change_weight=0.0,
            max_orders_per_rebalance=3,
            max_new_positions_per_rebalance=2,
        ),
        instruments=instruments,
        state=StateStore(tmp_path / "state.sqlite3"),
        bot_name="bot",
        live_orders=False,
    )
    orders = manager.plan_orders(
        target_weights={f"T{i}": 0.1 for i in range(6)},
        target_scores={f"T{i}": 0.9 for i in range(6)},
        positions={},
        prices={f"T{i}": 100.0 for i in range(6)},
        cash=100000,
    )
    assert len(orders) == 2

