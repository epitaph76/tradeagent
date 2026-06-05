from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from arena_bot.market_data import SavedCandleMarketDataProvider
from arena_bot.runtime import RuntimeEngine
from arena_bot.signals import EmptyKronosSignalProvider
from arena_bot.storage import StateStore
from arena_bot.types import (
    BaseSelectorConfig,
    Instrument,
    KronosConfig,
    LightGBMConfig,
    PortfolioConfig,
    RebalanceConfig,
    RiskConfig,
    RuntimeConfig,
    TradingSessionConfig,
)


def test_saved_candle_provider_reads_history_before_as_of(tmp_path: Path):
    data_dir = tmp_path / "candles"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "timestamps": ["2026-05-01 10:00:00", "2026-05-01 11:00:00", "2026-05-01 12:00:00"],
            "open": [100.0, 102.0, 99.0],
            "high": [103.0, 104.0, 100.0],
            "low": [99.0, 101.0, 98.0],
            "close": [102.0, 103.0, 99.0],
            "volume": [1000, 1000, 1000],
            "amount": [102000, 103000, 99000],
        }
    ).to_csv(data_dir / "candles_SBER.csv", index=False)
    provider = SavedCandleMarketDataProvider(directories=[data_dir])
    snapshots = provider.snapshots(datetime(2026, 5, 1, 11, 30), [Instrument("SBER")])
    candles = provider.candles(datetime(2026, 5, 1, 11, 30), [Instrument("SBER")])
    assert snapshots["SBER"].last_price == 103.0
    assert len(candles["SBER"]) == 2


def test_saved_candle_provider_hardcodes_crypto_spread(tmp_path: Path):
    data_dir = tmp_path / "candles"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "timestamps": ["2026-05-01 10:00:00"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1.0],
            "amount": [100.0],
        }
    ).to_csv(data_dir / "candles_BTCUSDT.csv", index=False)
    provider = SavedCandleMarketDataProvider(directories=[data_dir])

    snapshot = provider.snapshots(datetime(2026, 5, 1, 10, 0), [Instrument("BTCUSDT", asset_class="crypto")])["BTCUSDT"]

    assert snapshot.bid == 99.99
    assert snapshot.ask == 100.01
    assert round(snapshot.spread_pct, 8) == 0.0002


def test_saved_candle_provider_uses_fallback_spread_for_equity_and_future(tmp_path: Path):
    data_dir = tmp_path / "candles"
    data_dir.mkdir()
    for secid in ("SBER", "BRN6"):
        pd.DataFrame(
            {
                "timestamps": ["2026-05-01 10:00:00"],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1.0],
                "amount": [100.0],
            }
        ).to_csv(data_dir / f"candles_{secid}.csv", index=False)
    provider = SavedCandleMarketDataProvider(directories=[data_dir])

    snapshots = provider.snapshots(
        datetime(2026, 5, 1, 10, 0),
        [Instrument("SBER", asset_class="equity"), Instrument("BRN6", asset_class="future")],
    )

    assert snapshots["SBER"].bid < 100.0 < snapshots["SBER"].ask
    assert round(snapshots["SBER"].spread_pct, 8) == 0.0005
    assert snapshots["BRN6"].bid < 100.0 < snapshots["BRN6"].ask
    assert round(snapshots["BRN6"].spread_pct, 8) == 0.0002


def test_runtime_can_run_paper_on_saved_candles(tmp_path: Path):
    data_dir = tmp_path / "candles"
    data_dir.mkdir()
    instruments = tuple(Instrument(secid) for secid in ("SBER", "LKOH", "GAZP"))
    for idx, instrument in enumerate(instruments):
        base = 100 + idx * 10
        pd.DataFrame(
            {
                "timestamps": pd.date_range("2026-05-01 10:00:00", periods=8, freq="h"),
                "open": [base + i for i in range(8)],
                "high": [base + i + 1 for i in range(8)],
                "low": [base + i - 1 for i in range(8)],
                "close": [base + i for i in range(8)],
                "volume": [1000] * 8,
                "amount": [(base + i) * 1000 for i in range(8)],
            }
        ).to_csv(data_dir / f"candles_{instrument.secid}.csv", index=False)
    config = RuntimeConfig(
        mode="paper",
        bot_name="paper",
        data_dir=str(tmp_path / "state"),
        instruments=instruments,
        base_selectors=(
            BaseSelectorConfig(
                name="selector_kronos_core",
                signal_weights={"kronos": 1.0},
                threshold=0.55,
                rank_power=2.0,
                max_positions=2,
                max_long_positions=2,
                max_short_positions=0,
                asset_filter=("equity",),
            ),
        ),
        portfolio=PortfolioConfig(max_positions=2, min_abs_weight=0.01),
        risk=RiskConfig(starting_cash=100000, min_order_value_rub=100, min_position_change_weight=0.0),
        rebalance=RebalanceConfig(),
        lightgbm=LightGBMConfig(),
        kronos=KronosConfig(enabled=False),
        max_equities=3,
        trading_session=TradingSessionConfig(enabled=False),
    )
    engine = RuntimeEngine(
        config=config,
        market_data=SavedCandleMarketDataProvider(directories=[data_dir]),
        kronos_provider=EmptyKronosSignalProvider(),
        state=StateStore(tmp_path / "state.sqlite3"),
    )
    result = engine.run_once(datetime(2026, 5, 1, 17, 0))
    assert result.target_weights
    assert result.orders
    assert all(order.status == "dry_run" for order in result.orders)
