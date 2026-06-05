from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from arena_bot.historical import run_historical_batch
from arena_bot.market_data import SavedCandleMarketDataProvider
from arena_bot.signals import StaticSignalProvider
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


def test_historical_batch_creates_paired_lightgbm_rows(tmp_path: Path):
    candle_dir = tmp_path / "candles"
    candle_dir.mkdir()
    instruments = tuple(Instrument(secid) for secid in ("SBER", "LKOH", "GAZP"))
    for idx, instrument in enumerate(instruments):
        base = 100.0 + idx * 20
        closes = [base + i * (idx + 1) for i in range(55)]
        pd.DataFrame(
            {
                "timestamps": pd.date_range("2026-05-01 10:00:00", periods=55, freq="h"),
                "open": closes,
                "high": [value + 1 for value in closes],
                "low": [value - 1 for value in closes],
                "close": closes,
                "volume": [1000] * 55,
                "amount": [value * 1000 for value in closes],
            }
        ).to_csv(candle_dir / f"candles_{instrument.secid}.csv", index=False)

    config = RuntimeConfig(
        mode="paper",
        bot_name="paper",
        data_dir=str(tmp_path / "state"),
        instruments=instruments,
        base_selectors=(
            BaseSelectorConfig(
                name="selector_kronos_core",
                signal_weights={"kronos": 1.0},
                threshold=0.50,
                rank_power=2.0,
                max_positions=2,
                max_long_positions=2,
                max_short_positions=1,
                asset_filter=("equity",),
            ),
            BaseSelectorConfig(
                name="selector_kronos_conservative",
                signal_weights={"kronos": 1.0},
                threshold=0.75,
                rank_power=2.0,
                max_positions=1,
                max_long_positions=1,
                max_short_positions=1,
                asset_filter=("equity",),
            ),
        ),
        portfolio=PortfolioConfig(max_positions=2, min_abs_weight=0.01),
        risk=RiskConfig(starting_cash=100000, min_order_value_rub=100),
        rebalance=RebalanceConfig(),
        lightgbm=LightGBMConfig(min_train_intervals=48),
        kronos=KronosConfig(enabled=False),
        max_equities=3,
        trading_session=TradingSessionConfig(enabled=False),
    )
    state = StateStore(tmp_path / "state.sqlite3")
    result = run_historical_batch(
        config=config,
        market_data=SavedCandleMarketDataProvider(directories=[candle_dir], history_rows=20),
        state=state,
        kronos_provider=StaticSignalProvider("kronos", {"SBER": 1.0, "LKOH": 0.5, "GAZP": 0.0}),
        intervals=50,
        from_dt=datetime(2026, 5, 1, 10, 0),
        till_dt=datetime(2026, 5, 3, 16, 0),
    )

    rows = state.load_lightgbm_training_rows(limit=100)
    assert result.replayed_intervals == 50
    assert len(rows) == 50
    assert all(set(row["returns"]) == {"selector_kronos_core", "selector_kronos_conservative"} for row in rows)


def test_historical_batch_charges_turnover_not_hourly_roundtrip(tmp_path: Path):
    candle_dir = tmp_path / "candles"
    candle_dir.mkdir()
    pd.DataFrame(
        {
            "timestamps": pd.date_range("2026-05-01 10:00:00", periods=3, freq="h"),
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0],
            "volume": [1000, 1000, 1000],
            "amount": [100000, 100000, 100000],
        }
    ).to_csv(candle_dir / "candles_SBER.csv", index=False)

    config = RuntimeConfig(
        mode="paper",
        bot_name="paper",
        data_dir=str(tmp_path / "state"),
        instruments=(Instrument("SBER"),),
        base_selectors=(
            BaseSelectorConfig(
                name="selector_kronos_core",
                signal_weights={"kronos": 1.0},
                threshold=0.50,
                rank_power=2.0,
                max_positions=1,
                max_long_positions=1,
                max_short_positions=0,
                asset_filter=("equity",),
            ),
        ),
        portfolio=PortfolioConfig(max_positions=1, min_abs_weight=0.01),
        risk=RiskConfig(starting_cash=100000, min_order_value_rub=100, commission_rate=0.0005),
        rebalance=RebalanceConfig(),
        lightgbm=LightGBMConfig(min_train_intervals=48),
        kronos=KronosConfig(enabled=False),
        max_equities=1,
        trading_session=TradingSessionConfig(enabled=False),
    )
    state = StateStore(tmp_path / "state.sqlite3")
    run_historical_batch(
        config=config,
        market_data=SavedCandleMarketDataProvider(directories=[candle_dir], history_rows=3),
        state=state,
        kronos_provider=StaticSignalProvider("kronos", {"SBER": 1.0}),
        intervals=2,
        from_dt=datetime(2026, 5, 1, 10, 0),
        till_dt=datetime(2026, 5, 1, 12, 0),
    )

    returns = state.connect().execute(
        """
        SELECT as_of, return_value
        FROM selector_returns
        WHERE selector = 'selector_kronos_core'
        ORDER BY as_of
        """
    ).fetchall()

    assert [round(float(row["return_value"]), 7) for row in returns] == [-0.00075, 0.0]
