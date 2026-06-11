from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from arena_bot.market_data import SavedCandleMarketDataProvider
from arena_bot.runtime_backtest import _runtime_decision_times, _runtime_replay_timestamps
from arena_bot.storage import StateStore
from arena_bot.types import Instrument, TradingSessionConfig


def test_runtime_backtest_replay_timestamps_uses_union(tmp_path: Path):
    candle_dir = tmp_path / "candles"
    candle_dir.mkdir()
    pd.DataFrame(
        {
            "timestamps": ["2026-06-03 10:00:00", "2026-06-03 11:00:00"],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [1000.0, 1000.0],
            "amount": [100000.0, 101000.0],
        }
    ).to_csv(candle_dir / "candles_SBER.csv", index=False)
    pd.DataFrame(
        {
            "timestamps": ["2026-06-03 11:00:00", "2026-06-03 12:00:00"],
            "open": [200.0, 201.0],
            "high": [201.0, 202.0],
            "low": [199.0, 200.0],
            "close": [200.0, 201.0],
            "volume": [1.0, 1.0],
            "amount": [200.0, 201.0],
        }
    ).to_csv(candle_dir / "candles_BTCUSDT.csv", index=False)

    timestamps = _runtime_replay_timestamps(
        SavedCandleMarketDataProvider(directories=[candle_dir]),
        [Instrument("SBER"), Instrument("BTCUSDT", "crypto")],
        from_dt=datetime(2026, 6, 3, 10, 0),
        till_dt=datetime(2026, 6, 3, 12, 0),
    )

    assert timestamps == [
        datetime(2026, 6, 3, 10, 0),
        datetime(2026, 6, 3, 11, 0),
        datetime(2026, 6, 3, 12, 0),
    ]


def test_runtime_decision_times_adds_force_flat_for_each_moex_session(tmp_path: Path):
    session = TradingSessionConfig(
        enabled=True,
        session_templates={
            "moex_stock": {
                "timezone": "Europe/Moscow",
                "sessions": [
                    {"type": "main", "open": "09:50", "close": "18:50"},
                    {"type": "evening", "open": "19:00", "close": "23:49:59"},
                ],
            }
        },
    )
    config = SimpleNamespace(trading_session=session, instruments=(Instrument("SBER"),))
    timestamps = [datetime(2026, 6, 3, hour, 0) for hour in (10, 11, 12, 20, 21)]

    decision_times = _runtime_decision_times(
        timestamps,
        config=config,
        state=StateStore(tmp_path / "state.sqlite3"),
        from_dt=datetime(2026, 6, 3, 0, 0),
        till_dt=datetime(2026, 6, 3, 23, 59),
    )

    assert datetime(2026, 6, 3, 18, 40) in decision_times
    assert datetime(2026, 6, 3, 23, 39, 59) in decision_times
