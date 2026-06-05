from __future__ import annotations

import pandas as pd

from arena_bot.config import load_config
from arena_bot.crypto_download import binance_klines_to_candles
from arena_bot.moex_download import resolve_futures_roots


def test_binance_klines_to_candles_uses_quote_amount_and_msk_offset():
    rows = [
        [
            1_777_593_600_000,
            "100.0",
            "105.0",
            "99.0",
            "103.0",
            "2.5",
            1_777_597_199_999,
            "257.5",
        ]
    ]

    df = binance_klines_to_candles(rows, timestamp_offset_hours=3)

    assert list(df.columns) == ["timestamps", "end", "open", "high", "low", "close", "volume", "amount"]
    assert df["timestamps"].iloc[0] == pd.Timestamp("2026-05-01 03:00:00")
    assert df["open"].iloc[0] == 100.0
    assert df["amount"].iloc[0] == 257.5


def test_resolve_futures_roots_chooses_front_contract_after_till(monkeypatch):
    securities = pd.DataFrame(
        [
            {"SECID": "BRM6", "ASSETCODE": "BR", "LASTTRADEDATE": "2026-05-01"},
            {"SECID": "BRN6", "ASSETCODE": "BR", "LASTTRADEDATE": "2026-07-01"},
            {"SECID": "SiM6", "ASSETCODE": "Si", "LASTTRADEDATE": "2026-06-18"},
            {"SECID": "SILVM6", "ASSETCODE": "SILV", "LASTTRADEDATE": "2026-06-18"},
            {"SECID": "GDM6", "ASSETCODE": "GOLD", "LASTTRADEDATE": "2026-06-19"},
            {"SECID": "MMM6", "ASSETCODE": "MXI", "LASTTRADEDATE": "2026-06-18"},
        ]
    )
    monkeypatch.setattr("arena_bot.moex_download.fetch_moex_futures_securities", lambda: securities)

    resolved = resolve_futures_roots(roots=["BR", "Si", "GD", "IMOEX"], till_date="2026-05-14")

    assert resolved == {"BR": "BRN6", "Si": "SiM6", "GD": "GDM6", "IMOEX": "MMM6"}


def test_universe_v1_config_loads_20_instruments():
    config = load_config("configs/universe_v1_may1_14.yaml")

    assert len(config.instruments) == 20
    assert [instrument.secid for instrument in config.instruments if instrument.asset_class == "crypto"] == [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "BNBUSDT",
    ]
    assert [instrument.secid for instrument in config.instruments if instrument.asset_class == "future"] == [
        "BRN6",
        "NGM6",
        "GDM6",
        "SiM6",
        "CRM6",
        "MMM6",
    ]
