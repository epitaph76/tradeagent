from __future__ import annotations

from datetime import datetime

import pandas as pd

from arena_bot.cli import _static_market_data_from_config
from arena_bot.market_data import AlgoPackRealtimeMarketDataProvider, BinanceRealtimeMarketDataProvider, SavedCandleMarketDataProvider, StaticMarketDataProvider
from arena_bot.types import Instrument, MarketSnapshot


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, *, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {}), "timeout": timeout})
        if "orderbook" in url:
            return FakeResponse(
                {
                    "orderbook": {
                        "columns": ["BUYSELL", "PRICE", "QUANTITY"],
                        "data": [
                            ["B", 100.10, 10],
                            ["B", 100.20, 5],
                            ["S", 100.50, 7],
                            ["S", 100.40, 3],
                        ],
                    }
                }
            )
        return FakeResponse(
            {
                "marketdata": {
                    "columns": ["SECID", "BID", "OFFER", "LAST", "VALTODAY"],
                    "data": [["SBER", 99.0, 101.0, 100.30, 123456.0]],
                }
            }
        )


class FakeBinanceSession:
    def __init__(self):
        self.calls = []

    def get(self, url, *, params=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}), "timeout": timeout})
        if "bookTicker" in url:
            return FakeResponse(
                {
                    "symbol": "BTCUSDT",
                    "bidPrice": "50000.10",
                    "bidQty": "1.2",
                    "askPrice": "50000.30",
                    "askQty": "0.8",
                }
            )
        return FakeResponse(
            {
                "symbol": "BTCUSDT",
                "lastPrice": "50000.20",
                "quoteVolume": "123456789.0",
            }
        )


def test_algopack_realtime_provider_uses_orderbook_top_for_moex_snapshot():
    fallback = StaticMarketDataProvider(
        snapshot_rows={
            "SBER": MarketSnapshot("SBER", last_price=100.0, bid=99.0, ask=101.0, source="fallback"),
            "BTCUSDT": MarketSnapshot("BTCUSDT", last_price=50000.0, bid=49990.0, ask=50010.0, source="fallback"),
        }
    )
    session = FakeSession()
    provider = AlgoPackRealtimeMarketDataProvider(fallback=fallback, token="secret", session=session)

    snapshots = provider.snapshots(datetime(2026, 6, 12, 12, 0), [Instrument("SBER"), Instrument("BTCUSDT", asset_class="crypto")])

    assert snapshots["SBER"].bid == 100.20
    assert snapshots["SBER"].ask == 100.40
    assert snapshots["SBER"].last_price == 100.30
    assert snapshots["SBER"].volume_value == 123456.0
    assert snapshots["SBER"].source == "algopack_orderbook"
    assert snapshots["BTCUSDT"].source == "fallback"
    assert len(session.calls) == 2
    assert all(call["headers"]["Authorization"] == "Bearer secret" for call in session.calls)


def test_algopack_realtime_provider_falls_back_to_marketdata_bid_offer_when_orderbook_empty():
    class EmptyBookSession(FakeSession):
        def get(self, url, *, params=None, headers=None, timeout=None):
            if "orderbook" in url:
                self.calls.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {}), "timeout": timeout})
                return FakeResponse({"orderbook": {"columns": ["BUYSELL", "PRICE"], "data": []}})
            return super().get(url, params=params, headers=headers, timeout=timeout)

    provider = AlgoPackRealtimeMarketDataProvider(
        fallback=StaticMarketDataProvider(snapshot_rows={"SBER": MarketSnapshot("SBER", last_price=100.0)}),
        token="Bearer secret",
        session=EmptyBookSession(),
    )

    snapshot = provider.snapshots(datetime(2026, 6, 12, 12, 0), [Instrument("SBER")])["SBER"]

    assert snapshot.bid == 99.0
    assert snapshot.ask == 101.0
    assert snapshot.source == "algopack_marketdata"


def test_binance_realtime_provider_uses_book_ticker_for_crypto_snapshot():
    fallback = StaticMarketDataProvider(
        snapshot_rows={
            "BTCUSDT": MarketSnapshot("BTCUSDT", last_price=49000.0, bid=48990.0, ask=49010.0, source="fallback"),
            "SBER": MarketSnapshot("SBER", last_price=100.0, bid=99.0, ask=101.0, source="fallback"),
        }
    )
    session = FakeBinanceSession()
    provider = BinanceRealtimeMarketDataProvider(fallback=fallback, session=session)

    snapshots = provider.snapshots(datetime(2026, 6, 12, 12, 0), [Instrument("BTCUSDT", asset_class="crypto"), Instrument("SBER")])

    assert snapshots["BTCUSDT"].bid == 50000.10
    assert snapshots["BTCUSDT"].ask == 50000.30
    assert snapshots["BTCUSDT"].last_price == 50000.20
    assert snapshots["BTCUSDT"].volume_value == 123456789.0
    assert snapshots["BTCUSDT"].source == "binance_book_ticker"
    assert snapshots["SBER"].source == "fallback"
    assert len(session.calls) == 2


def test_binance_realtime_provider_falls_back_when_book_ticker_is_missing():
    class EmptyBinanceSession(FakeBinanceSession):
        def get(self, url, *, params=None, timeout=None):
            self.calls.append({"url": url, "params": dict(params or {}), "timeout": timeout})
            return FakeResponse({"symbol": "BTCUSDT", "bidPrice": "0", "askPrice": "0"})

    provider = BinanceRealtimeMarketDataProvider(
        fallback=StaticMarketDataProvider(snapshot_rows={"BTCUSDT": MarketSnapshot("BTCUSDT", last_price=49000.0, bid=48990.0, ask=49010.0, source="fallback")}),
        session=EmptyBinanceSession(),
    )

    snapshot = provider.snapshots(datetime(2026, 6, 12, 12, 0), [Instrument("BTCUSDT", asset_class="crypto")])["BTCUSDT"]

    assert snapshot.source == "fallback"


def test_static_market_data_builder_keeps_saved_candles_unwrapped_for_historical_path(tmp_path):
    candles_dir = tmp_path / "candles"
    candles_dir.mkdir()
    pd.DataFrame(
        {
            "timestamps": ["2026-06-12 12:00:00"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1.0],
            "amount": [100.0],
        }
    ).to_csv(candles_dir / "candles_SBER.csv", index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
market_data:
  saved_candles:
    directories:
      - {candles_dir.as_posix()}
""",
        encoding="utf-8",
    )

    provider = _static_market_data_from_config(str(config_path))

    assert isinstance(provider, SavedCandleMarketDataProvider)


def test_static_market_data_builder_wraps_realtime_overlay_when_enabled(tmp_path, monkeypatch):
    candles_dir = tmp_path / "candles"
    candles_dir.mkdir()
    pd.DataFrame(
        {
            "timestamps": ["2026-06-12 12:00:00"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1.0],
            "amount": [100.0],
        }
    ).to_csv(candles_dir / "candles_SBER.csv", index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
market_data:
  algopack:
    token_env: TEST_MOEX_TOKEN
  saved_candles:
    directories:
      - {candles_dir.as_posix()}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_MOEX_TOKEN", "secret")

    provider = _static_market_data_from_config(str(config_path), enable_realtime_overlays=True)

    assert isinstance(provider, BinanceRealtimeMarketDataProvider)
    assert isinstance(provider.fallback, AlgoPackRealtimeMarketDataProvider)


def test_static_market_data_builder_can_disable_binance_overlay(tmp_path, monkeypatch):
    candles_dir = tmp_path / "candles"
    candles_dir.mkdir()
    pd.DataFrame(
        {
            "timestamps": ["2026-06-12 12:00:00"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1.0],
            "amount": [100.0],
        }
    ).to_csv(candles_dir / "candles_SBER.csv", index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
market_data:
  algopack:
    token_env: TEST_MOEX_TOKEN
  binance:
    enabled: false
  saved_candles:
    directories:
      - {candles_dir.as_posix()}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_MOEX_TOKEN", "secret")

    provider = _static_market_data_from_config(str(config_path), enable_realtime_overlays=True)

    assert isinstance(provider, AlgoPackRealtimeMarketDataProvider)
