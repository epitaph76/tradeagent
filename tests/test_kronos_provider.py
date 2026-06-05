from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from arena_bot.kronos_provider import KLINE_COLS, RealKronosSignalProvider, predicted_returns_to_bullish_scores
from arena_bot.types import Instrument, KronosConfig


def test_predicted_returns_convert_to_percentile_bullish_scores():
    scores = predicted_returns_to_bullish_scores({"SBER": 0.02, "LKOH": -0.01, "GAZP": 0.00})
    assert scores == {"LKOH": 0.0, "GAZP": 0.5, "SBER": 1.0}


def test_single_kronos_return_is_neutral():
    assert predicted_returns_to_bullish_scores({"SBER": 0.02}) == {"SBER": 0.5}


def test_kronos_multistep_forecast_uses_last_predicted_close(monkeypatch):
    class FakePredictor:
        def predict_samples(self, **kwargs):
            samples = np.zeros((2, 4, len(KLINE_COLS)))
            samples[:, 0, KLINE_COLS.index("close")] = 90.0
            samples[:, 3, KLINE_COLS.index("open")] = [100.0, 104.0]
            samples[:, 3, KLINE_COLS.index("high")] = [112.0, 132.0]
            samples[:, 3, KLINE_COLS.index("low")] = [98.0, 118.0]
            samples[:, 3, KLINE_COLS.index("close")] = [110.0, 130.0]
            samples[:, 3, KLINE_COLS.index("volume")] = [1000.0, 3000.0]
            samples[:, 3, KLINE_COLS.index("amount")] = [110000.0, 390000.0]
            return samples

    provider = RealKronosSignalProvider(config=KronosConfig(enabled=True, pred_len=4, sample_count=2))
    monkeypatch.setattr(provider, "_ensure_predictor", lambda: FakePredictor())
    candles = pd.DataFrame(
        {
            "timestamps": pd.date_range("2026-05-01 10:00:00", periods=5, freq="h"),
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.0] * 5,
            "volume": [1000.0] * 5,
            "amount": [100000.0] * 5,
        }
    )

    row = provider._forecast_one(datetime(2026, 5, 1, 14, 0), Instrument("SBER"), candles)

    assert row is not None
    assert row["pred_close"] == 120.0
    assert round(row["pred_return"], 8) == 0.2
    assert row["metadata"]["pred_ohlcv"] == {
        "open": 102.0,
        "high": 122.0,
        "low": 108.0,
        "close": 120.0,
        "volume": 2000.0,
        "amount": 250000.0,
    }
    assert row["metadata"]["pred_ohlcv"]["close"] == row["pred_close"]
    assert row["metadata"]["target_timestamp"] == "2026-05-01T18:00:00"


def test_kronos_forecast_paths_returns_json_safe_samples(monkeypatch):
    class FakePredictor:
        def predict_samples(self, **kwargs):
            samples = np.zeros((20, 8, len(KLINE_COLS)))
            for path_idx in range(20):
                for step_idx in range(8):
                    price = 100.0 + path_idx * 0.1 + step_idx
                    samples[path_idx, step_idx, KLINE_COLS.index("open")] = price
                    samples[path_idx, step_idx, KLINE_COLS.index("high")] = price + 1.0
                    samples[path_idx, step_idx, KLINE_COLS.index("low")] = price - 1.0
                    samples[path_idx, step_idx, KLINE_COLS.index("close")] = price + 0.5
                    samples[path_idx, step_idx, KLINE_COLS.index("volume")] = 1000.0 + path_idx
                    samples[path_idx, step_idx, KLINE_COLS.index("amount")] = (price + 0.5) * (1000.0 + path_idx)
            return samples

    provider = RealKronosSignalProvider(config=KronosConfig(enabled=True, pred_len=1, sample_count=2))
    monkeypatch.setattr(provider, "_ensure_predictor", lambda: FakePredictor())
    candles = pd.DataFrame(
        {
            "timestamps": pd.date_range("2026-05-01 10:00:00", periods=5, freq="h"),
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.0] * 5,
            "volume": [1000.0] * 5,
            "amount": [100000.0] * 5,
        }
    )

    paths = provider.forecast_paths(
        datetime(2026, 5, 1, 14, 0),
        [Instrument("SBER")],
        {"SBER": candles},
        pred_len=8,
        sample_count=20,
    )

    row = paths["SBER"]
    assert row["horizon"] == 8
    assert row["sample_count"] == 20
    assert len(row["timestamps"]) == 8
    assert len(row["paths"]) == 20
    assert row["paths"][0][0]["close"] == 100.5
    assert isinstance(row["paths"][0][0]["close"], float)


def test_kronos_forecast_paths_returns_empty_when_horizon_exceeds_limit(monkeypatch):
    class FakePredictor:
        def predict_samples(self, **kwargs):
            raise AssertionError("predict_samples should not be called after session horizon rejection")

    provider = RealKronosSignalProvider(config=KronosConfig(enabled=True, pred_len=1, sample_count=20))
    monkeypatch.setattr(provider, "_ensure_predictor", lambda: FakePredictor())
    candles = pd.DataFrame(
        {
            "timestamps": pd.date_range("2026-05-01 10:00:00", periods=5, freq="h"),
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.0] * 5,
            "volume": [1000.0] * 5,
            "amount": [100000.0] * 5,
        }
    )

    paths = provider.forecast_paths(
        datetime(2026, 5, 1, 14, 0),
        [Instrument("SBER")],
        {"SBER": candles},
        pred_len=8,
        sample_count=20,
        max_target_time=datetime(2026, 5, 1, 18, 30),
    )

    assert paths == {}
