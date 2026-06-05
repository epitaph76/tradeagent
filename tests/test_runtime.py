from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from arena_bot.logging import JsonlLogger
from arena_bot.market_data import StaticMarketDataProvider
from arena_bot.runtime import RuntimeEngine, _trading_session_state
from arena_bot.signals import StaticSignalProvider
from arena_bot.storage import StateStore
from arena_bot.types import (
    BaseSelectorConfig,
    Instrument,
    KronosConfig,
    LightGBMConfig,
    MarketMetrics,
    MarketSnapshot,
    PortfolioConfig,
    RebalanceConfig,
    RiskConfig,
    RuntimeConfig,
    SignalRow,
    TradeLifecycleConfig,
    TradeLifecycleEntryConfig,
    TradeLifecycleExitConfig,
    TradingSessionConfig,
)


class FakeArenaClient:
    def __init__(self):
        self.submitted = []

    def positions(self, portfolio):
        return type("R", (), {"ok": True, "payload": {"positions": []}})()

    def submit_order(self, **request):
        self.submitted.append(request)
        return type("R", (), {"ok": True, "payload": {"ok": True}, "error": ""})()


class ForecastSignalProvider:
    name = "kronos"

    def __init__(self, returns: dict[str, float]):
        self.returns = dict(returns)
        self.calls = 0

    def score(self, as_of, instruments, candles):
        self.calls += 1
        return [
            SignalRow(
                as_of=as_of,
                secid=instrument.secid,
                signal_name="kronos",
                bullish_score=0.75 if self.returns[instrument.secid] >= 0 else 0.25,
                confidence=1.0,
                metadata={"pred_return": self.returns[instrument.secid]},
            )
            for instrument in instruments
            if instrument.secid in self.returns
        ]


class ParticleForecastProvider:
    name = "kronos"

    def __init__(self, payloads: dict[str, dict] | None = None, returns: dict[str, float] | None = None):
        self.payloads = dict(payloads or {})
        self.returns = dict(returns or {})
        self.path_calls = 0
        self.path_call_args: list[dict] = []

    def forecast_paths(self, as_of, instruments, candles, *, pred_len=None, sample_count=None, max_target_time=None):
        self.path_calls += 1
        self.path_call_args.append({"pred_len": pred_len, "sample_count": sample_count, "max_target_time": max_target_time})
        out = {}
        for instrument in instruments:
            payload = self.payloads.get(instrument.secid)
            if payload:
                limit = int(pred_len or payload.get("horizon") or len(payload.get("timestamps") or []))
                paths = [list(path[:limit]) for path in list(payload.get("paths") or [])]
                timestamps = list(payload.get("timestamps") or [])[:limit]
                out[instrument.secid] = {
                    **payload,
                    "secid": instrument.secid,
                    "as_of": as_of.isoformat(timespec="seconds"),
                    "horizon": len(timestamps),
                    "timestamps": timestamps,
                    "paths": paths,
                }
        return out

    def score(self, as_of, instruments, candles):
        return [
            SignalRow(
                as_of=as_of,
                secid=instrument.secid,
                signal_name="kronos",
                bullish_score=0.75 if self.returns[instrument.secid] >= 0 else 0.25,
                confidence=1.0,
                metadata={"pred_return": self.returns[instrument.secid]},
            )
            for instrument in instruments
            if instrument.secid in self.returns
        ]


def _path_payload(secid: str, timestamps: list[datetime], closes_by_path: list[list[float]], *, last_close: float = 100.0) -> dict:
    paths = []
    for closes in closes_by_path:
        prev = last_close
        path = []
        for close in closes:
            high = max(prev, close) + 0.1
            low = min(prev, close) - 0.1
            path.append({"open": prev, "high": high, "low": low, "close": close, "volume": 1000.0, "amount": close * 1000.0})
            prev = close
        paths.append(path)
    return {
        "secid": secid,
        "last_close": last_close,
        "horizon": len(timestamps),
        "sample_count": len(paths),
        "timestamps": [value.isoformat(timespec="seconds") for value in timestamps],
        "paths": paths,
    }


def _candles(timestamps: list[datetime], closes: list[float], *, first_open: float = 100.0) -> pd.DataFrame:
    rows = []
    prev = first_open
    for ts, close in zip(timestamps, closes):
        rows.append(
            {
                "timestamps": ts,
                "open": prev,
                "high": max(prev, close) + 0.1,
                "low": min(prev, close) - 0.1,
                "close": close,
                "volume": 1000.0,
                "amount": close * 1000.0,
            }
        )
        prev = close
    return pd.DataFrame(rows)


def _session_config() -> TradingSessionConfig:
    return TradingSessionConfig(enabled=True)


def test_trading_session_state_boundaries():
    config = _session_config()
    assert _trading_session_state(datetime(2026, 6, 3, 9, 59), config)["session_state"] == "pre_session"
    assert _trading_session_state(datetime(2026, 6, 3, 10, 30), config)["session_state"] == "warmup"
    assert _trading_session_state(datetime(2026, 6, 3, 11, 0), config)["session_state"] == "trade"
    assert _trading_session_state(datetime(2026, 6, 3, 17, 45), config)["session_state"] == "no_new_entries"
    assert _trading_session_state(datetime(2026, 6, 3, 18, 31), config)["session_state"] == "force_flat"
    assert _trading_session_state(datetime(2026, 6, 3, 18, 41), config)["session_state"] == "closed"


def test_session_warmup_blocks_entries_and_kronos(tmp_path: Path):
    provider = ForecastSignalProvider({"SBER": 0.05})
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={"SBER": 0.95},
        entry_mode="kronos_rank",
        trading_session=_session_config(),
        kronos_provider=provider,
    )

    result = engine.run_once(datetime(2026, 6, 3, 10, 30))
    payload = _decision_payload(engine, result.decision_id)

    assert not result.orders
    assert provider.calls == 0
    assert payload["session_state"] == "warmup"
    assert payload["entry_diagnostics"]["reason"] == "session_warmup_no_entry"


def test_session_entry_cutoff_blocks_new_entries(tmp_path: Path):
    provider = ForecastSignalProvider({"SBER": 0.05})
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={"SBER": 0.95},
        entry_mode="kronos_rank",
        trading_session=_session_config(),
        kronos_provider=provider,
    )

    result = engine.run_once(datetime(2026, 6, 3, 17, 45))
    payload = _decision_payload(engine, result.decision_id)

    assert not result.orders
    assert provider.calls == 0
    assert payload["session_state"] == "no_new_entries"
    assert payload["entry_diagnostics"]["reason"] == "session_entry_cutoff"


def test_session_force_flat_closes_equity_and_crypto(tmp_path: Path):
    as_of = datetime(2026, 6, 3, 18, 31)
    instruments = (Instrument("SBER", "equity"), Instrument("BTCUSDT", "crypto"))
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={"SBER": 0.95, "BTCUSDT": 0.95},
        instruments=instruments,
        trading_session=_session_config(),
    )
    engine.state.upsert_paper_position("SBER", 10, 0.1, (as_of - timedelta(hours=2)).isoformat(timespec="seconds"))
    engine.state.upsert_paper_position("BTCUSDT", 2, 0.1, (as_of - timedelta(hours=2)).isoformat(timespec="seconds"))

    result = engine.run_once(as_of)
    payload = _decision_payload(engine, result.decision_id)

    assert {order.secid for order in result.orders} == {"SBER", "BTCUSDT"}
    assert all(order.request.get("reason") == "exit_pass" for order in result.orders)
    assert payload["session_state"] == "force_flat"
    assert payload["exit_diagnostics"]["held"]["SBER"]["action_reason"] == "session_force_flat"
    assert payload["exit_diagnostics"]["held"]["BTCUSDT"]["action_reason"] == "session_force_flat"
    assert engine.state.load_paper_positions() == {}


def test_session_closed_does_not_call_kronos(tmp_path: Path):
    provider = ForecastSignalProvider({"SBER": 0.05})
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={"SBER": 0.95},
        entry_mode="kronos_rank",
        trading_session=_session_config(),
        kronos_provider=provider,
    )

    result = engine.run_once(datetime(2026, 6, 3, 18, 41))
    payload = _decision_payload(engine, result.decision_id)

    assert not result.orders
    assert provider.calls == 0
    assert payload["session_state"] == "closed"
    assert payload["entry_diagnostics"]["reason"] == "session_closed"


def test_session_blocks_entry_when_kronos_horizon_exceeds_cutoff(tmp_path: Path):
    as_of = datetime(2026, 6, 3, 17, 0)
    provider = ForecastSignalProvider({"SBER": 0.05})
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={"SBER": 0.95},
        entry_mode="kronos_rank",
        trading_session=_session_config(),
        kronos_provider=provider,
        candles_by_secid={"SBER": _candles([as_of - timedelta(hours=1), as_of], [99.0, 100.0])},
    )

    result = engine.run_once(as_of)
    payload = _decision_payload(engine, result.decision_id)

    assert not result.orders
    assert provider.calls == 0
    assert payload["session_state"] == "trade"
    assert payload["entry_diagnostics"]["reason"] == "session_kronos_horizon_exceeds_close"


def test_session_clips_particle_tracker_horizon_to_force_flat(tmp_path: Path):
    as_of = datetime(2026, 6, 3, 16, 0)
    timestamps = [as_of + timedelta(hours=idx) for idx in range(1, 5)]
    provider = ParticleForecastProvider(
        {"SBER": _path_payload("SBER", timestamps, [[101.0, 102.0, 103.0, 104.0], [101.0, 102.0, 103.0, 104.0]])}
    )
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={"SBER": 0.95},
        exit_provider=provider,
        particle_exit_enabled=True,
        particle_horizon=4,
        particle_sample_count=2,
        trading_session=_session_config(),
        candles_by_secid={"SBER": _candles([as_of - timedelta(hours=1), as_of], [99.0, 100.0])},
    )
    engine.state.upsert_paper_position("SBER", 10, 0.1, (as_of - timedelta(hours=1)).isoformat(timespec="seconds"))

    result = engine.run_once(as_of)
    payload = _decision_payload(engine, result.decision_id)
    tracker = engine.state.load_kronos_exit_tracker("SBER")

    assert provider.path_calls == 1
    assert provider.path_call_args[0]["pred_len"] == 1
    assert tracker is not None
    assert tracker["horizon"] == 1
    assert payload["exit_diagnostics"]["held"]["SBER"]["effective_horizon"] == 1
    assert payload["exit_diagnostics"]["held"]["SBER"]["kronos_horizon"]["available_pred_len"] == 2
    assert payload["exit_diagnostics"]["held"]["SBER"]["kronos_horizon"]["clipped"] is True
    assert payload["exit_diagnostics"]["held"]["SBER"]["kronos_horizon"]["short_horizon_next_candle"] is True
    assert payload["exit_diagnostics"]["held"]["SBER"]["planned_exit_at"] == timestamps[0].isoformat(timespec="seconds")
    assert not [order for order in result.orders if order.secid == "SBER" and order.request.get("reason") == "exit_pass"]


def test_session_disabled_keeps_old_after_close_behavior(tmp_path: Path):
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={"SBER": 0.95},
        trading_session=TradingSessionConfig(enabled=False),
    )

    result = engine.run_once(datetime(2026, 6, 3, 18, 41))

    assert result.orders


def test_run_once_paper_logs_decision_and_never_submits(tmp_path: Path):
    client = FakeArenaClient()
    engine = _engine(tmp_path, mode="paper", client=client, scores={"SBER": 0.95, "LKOH": 0.05})
    result = engine.run_once(datetime(2026, 6, 3, 12, 0))
    assert result.orders
    assert all(order.status == "dry_run" for order in result.orders)
    assert client.submitted == []
    assert result.blend_diagnostics["final_target_positions_count"] <= 2
    assert result.selector_diagnostics["selector_kronos_core"]["selected_tickers_count"] <= 2
    assert result.selector_weights
    assert _decision_count(tmp_path) == 1


def test_live_mode_calls_submit_only_when_enabled(tmp_path: Path):
    client = FakeArenaClient()
    engine = _engine(tmp_path, mode="live", client=client, scores={"SBER": 0.95})
    result = engine.run_once(datetime(2026, 6, 3, 12, 0))
    assert result.orders
    assert client.submitted
    assert all(order.status == "submitted" for order in result.orders)


def test_news_is_disabled_in_runtime_payload(tmp_path: Path):
    engine = _engine(tmp_path, mode="paper", scores={"SBER": 0.95})
    result = engine.run_once(datetime(2026, 6, 3, 12, 0))
    row = engine.state.connect().execute("SELECT payload_json FROM decisions WHERE decision_id = ?", (result.decision_id,)).fetchone()
    assert '"news_enabled": false' in row["payload_json"]
    assert "llm" not in result.selector_diagnostics["selector_kronos_core"]["selector"]


def test_entry_pass_does_not_flip_held_position(tmp_path: Path):
    as_of = datetime(2026, 6, 3, 12, 0)
    engine = _engine(tmp_path, mode="paper", scores={"SBER": 0.20})
    engine.state.upsert_paper_position("SBER", 10, 0.1, (as_of - timedelta(minutes=20)).isoformat(timespec="seconds"))
    result = engine.run_once(as_of)
    assert result.target_weights.get("SBER", 0.0) >= 0.0
    row = engine.state.connect().execute("SELECT payload_json FROM decisions WHERE decision_id = ?", (result.decision_id,)).fetchone()
    assert "held_opposite_side_entry_skipped" in row["payload_json"]


def test_selector_returns_are_aligned_to_previous_decision_timestamp(tmp_path: Path):
    first = datetime(2026, 6, 3, 12, 0)
    second = datetime(2026, 6, 3, 13, 0)
    engine = _engine(tmp_path, mode="paper", scores={"SBER": 0.95})
    engine.run_once(first)
    engine.market_data.snapshot_rows = {
        "SBER": MarketSnapshot("SBER", last_price=110.0, bid=109.9, ask=110.1),
        "LKOH": MarketSnapshot("LKOH", last_price=100.0, bid=99.9, ask=100.1),
    }
    engine.run_once(second)
    rows = engine.state.connect().execute("SELECT DISTINCT as_of FROM selector_returns").fetchall()
    assert [row["as_of"] for row in rows] == [first.isoformat(timespec="seconds")]


def test_entry_pass_opens_at_most_five_total_positions(tmp_path: Path):
    instruments = tuple(Instrument(f"T{i:02d}") for i in range(7))
    scores = {instrument.secid: 0.99 - idx * 0.01 for idx, instrument in enumerate(instruments)}
    engine = _engine(tmp_path, mode="paper", scores=scores, instruments=instruments, selector_max_positions=7, portfolio_max_positions=10)

    result = engine.run_once(datetime(2026, 6, 3, 12, 0))
    positions = engine.state.load_paper_positions()

    assert len(positions) <= 5
    assert len([order for order in result.orders if order.status == "dry_run"]) <= 5
    assert '"status": "skipped_no_positions"' in engine.state.connect().execute(
        "SELECT payload_json FROM decisions WHERE decision_id = ?", (result.decision_id,)
    ).fetchone()["payload_json"]


def test_missing_exit_forecast_keeps_held_position(tmp_path: Path):
    as_of = datetime(2026, 6, 3, 12, 0)
    engine = _engine(tmp_path, mode="paper", scores={"SBER": 0.95}, exit_returns={})
    engine.state.upsert_paper_position("SBER", 10, 0.1, (as_of - timedelta(hours=2)).isoformat(timespec="seconds"))

    result = engine.run_once(as_of)

    assert "SBER" in engine.state.load_paper_positions()
    assert not [order for order in result.orders if order.secid == "SBER" and order.request.get("reason") == "exit_pass"]


def test_exit_pass_closes_weak_held_before_entry_opens_new_position(tmp_path: Path):
    as_of = datetime(2026, 6, 3, 12, 0)
    engine = _engine(tmp_path, mode="paper", scores={"LKOH": 0.95, "SBER": 0.95}, exit_returns={"SBER": -0.02})
    engine.state.upsert_paper_position("SBER", 10, 0.1, (as_of - timedelta(hours=2)).isoformat(timespec="seconds"))

    result = engine.run_once(as_of)
    positions = engine.state.load_paper_positions()

    assert any(order.secid == "SBER" and order.direction == "S" and order.request.get("reason") == "exit_pass" for order in result.orders)
    assert "SBER" not in positions
    assert "LKOH" in positions


def test_held_strong_asset_can_receive_entry_topup(tmp_path: Path):
    as_of = datetime(2026, 6, 3, 12, 0)
    engine = _engine(tmp_path, mode="paper", scores={"SBER": 0.99, "LKOH": 0.90}, exit_returns={"SBER": 0.02})
    engine.state.upsert_paper_position("SBER", 1, 0.001, (as_of - timedelta(hours=2)).isoformat(timespec="seconds"))

    result = engine.run_once(as_of)
    topups = [order for order in result.orders if order.secid == "SBER" and order.request.get("reason") == "entry_pass"]

    assert topups
    assert topups[0].request["target_lots"] > topups[0].request["current_lots"]


def test_kronos_rank_maps_return_sign_to_entry_side(tmp_path: Path):
    instruments = (Instrument("LONG"), Instrument("SHORT"))
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={},
        forecast_returns={"LONG": 0.02, "SHORT": -0.03},
        instruments=instruments,
        entry_mode="kronos_rank",
        portfolio_max_positions=5,
    )

    result = engine.run_once(datetime(2026, 6, 3, 12, 0))
    payload = _decision_payload(engine, result.decision_id)
    by_secid = {row["secid"]: row for row in payload["entry_ranked_candidates"]}

    assert by_secid["LONG"]["side"] == "long"
    assert by_secid["SHORT"]["side"] == "short"
    assert any(order.secid == "LONG" and order.direction == "B" for order in result.orders)
    assert any(order.secid == "SHORT" and order.direction == "S" for order in result.orders)


def test_kronos_rank_sorts_by_abs_edge_not_bullish_score(tmp_path: Path):
    instruments = (Instrument("SMALL_LONG"), Instrument("BIG_SHORT"), Instrument("MID_LONG"))
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={},
        forecast_returns={"SMALL_LONG": 0.002, "BIG_SHORT": -0.03, "MID_LONG": 0.01},
        instruments=instruments,
        entry_mode="kronos_rank",
        portfolio_max_positions=5,
    )

    result = engine.run_once(datetime(2026, 6, 3, 12, 0))
    payload = _decision_payload(engine, result.decision_id)
    ranked = [row for row in payload["entry_ranked_candidates"] if row["rank"] is not None]

    assert ranked[0]["secid"] == "BIG_SHORT"
    assert ranked[0]["side"] == "short"


def test_kronos_rank_filters_selected_candidate_when_round_trip_cost_exceeds_return(tmp_path: Path):
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={},
        forecast_returns={"SBER": 0.0005},
        entry_mode="kronos_rank",
        portfolio_max_positions=5,
    )

    result = engine.run_once(datetime(2026, 6, 3, 12, 0))
    payload = _decision_payload(engine, result.decision_id)
    by_secid = {row["secid"]: row for row in payload["entry_ranked_candidates"]}

    assert not result.orders
    assert by_secid["SBER"]["reason"] == "cost_exceeds_pred_return"
    assert by_secid["SBER"]["gross_pred_return"] == 0.0005
    assert by_secid["SBER"]["round_trip_cost"] > by_secid["SBER"]["gross_pred_return"]
    assert by_secid["SBER"]["net_edge"] < 0


def test_kronos_rank_cost_filter_does_not_backfill_after_top_n(tmp_path: Path):
    instruments = (Instrument("BAD"),) + tuple(Instrument(f"T{i:02d}") for i in range(5))
    returns = {
        "BAD": 0.03,
        "T00": 0.020,
        "T01": 0.019,
        "T02": 0.018,
        "T03": 0.017,
        "T04": 0.016,
    }
    spreads = {"BAD": 0.020, **{f"T{i:02d}": 0.001 for i in range(5)}}
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={},
        forecast_returns=returns,
        instruments=instruments,
        entry_mode="kronos_rank",
        portfolio_max_positions=10,
        spread_pct_by_secid=spreads,
    )

    result = engine.run_once(datetime(2026, 6, 3, 12, 0))
    payload = _decision_payload(engine, result.decision_id)
    by_secid = {row["secid"]: row for row in payload["entry_ranked_candidates"]}
    open_orders = [order for order in result.orders if order.request.get("order_kind") == "open"]

    assert by_secid["BAD"]["rank"] == 1
    assert by_secid["BAD"]["reason"] == "cost_exceeds_pred_return"
    assert by_secid["BAD"]["selected"] is False
    assert by_secid["T04"]["reason"] == "no_free_slot"
    assert by_secid["T04"]["selected"] is False
    assert len(open_orders) == 4


def test_kronos_rank_slippage_multiplier_tightens_cost_filter(tmp_path: Path):
    instruments = (Instrument("EDGE"),)
    loose = _engine(
        tmp_path / "loose",
        mode="paper",
        scores={},
        forecast_returns={"EDGE": 0.0035},
        instruments=instruments,
        entry_mode="kronos_rank",
        portfolio_max_positions=5,
        slippage_spread_multiplier=0.5,
    )
    strict = _engine(
        tmp_path / "strict",
        mode="paper",
        scores={},
        forecast_returns={"EDGE": 0.0035},
        instruments=instruments,
        entry_mode="kronos_rank",
        portfolio_max_positions=5,
        slippage_spread_multiplier=1.0,
    )

    loose_result = loose.run_once(datetime(2026, 6, 3, 12, 0))
    strict_result = strict.run_once(datetime(2026, 6, 3, 12, 0))
    loose_row = _decision_payload(loose, loose_result.decision_id)["entry_ranked_candidates"][0]
    strict_row = _decision_payload(strict, strict_result.decision_id)["entry_ranked_candidates"][0]

    assert loose_result.orders
    assert loose_row["selected"] is True
    assert loose_row["net_edge"] > 0
    assert not strict_result.orders
    assert strict_row["reason"] == "cost_exceeds_pred_return"
    assert strict_row["round_trip_cost"] > loose_row["round_trip_cost"]


def test_kronos_single_top_opens_only_rank_one_when_filters_pass(tmp_path: Path):
    instruments = (Instrument("TOP"), Instrument("SECOND"))
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={},
        forecast_returns={"TOP": 0.006, "SECOND": -0.005},
        instruments=instruments,
        entry_mode="kronos_single_top",
        exit_enabled=False,
        portfolio_max_positions=1,
    )

    result = engine.run_once(datetime(2026, 6, 3, 12, 0))
    payload = _decision_payload(engine, result.decision_id)
    by_secid = {row["secid"]: row for row in payload["entry_ranked_candidates"]}

    assert len(result.orders) == 1
    assert result.orders[0].secid == "TOP"
    assert result.orders[0].direction == "B"
    assert result.orders[0].request["reason"] == "single_top_rebalance"
    assert by_secid["TOP"]["selected"] is True
    assert by_secid["TOP"]["reason"] == "single_top_entry"
    assert by_secid["TOP"]["target_weight"] > 0.9
    assert by_secid["SECOND"]["selected"] is False
    assert by_secid["SECOND"]["reason"] == "not_top_1"
    assert payload["entry_diagnostics"]["ranking_mode"] == "kronos_single_top"
    assert payload["entry_diagnostics"]["target_action"] == "open"


def test_kronos_single_top_does_not_backfill_rank_two_when_top_fails(tmp_path: Path):
    instruments = (Instrument("OVERCONFIDENT"), Instrument("VALID"))
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={},
        forecast_returns={"OVERCONFIDENT": 0.009, "VALID": 0.006},
        instruments=instruments,
        entry_mode="kronos_single_top",
        exit_enabled=False,
        portfolio_max_positions=1,
    )

    result = engine.run_once(datetime(2026, 6, 3, 12, 0))
    payload = _decision_payload(engine, result.decision_id)
    by_secid = {row["secid"]: row for row in payload["entry_ranked_candidates"]}

    assert not result.orders
    assert by_secid["OVERCONFIDENT"]["rank"] == 1
    assert by_secid["OVERCONFIDENT"]["reason"] == "single_top_gross_pred_return_cap"
    assert by_secid["OVERCONFIDENT"]["selected"] is False
    assert by_secid["VALID"]["rank"] == 2
    assert by_secid["VALID"]["reason"] == "not_top_1"
    assert payload["entry_diagnostics"]["target_action"] == "skip_flat"
    assert payload["entry_diagnostics"]["selected_count"] == 0


def test_kronos_single_top_holds_same_instrument_and_side(tmp_path: Path):
    as_of = datetime(2026, 6, 3, 12, 0)
    instruments = (Instrument("TOP"), Instrument("SECOND"))
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={},
        forecast_returns={"TOP": 0.006, "SECOND": 0.005},
        instruments=instruments,
        entry_mode="kronos_single_top",
        exit_enabled=False,
        portfolio_max_positions=1,
    )
    engine.state.upsert_paper_position("TOP", 10, 0.01, (as_of - timedelta(hours=1)).isoformat(timespec="seconds"))

    result = engine.run_once(as_of)
    payload = _decision_payload(engine, result.decision_id)
    by_secid = {row["secid"]: row for row in payload["entry_ranked_candidates"]}

    assert not result.orders
    assert by_secid["TOP"]["reason"] == "same_top_hold"
    assert by_secid["TOP"]["same_as_current_position"] is True
    assert payload["entry_diagnostics"]["target_action"] == "hold_same"


def test_kronos_single_top_switches_when_top_changes(tmp_path: Path):
    as_of = datetime(2026, 6, 3, 12, 0)
    instruments = (Instrument("OLD"), Instrument("NEW"))
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={},
        forecast_returns={"OLD": 0.005, "NEW": 0.006},
        instruments=instruments,
        entry_mode="kronos_single_top",
        exit_enabled=False,
        portfolio_max_positions=1,
    )
    engine.state.upsert_paper_position("OLD", 10, 0.01, (as_of - timedelta(hours=1)).isoformat(timespec="seconds"))

    result = engine.run_once(as_of)
    payload = _decision_payload(engine, result.decision_id)
    by_order = {(order.secid, order.direction): order for order in result.orders}

    assert ("OLD", "S") in by_order
    assert ("NEW", "B") in by_order
    assert all(order.request["reason"] == "single_top_rebalance" for order in result.orders)
    assert payload["entry_diagnostics"]["target_action"] == "switch"


def test_kronos_single_top_closes_to_cash_when_top_fails_filters(tmp_path: Path):
    as_of = datetime(2026, 6, 3, 12, 0)
    instruments = (Instrument("OLD"), Instrument("OVERCONFIDENT"))
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={},
        forecast_returns={"OLD": 0.005, "OVERCONFIDENT": 0.009},
        instruments=instruments,
        entry_mode="kronos_single_top",
        exit_enabled=False,
        portfolio_max_positions=1,
    )
    engine.state.upsert_paper_position("OLD", 10, 0.01, (as_of - timedelta(hours=1)).isoformat(timespec="seconds"))

    result = engine.run_once(as_of)
    payload = _decision_payload(engine, result.decision_id)
    by_secid = {row["secid"]: row for row in payload["entry_ranked_candidates"]}

    assert len(result.orders) == 1
    assert result.orders[0].secid == "OLD"
    assert result.orders[0].direction == "S"
    assert result.orders[0].request["reason"] == "single_top_rebalance"
    assert by_secid["OVERCONFIDENT"]["reason"] == "single_top_gross_pred_return_cap"
    assert payload["entry_diagnostics"]["target_action"] == "close_to_cash"
    assert payload["entry_diagnostics"]["action_reason"] == "single_top_close_to_cash"


def test_kronos_single_top_gross_cap_blocks_overconfident_signal(tmp_path: Path):
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={},
        forecast_returns={"SBER": 0.008},
        entry_mode="kronos_single_top",
        exit_enabled=False,
        portfolio_max_positions=1,
    )

    result = engine.run_once(datetime(2026, 6, 3, 12, 0))
    payload = _decision_payload(engine, result.decision_id)
    row = payload["entry_ranked_candidates"][0]

    assert not result.orders
    assert row["reason"] == "single_top_gross_pred_return_cap"
    assert row["single_top_passed_filters"] is False
    assert payload["entry_diagnostics"]["top1_passed_filters"] is False


def test_kronos_single_top_keeps_ranked_candidates_when_no_order(tmp_path: Path):
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={},
        forecast_returns={"SBER": 0.003},
        entry_mode="kronos_single_top",
        exit_enabled=False,
        portfolio_max_positions=1,
    )

    result = engine.run_once(datetime(2026, 6, 3, 12, 0))
    payload = _decision_payload(engine, result.decision_id)
    row = payload["entry_ranked_candidates"][0]

    assert not result.orders
    assert row["secid"] == "SBER"
    assert row["selected"] is False
    assert row["reason"] == "single_top_net_edge_below_min"
    assert row["target_weight"] == 0.0
    assert payload["entry_diagnostics"]["ranked_candidates"]


def test_kronos_rank_with_zero_held_opens_at_most_five_positions(tmp_path: Path):
    instruments = tuple(Instrument(f"T{i:02d}") for i in range(7))
    returns = {instrument.secid: 0.03 - idx * 0.001 for idx, instrument in enumerate(instruments)}
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={},
        forecast_returns=returns,
        instruments=instruments,
        entry_mode="kronos_rank",
        portfolio_max_positions=10,
    )

    result = engine.run_once(datetime(2026, 6, 3, 12, 0))
    positions = engine.state.load_paper_positions()
    payload = _decision_payload(engine, result.decision_id)

    assert len(positions) <= 5
    assert len([order for order in result.orders if order.request.get("order_kind") == "open"]) <= 5
    assert payload["selector_model"]["mode"] == "bypassed_kronos_rank"
    assert payload["selector_weights"] == {"kronos_rank": 1.0}


def test_kronos_rank_unorderable_lot_does_not_consume_flat_slot(tmp_path: Path):
    instruments = (Instrument("TOO_BIG", lot_size=10000),) + tuple(Instrument(f"T{i:02d}") for i in range(5))
    returns = {"TOO_BIG": 0.05, **{f"T{i:02d}": 0.04 - i * 0.001 for i in range(5)}}
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={},
        forecast_returns=returns,
        instruments=instruments,
        entry_mode="kronos_rank",
        portfolio_max_positions=10,
    )

    result = engine.run_once(datetime(2026, 6, 3, 12, 0))
    payload = _decision_payload(engine, result.decision_id)
    by_secid = {row["secid"]: row for row in payload["entry_ranked_candidates"]}

    assert by_secid["TOO_BIG"]["reason"] == "unorderable_zero_delta"
    assert by_secid["TOO_BIG"]["selected"] is False
    assert len([order for order in result.orders if order.request.get("order_kind") == "open"]) == 5
    assert "TOO_BIG" not in engine.state.load_paper_positions()


def test_kronos_rank_with_three_held_opens_at_most_two_new_flats(tmp_path: Path):
    as_of = datetime(2026, 6, 3, 12, 0)
    instruments = tuple(Instrument(f"T{i:02d}") for i in range(7))
    returns = {instrument.secid: 0.03 - idx * 0.001 for idx, instrument in enumerate(instruments)}
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={},
        forecast_returns=returns,
        instruments=instruments,
        entry_mode="kronos_rank",
        portfolio_max_positions=10,
    )
    for secid in ("T00", "T01", "T02"):
        engine.state.upsert_paper_position(secid, 1, 0.001, (as_of - timedelta(hours=2)).isoformat(timespec="seconds"))

    result = engine.run_once(as_of)
    new_open_orders = [order for order in result.orders if order.request.get("order_kind") == "open"]

    assert len(new_open_orders) <= 2
    assert len(engine.state.load_paper_positions()) <= 5


def test_kronos_rank_held_same_side_can_topup_without_new_slot(tmp_path: Path):
    as_of = datetime(2026, 6, 3, 12, 0)
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={},
        forecast_returns={"SBER": 0.05, "LKOH": 0.04},
        entry_mode="kronos_rank",
        portfolio_max_positions=5,
    )
    engine.state.upsert_paper_position("SBER", 1, 0.001, (as_of - timedelta(hours=2)).isoformat(timespec="seconds"))

    result = engine.run_once(as_of)
    payload = _decision_payload(engine, result.decision_id)
    sber = {row["secid"]: row for row in payload["entry_ranked_candidates"]}["SBER"]
    topups = [order for order in result.orders if order.secid == "SBER" and order.request.get("reason") == "entry_pass"]

    assert sber["reason"] == "topup"
    assert sber["selected"] is True
    assert topups and topups[0].request["target_lots"] > topups[0].request["current_lots"]


def test_kronos_rank_held_opposite_side_is_skipped_not_flipped(tmp_path: Path):
    as_of = datetime(2026, 6, 3, 12, 0)
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={},
        forecast_returns={"SBER": -0.05, "LKOH": 0.04},
        entry_mode="kronos_rank",
        portfolio_max_positions=5,
    )
    engine.state.upsert_paper_position("SBER", 1, 0.001, (as_of - timedelta(hours=2)).isoformat(timespec="seconds"))

    result = engine.run_once(as_of)
    payload = _decision_payload(engine, result.decision_id)
    sber = {row["secid"]: row for row in payload["entry_ranked_candidates"]}["SBER"]

    assert sber["reason"] == "held_opposite_side_entry_skipped"
    assert not [order for order in result.orders if order.secid == "SBER" and order.request.get("reason") == "entry_pass"]
    assert engine.state.load_paper_positions()["SBER"]["lots"] == 1


def test_kronos_rank_closed_in_exit_pass_is_not_reopened_same_tick(tmp_path: Path):
    as_of = datetime(2026, 6, 3, 12, 0)
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={},
        forecast_returns={"SBER": 0.05, "LKOH": 0.04},
        exit_returns={"SBER": -0.02},
        entry_mode="kronos_rank",
        portfolio_max_positions=5,
    )
    engine.state.upsert_paper_position("SBER", 10, 0.1, (as_of - timedelta(hours=2)).isoformat(timespec="seconds"))

    result = engine.run_once(as_of)
    payload = _decision_payload(engine, result.decision_id)
    sber = {row["secid"]: row for row in payload["entry_ranked_candidates"]}["SBER"]

    assert any(order.secid == "SBER" and order.request.get("reason") == "exit_pass" for order in result.orders)
    assert sber["reason"] == "closed_in_exit_pass"
    assert not [order for order in result.orders if order.secid == "SBER" and order.request.get("reason") == "entry_pass"]


def test_runtime_persists_account_and_does_not_reuse_starting_cash_for_topups(tmp_path: Path):
    first = datetime(2026, 6, 3, 12, 0)
    second = datetime(2026, 6, 3, 13, 0)
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={},
        forecast_returns={"SBER": 0.05, "LKOH": 0.04},
        exit_returns={"SBER": 0.05, "LKOH": 0.05},
        entry_mode="kronos_rank",
        portfolio_max_positions=5,
    )

    first_result = engine.run_once(first)
    first_payload = _decision_payload(engine, first_result.decision_id)
    second_result = engine.run_once(second)
    second_payload = _decision_payload(engine, second_result.decision_id)

    assert first_payload["account_after_orders"]["cash"] < 100000
    assert second_payload["account_before"]["cash"] == first_payload["account_after_orders"]["cash"]
    assert second_payload["account_after_orders"]["gross"] <= second_payload["account_after_orders"]["equity"]
    assert second_payload["account_after_orders"]["cash"] >= second_payload["account_after_orders"]["cash_buffer"]


def test_runtime_reduces_existing_positions_when_mark_to_market_breaches_gross_cap(tmp_path: Path):
    as_of = datetime(2026, 6, 3, 12, 0)
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={},
        forecast_returns={"SBER": -0.05},
        exit_returns={"SBER": -0.05},
        entry_mode="kronos_rank",
        portfolio_max_positions=5,
    )
    engine.market_data.snapshot_rows = {
        "SBER": MarketSnapshot("SBER", last_price=125.0, bid=124.9, ask=125.1),
        "LKOH": MarketSnapshot("LKOH", last_price=100.0, bid=99.9, ask=100.1),
    }
    engine.state.upsert_paper_position("SBER", -800, -0.8, (as_of - timedelta(hours=2)).isoformat(timespec="seconds"))
    engine.state.save_account_state(
        {
            "cash": 180000.0,
            "equity": 80000.0,
            "gross": 100000.0,
            "net": -100000.0,
            "margin_used": 100000.0,
            "available_cash": 0.0,
            "available_gross": 0.0,
            "cash_buffer": 1600.0,
        },
        account_id="bot",
        as_of=(as_of - timedelta(hours=1)).isoformat(timespec="seconds"),
    )

    result = engine.run_once(as_of)
    payload = _decision_payload(engine, result.decision_id)
    risk_orders = [order for order in result.orders if order.request.get("reason") == "risk_cap_pass"]

    assert risk_orders
    assert risk_orders[0].secid == "SBER"
    assert risk_orders[0].direction == "B"
    assert payload["risk_cap_diagnostics"]["breached"] is True
    assert payload["account_after_risk"]["gross"] <= payload["account_after_risk"]["equity"]


def test_particle_exit_creates_tracker_and_planned_exit_for_held_position(tmp_path: Path):
    as_of = datetime(2026, 6, 3, 12, 0)
    timestamps = [as_of + timedelta(hours=idx) for idx in range(1, 4)]
    provider = ParticleForecastProvider(
        {"SBER": _path_payload("SBER", timestamps, [[101.0, 102.0, 103.0], [100.8, 101.5, 102.0]])}
    )
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={"SBER": 0.95},
        exit_provider=provider,
        particle_exit_enabled=True,
        particle_horizon=3,
        particle_sample_count=2,
        candles_by_secid={"SBER": _candles([as_of], [100.0])},
    )
    engine.state.upsert_paper_position("SBER", 10, 0.1, (as_of - timedelta(hours=1)).isoformat(timespec="seconds"))

    result = engine.run_once(as_of)
    payload = _decision_payload(engine, result.decision_id)
    tracker = engine.state.load_kronos_exit_tracker("SBER")

    assert tracker is not None
    assert tracker["planned_exit_at"] == timestamps[-1].isoformat(timespec="seconds")
    assert payload["exit_diagnostics"]["held"]["SBER"]["particle_enabled"] is True
    assert payload["exit_diagnostics"]["held"]["SBER"]["planned_exit_at"] == tracker["planned_exit_at"]
    assert not [order for order in result.orders if order.secid == "SBER" and order.request.get("reason") == "exit_pass"]


def test_particle_exit_updates_weights_toward_matching_path(tmp_path: Path):
    first = datetime(2026, 6, 3, 12, 0)
    second = datetime(2026, 6, 3, 13, 0)
    timestamps = [first + timedelta(hours=idx) for idx in range(1, 4)]
    provider = ParticleForecastProvider(
        {"SBER": _path_payload("SBER", timestamps, [[101.0, 102.0, 103.0], [100.2, 101.0, 101.5]])}
    )
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={"SBER": 0.95},
        exit_provider=provider,
        particle_exit_enabled=True,
        particle_horizon=3,
        particle_sample_count=2,
        candles_by_secid={"SBER": _candles([first], [100.0])},
    )
    engine.state.upsert_paper_position("SBER", 10, 0.1, (first - timedelta(hours=1)).isoformat(timespec="seconds"))
    engine.run_once(first)
    engine.market_data.candle_rows = {"SBER": _candles([first, second], [100.0, 101.0])}

    result = engine.run_once(second)
    tracker = engine.state.load_kronos_exit_tracker("SBER")
    payload = _decision_payload(engine, result.decision_id)
    weights = tracker["state"]["weights"]

    assert tracker["current_step"] == 1
    assert weights[0] > weights[1]
    assert payload["exit_diagnostics"]["held"]["SBER"]["updated_steps"] == 1


def test_particle_exit_closes_when_planned_exit_is_due(tmp_path: Path):
    as_of = datetime(2026, 6, 3, 13, 0)
    timestamps = [as_of + timedelta(hours=idx) for idx in range(1, 4)]
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={"SBER": 0.95},
        exit_provider=ParticleForecastProvider({}),
        particle_exit_enabled=True,
        particle_horizon=3,
        particle_sample_count=2,
    )
    engine.state.upsert_paper_position("SBER", 10, 0.1, (as_of - timedelta(hours=2)).isoformat(timespec="seconds"))
    engine.state.save_kronos_exit_tracker(
        secid="SBER",
        side="long",
        created_at=(as_of - timedelta(hours=2)).isoformat(timespec="seconds"),
        last_updated_at=(as_of - timedelta(hours=1)).isoformat(timespec="seconds"),
        horizon=3,
        sample_count=2,
        current_step=0,
        planned_exit_at=as_of.isoformat(timespec="seconds"),
        confidence=1.0,
        state={
            "paths": _path_payload("SBER", timestamps, [[101.0, 102.0, 103.0], [101.0, 102.0, 103.0]])["paths"],
            "timestamps": [value.isoformat(timespec="seconds") for value in timestamps],
            "weights": [0.5, 0.5],
            "extension_count": 0,
        },
    )

    result = engine.run_once(as_of)
    payload = _decision_payload(engine, result.decision_id)

    assert any(order.secid == "SBER" and order.request.get("reason") == "exit_pass" for order in result.orders)
    assert payload["exit_diagnostics"]["held"]["SBER"]["action_reason"] == "planned_exit_due"
    assert engine.state.load_kronos_exit_tracker("SBER") is None


def test_particle_exit_closes_when_expected_net_turns_negative(tmp_path: Path):
    as_of = datetime(2026, 6, 3, 12, 0)
    timestamps = [as_of + timedelta(hours=idx) for idx in range(1, 3)]
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={"SBER": 0.95},
        exit_provider=ParticleForecastProvider({}),
        particle_exit_enabled=True,
        particle_horizon=2,
        particle_sample_count=2,
    )
    engine.state.upsert_paper_position("SBER", 10, 0.1, (as_of - timedelta(hours=1)).isoformat(timespec="seconds"))
    engine.state.save_kronos_exit_tracker(
        secid="SBER",
        side="long",
        created_at=(as_of - timedelta(hours=1)).isoformat(timespec="seconds"),
        last_updated_at=(as_of - timedelta(hours=1)).isoformat(timespec="seconds"),
        horizon=2,
        sample_count=2,
        current_step=0,
        planned_exit_at=(as_of + timedelta(hours=2)).isoformat(timespec="seconds"),
        confidence=1.0,
        state={
            "paths": _path_payload("SBER", timestamps, [[99.0, 98.0], [99.5, 98.5]])["paths"],
            "timestamps": [value.isoformat(timespec="seconds") for value in timestamps],
            "weights": [0.5, 0.5],
            "extension_count": 0,
        },
    )

    result = engine.run_once(as_of)
    payload = _decision_payload(engine, result.decision_id)

    assert any(order.secid == "SBER" and order.request.get("reason") == "exit_pass" for order in result.orders)
    assert payload["exit_diagnostics"]["held"]["SBER"]["action_reason"] == "expected_net_non_positive"


def test_particle_exit_does_not_extend_after_extension_limit(tmp_path: Path):
    as_of = datetime(2026, 6, 3, 12, 0)
    old_plan = as_of + timedelta(hours=2)
    timestamps = [as_of + timedelta(hours=idx) for idx in range(1, 5)]
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={"SBER": 0.95},
        exit_provider=ParticleForecastProvider({}),
        particle_exit_enabled=True,
        particle_horizon=4,
        particle_sample_count=2,
    )
    engine.state.upsert_paper_position("SBER", 10, 0.1, (as_of - timedelta(hours=1)).isoformat(timespec="seconds"))
    engine.state.save_kronos_exit_tracker(
        secid="SBER",
        side="long",
        created_at=(as_of - timedelta(hours=1)).isoformat(timespec="seconds"),
        last_updated_at=(as_of - timedelta(hours=1)).isoformat(timespec="seconds"),
        horizon=4,
        sample_count=2,
        current_step=0,
        planned_exit_at=old_plan.isoformat(timespec="seconds"),
        confidence=1.0,
        state={
            "paths": _path_payload("SBER", timestamps, [[101.0, 102.0, 105.0, 106.0], [101.0, 102.0, 105.0, 106.0]])["paths"],
            "timestamps": [value.isoformat(timespec="seconds") for value in timestamps],
            "weights": [0.5, 0.5],
            "extension_count": 1,
            "planned_expected_net": 0.002,
        },
    )

    result = engine.run_once(as_of)
    payload = _decision_payload(engine, result.decision_id)
    tracker = engine.state.load_kronos_exit_tracker("SBER")

    assert tracker["planned_exit_at"] == old_plan.isoformat(timespec="seconds")
    assert payload["exit_diagnostics"]["held"]["SBER"]["action_reason"] == "planned_exit_kept"


def test_particle_exit_refresh_failure_falls_back_to_edge_exit(tmp_path: Path):
    as_of = datetime(2026, 6, 3, 12, 0)
    timestamps = [as_of + timedelta(hours=idx) for idx in range(1, 3)]
    provider = ParticleForecastProvider({}, returns={"SBER": -0.02})
    engine = _engine(
        tmp_path,
        mode="paper",
        scores={"SBER": 0.95},
        exit_provider=provider,
        particle_exit_enabled=True,
        particle_horizon=2,
        particle_sample_count=4,
        particle_ess_refresh_fraction=0.5,
    )
    engine.state.upsert_paper_position("SBER", 10, 0.1, (as_of - timedelta(hours=1)).isoformat(timespec="seconds"))
    engine.state.save_kronos_exit_tracker(
        secid="SBER",
        side="long",
        created_at=(as_of - timedelta(hours=1)).isoformat(timespec="seconds"),
        last_updated_at=(as_of - timedelta(hours=1)).isoformat(timespec="seconds"),
        horizon=2,
        sample_count=4,
        current_step=0,
        planned_exit_at=(as_of + timedelta(hours=1)).isoformat(timespec="seconds"),
        confidence=0.25,
        state={
            "paths": _path_payload("SBER", timestamps, [[101.0, 102.0], [101.0, 102.0], [101.0, 102.0], [101.0, 102.0]])["paths"],
            "timestamps": [value.isoformat(timespec="seconds") for value in timestamps],
            "weights": [1.0, 0.0, 0.0, 0.0],
            "extension_count": 0,
        },
    )

    result = engine.run_once(as_of)
    payload = _decision_payload(engine, result.decision_id)

    assert any(order.secid == "SBER" and order.request.get("reason") == "exit_pass" for order in result.orders)
    assert payload["exit_diagnostics"]["held"]["SBER"]["fallback"] == "edge_exit"
    assert payload["exit_diagnostics"]["held"]["SBER"]["refresh_reason"] == "ess_below_threshold"


def _engine(
    tmp_path: Path,
    *,
    mode: str,
    scores: dict[str, float],
    client=None,
    exit_returns: dict[str, float] | None = None,
    forecast_returns: dict[str, float] | None = None,
    instruments: tuple[Instrument, ...] | None = None,
    selector_max_positions: int = 2,
    portfolio_max_positions: int = 2,
    entry_mode: str = "selectors",
    exit_enabled: bool = True,
    slippage_spread_multiplier: float = 0.5,
    spread_pct_by_secid: dict[str, float] | None = None,
    exit_provider=None,
    particle_exit_enabled: bool = False,
    particle_horizon: int = 8,
    particle_sample_count: int = 20,
    particle_ess_refresh_fraction: float = 0.25,
    single_top_min_net_edge: float = 0.0015,
    single_top_max_gross_pred_return: float = 0.0075,
    single_top_target_weight: float = 1.0,
    candles_by_secid: dict[str, pd.DataFrame] | None = None,
    trading_session: TradingSessionConfig | None = None,
    kronos_provider=None,
) -> RuntimeEngine:
    instruments = instruments or (Instrument("SBER"), Instrument("LKOH"))
    config = RuntimeConfig(
        mode=mode,  # type: ignore[arg-type]
        bot_name="bot",
        data_dir=str(tmp_path),
        instruments=instruments,
        base_selectors=(
            BaseSelectorConfig(
                name="selector_kronos_core",
                signal_weights={"kronos": 1.0},
                threshold=0.65,
                rank_power=2.0,
                max_positions=selector_max_positions,
                max_long_positions=selector_max_positions,
                max_short_positions=1,
                asset_filter=("equity",),
            ),
        ),
        portfolio=PortfolioConfig(max_positions=portfolio_max_positions, min_abs_weight=0.01, min_hold_minutes=60, strong_flip_threshold=0.85),
        risk=RiskConfig(
            starting_cash=100000,
            min_order_value_rub=100,
            min_position_change_weight=0.0,
            slippage_spread_multiplier=slippage_spread_multiplier,
        ),
        rebalance=RebalanceConfig(decision_interval_minutes=60),
        lightgbm=LightGBMConfig(min_train_intervals=48),
        kronos=KronosConfig(enabled=False),
        max_equities=len(instruments),
        trading_session=trading_session or TradingSessionConfig(enabled=False),
        trade_lifecycle=TradeLifecycleConfig(
            exit=TradeLifecycleExitConfig(
                enabled=exit_enabled,
                particle_enabled=particle_exit_enabled,
                particle_horizon=particle_horizon,
                particle_sample_count=particle_sample_count,
                particle_ess_refresh_fraction=particle_ess_refresh_fraction,
            ),
            entry=TradeLifecycleEntryConfig(
                mode=entry_mode,
                single_top_min_net_edge=single_top_min_net_edge,
                single_top_max_gross_pred_return=single_top_max_gross_pred_return,
                single_top_target_weight=single_top_target_weight,
            ),
        ),
    )
    spreads = dict(spread_pct_by_secid or {})
    snapshots = {
        instrument.secid: MarketSnapshot(
            instrument.secid,
            last_price=100.0,
            bid=100.0 * (1.0 - float(spreads.get(instrument.secid, 0.001)) / 2.0),
            ask=100.0 * (1.0 + float(spreads.get(instrument.secid, 0.001)) / 2.0),
        )
        for instrument in instruments
    }
    metrics = {
        instrument.secid: MarketMetrics(
            instrument.secid,
            0.02,
            0.02,
            1_000_000,
            float(spreads.get(instrument.secid, 0.001)),
            0,
            20,
        )
        for instrument in instruments
    }
    candles = candles_by_secid or {
        instrument.secid: pd.DataFrame({"close": [90, 100], "open": [90, 95], "high": [91, 101], "low": [89, 94], "volume": [1000, 1000]})
        for instrument in instruments
    }
    return RuntimeEngine(
        config=config,
        market_data=StaticMarketDataProvider(snapshot_rows=snapshots, metric_rows=metrics, candle_rows=candles),
        kronos_provider=kronos_provider
        or (ForecastSignalProvider(forecast_returns) if forecast_returns is not None else StaticSignalProvider("kronos", scores)),
        exit_kronos_provider=exit_provider or ForecastSignalProvider(exit_returns or {}),
        state=StateStore(tmp_path / "state.sqlite3"),
        logger=JsonlLogger(tmp_path / "logs", stdout=False),
        arenago_client=client,
    )


def _decision_count(tmp_path: Path) -> int:
    state = StateStore(tmp_path / "state.sqlite3")
    row = state.connect().execute("SELECT COUNT(*) AS c FROM decisions").fetchone()
    return int(row["c"])


def _decision_payload(engine: RuntimeEngine, decision_id: str) -> dict:
    row = engine.state.connect().execute("SELECT payload_json FROM decisions WHERE decision_id = ?", (decision_id,)).fetchone()
    return json.loads(row["payload_json"])
