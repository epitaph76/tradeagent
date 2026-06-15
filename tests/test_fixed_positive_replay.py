from __future__ import annotations

import pytest

from arena_bot.optimization.fixed_positive_replay import (
    SingleTopRuntimeLikeParams,
    normalized_fixed_risk_weights,
    replay_candidate_rows,
    replay_candidate_rows_single_top_runtime_like,
)
from arena_bot.ranking.scoring import RISK_METRICS


def _row(
    as_of: str,
    *,
    secid: str = "AAA",
    side: str = "long",
    score_level: float = 0.8,
    net_edge: float = 0.01,
    net_edge_score: float = 0.8,
    bid: float = 100.0,
    ask: float = 100.1,
    future_bid: float = 101.0,
    future_ask: float = 101.1,
    risk_value: float = 0.2,
    mae_pct: float = 0.001,
    rr: float = 2.0,
) -> dict:
    positive_vector = {
        "net_edge_score": net_edge_score,
        "edge_z_score": score_level,
        "rr_score": score_level,
        "mae_score": score_level,
        "close_score": score_level,
        "body_score": score_level,
        "wick_score": score_level,
        "candle_quality": score_level,
        "edge_risk_quality": score_level,
    }
    return {
        "as_of": as_of,
        "trade_date": as_of[:10],
        "secid": secid,
        "asset_class": "equity",
        "side": side,
        "positive_vector": positive_vector,
        "risk_vector": {metric: risk_value for metric in RISK_METRICS},
        "raw_vector_metrics": {"net_edge": net_edge, "mae_pct": mae_pct, "rr": rr},
        "current_bid": bid,
        "current_ask": ask,
        "future_bid": future_bid,
        "future_ask": future_ask,
        "future_as_of": "2026-04-14T12:00:00",
        "commission_rate": 0.0,
        "slippage_rate": 0.0,
        "realized_net_return": 0.0,
    }


def test_fixed_risk_weights_normalize_to_one():
    weights = normalized_fixed_risk_weights()

    assert set(weights) == set(RISK_METRICS)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_top1_without_positive_edge_does_not_backfill_top2():
    rows = [
        _row("2026-04-14T11:00:00", secid="BAD", score_level=1.0, net_edge=-0.01, net_edge_score=0.0),
        _row("2026-04-14T11:00:00", secid="GOOD", score_level=0.4, net_edge=0.02, net_edge_score=0.4),
    ]

    result = replay_candidate_rows(rows)

    assert result["summary"]["trade_count"] == 0
    assert not [event for event in result["events"] if event["event"] == "entry"]
    assert result["events"][0]["event"] == "skip_no_edge"
    assert result["events"][0]["top1_secid"] == "BAD"


def test_same_top1_is_held_without_reopen():
    rows = [
        _row("2026-04-14T11:00:00", secid="AAA", score_level=0.9, net_edge=0.02, net_edge_score=0.9, bid=100.0, ask=100.1),
        _row("2026-04-14T12:00:00", secid="AAA", score_level=0.9, net_edge=0.02, net_edge_score=0.9, bid=101.0, ask=101.1),
    ]

    result = replay_candidate_rows(rows)
    events = [event["event"] for event in result["events"]]

    assert events.count("entry") == 1
    assert "hold_same_top1" in events
    assert result["summary"]["trade_count"] == 1
    assert result["trades"][0]["exit_reason"] == "final_close"


def test_high_risk_score_is_logged_but_does_not_block_entry():
    rows = [
        _row(
            "2026-04-14T11:00:00",
            secid="RISKY",
            score_level=0.9,
            net_edge=0.02,
            net_edge_score=0.9,
            risk_value=1.0,
        )
    ]

    result = replay_candidate_rows(rows)

    assert any(event["event"] == "entry" and event["top1_secid"] == "RISKY" for event in result["events"])
    assert result["decision_rows"][0]["risk_score"] == pytest.approx(1.0)
    assert result["summary"]["trade_count"] == 1


def test_max_positive_score_blocks_top1_without_backfill():
    rows = [
        _row("2026-04-14T11:00:00", secid="HIGH", score_level=1.0, net_edge=0.02, net_edge_score=1.0),
        _row("2026-04-14T11:00:00", secid="OK", score_level=0.2, net_edge=0.01, net_edge_score=0.2),
    ]

    result = replay_candidate_rows_single_top_runtime_like(
        rows,
        params=SingleTopRuntimeLikeParams(
            min_net_edge=0.0,
            min_rank_gap=0.0,
            max_gross_pred_return=1.0,
            target_abs_weight=1.0,
            max_positive_score=0.7,
        ),
        ranking_metric="net_edge",
    )

    assert result["summary"]["trade_count"] == 0
    assert result["decision_rows"][0]["secid"] == "HIGH"
    assert result["decision_rows"][0]["reason"] == "positive_score_above_max"
    assert result["decision_rows"][1]["secid"] == "OK"
    assert result["decision_rows"][1]["selected"] is False


def test_min_raw_mae_pct_blocks_top1_without_backfill():
    rows = [
        _row("2026-04-14T11:00:00", secid="ZERO_MAE", score_level=0.9, net_edge=0.02, net_edge_score=0.9, mae_pct=0.0),
        _row("2026-04-14T11:00:00", secid="OK", score_level=0.8, net_edge=0.01, net_edge_score=0.8, mae_pct=0.001),
    ]

    result = replay_candidate_rows_single_top_runtime_like(
        rows,
        params=SingleTopRuntimeLikeParams(
            min_net_edge=0.0,
            min_rank_gap=0.0,
            max_gross_pred_return=1.0,
            target_abs_weight=1.0,
            min_raw_mae_pct=0.0,
        ),
        ranking_metric="net_edge",
    )

    assert result["summary"]["trade_count"] == 0
    assert result["decision_rows"][0]["secid"] == "ZERO_MAE"
    assert result["decision_rows"][0]["reason"] == "single_top_raw_mae_pct_below_min"
    assert result["decision_rows"][1]["secid"] == "OK"
    assert result["decision_rows"][1]["selected"] is False


def test_max_raw_rr_blocks_top1_without_backfill():
    rows = [
        _row("2026-04-14T11:00:00", secid="HIGH_RR", score_level=0.9, net_edge=0.02, net_edge_score=0.9, rr=40.0),
        _row("2026-04-14T11:00:00", secid="OK", score_level=0.8, net_edge=0.01, net_edge_score=0.8, rr=2.0),
    ]

    result = replay_candidate_rows_single_top_runtime_like(
        rows,
        params=SingleTopRuntimeLikeParams(
            min_net_edge=0.0,
            min_rank_gap=0.0,
            max_gross_pred_return=1.0,
            target_abs_weight=1.0,
            max_raw_rr=40.0,
        ),
        ranking_metric="net_edge",
    )

    assert result["summary"]["trade_count"] == 0
    assert result["decision_rows"][0]["secid"] == "HIGH_RR"
    assert result["decision_rows"][0]["reason"] == "single_top_raw_rr_above_max"
    assert result["decision_rows"][1]["secid"] == "OK"
    assert result["decision_rows"][1]["selected"] is False


def test_close_before_session_gap_closes_at_future_price():
    rows = [
        _row(
            "2026-04-14T11:00:00",
            secid="AAA",
            score_level=0.9,
            net_edge=0.02,
            net_edge_score=0.9,
            bid=100.0,
            ask=100.1,
            future_bid=102.0,
            future_ask=102.1,
        )
    ]

    result = replay_candidate_rows_single_top_runtime_like(
        rows,
        params=SingleTopRuntimeLikeParams(
            min_net_edge=0.0,
            min_rank_gap=0.0,
            max_gross_pred_return=1.0,
            target_abs_weight=1.0,
        ),
        ranking_metric="net_edge",
        close_before_session_gap=True,
    )

    assert result["summary"]["trade_count"] == 1
    assert result["summary"]["session_gap_close_count"] == 1
    assert result["trades"][0]["exit_reason"] == "session_gap_close"
    assert result["trades"][0]["exit_as_of"] == "2026-04-14T12:00:00"
    assert result["trades"][0]["exit_price"] == pytest.approx(102.0)
    assert result["decision_rows"][0]["session_gap_after"] is True
    assert result["decision_rows"][0]["position_after"] == {}
    assert any(event["event"] == "session_gap_close" for event in result["events"])
