from __future__ import annotations

from datetime import datetime

from arena_bot.portfolio import prune_blended_weights
from arena_bot.selectors import build_selector_portfolio
from arena_bot.types import BaseSelectorConfig, Instrument, PortfolioConfig, SignalRow


def test_selector_with_twenty_passing_tickers_obeys_max_positions():
    instruments = tuple(Instrument(secid=f"T{i:02d}") for i in range(20))
    scores = [0.05 + i * 0.05 for i in range(20)]
    signals = [SignalRow(datetime(2026, 1, 1), f"T{i:02d}", "kronos", score, 1.0) for i, score in enumerate(scores)]
    decision = build_selector_portfolio(
        BaseSelectorConfig(
            name="wide_selector",
            signal_weights={"kronos": 1.0},
            threshold=0.55,
            rank_power=0.0,
            max_positions=6,
            max_long_positions=3,
            max_short_positions=3,
        ),
        instruments=instruments,
        signals=signals,
    )
    assert len(decision.weights) <= 6
    assert decision.diagnostics["candidate_count_before_caps"] > 6
    assert decision.diagnostics["long_count"] <= 3
    assert decision.diagnostics["short_count"] <= 3


def test_long_short_caps_are_enforced_independently():
    instruments = tuple(Instrument(secid=f"T{i}") for i in range(8))
    scores = [0.95, 0.9, 0.85, 0.8, 0.05, 0.1, 0.15, 0.2]
    signals = [SignalRow(datetime(2026, 1, 1), f"T{i}", "kronos", score, 1.0) for i, score in enumerate(scores)]
    decision = build_selector_portfolio(
        BaseSelectorConfig(
            name="capped",
            signal_weights={"kronos": 1.0},
            threshold=0.75,
            rank_power=2.0,
            max_positions=4,
            max_long_positions=2,
            max_short_positions=1,
        ),
        instruments=instruments,
        signals=signals,
    )
    assert decision.diagnostics["long_count"] == 2
    assert decision.diagnostics["short_count"] == 1
    assert len(decision.weights) == 3


def test_rank_power_zero_still_cannot_expand_beyond_caps():
    instruments = tuple(Instrument(secid=f"T{i}") for i in range(12))
    signals = [SignalRow(datetime(2026, 1, 1), f"T{i}", "kronos", 0.95, 1.0) for i in range(12)]
    decision = build_selector_portfolio(
        BaseSelectorConfig(
            name="flat",
            signal_weights={"kronos": 1.0},
            threshold=0.55,
            rank_power=0.0,
            max_positions=5,
            max_long_positions=5,
        ),
        instruments=instruments,
        signals=signals,
    )
    assert len(decision.weights) == 5
    assert set(round(value, 6) for value in decision.weights.values()) == {0.2}


def test_post_blend_pruning_keeps_top_eight_and_renormalizes():
    weights = {f"T{i}": 0.20 - i * 0.015 for i in range(12)}
    final, diag = prune_blended_weights(
        weights,
        PortfolioConfig(max_positions=8, min_abs_weight=0.03, max_gross=1.0),
    )
    assert len(final) == 8
    assert diag["blended_positions_before_prune_count"] == 12
    assert diag["blended_positions_after_prune_count"] == 8
    assert abs(sum(abs(value) for value in final.values()) - 1.0) < 1e-9
    assert "T11" in diag["pruned_tickers"]

