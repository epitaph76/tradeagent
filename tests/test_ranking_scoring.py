from __future__ import annotations

import pytest

from arena_bot.ranking.optuna_per_instrument import extract_normalized_weights_from_best_params, sample_normalized_weights
from arena_bot.ranking.scoring import (
    BASELINE_POSITIVE_WEIGHTS,
    BASELINE_RISK_WEIGHTS,
    POSITIVE_METRICS,
    RISK_METRICS,
    compute_candidate_vectors,
    cosine_strength_score,
    load_instrument_weights,
    score_candidate,
    weighted_sum,
)


class FakeTrial:
    def suggest_float(self, name, low, high):
        return {
            "x_a": 1.0,
            "x_b": 3.0,
        }.get(name, 2.0)


def test_weighted_sum_uses_separate_vector_and_weights():
    assert weighted_sum({"a": 0.25, "b": 0.5}, {"a": 0.2, "b": 0.8}) == pytest.approx(0.45)


def test_cosine_strength_prefers_stronger_vector_with_same_direction():
    weak = {name: 0.25 for name in POSITIVE_METRICS}
    strong = {name: 0.75 for name in POSITIVE_METRICS}

    assert cosine_strength_score(strong, BASELINE_POSITIVE_WEIGHTS) > cosine_strength_score(weak, BASELINE_POSITIVE_WEIGHTS)


def test_compute_candidate_vectors_normalizes_long_and_short_to_same_semantics():
    scales = {
        "edge_good_pct": 0.01,
        "close_good_pct": 0.01,
        "body_good_pct": 0.01,
        "mae_bad_pct": 0.03,
        "spread_bad_pct": 0.01,
        "range_min_good_pct": 0.005,
        "range_max_good_pct": 0.05,
        "edge_mean_pct": 0.0,
        "edge_std_pct": 0.01,
    }
    base = {
        "secid": "SBER",
        "bid": 100.0,
        "ask": 100.0,
        "commission_rate": 0.0,
        "slippage_rate": 0.0,
    }
    long_vectors = compute_candidate_vectors(
        {
            **base,
            "direction": "long",
            "pred_open": 100.0,
            "pred_high": 104.0,
            "pred_low": 99.5,
            "pred_close": 103.0,
        },
        scales,
    )
    short_vectors = compute_candidate_vectors(
        {
            **base,
            "direction": "short",
            "pred_open": 100.0,
            "pred_high": 100.5,
            "pred_low": 96.0,
            "pred_close": 97.0,
        },
        scales,
    )

    for payload in (long_vectors, short_vectors):
        assert set(payload["positive_vector"]) == set(POSITIVE_METRICS)
        assert set(payload["risk_vector"]) == set(RISK_METRICS)
        assert all(0.0 <= value <= 1.0 for value in payload["positive_vector"].values())
        assert all(0.0 <= value <= 1.0 for value in payload["risk_vector"].values())
        assert payload["positive_vector"]["close_score"] > 0.0
        assert payload["positive_vector"]["body_score"] > 0.0


def test_score_candidate_uses_cosine_for_positive_and_weighted_sum_for_risk():
    candidate = {
        "positive_vector": {name: 1.0 for name in POSITIVE_METRICS},
        "risk_vector": {name: 0.0 for name in RISK_METRICS},
    }
    scores = score_candidate(
        candidate,
        {
            "positive_weights": BASELINE_POSITIVE_WEIGHTS,
            "risk_weights": BASELINE_RISK_WEIGHTS,
            "risk_threshold": 0.35,
        },
    )

    assert scores["positive_score"] == pytest.approx(1.0)
    assert scores["risk_score"] == pytest.approx(0.0)


def test_load_instrument_weights_falls_back_to_baseline_for_missing_secid(tmp_path):
    path = tmp_path / "weights.yaml"
    path.write_text(
        """
instrument_weights:
  SBER:
    positive_weights:
      net_edge_score: 1
      edge_z_score: 1
      rr_score: 1
      mae_score: 1
      close_score: 1
      body_score: 1
      wick_score: 1
      candle_quality: 1
      edge_risk_quality: 1
    risk_weights:
      false_breakout_risk: 1
      wide_spread_risk: 1
      late_entry_risk: 1
      high_mae_risk: 1
      direction_conflict_risk: 1
    risk_threshold: 0.4
""",
        encoding="utf-8",
    )

    weights = load_instrument_weights(path=path, instruments=["SBER", "BRN6"])

    assert weights["SBER"]["risk_threshold"] == 0.4
    assert sum(weights["SBER"]["positive_weights"].values()) == pytest.approx(1.0)
    assert weights["BRN6"]["positive_weights"] == pytest.approx(BASELINE_POSITIVE_WEIGHTS)


def test_optuna_weight_helpers_normalize_values():
    sampled = sample_normalized_weights(FakeTrial(), "x", ["a", "b"])
    extracted = extract_normalized_weights_from_best_params({"p_a": 2.0, "p_b": 6.0}, "p", ["a", "b"])

    assert sampled == pytest.approx({"a": 0.25, "b": 0.75})
    assert extracted == pytest.approx({"a": 0.25, "b": 0.75})
