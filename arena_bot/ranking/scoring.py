from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

EPS = 1e-9

POSITIVE_METRICS = [
    "net_edge_score",
    "edge_z_score",
    "rr_score",
    "mae_score",
    "close_score",
    "body_score",
    "wick_score",
    "candle_quality",
    "edge_risk_quality",
]

RISK_METRICS = [
    "false_breakout_risk",
    "wide_spread_risk",
    "late_entry_risk",
    "high_mae_risk",
    "direction_conflict_risk",
]

BASELINE_POSITIVE_WEIGHTS = {
    "net_edge_score": 0.25,
    "edge_z_score": 0.10,
    "rr_score": 0.15,
    "mae_score": 0.10,
    "close_score": 0.15,
    "body_score": 0.07,
    "wick_score": 0.05,
    "candle_quality": 0.08,
    "edge_risk_quality": 0.05,
}

BASELINE_RISK_WEIGHTS = {
    "false_breakout_risk": 0.25,
    "wide_spread_risk": 0.25,
    "late_entry_risk": 0.15,
    "high_mae_risk": 0.20,
    "direction_conflict_risk": 0.15,
}

BASELINE_RISK_THRESHOLD = 0.35

DEFAULT_INSTRUMENT_SCALES = {
    "edge_good_pct": 0.006,
    "close_good_pct": 0.006,
    "body_good_pct": 0.004,
    "mae_bad_pct": 0.010,
    "spread_bad_pct": 0.004,
    "range_min_good_pct": 0.002,
    "range_max_good_pct": 0.020,
    "edge_mean_pct": 0.0,
    "edge_std_pct": 0.003,
    "rr_good": 3.0,
    "rr_min": 1.0,
    "z_good": 3.0,
    "edge_risk_min": 1.0,
    "edge_risk_good": 3.0,
}


def clip01(x: float) -> float:
    try:
        out = float(x)
    except Exception:
        return 0.0
    if not math.isfinite(out):
        return 0.0
    return max(0.0, min(1.0, out))


def safe_div(a: float, b: float, eps: float = EPS) -> float:
    return float(a) / (float(b) + eps)


def normalize_positive(value: float, good_value: float) -> float:
    return clip01(safe_div(value, good_value))


def normalize_inverse_bad(value: float, bad_value: float) -> float:
    return clip01(1.0 - safe_div(value, bad_value))


def weighted_sum(vector: Mapping[str, float], weights: Mapping[str, float]) -> float:
    return float(sum(float(vector[name]) * float(weights[name]) for name in weights))


def cosine_strength_score(vector: Mapping[str, float], weights: Mapping[str, float]) -> float:
    candidate = []
    ideal = []
    for name in POSITIVE_METRICS:
        weight = max(float(weights[name]), 0.0)
        root_weight = math.sqrt(weight)
        candidate.append(clip01(float(vector.get(name, 0.0))) * root_weight)
        ideal.append(root_weight)
    dot = sum(a * b for a, b in zip(candidate, ideal))
    candidate_norm = math.sqrt(sum(value * value for value in candidate))
    ideal_norm = math.sqrt(sum(value * value for value in ideal))
    if candidate_norm <= 0.0 or ideal_norm <= 0.0:
        return 0.0
    cosine = dot / (candidate_norm * ideal_norm + EPS)
    strength = candidate_norm / (ideal_norm + EPS)
    return clip01(cosine * strength)


def score_candidate(candidate: Mapping[str, Any], weights: Mapping[str, Any]) -> dict[str, float]:
    positive_weights = validate_metric_weights(weights.get("positive_weights", {}), POSITIVE_METRICS)
    risk_weights = validate_metric_weights(weights.get("risk_weights", {}), RISK_METRICS)
    positive_vector = _require_vector(candidate, "positive_vector", POSITIVE_METRICS)
    risk_vector = _require_vector(candidate, "risk_vector", RISK_METRICS)
    return {
        "positive_score": cosine_strength_score(positive_vector, positive_weights),
        "risk_score": weighted_sum(risk_vector, risk_weights),
    }


def compute_candidate_vectors(
    candidate: Mapping[str, Any],
    instrument_scales: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    scales = {**DEFAULT_INSTRUMENT_SCALES, **dict(instrument_scales or {})}
    direction = str(candidate.get("direction") or candidate.get("side") or "").lower()
    if direction not in {"long", "short"}:
        raise ValueError(f"unsupported candidate direction: {direction}")

    bid = _positive_float(candidate.get("bid"))
    ask = _positive_float(candidate.get("ask"))
    pred_open = _positive_float(candidate.get("pred_open"))
    pred_high = _positive_float(candidate.get("pred_high"))
    pred_low = _positive_float(candidate.get("pred_low"))
    pred_close = _positive_float(candidate.get("pred_close"))
    if bid <= 0.0 or ask <= 0.0 or ask < bid:
        raise ValueError("candidate requires positive bid/ask with ask >= bid")
    if min(pred_open, pred_high, pred_low, pred_close) <= 0.0 or pred_high < pred_low:
        raise ValueError("candidate requires valid predicted OHLC")

    mid = (bid + ask) / 2.0
    spread_rate = (ask - bid) / (mid + EPS)
    commission_rate = _non_negative_float(candidate.get("commission_rate"), default=0.0)
    slippage_rate = _non_negative_float(candidate.get("slippage_rate"), default=0.0)
    cost_rate = commission_rate + slippage_rate + spread_rate
    candle_range = max(pred_high - pred_low, EPS)
    range_pct = candle_range / max(pred_open, EPS)
    upper_wick = max(pred_high - max(pred_open, pred_close), 0.0)
    lower_wick = max(min(pred_open, pred_close) - pred_low, 0.0)
    upper_wick_ratio = clip01(upper_wick / candle_range)
    lower_wick_ratio = clip01(lower_wick / candle_range)

    if direction == "long":
        entry_price = ask
        gross_edge = (pred_close - ask) / ask
        reward = max(0.0, (pred_high - ask) / ask - cost_rate)
        mae_pct = max(0.0, (ask - pred_low) / ask)
        close_edge = gross_edge - cost_rate
        body_return = (pred_close - pred_open) / pred_open
        close_location = clip01((pred_close - pred_low) / candle_range)
        bad_wick_ratio = upper_wick_ratio
        direction_consistency = 1.0 if pred_close > pred_open else 0.0
    else:
        entry_price = bid
        gross_edge = (bid - pred_close) / bid
        reward = max(0.0, (bid - pred_low) / bid - cost_rate)
        mae_pct = max(0.0, (pred_high - bid) / bid)
        close_edge = gross_edge - cost_rate
        body_return = (pred_open - pred_close) / pred_open
        close_location = clip01((pred_high - pred_close) / candle_range)
        bad_wick_ratio = lower_wick_ratio
        direction_consistency = 1.0 if pred_close < pred_open else 0.0

    net_edge = gross_edge - cost_rate
    edge_mean = float(scales.get("edge_mean_pct", 0.0))
    edge_std = max(float(scales.get("edge_std_pct", DEFAULT_INSTRUMENT_SCALES["edge_std_pct"])), EPS)
    edge_z = (net_edge - edge_mean) / (edge_std + EPS)
    rr = reward / (mae_pct + EPS)

    net_edge_score = normalize_positive(net_edge, _scale(scales, "edge_good_pct"))
    edge_z_score = clip01(edge_z / _scale(scales, "z_good"))
    rr_min = float(scales.get("rr_min", 1.0))
    rr_good = max(float(scales.get("rr_good", 3.0)), rr_min + EPS)
    rr_score = clip01((rr - rr_min) / (rr_good - rr_min + EPS))
    mae_score = normalize_inverse_bad(mae_pct, _scale(scales, "mae_bad_pct"))
    close_score = normalize_positive(close_edge, _scale(scales, "close_good_pct"))
    body_score = normalize_positive(body_return, _scale(scales, "body_good_pct"))
    wick_score = clip01(0.5 * close_location + 0.5 * (1.0 - bad_wick_ratio))
    range_score = _range_score(
        range_pct=range_pct,
        range_min_good_pct=_scale(scales, "range_min_good_pct"),
        range_max_good_pct=_scale(scales, "range_max_good_pct"),
    )
    candle_quality = clip01(
        0.35 * body_score
        + 0.35 * wick_score
        + 0.20 * range_score
        + 0.10 * direction_consistency
    )
    edge_risk_ratio = net_edge / (mae_pct + cost_rate + EPS)
    edge_risk_min = float(scales.get("edge_risk_min", 1.0))
    edge_risk_good = max(float(scales.get("edge_risk_good", 3.0)), edge_risk_min + EPS)
    edge_risk_quality = clip01((edge_risk_ratio - edge_risk_min) / (edge_risk_good - edge_risk_min + EPS))

    positive_vector = {
        "net_edge_score": net_edge_score,
        "edge_z_score": edge_z_score,
        "rr_score": rr_score,
        "mae_score": mae_score,
        "close_score": close_score,
        "body_score": body_score,
        "wick_score": wick_score,
        "candle_quality": candle_quality,
        "edge_risk_quality": edge_risk_quality,
    }
    false_breakout_risk = clip01(0.5 * bad_wick_ratio + 0.5 * (1.0 - close_location))
    wide_spread_risk = clip01(spread_rate / _scale(scales, "spread_bad_pct"))
    late_entry_risk = clip01(1.0 - reward / (_scale(scales, "edge_good_pct") + EPS))
    high_mae_risk = clip01(1.0 - mae_score)
    direction_conflict_risk = clip01(1.0 - (0.5 * close_score + 0.5 * body_score))
    risk_vector = {
        "false_breakout_risk": false_breakout_risk,
        "wide_spread_risk": wide_spread_risk,
        "late_entry_risk": late_entry_risk,
        "high_mae_risk": high_mae_risk,
        "direction_conflict_risk": direction_conflict_risk,
    }
    raw_metrics = {
        "entry_price": entry_price,
        "spread_rate": spread_rate,
        "cost_rate": cost_rate,
        "gross_edge": gross_edge,
        "net_edge": net_edge,
        "reward": reward,
        "mae_pct": mae_pct,
        "edge_z": edge_z,
        "rr": rr,
        "range_pct": range_pct,
        "range_score": range_score,
        "close_location": close_location,
        "upper_wick_ratio": upper_wick_ratio,
        "lower_wick_ratio": lower_wick_ratio,
        "bad_wick_ratio": bad_wick_ratio,
        "direction_consistency": direction_consistency,
        "edge_risk_ratio": edge_risk_ratio,
    }
    return {
        "positive_vector": positive_vector,
        "risk_vector": risk_vector,
        "raw_metrics": _json_safe_floats(raw_metrics),
        "instrument_scales": _json_safe_floats(scales),
    }


def instrument_scales_from_history(
    candles: Any,
    *,
    snapshot: Any = None,
    metric: Any = None,
) -> dict[str, float]:
    values = dict(DEFAULT_INSTRUMENT_SCALES)
    spread_pct = _non_negative_float(getattr(metric, "spread_pct", None), default=0.0)
    if spread_pct <= 0.0 and snapshot is not None:
        spread_pct = _non_negative_float(getattr(snapshot, "spread_pct", None), default=0.0)

    realized_vol = _non_negative_float(getattr(metric, "realized_volatility", None), default=0.0)
    atr_pct = _non_negative_float(getattr(metric, "atr_pct", None), default=0.0)
    returns_abs: pd.Series | None = None
    if isinstance(candles, pd.DataFrame) and "close" in candles and len(candles) >= 3:
        close = pd.to_numeric(candles["close"], errors="coerce").dropna()
        returns = close.pct_change().dropna()
        if not returns.empty:
            returns_abs = returns.abs()
            realized_vol = realized_vol or float(returns.std(ddof=0) or 0.0)

    typical_move = max(realized_vol, atr_pct * 0.5, 0.001)
    if returns_abs is not None and not returns_abs.empty:
        edge_mean = float(returns_abs.tail(128).mean() or 0.0)
        edge_std = float(returns_abs.tail(128).std(ddof=0) or 0.0)
    else:
        edge_mean = 0.0
        edge_std = typical_move

    values.update(
        {
            "edge_good_pct": max(1.5 * typical_move, 0.0015),
            "close_good_pct": max(1.2 * typical_move, 0.0015),
            "body_good_pct": max(0.8 * typical_move, 0.0010),
            "mae_bad_pct": max(2.0 * typical_move, atr_pct, 0.0020),
            "spread_bad_pct": max(4.0 * spread_pct, spread_pct + 0.0005, 0.0010),
            "range_min_good_pct": max(0.5 * typical_move, 0.0008),
            "range_max_good_pct": max(4.0 * typical_move, 2.0 * atr_pct, 0.0060),
            "edge_mean_pct": edge_mean,
            "edge_std_pct": max(edge_std, typical_move * 0.5, 0.0005),
        }
    )
    return values


def load_instrument_weights(
    *,
    path: str | Path | None = None,
    inline: Mapping[str, Any] | None = None,
    instruments: Sequence[Any] = (),
) -> dict[str, dict[str, Any]]:
    secids = [str(getattr(item, "secid", item)) for item in instruments]
    payload: dict[str, Any] = {}
    if path:
        weight_path = Path(path)
        if weight_path.exists():
            with weight_path.open("r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
            if isinstance(loaded, Mapping):
                payload.update(dict(loaded.get("instrument_weights", loaded)))
    if inline:
        raw_inline = inline.get("instrument_weights", inline) if isinstance(inline, Mapping) else inline
        if isinstance(raw_inline, Mapping):
            payload.update(dict(raw_inline))

    if not payload:
        return baseline_instrument_weights(secids)

    out: dict[str, dict[str, Any]] = {}
    for secid in secids:
        raw = payload.get(secid) or payload.get(str(secid))
        out[secid] = _normalize_instrument_weight_row(raw, secid=secid)
    for secid, raw in payload.items():
        secid_s = str(secid)
        if secid_s not in out:
            out[secid_s] = _normalize_instrument_weight_row(raw, secid=secid_s)
    return out


def baseline_instrument_weights(secids: Sequence[str]) -> dict[str, dict[str, Any]]:
    return {
        str(secid): {
            "positive_weights": dict(BASELINE_POSITIVE_WEIGHTS),
            "risk_weights": dict(BASELINE_RISK_WEIGHTS),
            "risk_threshold": BASELINE_RISK_THRESHOLD,
        }
        for secid in secids
    }


def save_instrument_weights_yaml(instrument_weights: Mapping[str, Any], path: str | Path) -> None:
    payload = {"instrument_weights": dict(instrument_weights)}
    weight_path = Path(path)
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    with weight_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, allow_unicode=True, sort_keys=False)


def validate_metric_weights(raw: Mapping[str, Any], metrics: Sequence[str]) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        raise ValueError("metric weights must be a mapping")
    missing = [name for name in metrics if name not in raw]
    if missing:
        raise ValueError(f"missing metric weights: {', '.join(missing)}")
    values = {name: _non_negative_float(raw.get(name), default=-1.0) for name in metrics}
    invalid = [name for name, value in values.items() if value < 0.0]
    if invalid:
        raise ValueError(f"metric weights must be non-negative: {', '.join(invalid)}")
    total = sum(values.values())
    if total <= 0.0:
        raise ValueError("metric weights total must be positive")
    return {name: values[name] / (total + EPS) for name in metrics}


def _normalize_instrument_weight_row(raw: Any, *, secid: str) -> dict[str, Any]:
    if raw is None:
        return {
            "positive_weights": dict(BASELINE_POSITIVE_WEIGHTS),
            "risk_weights": dict(BASELINE_RISK_WEIGHTS),
            "risk_threshold": BASELINE_RISK_THRESHOLD,
        }
    if not isinstance(raw, Mapping):
        raise ValueError(f"instrument weights for {secid} must be a mapping")
    threshold = _finite_float(raw.get("risk_threshold", BASELINE_RISK_THRESHOLD), default=BASELINE_RISK_THRESHOLD)
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"risk_threshold for {secid} must be in [0, 1]")
    out = {
        "positive_weights": validate_metric_weights(raw.get("positive_weights", {}), POSITIVE_METRICS),
        "risk_weights": validate_metric_weights(raw.get("risk_weights", {}), RISK_METRICS),
        "risk_threshold": threshold,
    }
    if "objective_value" in raw:
        out["objective_value"] = _finite_float(raw.get("objective_value"), default=0.0)
    return out


def _require_vector(candidate: Mapping[str, Any], key: str, metrics: Sequence[str]) -> dict[str, float]:
    raw = candidate.get(key)
    if not isinstance(raw, Mapping):
        raise ValueError(f"candidate requires {key}")
    return {name: clip01(float(raw.get(name, 0.0) or 0.0)) for name in metrics}


def _range_score(*, range_pct: float, range_min_good_pct: float, range_max_good_pct: float) -> float:
    if range_pct <= 0.0:
        return 0.0
    if range_pct < range_min_good_pct:
        return clip01(range_pct / (range_min_good_pct + EPS))
    if range_pct <= range_max_good_pct:
        return 1.0
    return clip01(1.0 - ((range_pct - range_max_good_pct) / (range_max_good_pct + EPS)))


def _scale(scales: Mapping[str, float], key: str) -> float:
    return max(float(scales.get(key, DEFAULT_INSTRUMENT_SCALES[key])), EPS)


def _positive_float(value: Any) -> float:
    out = _finite_float(value, default=0.0)
    return out if out > 0.0 else 0.0


def _non_negative_float(value: Any, *, default: float) -> float:
    out = _finite_float(value, default=default)
    return out if out >= 0.0 else default


def _finite_float(value: Any, *, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _json_safe_floats(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_floats(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    return value
