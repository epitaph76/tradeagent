from __future__ import annotations

from typing import Any, Mapping

from .types import PortfolioConfig, SelectorDecision


def blend_selector_portfolios(
    selector_decisions: Mapping[str, SelectorDecision],
    selector_weights: Mapping[str, float],
) -> dict[str, float]:
    blended: dict[str, float] = {}
    for selector_name, decision in selector_decisions.items():
        selector_weight = float(selector_weights.get(selector_name, 0.0) or 0.0)
        if abs(selector_weight) <= 1e-12:
            continue
        for secid, weight in decision.weights.items():
            blended[secid] = blended.get(secid, 0.0) + selector_weight * float(weight)
    return blended


def prune_blended_weights(
    weights: Mapping[str, float],
    config: PortfolioConfig,
) -> tuple[dict[str, float], dict[str, Any]]:
    before = {str(k): float(v) for k, v in weights.items() if abs(float(v)) > 1e-12}
    normalized = normalize_gross(before, max_gross=config.max_gross)
    above_min = {secid: weight for secid, weight in normalized.items() if abs(weight) >= config.min_abs_weight}
    ordered = sorted(above_min.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
    kept_pairs = ordered[: max(config.max_positions, 0)]
    kept = normalize_gross(dict(kept_pairs), max_gross=config.max_gross, scale_up=True)
    pruned = sorted(set(before) - set(kept))
    diagnostics = {
        "blended_positions_before_prune_count": len(before),
        "blended_positions_after_prune_count": len(kept),
        "pruned_tickers": pruned,
        "final_target_positions_count": len(kept),
        "min_abs_weight": config.min_abs_weight,
        "max_positions": config.max_positions,
        "gross_before_prune": sum(abs(v) for v in before.values()),
        "gross_after_prune": sum(abs(v) for v in kept.values()),
    }
    return kept, diagnostics


def normalize_gross(weights: Mapping[str, float], *, max_gross: float, scale_up: bool = False) -> dict[str, float]:
    clean = {str(k): float(v) for k, v in weights.items() if abs(float(v)) > 1e-12}
    gross = sum(abs(value) for value in clean.values())
    if gross <= 0:
        return {}
    if gross <= max_gross and not scale_up:
        return clean
    scale = float(max_gross) / gross
    return {secid: value * scale for secid, value in clean.items()}
