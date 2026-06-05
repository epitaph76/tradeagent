from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib


@dataclass(frozen=True)
class MetaSelectorResult:
    selector_weights: dict[str, float]
    selector_scores: dict[str, float]
    mode: str
    reason: str = ""
    trained_rows: int = 0
    model_path: str = ""
    metadata: Mapping[str, Any] | None = None


class RollingRankWeightedMetaSelector:
    def __init__(self, *, base_selectors: Sequence[str], lookback: int = 24, rank_power: float = 2.0):
        self.base_selectors = tuple(base_selectors)
        self.lookback = lookback
        self.rank_power = rank_power

    def weights(self, history: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        rows = list(history)[-self.lookback :]
        if not rows:
            return _rank_weights_from_order(self.base_selectors, self.rank_power)
        scores = {
            selector: sum(float((row.get("returns") or {}).get(selector, 0.0) or 0.0) for row in rows)
            for selector in self.base_selectors
        }
        return rank_weights_from_scores(scores, rank_power=self.rank_power, base_selectors=self.base_selectors)


class LightGBMMetaSelector:
    def __init__(
        self,
        *,
        model_dir: str | Path,
        base_selectors: Sequence[str],
        rank_power: float = 2.0,
        max_model_age_hours: int = 36,
    ):
        self.model_dir = Path(model_dir)
        self.base_selectors = tuple(base_selectors)
        self.rank_power = rank_power
        self.max_model_age = timedelta(hours=max_model_age_hours)

    @property
    def latest_dir(self) -> Path:
        return self.model_dir / "latest"

    def predict_weights(self, current_features: Mapping[str, float]) -> MetaSelectorResult:
        metadata_path = self.latest_dir / "metadata.json"
        model_path = self.latest_dir / "model.joblib"
        if not metadata_path.exists() or not model_path.exists():
            return MetaSelectorResult({}, {}, mode="fallback", reason="missing_model_artifact")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if _is_stale(metadata.get("created_at_utc"), self.max_model_age):
            return MetaSelectorResult({}, {}, mode="fallback", reason="stale_model_artifact", metadata=metadata)
        try:
            model = joblib.load(model_path)
            feature_names = list(metadata.get("feature_names") or [])
            base_selectors = tuple(metadata.get("base_selectors") or self.base_selectors)
            scores = {
                selector: float(model.predict([_vector(current_features, feature_names, idx, len(base_selectors))])[0])
                for idx, selector in enumerate(base_selectors)
            }
        except Exception as exc:
            return MetaSelectorResult({}, {}, mode="fallback", reason=f"model_load_or_predict_error: {str(exc)[:240]}", metadata=metadata)
        weights = rank_weights_from_scores(scores, rank_power=float(metadata.get("rank_power", self.rank_power)), base_selectors=base_selectors)
        return MetaSelectorResult(
            selector_weights=weights,
            selector_scores=scores,
            mode="lightgbm",
            trained_rows=int(metadata.get("trained_rows", 0) or 0),
            model_path=str(model_path),
            metadata=metadata,
        )


def train_daily_lightgbm(
    *,
    rows: Sequence[Mapping[str, Any]],
    model_dir: str | Path,
    base_selectors: Sequence[str],
    min_train_intervals: int = 48,
    train_lookback_intervals: int = 512,
    rank_power: float = 2.0,
    n_estimators: int = 60,
) -> dict[str, Any]:
    model_dir = Path(model_dir)
    latest_dir = model_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    rows = list(rows)[-train_lookback_intervals:]
    metadata_base = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_selectors": list(base_selectors),
        "rank_power": float(rank_power),
        "trained_rows": len(rows),
        "min_train_intervals": int(min_train_intervals),
        "train_lookback_intervals": int(train_lookback_intervals),
    }
    if len(rows) < min_train_intervals:
        metadata = {**metadata_base, "status": "skipped", "reason": f"insufficient_rows: {len(rows)}/{min_train_intervals}"}
        _write_metadata(latest_dir, metadata)
        return metadata
    try:
        import lightgbm as lgb  # type: ignore
    except Exception as exc:
        metadata = {**metadata_base, "status": "skipped", "reason": f"lightgbm_unavailable: {str(exc)[:240]}"}
        _write_metadata(latest_dir, metadata)
        return metadata

    feature_names = sorted({key for row in rows for key in (row.get("features") or {}).keys()})
    x_train: list[list[float]] = []
    y_train: list[float] = []
    for row in rows:
        features = row.get("features") or {}
        returns = row.get("returns") or {}
        for idx, selector in enumerate(base_selectors):
            x_train.append(_vector(features, feature_names, idx, len(base_selectors)))
            y_train.append(float(returns.get(selector, 0.0) or 0.0))
    if len({round(value, 12) for value in y_train}) <= 1:
        metadata = {**metadata_base, "status": "skipped", "reason": "constant_target", "feature_names": feature_names}
        _write_metadata(latest_dir, metadata)
        return metadata

    model = lgb.LGBMRegressor(
        n_estimators=int(n_estimators),
        learning_rate=0.05,
        max_depth=3,
        min_child_samples=8,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=1,
        verbosity=-1,
    )
    model.fit(x_train, y_train)
    joblib.dump(model, latest_dir / "model.joblib")
    metadata = {**metadata_base, "status": "trained", "reason": "", "feature_names": feature_names}
    _write_metadata(latest_dir, metadata)
    return metadata


def rank_weights_from_scores(
    scores: Mapping[str, float],
    *,
    rank_power: float,
    base_selectors: Sequence[str],
) -> dict[str, float]:
    order = sorted(tuple(base_selectors), key=lambda selector: float(scores.get(selector, 0.0)), reverse=True)
    return _rank_weights_from_order(order, rank_power)


def _rank_weights_from_order(order: Sequence[str], rank_power: float) -> dict[str, float]:
    if not order:
        return {}
    raw = [1.0 if rank_power == 0 else (rank + 1) ** (-float(rank_power)) for rank in range(len(order))]
    total = sum(raw)
    return {selector: value / total for selector, value in zip(order, raw)}


def _vector(features: Mapping[str, Any], feature_names: Sequence[str], selector_index: int, selector_count: int) -> list[float]:
    one_hot = [1.0 if selector_index == idx else 0.0 for idx in range(selector_count)]
    return [_float(features.get(name), 0.0) for name in feature_names] + [float(selector_index)] + one_hot


def _float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _write_metadata(latest_dir: Path, metadata: Mapping[str, Any]) -> None:
    (latest_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _is_stale(created_at_utc: str | None, max_age: timedelta) -> bool:
    if not created_at_utc:
        return True
    try:
        created = datetime.fromisoformat(created_at_utc)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
    except Exception:
        return True
    return datetime.now(timezone.utc) - created > max_age

