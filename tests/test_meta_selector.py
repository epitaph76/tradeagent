from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from arena_bot.meta_selector import LightGBMMetaSelector, train_daily_lightgbm


class DummySelectorModel:
    def predict(self, rows):
        return [float(row[1]) for row in rows]


def test_lightgbm_inference_uses_latest_persisted_model(tmp_path: Path):
    latest = tmp_path / "latest"
    latest.mkdir()
    (latest / "metadata.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "feature_names": ["x"],
                "base_selectors": ["a", "b", "c"],
                "rank_power": 2.0,
                "trained_rows": 50,
            }
        ),
        encoding="utf-8",
    )
    joblib.dump(DummySelectorModel(), latest / "model.joblib")
    result = LightGBMMetaSelector(model_dir=tmp_path, base_selectors=["a", "b", "c"]).predict_weights({"x": 1.0})
    assert result.mode == "lightgbm"
    assert result.selector_scores == {"a": 0.0, "b": 1.0, "c": 2.0}
    assert result.selector_weights["c"] > result.selector_weights["b"] > result.selector_weights["a"]


def test_missing_model_falls_back(tmp_path: Path):
    result = LightGBMMetaSelector(model_dir=tmp_path, base_selectors=["a", "b"]).predict_weights({"x": 1.0})
    assert result.mode == "fallback"
    assert result.reason == "missing_model_artifact"


def test_daily_training_writes_metadata_even_when_skipped(tmp_path: Path):
    metadata = train_daily_lightgbm(
        rows=[],
        model_dir=tmp_path,
        base_selectors=["a", "b"],
        min_train_intervals=48,
    )
    metadata_path = tmp_path / "latest" / "metadata.json"
    assert metadata_path.exists()
    saved = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert saved["status"] == "skipped"
    assert saved["reason"].startswith("insufficient_rows")
    assert metadata["trained_rows"] == 0

