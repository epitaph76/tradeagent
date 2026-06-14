from __future__ import annotations

from arena_bot.config import load_config


def test_entry_metrics_config_can_be_overridden_from_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
mode: paper
trade_lifecycle:
  entry:
    metrics:
      enabled: false
      min_edge_bps: 12
      edge_scale_bps: 80
      vol_floor_bps: 15
      max_allowed_spread_bps: 25
      required_recheck_minutes: 90
      eps: 0.000001
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.trade_lifecycle.entry.metrics.enabled is False
    assert config.trade_lifecycle.entry.metrics.min_edge_bps == 12
    assert config.trade_lifecycle.entry.metrics.edge_scale_bps == 80
    assert config.trade_lifecycle.entry.metrics.vol_floor_bps == 15
    assert config.trade_lifecycle.entry.metrics.max_allowed_spread_bps == 25
    assert config.trade_lifecycle.entry.metrics.required_recheck_minutes == 90
    assert config.trade_lifecycle.entry.metrics.eps == 0.000001


def test_entry_instrument_weights_path_is_resolved_relative_to_config(tmp_path):
    path = tmp_path / "configs" / "config.yaml"
    path.parent.mkdir()
    path.write_text(
        """
mode: paper
trade_lifecycle:
  entry:
    mode: kronos_vector_research
    instrument_weights_path: baseline_per_instrument_weights.yaml
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.trade_lifecycle.entry.mode == "kronos_vector_research"
    assert config.trade_lifecycle.entry.instrument_weights_path == str(path.parent / "baseline_per_instrument_weights.yaml")
