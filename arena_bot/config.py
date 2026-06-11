from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from .types import (
    BaseSelectorConfig,
    Instrument,
    KronosConfig,
    LightGBMConfig,
    PortfolioConfig,
    RebalanceConfig,
    RiskConfig,
    RuntimeConfig,
    TradeLifecycleConfig,
    TradeLifecycleEntryConfig,
    TradeLifecycleExitConfig,
    TradingSessionConfig,
)

TOP20_EQUITIES: tuple[str, ...] = (
    "LKOH",
    "SBER",
    "ROSN",
    "GAZP",
    "VTBR",
    "YDEX",
    "PLZL",
    "T",
    "NVTK",
    "X5",
    "GMKN",
    "MGNT",
    "ALRS",
    "AFLT",
    "CHMF",
    "NLMK",
    "MOEX",
    "SNGSP",
    "MTSS",
    "PIKK",
)

DEFAULT_LOT_SIZES = {
    "GAZP": 10,
    "GMKN": 10,
    "ALRS": 10,
    "AFLT": 10,
    "NLMK": 10,
    "MOEX": 10,
    "SNGSP": 10,
    "MTSS": 10,
}


def default_base_selectors() -> tuple[BaseSelectorConfig, ...]:
    return (
        BaseSelectorConfig(
            name="selector_kronos_core",
            signal_weights={"kronos": 1.0},
            threshold=0.65,
            rank_power=2.0,
            max_positions=6,
            max_long_positions=3,
            max_short_positions=3,
            asset_filter=("equity",),
        ),
        BaseSelectorConfig(
            name="selector_kronos_aggressive",
            signal_weights={"kronos": 1.0},
            threshold=0.60,
            rank_power=1.5,
            max_positions=8,
            max_long_positions=4,
            max_short_positions=4,
            asset_filter=("equity",),
        ),
        BaseSelectorConfig(
            name="selector_kronos_conservative",
            signal_weights={"kronos": 1.0},
            threshold=0.75,
            rank_power=2.5,
            max_positions=4,
            max_long_positions=2,
            max_short_positions=2,
            asset_filter=("equity",),
        ),
        BaseSelectorConfig(
            name="selector_volatility_filtered",
            signal_weights={"kronos": 1.0},
            threshold=0.65,
            rank_power=2.0,
            max_positions=5,
            max_long_positions=3,
            max_short_positions=2,
            asset_filter=("equity",),
        ),
        BaseSelectorConfig(
            name="selector_futures_view",
            signal_weights={"momentum": 1.0},
            threshold=0.65,
            rank_power=2.0,
            max_positions=3,
            max_long_positions=2,
            max_short_positions=1,
            asset_filter=("future",),
        ),
        BaseSelectorConfig(
            name="selector_crypto_view",
            signal_weights={"momentum": 1.0},
            threshold=0.70,
            rank_power=2.0,
            max_positions=2,
            max_long_positions=1,
            max_short_positions=1,
            asset_filter=("crypto",),
        ),
    )


def load_config(path: str | Path) -> RuntimeConfig:
    path = Path(path)
    raw = _load_yaml(path)
    mode = str(os.environ.get("ARENA_MODE", raw.get("mode", "paper"))).lower()
    if _bool_env("ARENA_LIVE_ORDERS", False):
        mode = "live"
    if mode not in {"paper", "live", "dry_run"}:
        raise ValueError(f"unsupported mode: {mode}")

    data_dir = str(raw.get("data_dir", path.parent / "data"))
    return RuntimeConfig(
        mode=mode,  # type: ignore[arg-type]
        bot_name=str(raw.get("bot_name", "ArenaCompactMeta")),
        data_dir=data_dir,
        instruments=_parse_instruments(raw.get("instruments")),
        base_selectors=_parse_base_selectors(raw.get("base_selectors")),
        portfolio=_dataclass_from_mapping(PortfolioConfig, raw.get("portfolio", {})),
        risk=_dataclass_from_mapping(RiskConfig, raw.get("risk", {})),
        rebalance=_dataclass_from_mapping(RebalanceConfig, raw.get("rebalance", {})),
        lightgbm=_dataclass_from_mapping(LightGBMConfig, raw.get("lightgbm", {})),
        kronos=_dataclass_from_mapping(KronosConfig, raw.get("kronos", {})),
        max_equities=int(raw.get("universe", {}).get("max_equities", raw.get("max_equities", 10))),
        trading_session=_dataclass_from_mapping(TradingSessionConfig, raw.get("trading_session", {})),
        trade_lifecycle=_parse_trade_lifecycle(raw.get("trade_lifecycle", {})),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _parse_instruments(raw: Any) -> tuple[Instrument, ...]:
    if not raw:
        return tuple(
            Instrument(secid=secid, asset_class="equity", lot_size=DEFAULT_LOT_SIZES.get(secid, 1))
            for secid in TOP20_EQUITIES
        )
    rows = raw.values() if isinstance(raw, Mapping) else raw
    instruments = []
    for item in rows:
        if isinstance(item, str):
            instruments.append(Instrument(secid=item, asset_class="equity", lot_size=DEFAULT_LOT_SIZES.get(item, 1)))
            continue
        secid = str(item["secid"])
        instruments.append(
            Instrument(
                secid=secid,
                asset_class=str(item.get("asset_class", "equity")),  # type: ignore[arg-type]
                lot_size=int(item.get("lot_size", DEFAULT_LOT_SIZES.get(secid, 1))),
                enabled=bool(item.get("enabled", True)),
                price_step=_optional_float(item.get("price_step")),
                quote_currency=str(item.get("quote_currency", "RUB")),
                venue=str(item["venue"]) if item.get("venue") is not None else None,
                boardid=str(item["boardid"]) if item.get("boardid") is not None else None,
            )
        )
    return tuple(instruments)


def _parse_base_selectors(raw: Any) -> tuple[BaseSelectorConfig, ...]:
    if not raw:
        return default_base_selectors()
    rows = raw.values() if isinstance(raw, Mapping) else raw
    out = []
    for item in rows:
        out.append(
            BaseSelectorConfig(
                name=str(item["name"]),
                signal_weights={str(k): float(v) for k, v in dict(item.get("signal_weights", {})).items()},
                threshold=float(item.get("threshold", 0.65)),
                rank_power=float(item.get("rank_power", 2.0)),
                max_positions=_optional_int(item.get("max_positions")),
                max_long_positions=_optional_int(item.get("max_long_positions")),
                max_short_positions=_optional_int(item.get("max_short_positions")),
                max_gross=float(item.get("max_gross", 1.0)),
                allow_short=bool(item.get("allow_short", True)),
                asset_filter=tuple(item.get("asset_filter") or ()) or None,  # type: ignore[arg-type]
            )
        )
    return tuple(out)


def _dataclass_from_mapping(cls, raw: Mapping[str, Any] | None):
    raw = dict(raw or {})
    allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in raw.items() if k in allowed})


def _parse_trade_lifecycle(raw: Mapping[str, Any] | None) -> TradeLifecycleConfig:
    raw = dict(raw or {})
    exit_raw = raw.get("exit", {})
    entry_raw = raw.get("entry", {})
    return TradeLifecycleConfig(
        max_total_positions=int(raw.get("max_total_positions", 5)),
        exit=_dataclass_from_mapping(TradeLifecycleExitConfig, exit_raw if isinstance(exit_raw, Mapping) else {}),
        entry=_dataclass_from_mapping(TradeLifecycleEntryConfig, entry_raw if isinstance(entry_raw, Mapping) else {}),
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
