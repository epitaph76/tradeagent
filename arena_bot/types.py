from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Mapping

AssetClass = Literal["equity", "future", "crypto"]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        out = float(value)
    except Exception:
        return low
    if out != out:
        return low
    return min(high, max(low, out))


@dataclass(frozen=True)
class Instrument:
    secid: str
    asset_class: AssetClass = "equity"
    lot_size: int = 1
    enabled: bool = True
    price_step: float | None = None
    quote_currency: str = "RUB"
    venue: str | None = None
    boardid: str | None = None

    def __post_init__(self) -> None:
        if self.asset_class not in {"equity", "future", "crypto"}:
            raise ValueError(f"unsupported asset_class: {self.asset_class}")
        if not self.secid:
            raise ValueError("instrument secid is required")
        if self.venue is None:
            venue = {
                "equity": "moex_stock",
                "future": "moex_futures",
                "crypto": "binance_spot",
            }[self.asset_class]
            object.__setattr__(self, "venue", venue)
        if self.boardid is None:
            boardid = {
                "equity": "TQBR",
                "future": "RFUD",
                "crypto": "",
            }[self.asset_class]
            object.__setattr__(self, "boardid", boardid)


@dataclass(frozen=True)
class SignalRow:
    as_of: datetime
    secid: str
    signal_name: str
    bullish_score: float
    confidence: float = 1.0
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bullish_score", clamp(self.bullish_score))
        object.__setattr__(self, "confidence", clamp(self.confidence))


@dataclass(frozen=True)
class BaseSelectorConfig:
    name: str
    signal_weights: Mapping[str, float]
    threshold: float
    rank_power: float
    max_positions: int | None
    max_long_positions: int | None = None
    max_short_positions: int | None = None
    max_gross: float = 1.0
    allow_short: bool = True
    asset_filter: tuple[AssetClass, ...] | None = None


@dataclass(frozen=True)
class PortfolioConfig:
    max_positions: int = 8
    min_abs_weight: float = 0.03
    max_gross: float = 1.0
    min_hold_minutes: int = 60
    strong_flip_threshold: float = 0.85


@dataclass(frozen=True)
class RiskConfig:
    starting_cash: float = 100000.0
    min_order_value_rub: float = 500.0
    min_position_change_weight: float = 0.01
    max_orders_per_rebalance: int = 12
    max_new_positions_per_rebalance: int = 4
    max_daily_orders: int = 100
    commission_rate: float = 0.0005
    slippage_spread_multiplier: float = 0.5
    cash_buffer_pct: float = 0.02
    sizing_safety_pct: float = 0.005
    long_margin_rate: float = 1.0
    short_margin_rate: float = 1.0
    future_margin_rate: float = 1.0


@dataclass(frozen=True)
class RebalanceConfig:
    decision_interval_minutes: int = 60
    exit_interval_minutes: int = 1


@dataclass(frozen=True)
class TradingSessionConfig:
    enabled: bool = True
    timezone: str = "Europe/Moscow"
    session_open: str = "10:00"
    entry_start: str = "11:00"
    new_entry_cutoff: str = "17:40"
    kronos_cutoff: str = "17:40"
    force_flat_time: str = "18:30"
    session_close: str = "18:40"
    flat_all_asset_classes: bool = True
    calendar_cache_enabled: bool = True
    calendar_cache_ttl_minutes: int = 1440
    force_flat_minutes_before_close: int = 10
    entry_warmup_minutes_after_open: int = 60
    new_entry_cutoff_minutes_before_close: int = 60
    session_templates: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LightGBMConfig:
    model_dir: str = "models/lightgbm_meta"
    min_train_intervals: int = 48
    train_lookback_intervals: int = 512
    rank_power: float = 2.0
    n_estimators: int = 60
    max_model_age_hours: int = 36
    rolling_lookback_intervals: int = 24


@dataclass(frozen=True)
class KronosConfig:
    enabled: bool = False
    code_dir: str = "kronos/model"
    weights_dir: str = "kronos/weights"
    model: str = "base"
    device: str = "auto"
    context_rows: int = 512
    pred_len: int = 1
    sample_count: int = 10
    temperature: float = 0.6
    top_p: float = 0.9
    cache_enabled: bool = True


@dataclass(frozen=True)
class TradeLifecycleExitConfig:
    enabled: bool = True
    pred_len: int = 4
    sample_count: int = 15
    edge_threshold: float = 0.0
    edge_enabled: bool = True
    giveback_enabled: bool = False
    giveback_min_arm_profit: float = 0.012
    giveback_ratio: float = 0.60
    particle_enabled: bool = False
    particle_horizon: int = 8
    particle_sample_count: int = 20
    particle_min_expected_profit: float = 0.0015
    particle_min_plan_probability: float = 0.60
    particle_close_probability: float = 0.52
    particle_ess_refresh_fraction: float = 0.25
    particle_extend_min_improvement: float = 0.0015
    particle_max_extensions: int = 1


@dataclass(frozen=True)
class TradeLifecycleEntryConfig:
    mode: str = "selectors"
    include_held_for_topup: bool = True
    capital_mode: str = "free_only"
    topup_sizing: str = "rank_budget"
    rank_power: float = 2.0
    ranking_metric: str = "abs_edge"
    single_top_min_net_edge: float = 0.0015
    single_top_min_rank_gap: float = 0.0
    single_top_max_gross_pred_return: float = 0.0075
    single_top_target_weight: float = 1.0


@dataclass(frozen=True)
class TradeLifecycleConfig:
    max_total_positions: int = 5
    exit: TradeLifecycleExitConfig = field(default_factory=TradeLifecycleExitConfig)
    entry: TradeLifecycleEntryConfig = field(default_factory=TradeLifecycleEntryConfig)


@dataclass(frozen=True)
class RuntimeConfig:
    mode: Literal["paper", "live", "dry_run"]
    bot_name: str
    data_dir: str
    instruments: tuple[Instrument, ...]
    base_selectors: tuple[BaseSelectorConfig, ...]
    portfolio: PortfolioConfig
    risk: RiskConfig
    rebalance: RebalanceConfig
    lightgbm: LightGBMConfig
    kronos: KronosConfig
    max_equities: int = 10
    trading_session: TradingSessionConfig = field(default_factory=TradingSessionConfig)
    trade_lifecycle: TradeLifecycleConfig = field(default_factory=TradeLifecycleConfig)

    @property
    def live_orders(self) -> bool:
        return self.mode == "live"


@dataclass(frozen=True)
class MarketSnapshot:
    secid: str
    last_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    volume_value: float = 0.0
    source: str = ""

    @property
    def spread_pct(self) -> float:
        if self.bid > 0 and self.ask > 0 and self.ask >= self.bid:
            mid = (self.bid + self.ask) / 2.0
            return (self.ask - self.bid) / mid if mid > 0 else 1.0
        return 1.0

    @property
    def tradable(self) -> bool:
        return self.last_price > 0


@dataclass(frozen=True)
class MarketMetrics:
    secid: str
    realized_volatility: float = 0.0
    atr_pct: float = 0.0
    volume_value: float = 0.0
    spread_pct: float = 1.0
    missing_candles: int = 0
    candle_count: int = 0


@dataclass(frozen=True)
class SelectorDecision:
    name: str
    weights: Mapping[str, float]
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class PlannedOrder:
    secid: str
    direction: Literal["B", "S"]
    quantity: int
    current_lots: int
    target_lots: int
    price: float
    lot_size: int
    order_value: float
    target_weight: float
    score: float
    order_kind: str
    reason: str = ""


@dataclass(frozen=True)
class AccountState:
    cash: float
    equity: float
    gross: float
    net: float
    margin_used: float
    available_cash: float = 0.0
    available_gross: float = 0.0
    cash_buffer: float = 0.0


@dataclass(frozen=True)
class OrderResult:
    secid: str
    direction: str
    quantity: int
    status: str
    request: Mapping[str, Any]
    response: Mapping[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass(frozen=True)
class DecisionResult:
    decision_id: str
    as_of: datetime
    selector_weights: Mapping[str, float]
    target_weights: Mapping[str, float]
    selector_diagnostics: Mapping[str, Any]
    blend_diagnostics: Mapping[str, Any]
    orders: tuple[OrderResult, ...]
