"""
Data models shared across the TopstepX trading framework.

Each dataclass is intentionally lightweight so that it can be serialised easily
for persistence or downstream analytics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class SignalAction(str, Enum):
    """Supported signal actions emitted by strategies."""

    ENTER_LONG = "ENTER_LONG"
    ENTER_SHORT = "ENTER_SHORT"
    EXIT = "EXIT"


class ExitReason(str, Enum):
    """Categorised exit reasons for trade audit trail."""

    STOP = "STOP"
    TARGET = "TARGET"
    PARTIAL_TP = "PARTIAL_TP"
    SESSION_FLAT = "SESSION_FLAT"
    FINAL_BAR_FLAT = "FINAL_BAR_FLAT"
    DAILY_LOSS = "DAILY_LOSS"
    TRAILING_DD = "TRAILING_DD"
    MANUAL = "MANUAL"


@dataclass(slots=True)
class Bar:
    """Single OHLCV bar."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(slots=True)
class VolumeProfileNode:
    """Single price level in the volume profile."""

    price_level: float
    volume: float


@dataclass(slots=True)
class VolumeProfile:
    """Volume Profile analysis for a session or lookback period."""

    poc: Optional[float] = None  # Point of Control (highest volume price)
    value_area_high: Optional[float] = None  # Top of 70% volume zone
    value_area_low: Optional[float] = None  # Bottom of 70% volume zone
    total_volume: float = 0.0
    nodes: List[VolumeProfileNode] = field(default_factory=list)


@dataclass(slots=True)
class IndicatorSnapshot:
    """Convenience bundle of computed indicators aligned with a bar."""

    fast_ema: float
    slow_ema: float
    atr: Optional[float] = None
    micro_ema: Optional[float] = None  # For scalping micro-trend
    vwap: Optional[float] = None  # Volume-Weighted Average Price
    volume_profile: Optional[VolumeProfile] = None  # Session volume profile


@dataclass(slots=True)
class Signal:
    """Strategy output describing an intent but not an executable order yet."""

    action: SignalAction
    symbol: str
    timestamp: datetime
    stop_price: float
    target_price: float
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Order:
    """Represents an order routed by the execution engine."""

    id: str
    signal: Signal
    side: SignalAction
    contracts: int
    entry_price: float
    stop_price: float
    target_price: float
    status: str = "PENDING"


@dataclass(slots=True)
class Position:
    """Open position details used by risk management and performance tracking."""

    symbol: str
    contracts: int
    direction: SignalAction
    entry_price: float
    stop_price: float
    target_price: float
    entry_time: datetime

    def unrealized_pnl(self, last_price: float, tick_value: float, tick_size: float) -> float:
        """
        Compute current unrealised PnL in dollars.

        Args:
            last_price: The most recent traded price.
            tick_value: Dollar value per tick move for the instrument.
            tick_size: Price increment per tick.

        Returns:
            Unrealised PnL in dollars.
        """
        if tick_size <= 0:
            return 0.0
        ticks = (last_price - self.entry_price) / tick_size
        if self.direction == SignalAction.ENTER_SHORT:
            ticks *= -1
        return ticks * tick_value * self.contracts


@dataclass(slots=True)
class Trade:
    """Completed trade statistics with full audit trail."""

    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    direction: SignalAction
    contracts: int
    pnl: float
    risk_multiple: float
    reason: str
    # Additional audit fields
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    risk_per_contract: Optional[float] = None

    @property
    def is_winner(self) -> bool:
        """Return True if trade was profitable."""
        return self.pnl > 0

    @property
    def duration_minutes(self) -> float:
        """Return trade duration in minutes."""
        return (self.exit_time - self.entry_time).total_seconds() / 60


@dataclass(slots=True)
class RiskState:
    """Holds intra-day risk metrics enforced across strategy + execution."""

    trading_locked: bool = False
    realized_daily_pnl: float = 0.0
    trailing_peak: float = 0.0
    cumulative_pnl: float = 0.0
    open_positions: int = 0  # CRITICAL: Tracks TOTAL CONTRACTS in use, not position count


@dataclass(slots=True)
class DailyPerformance:
    """Aggregated metrics for a trading day."""

    date: datetime
    pnl: float
    max_drawdown: float
    trades: List[Trade]


@dataclass(slots=True)
class BacktestStats:
    """Summary metrics from a completed backtest run."""

    total_trades: int
    win_rate: float
    avg_r_multiple: float
    max_drawdown: float
    max_daily_drawdown: float
    profit_factor: float
    expectancy: float
    equity_curve: List[float]
    trades: List[Trade]
    # Extended metrics
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    max_win_streak: int = 0
    max_loss_streak: int = 0


@dataclass(slots=True)
class ScalpingParams:
    """
    Pure value object for scalping strategy parameters.

    Extracted from StrategyConfig for cleaner passing between functions.
    """

    fast_ema_period: int
    slow_ema_period: int
    micro_trend_period: Optional[int]
    atr_period: int
    pullback_ticks: int
    breakout_body_ticks: int
    stop_ticks: int
    target_ticks: int
    max_trades_per_day: int
    cooldown_bars_after_exit: int
    intraday_start_time: str
    intraday_end_time: str
    min_atr_ticks: int
    max_atr_ticks: int
    pullback_pierce_ticks: int
    pullback_pause_bars: int
    # Volume Profile and VWAP parameters
    enable_vwap_filter: bool = False
    vwap_min_distance_ticks: int = 1
    enable_volume_profile: bool = False
    vp_lookback_bars: int = 390  # ~1 session for MES
    vp_poc_distance_ticks: int = 3  # How close to POC/VA qualifies as "at node"

    @classmethod
    def from_strategy_config(cls, cfg: Any) -> "ScalpingParams":
        """Create ScalpingParams from a StrategyConfig object."""
        return cls(
            fast_ema_period=cfg.scalping_fast_ema_period,
            slow_ema_period=cfg.scalping_slow_ema_period,
            micro_trend_period=cfg.scalping_micro_trend_period,
            atr_period=cfg.scalping_atr_period,
            pullback_ticks=cfg.scalping_pullback_ticks,
            breakout_body_ticks=cfg.scalping_breakout_body_ticks,
            stop_ticks=cfg.scalping_stop_ticks,
            target_ticks=cfg.scalping_target_ticks,
            max_trades_per_day=cfg.scalping_max_trades_per_day,
            cooldown_bars_after_exit=cfg.scalping_cooldown_bars_after_exit,
            intraday_start_time=cfg.scalping_intraday_start_time,
            intraday_end_time=cfg.scalping_intraday_end_time,
            min_atr_ticks=cfg.scalping_min_atr_ticks,
            max_atr_ticks=cfg.scalping_max_atr_ticks,
            pullback_pierce_ticks=getattr(cfg, "scalping_pullback_pierce_ticks", 0),
            pullback_pause_bars=getattr(cfg, "scalping_pullback_pause_bars", 0),
            enable_vwap_filter=getattr(cfg, "scalping_enable_vwap_filter", False),
            vwap_min_distance_ticks=getattr(cfg, "scalping_vwap_min_distance_ticks", 1),
            enable_volume_profile=getattr(cfg, "scalping_enable_volume_profile", False),
            vp_lookback_bars=getattr(cfg, "scalping_vp_lookback_bars", 390),
            vp_poc_distance_ticks=getattr(cfg, "scalping_vp_poc_distance_ticks", 3),
        )
