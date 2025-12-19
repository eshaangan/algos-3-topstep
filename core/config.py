"""
Centralised configuration objects and helpers for the TopstepX trading framework.

All runtime parameters live here and can optionally be overridden via a YAML file.
The module must remain importable inside the activated `.venv`.

Supports named profiles for scalping strategies via the `scalping_profile` field.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Scalping Profile Presets
# ---------------------------------------------------------------------------

SCALPING_PROFILES: Dict[str, Dict[str, Any]] = {
    "baseline": {
        # Optimized parameters from sweep: PF=1.12, MDD=$2054, 69 trades
        "scalping_fast_ema_period": 13,
        "scalping_slow_ema_period": 34,
        "scalping_micro_trend_period": 89,
        "scalping_atr_period": 14,
        "scalping_pullback_ticks": 6,
        "scalping_breakout_body_ticks": 2,
        "scalping_stop_ticks": 8,
        "scalping_target_ticks": 10,
        "scalping_max_trades_per_day": 20,
        "scalping_cooldown_bars_after_exit": 2,
        "scalping_intraday_start_time": "08:30",
        "scalping_intraday_end_time": "14:45",
        "scalping_min_atr_ticks": 3,
        "scalping_max_atr_ticks": 50,
    },
    "high_frequency": {
        # More trades: relaxed filters, tighter stops, faster EMAs
        # Target: >120 trades while PF >= 1.05
        "scalping_fast_ema_period": 8,
        "scalping_slow_ema_period": 21,
        "scalping_micro_trend_period": 55,
        "scalping_atr_period": 10,
        "scalping_pullback_ticks": 4,
        "scalping_breakout_body_ticks": 1,
        "scalping_stop_ticks": 6,
        "scalping_target_ticks": 8,
        "scalping_max_trades_per_day": 30,
        "scalping_cooldown_bars_after_exit": 1,
        "scalping_intraday_start_time": "08:30",
        "scalping_intraday_end_time": "14:55",
        "scalping_min_atr_ticks": 2,
        "scalping_max_atr_ticks": 60,
    },
    "conservative": {
        # Fewer trades, better quality: stricter filters, wider stops
        # Target: lower drawdown, higher win rate
        "scalping_fast_ema_period": 13,
        "scalping_slow_ema_period": 55,
        "scalping_micro_trend_period": 144,
        "scalping_atr_period": 21,
        "scalping_pullback_ticks": 8,
        "scalping_breakout_body_ticks": 3,
        "scalping_stop_ticks": 10,
        "scalping_target_ticks": 14,
        "scalping_max_trades_per_day": 10,
        "scalping_cooldown_bars_after_exit": 3,
        "scalping_intraday_start_time": "09:00",
        "scalping_intraday_end_time": "14:30",
        "scalping_min_atr_ticks": 4,
        "scalping_max_atr_ticks": 40,
    },
    "pierce_pause": {
        # Optimized for pullback pierce/pause filters (PF ~1.39, 124 trades)
        "scalping_fast_ema_period": 9,
        "scalping_slow_ema_period": 34,
        "scalping_micro_trend_period": 34,
        "scalping_atr_period": 21,
        "scalping_pullback_ticks": 5,
        "scalping_breakout_body_ticks": 3,
        "scalping_stop_ticks": 10,
        "scalping_target_ticks": 15,
        "scalping_max_trades_per_day": 15,
        "scalping_cooldown_bars_after_exit": 2,
        "scalping_intraday_start_time": "08:30",
        "scalping_intraday_end_time": "14:45",
        "scalping_min_atr_ticks": 3,
        "scalping_max_atr_ticks": 50,
        "scalping_pullback_pierce_ticks": 1,
        "scalping_pullback_pause_bars": 1,
    },
}


def get_scalping_profile(name: str) -> Dict[str, Any]:
    """
    Retrieve a named scalping profile's parameters.

    Args:
        name: Profile name ('baseline', 'high_frequency', 'conservative').

    Returns:
        Dictionary of scalping parameters to merge into StrategyConfig.

    Raises:
        ValueError: If profile name is not recognized.
    """
    if name not in SCALPING_PROFILES:
        available = ", ".join(SCALPING_PROFILES.keys())
        raise ValueError(f"Unknown scalping profile '{name}'. Available: {available}")
    return SCALPING_PROFILES[name].copy()


# ---------------------------------------------------------------------------
# Config Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SessionConfig:
    """Exchange session parameters used for time-based risk controls."""

    timezone: str = "America/Chicago"
    session_start: str = "08:30"  # Central Time
    session_end: str = "15:00"  # Must be flat by this time
    flat_buffer_minutes: int = 5  # stop trading n minutes before close


@dataclass
class DatabentoConfig:
    """Settings required to hydrate OHLCV data via the DataBento API."""

    dataset: str = "GLBX.MDP3"
    schema: str = "ohlcv-1m"
    symbol: Optional[str] = "ES"  # futures root symbol
    symbols: Optional[List[str]] = None  # explicit contract codes, overrides root inference
    start: str = "2024-01-01"
    end: str = "2024-06-01"
    cache_path: str = "data/ES_5min.csv"
    resample_minutes: Optional[int] = 5


@dataclass
class TopstepDataConfig:
    """Settings for calling ProjectX /api/History/retrieveBars."""

    contract_id: Optional[str] = None
    live: bool = True
    unit: int = 2  # minute bars
    unit_number: int = 5  # 5-minute aggregation
    limit: int = 500
    include_partial_bar: bool = False
    poll_seconds: int = 5
    fallback_to_sim: bool = True


@dataclass
class DataConfig:
    """Data ingestion settings for historical and live processing."""

    symbol: str = "ES"
    data_path: str = "data/ES_5min.csv"
    timeframe_minutes: int = 5
    tz_localize: bool = True
    timezone: str = "America/Chicago"
    timestamp_col: str = "timestamp"
    price_cols: Optional[Dict[str, str]] = None
    source: str = "csv"  # csv | databento | topstep
    databento: DatabentoConfig = field(default_factory=DatabentoConfig)
    topstep: TopstepDataConfig = field(default_factory=TopstepDataConfig)

    def __post_init__(self) -> None:
        if isinstance(self.databento, dict):
            self.databento = DatabentoConfig(**self.databento)
        if isinstance(self.topstep, dict):
            self.topstep = TopstepDataConfig(**self.topstep)
        if self.price_cols is None:
            self.price_cols = {
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }


@dataclass
class StrategyConfig:
    """Parameters for all strategy modes (EMA pullback, composite, scalping)."""

    # Global strategy selector: defaults to legacy EMA pullback for compatibility.
    mode: str = "ema_pullback"  # "ema_pullback" | "composite_v1" | "scalping_v1" | "ml_only"

    # Named profile for scalping strategies (baseline | high_frequency | conservative)
    # When set, profile values override individual scalping_* fields.
    scalping_profile: Optional[str] = None

    fast_ema_period: int = 50
    slow_ema_period: int = 200
    pullback_pct: float = 0.0010  # 0.10% pullback tolerance
    confirmation_body_ticks: int = 2
    stop_ticks: int = 24  # fallback distance in ticks
    target_rr_multiple: float = 1.2
    min_trend_bars: int = 50
    cooldown_bars_after_exit: int = 1
    confirmation_lookback: int = 2
    trend_bias_threshold: float = 0.0005
    reentry_buffer_pct: float = 0.0005
    use_adaptive_stop: bool = True
    atr_period: int = 21
    atr_multiplier: float = 1.2
    min_stop_ticks: int = 12
    max_stop_ticks: int = 32
    enable_momentum_entry: bool = True
    momentum_breakout_lookback: int = 3
    momentum_min_body_ticks: int = 4
    # Intraday trade quality filters
    enable_intraday_filter: bool = True
    intraday_start_time: str = "08:35"
    intraday_end_time: str = "14:30"
    enable_volatility_filter: bool = True
    min_atr_ticks: int = 8
    max_atr_ticks: int = 40
    # Trade management toggles
    enable_partial_profit: bool = True
    partial_profit_rr: float = 1.0
    partial_profit_fraction: float = 0.5
    enable_trailing_stop: bool = True
    trailing_start_rr: float = 1.0
    trailing_stop_lock_rr: float = 0.5
    # ML-only mode controls (reused from live config when mode == "ml_only")
    ml_threshold: float = 0.5
    ml_min_bars: int = 20
    ml_timeframe_minutes: int = 5
    ml_model_dir: str = "artifacts/ml/latest"
    # Composite ML gate (post-merge) settings
    composite_ml_gate_enabled: bool = False
    composite_ml_threshold: float = 0.5
    composite_ml_min_bars: int = 60
    composite_ml_timeframe_minutes: int = 5
    composite_ml_model_dir: str = "artifacts/ml/latest"

    # Moving-average cross leg
    ma_cross_enabled: bool = False

    # RSI + Bollinger mean-reversion leg
    rsi_bb_enabled: bool = False
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    bb_period: int = 20
    bb_stddev: float = 2.0

    # TTM Squeeze Momentum Leg
    squeeze_enabled: bool = False
    squeeze_bb_period: int = 20
    squeeze_bb_std: float = 2.0
    squeeze_kc_period: int = 20
    squeeze_kc_mult: float = 1.5
    squeeze_momentum_lookback: int = 20

    # Volatility Expansion Mode (allows entries on vol expansion without strict squeeze)
    volatility_expansion_enabled: bool = True
    volatility_expansion_threshold: float = 0.02  # BB width increase of 2%

    # Trend quality filters (all disabled by default to preserve backward compatibility)
    require_trend_acceleration: bool = False  # require EMA separation growing
    min_ema_separation_bars: int = 3  # require N bars with growing separation
    min_price_vs_slow_ema_pct: float = 0.003  # close must be X% above/below slow EMA
    require_consistent_trend_bars: int = 0  # require N bars all in same trend direction (0 = disabled)

    # Pullback quality filters (all disabled by default to preserve backward compatibility)
    require_volume_confirmation: bool = False  # require volume > avg * multiplier
    volume_threshold_multiplier: float = 1.2  # volume threshold multiplier
    volume_lookback: int = 20  # bars to calculate average volume
    volume_allow_low_vol_environment: bool = True  # allow entry in low-vol if avg volume < threshold
    volume_low_vol_threshold: float = 500.0  # if avg volume < this, don't filter on volume
    require_pullback_consolidation: bool = False  # wait for consolidation after pullback
    consolidation_bars_required: int = 2  # wait N bars after pullback before entry
    min_pullback_duration_bars: int = 0  # require pullback to last N bars (0 = disabled)

    # Scalping strategy (scalping_v1) parameters
    scalping_fast_ema_period: int = 9
    scalping_slow_ema_period: int = 21
    scalping_micro_trend_period: Optional[int] = 50
    scalping_atr_period: int = 14
    scalping_pullback_ticks: int = 4
    scalping_breakout_body_ticks: int = 3
    scalping_stop_ticks: int = 8
    scalping_target_ticks: int = 10
    scalping_max_trades_per_day: int = 15
    scalping_cooldown_bars_after_exit: int = 1
    scalping_intraday_start_time: str = "08:30"
    scalping_intraday_end_time: str = "14:45"
    scalping_min_atr_ticks: int = 4
    scalping_max_atr_ticks: int = 40
    scalping_diagnostics_enabled: bool = False
    scalping_diagnostics_path: Optional[str] = None
    # Pullback quality refinements
    scalping_pullback_pierce_ticks: int = 0  # require wick through fast EMA by N ticks (0 = disabled)
    scalping_pullback_pause_bars: int = 0  # require N bars pause after touch before breakout (0 = disabled)
    # Volume Profile and VWAP filters
    scalping_enable_vwap_filter: bool = False  # Use VWAP for trend bias filtering
    scalping_vwap_min_distance_ticks: int = 1  # Min distance from VWAP to avoid chop
    scalping_enable_volume_profile: bool = False  # Require pullbacks to high-volume nodes
    scalping_vp_lookback_bars: int = 390  # Lookback for Volume Profile calculation
    scalping_vp_poc_distance_ticks: int = 3  # Distance to POC/Value Area to qualify as "at node"


@dataclass
class RiskConfig:
    """TopstepX risk guardrails."""

    starting_balance: float = 50_000.0
    max_daily_loss: float = 1_000.0
    trailing_drawdown: float = 2_000.0
    daily_profit_target: Optional[float] = None  # Flatten all positions when daily profit reaches this amount
    fixed_risk_per_trade: float = 450.0
    scalping_risk_per_trade: Optional[float] = None  # Override for scalping_v1
    max_contracts: int = 3
    max_open_positions: int = 1
    tick_size: float = 0.25
    tick_value: float = 12.50
    flat_by_time: str = "14:55"
    # Optional drawdown-based risk scaling
    drawdown_risk_scale_enabled: bool = True
    drawdown_risk_scale_dd: float = 1_000.0
    drawdown_risk_scale_multiplier: float = 0.5
    # Entry cutoff buffer (minutes before flat_by_time/session end to stop new entries)
    entry_cutoff_buffer_minutes: int = 0
    # Optional ATR-based intraday risk scaling
    atr_risk_scale_enabled: bool = False
    atr_reference_ticks: float = 16.0
    atr_scale_floor: float = 0.5
    atr_scale_cap: float = 1.5
    # Live trading options
    use_live_account_state: bool = False
    live_state_refresh_seconds: int = 30
    live_trading_enabled: bool = False
    max_trades_per_day: int = 0  # 0 means no extra cap beyond Topstep rules
    bracket_stop_type: int = 4  # default stop bracket uses stop order
    bracket_target_type: int = 1  # default target bracket uses limit order

    # Adaptive sizing based on recent performance (disabled by default)
    adaptive_sizing_enabled: bool = False  # enable adaptive position sizing
    recent_trades_lookback: int = 10  # track last N trades for win/loss streak
    size_multiplier_on_losing_streak: float = 0.5  # reduce to 50% after N losses
    losing_streak_threshold: int = 3  # trigger after N consecutive losses
    size_multiplier_on_winning_streak: float = 1.0  # optional: increase after wins (1.0 = no increase)
    winning_streak_threshold: int = 0  # 0 = disabled, N+ = enable increase after N wins


@dataclass
class BacktestConfig:
    """Simulation-specific parameters."""

    slippage_ticks: float = 1.0
    commission_per_contract: float = 2.35
    initial_equity: float = 150_000.0
    results_path: str = "artifacts/backtests"


@dataclass
class LiveConfig:
    """Live trading configuration and guardrails."""

    live_trading_enabled: bool = False
    paper_trading: bool = True
    max_trade_count_per_day: int = 0  # 0 = unlimited (subject to risk rules)
    log_path: str = "artifacts/live_logs"
    ml_gate_enabled: bool = False
    ml_model_dir: str = "artifacts/ml/latest"
    ml_threshold: float = 0.75
    ml_timeframe_minutes: int = 5
    ml_min_bars: int = 20  # minimum bars before ML gating activates
    trade_cap_per_day: int = 20


@dataclass
class ProjectConfig:
    """
    Aggregate configuration wrapper used by the framework.

    Attributes hold nested configuration segments for data, strategy, risk, etc.
    """

    session: SessionConfig = field(default_factory=SessionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    live: LiveConfig = field(default_factory=LiveConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable dictionary representation."""
        return {
            "session": asdict(self.session),
            "data": asdict(self.data),
            "strategy": asdict(self.strategy),
            "risk": asdict(self.risk),
            "backtest": asdict(self.backtest),
            "live": asdict(self.live),
        }


DEFAULT_CONFIG_PATH = Path("config.yaml")


def _merge_dict(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override values into default dictionary."""
    result = dict(default)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _apply_scalping_profile(strategy_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply named scalping profile to strategy config dict if specified.

    Profile values override individual scalping_* fields but can themselves
    be overridden by explicit field values in the YAML.
    """
    profile_name = strategy_dict.get("scalping_profile")
    if not profile_name:
        return strategy_dict

    try:
        profile = get_scalping_profile(profile_name)
    except ValueError:
        return strategy_dict  # Unknown profile, skip

    # Profile provides defaults; explicit YAML values take precedence
    for key, value in profile.items():
        if key not in strategy_dict:
            strategy_dict[key] = value

    return strategy_dict


def load_config(yaml_path: Optional[str | Path] = None) -> ProjectConfig:
    """
    Load configuration from YAML if provided, otherwise fall back to defaults.

    Args:
        yaml_path: Optional path to a YAML file containing overrides.

    Returns:
        Fully populated ProjectConfig instance.
    """
    config = ProjectConfig()
    path = Path(yaml_path) if yaml_path else DEFAULT_CONFIG_PATH
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        merged = _merge_dict(config.to_dict(), payload)
        # Apply scalping profile before constructing StrategyConfig
        merged["strategy"] = _apply_scalping_profile(merged["strategy"])
        config = ProjectConfig(
            session=SessionConfig(**merged["session"]),
            data=DataConfig(**merged["data"]),
            strategy=StrategyConfig(**merged["strategy"]),
            risk=RiskConfig(**merged["risk"]),
            backtest=BacktestConfig(**merged["backtest"]),
            live=LiveConfig(**merged.get("live", {})),
        )
    return config


def save_default_config(path: Optional[str | Path] = None) -> Path:
    """
    Persist the default configuration to disk so it can be edited by the user.

    Returns:
        The path that was written to.
    """
    cfg = ProjectConfig().to_dict()
    target = Path(path) if path else DEFAULT_CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    return target
