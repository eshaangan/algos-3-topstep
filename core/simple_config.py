"""
Minimal configuration for ML training pipeline.
Standalone version without complex dependencies.
"""

from dataclasses import dataclass
from datetime import time
from typing import Optional


@dataclass
class RiskConfig:
    """TopstepX risk parameters."""

    starting_balance: float = 50000.0
    max_daily_loss: float = 500.0
    trailing_drawdown: float = 1800.0
    fixed_risk_per_trade: float = 200.0  # Max USD to risk per trade
    max_contracts: int = 5
    tick_size: float = 0.25
    tick_value: float = 1.25  # MES tick value

    # Trading hours (Central Time)
    session_start: time = time(8, 30)  # 8:30 AM CT
    session_end: time = time(14, 55)  # 2:55 PM CT


@dataclass
class TrainingConfig:
    """ML training parameters."""

    # ========== LABELS V2: Market-Centric Fixed-Horizon ==========
    # NEW: Replace TP/SL simulation with pure market movement labels
    use_labels_v2: bool = True  # Use fixed-horizon labels (recommended)
    horizon_bars: int = 12  # Look ahead H bars (12 bars @ 5min = 60 minutes)
    threshold_ticks: int = 10  # Minimum movement for directional label (10 ticks = 2.5 MES points)

    # OLD: TP/SL Simulation Labels (deprecated, keep for backward compatibility)
    lookback_bars: int = 100
    label_mode: str = "per_bar"  # "per_bar" (independent) or "sequential" (deprecated)
    stop_loss_ticks: int = 24  # Only for execution, NOT for labels
    target_multiplier: float = 2.0  # Only for execution, NOT for labels
    max_hold_bars: int = 12  # Only for execution, NOT for labels

    # ========== WALK-FORWARD SPLITS WITH LOCKBOX ==========
    train_fraction: float = 0.5  # Reduced from 0.6 to allow lockbox
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    lockbox_fraction: float = 0.1  # NEVER used for tuning, final evaluation only

    # ========== PURE ML STRATEGY: Score + Trade Budget ==========
    # NEW: Control frequency via ranking + daily budget, NOT high thresholds
    enable_long: bool = True
    enable_short: bool = True  # Enable shorts for symmetric ML selection

    session_mode: str = "RTH"  # "RTH" or "24H"

    # Score threshold (quantile-based, fitted on train)
    score_quantile: float = 0.975  # Default target for ~2 trades/day in RTH
    target_trades_per_day: int = 2
    auto_score_quantile: bool = True
    score_quantile_min: float = 0.90
    score_quantile_max: float = 0.995

    # Daily trade budget (hard limit)
    max_trades_per_day: int = 2  # Target ~1-2 trades/day on average
    min_bars_between_trades: int = 12  # Min 60 minutes between entries (avoid rapid-fire)

    # Deprecated: high fixed thresholds (old approach)
    min_probability_long: float = 0.50  # Lowered from 0.65 (use score_quantile instead)
    min_probability_short: float = 0.50

    # ========== MODEL REGULARIZATION ==========
    # Conservative RF parameters to prevent overfitting
    rf_n_estimators: int = 500           # More trees = better averaging
    rf_max_depth: int = 10                # Limit depth to prevent memorization
    rf_min_samples_leaf: int = 15         # Require meaningful patterns
    rf_min_samples_split: int = 30        # Conservative splitting
    rf_max_features: str = "sqrt"         # Reduce tree correlation

    # ========== FEATURE SELECTION ==========
    feature_selection_mode: str = "recommended"  # "all" (36), "recommended" (20), "auto"
    top_n_features: int = 20

    # ========== PERFORMANCE GATES ==========
    # Gates for live trading approval (conservative)
    min_win_rate: float = 0.52
    min_profit_factor: float = 1.3
    max_drawdown: float = 1500.0


@dataclass
class NNConfig:
    """Config for the tiny MLP pipeline."""

    horizon_bars: int = 12
    threshold_ticks: int = 12
    feature_lookback: int = 100
    embargo_bars: int = 0  # 0 means "compute from horizon + lookback"

    score_quantile: float = 0.975
    score_threshold: Optional[float] = None

    max_trades_per_day: int = 2
    min_bars_between_trades: int = 12
    enable_long: bool = True
    enable_short: bool = True

    session_mode: str = "RTH"  # "RTH" or "24H"
    target_trades_per_day: int = 2
    auto_score_quantile: bool = True
    score_quantile_min: float = 0.90
    score_quantile_max: float = 0.995
    deadline_time: time = time(11, 30)  # CT
    deadline_relax_factor: float = 0.98
    execution_mode: str = "time_exit"  # "time_exit" or "triple_barrier"
    exit_price_mode: str = "bar_close"  # "next_open" or "bar_close"

    train_fraction: float = 0.6
    val_fraction: float = 0.2
    test_fraction: float = 0.2

    folds: int = 1
    seed: int = 42
    feature_selection_mode: str = "recommended"


# Global instances
RISK_CONFIG = RiskConfig()
TRAINING_CONFIG = TrainingConfig()
NN_CONFIG = NNConfig()
