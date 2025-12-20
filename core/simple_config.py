"""
Minimal configuration for ML training pipeline.
Standalone version without complex dependencies.
"""

from dataclasses import dataclass
from datetime import time


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

    lookback_bars: int = 100
    label_mode: str = "per_bar"  # "per_bar" (independent) or "sequential" (one-trade-at-a-time labels)
    stop_loss_ticks: int = 32  # tuned on clean dataset: positive EV at 2.0R
    target_multiplier: float = 2.0
    max_hold_bars: int = 12

    # Walk-forward splits
    train_fraction: float = 0.6
    val_fraction: float = 0.2
    test_fraction: float = 0.2

    # Model thresholds
    min_probability_long: float = 0.65
    min_probability_short: float = 0.65
    enable_long: bool = True
    enable_short: bool = False  # Default off: shorts are currently structurally unfavorable

    # RandomForest regularization defaults (reduce overfitting)
    rf_n_estimators: int = 500
    rf_max_depth: int = 12
    rf_min_samples_leaf: int = 10
    rf_min_samples_split: int = 30
    rf_max_features: str = "sqrt"

    # Performance gates (must pass to use live)
    min_win_rate: float = 0.52
    min_profit_factor: float = 1.3
    max_drawdown: float = 1500.0


# Global instances
RISK_CONFIG = RiskConfig()
TRAINING_CONFIG = TrainingConfig()
