"""
Feature registry for V3 pipeline.

Defines FeatureSpec dataclass and registers the minimal feature set.
All features are causal (no lookahead) and deterministic.
"""

from dataclasses import dataclass
from typing import List, Literal


@dataclass
class FeatureSpec:
    """
    Feature specification for registry.

    Attributes:
        name: Feature column name
        lookback_bars: Number of historical bars required (0 = current bar only)
        uses_rolling_stats: Whether this feature uses rolling window computations
        requires_scaling: Whether this feature should be scaled before modeling
        fit_on_train_only: Whether transformations must be fit on train data only
        bar_sizes_supported: Which bar sizes this feature applies to
        description: Human-readable description
    """

    name: str
    lookback_bars: int
    uses_rolling_stats: bool
    requires_scaling: bool
    fit_on_train_only: bool
    bar_sizes_supported: List[Literal["1m", "5m"]]
    description: str


def get_feature_registry(config: dict) -> List[FeatureSpec]:
    """
    Get ordered feature registry based on config.

    This function defines the canonical feature set and ordering.
    The order is deterministic and must be stable for schema hashing.

    Args:
        config: Features config dict (from features.yaml)

    Returns:
        List of FeatureSpec in deterministic order

    ORDERING CONVENTION:
    1. Returns (single-bar, then multi-bar)
    2. Volatility (true_range, ATR)
    3. Trend (EMAs and derived)
    4. Structure (candle features)
    5. Time (cyclical encodings)
    6. Meta (is_synthetic, usable_for_training)
    """
    registry = []

    # -------------------------------------------------------------------------
    # 1. RETURNS
    # -------------------------------------------------------------------------
    # Single-bar log return (always included)
    registry.append(
        FeatureSpec(
            name="log_return_1",
            lookback_bars=1,
            uses_rolling_stats=False,
            requires_scaling=True,
            fit_on_train_only=False,
            bar_sizes_supported=["1m", "5m"],
            description="Log return over 1 bar: log(close_t / close_{t-1})",
        )
    )

    # Multi-bar log returns (bar-size specific)
    returns_config = config.get("returns", {})
    lookback_1m = returns_config.get("lookback_bars", {}).get("1m", [3, 6])
    lookback_5m = returns_config.get("lookback_bars", {}).get("5m", [2, 4])

    # Add 1m-specific returns
    for k in lookback_1m:
        registry.append(
            FeatureSpec(
                name=f"log_return_{k}",
                lookback_bars=k,
                uses_rolling_stats=False,
                requires_scaling=True,
                fit_on_train_only=False,
                bar_sizes_supported=["1m"],
                description=f"Log return over {k} bars (1m only)",
            )
        )

    # Add 5m-specific returns
    for k in lookback_5m:
        registry.append(
            FeatureSpec(
                name=f"log_return_{k}",
                lookback_bars=k,
                uses_rolling_stats=False,
                requires_scaling=True,
                fit_on_train_only=False,
                bar_sizes_supported=["5m"],
                description=f"Log return over {k} bars (5m only)",
            )
        )

    # PHASE 3: Multi-horizon returns (match label horizons)
    if returns_config.get("enable_multi_horizon", True):
        multi_horizon = returns_config.get("multi_horizon_bars", [6, 12, 24])
        for k in multi_horizon:
            registry.append(
                FeatureSpec(
                    name=f"log_return_{k}",
                    lookback_bars=k,
                    uses_rolling_stats=False,
                    requires_scaling=True,
                    fit_on_train_only=False,
                    bar_sizes_supported=["1m", "5m"],
                    description=f"Log return over {k} bars (multi-horizon)",
                )
            )

    # -------------------------------------------------------------------------
    # 2. VOLATILITY
    # -------------------------------------------------------------------------
    registry.append(
        FeatureSpec(
            name="true_range",
            lookback_bars=1,  # Needs previous close
            uses_rolling_stats=False,
            requires_scaling=True,
            fit_on_train_only=False,
            bar_sizes_supported=["1m", "5m"],
            description="True range: max(high-low, high-prev_close, prev_close-low)",
        )
    )

    atr_period = config.get("volatility", {}).get("atr_period", 14)
    registry.append(
        FeatureSpec(
            name=f"atr_{atr_period}",
            lookback_bars=atr_period,
            uses_rolling_stats=True,
            requires_scaling=True,
            fit_on_train_only=False,
            bar_sizes_supported=["1m", "5m"],
            description=f"Average True Range over {atr_period} bars (EMA of true_range)",
        )
    )

    # PHASE 3: Volatility regime indicators
    if config.get("volatility", {}).get("enable_regime_features", True):
        vol_regime_lookback = config.get("volatility", {}).get("vol_regime_lookback", 50)
        registry.extend([
            FeatureSpec(
                name="vol_20",
                lookback_bars=20,
                uses_rolling_stats=True,
                requires_scaling=True,
                fit_on_train_only=False,
                bar_sizes_supported=["1m", "5m"],
                description="Rolling volatility (std of returns over 20 bars)",
            ),
            FeatureSpec(
                name="vol_regime",
                lookback_bars=vol_regime_lookback,
                uses_rolling_stats=True,
                requires_scaling=True,
                fit_on_train_only=False,
                bar_sizes_supported=["1m", "5m"],
                description=f"Volatility regime: vol_20 / median(vol_20, {vol_regime_lookback} bars)",
            ),
            FeatureSpec(
                name="parkinson_vol",
                lookback_bars=20,
                uses_rolling_stats=True,
                requires_scaling=True,
                fit_on_train_only=False,
                bar_sizes_supported=["1m", "5m"],
                description="Parkinson volatility estimator (uses high-low range)",
            ),
            FeatureSpec(
                name="vol_forecast",
                lookback_bars=20,
                uses_rolling_stats=True,
                requires_scaling=True,
                fit_on_train_only=False,
                bar_sizes_supported=["1m", "5m"],
                description="EWMA volatility forecast (alpha=0.06)",
            ),
        ])

    # -------------------------------------------------------------------------
    # 3. TREND
    # -------------------------------------------------------------------------
    ema_fast = config.get("trend", {}).get("ema_fast_period", 13)
    ema_slow = config.get("trend", {}).get("ema_slow_period", 34)

    registry.append(
        FeatureSpec(
            name=f"ema_{ema_fast}",
            lookback_bars=ema_fast,
            uses_rolling_stats=True,
            requires_scaling=True,
            fit_on_train_only=False,
            bar_sizes_supported=["1m", "5m"],
            description=f"Exponential moving average of close, period {ema_fast}",
        )
    )

    registry.append(
        FeatureSpec(
            name=f"ema_{ema_slow}",
            lookback_bars=ema_slow,
            uses_rolling_stats=True,
            requires_scaling=True,
            fit_on_train_only=False,
            bar_sizes_supported=["1m", "5m"],
            description=f"Exponential moving average of close, period {ema_slow}",
        )
    )

    registry.append(
        FeatureSpec(
            name="ema_spread",
            lookback_bars=ema_slow,  # Max of fast/slow
            uses_rolling_stats=True,
            requires_scaling=True,
            fit_on_train_only=False,
            bar_sizes_supported=["1m", "5m"],
            description=f"ema_{ema_fast} - ema_{ema_slow}",
        )
    )

    registry.append(
        FeatureSpec(
            name="ema_ratio",
            lookback_bars=ema_slow,
            uses_rolling_stats=True,
            requires_scaling=True,
            fit_on_train_only=False,
            bar_sizes_supported=["1m", "5m"],
            description=f"ema_{ema_fast} / (ema_{ema_slow} + eps)",
        )
    )

    # PHASE 3: Advanced trend and mean reversion features
    if config.get("trend", {}).get("enable_advanced_features", True):
        sma_long_period = config.get("trend", {}).get("sma_long_period", 30)
        registry.extend([
            FeatureSpec(
                name="sma_20",
                lookback_bars=20,
                uses_rolling_stats=True,
                requires_scaling=True,
                fit_on_train_only=False,
                bar_sizes_supported=["1m", "5m"],
                description="Simple moving average of close, 20 bars",
            ),
            FeatureSpec(
                name=f"sma_{sma_long_period}",
                lookback_bars=sma_long_period,
                uses_rolling_stats=True,
                requires_scaling=True,
                fit_on_train_only=False,
                bar_sizes_supported=["1m", "5m"],
                description=f"Simple moving average of close, {sma_long_period} bars",
            ),
            FeatureSpec(
                name="trend_strength",
                lookback_bars=sma_long_period,
                uses_rolling_stats=True,
                requires_scaling=True,
                fit_on_train_only=False,
                bar_sizes_supported=["1m", "5m"],
                description=f"Normalized distance from SMA{sma_long_period}: (close - sma{sma_long_period}) / sma{sma_long_period}",
            ),
            FeatureSpec(
                name="autocorr_5",
                lookback_bars=20,
                uses_rolling_stats=True,
                requires_scaling=True,
                fit_on_train_only=False,
                bar_sizes_supported=["1m", "5m"],
                description="Autocorrelation lag-5 on 20-bar window (mean reversion indicator)",
            ),
            FeatureSpec(
                name="bb_position",
                lookback_bars=20,
                uses_rolling_stats=True,
                requires_scaling=False,  # Already normalized 0-1
                fit_on_train_only=False,
                bar_sizes_supported=["1m", "5m"],
                description="Position within Bollinger Bands (0=lower, 1=upper)",
            ),
        ])

    # PHASE 3: Microstructure features (order flow proxies)
    if config.get("microstructure", {}).get("enabled", True):
        registry.extend([
            FeatureSpec(
                name="volume_imbalance",
                lookback_bars=0,
                uses_rolling_stats=False,
                requires_scaling=True,
                fit_on_train_only=False,
                bar_sizes_supported=["1m", "5m"],
                description="Volume-weighted directional imbalance: (close-open)/(high-low)",
            ),
            FeatureSpec(
                name="price_vs_vwap",
                lookback_bars=50,
                uses_rolling_stats=True,
                requires_scaling=True,
                fit_on_train_only=False,
                bar_sizes_supported=["1m", "5m"],
                description="Price relative to 50-bar VWAP: (close - vwap) / vwap",
            ),
            FeatureSpec(
                name="relative_volume",
                lookback_bars=20,
                uses_rolling_stats=True,
                requires_scaling=True,
                fit_on_train_only=False,
                bar_sizes_supported=["1m", "5m"],
                description="Current volume relative to 20-bar average",
            ),
            FeatureSpec(
                name="large_move",
                lookback_bars=20,
                uses_rolling_stats=True,
                requires_scaling=False,  # Binary indicator
                fit_on_train_only=False,
                bar_sizes_supported=["1m", "5m"],
                description="Binary: 1 if |return| > 2× vol_20, else 0",
            ),
        ])

    # -------------------------------------------------------------------------
    # 4. STRUCTURE (Candle features)
    # -------------------------------------------------------------------------
    if config.get("structure", {}).get("enabled", True):
        registry.append(
            FeatureSpec(
                name="candle_body",
                lookback_bars=0,
                uses_rolling_stats=False,
                requires_scaling=True,
                fit_on_train_only=False,
                bar_sizes_supported=["1m", "5m"],
                description="close - open (positive = bullish)",
            )
        )

        registry.append(
            FeatureSpec(
                name="candle_range",
                lookback_bars=0,
                uses_rolling_stats=False,
                requires_scaling=True,
                fit_on_train_only=False,
                bar_sizes_supported=["1m", "5m"],
                description="high - low",
            )
        )

        registry.append(
            FeatureSpec(
                name="body_pct",
                lookback_bars=0,
                uses_rolling_stats=False,
                requires_scaling=True,
                fit_on_train_only=False,
                bar_sizes_supported=["1m", "5m"],
                description="candle_body / max(candle_range, eps)",
            )
        )

        registry.append(
            FeatureSpec(
                name="upper_wick",
                lookback_bars=0,
                uses_rolling_stats=False,
                requires_scaling=True,
                fit_on_train_only=False,
                bar_sizes_supported=["1m", "5m"],
                description="high - max(open, close)",
            )
        )

        registry.append(
            FeatureSpec(
                name="lower_wick",
                lookback_bars=0,
                uses_rolling_stats=False,
                requires_scaling=True,
                fit_on_train_only=False,
                bar_sizes_supported=["1m", "5m"],
                description="min(open, close) - low",
            )
        )

    # -------------------------------------------------------------------------
    # 5. TIME (Cyclical encodings)
    # -------------------------------------------------------------------------
    if config.get("time", {}).get("enabled", True):
        registry.append(
            FeatureSpec(
                name="minute_of_day_sin",
                lookback_bars=0,
                uses_rolling_stats=False,
                requires_scaling=False,  # Already bounded [-1, 1]
                fit_on_train_only=False,
                bar_sizes_supported=["1m", "5m"],
                description="sin(2π * minute_of_day / minutes_per_session)",
            )
        )

        registry.append(
            FeatureSpec(
                name="minute_of_day_cos",
                lookback_bars=0,
                uses_rolling_stats=False,
                requires_scaling=False,  # Already bounded [-1, 1]
                fit_on_train_only=False,
                bar_sizes_supported=["1m", "5m"],
                description="cos(2π * minute_of_day / minutes_per_session)",
            )
        )

        registry.append(
            FeatureSpec(
                name="day_of_week",
                lookback_bars=0,
                uses_rolling_stats=False,
                requires_scaling=False,  # Categorical, will be one-hot later
                fit_on_train_only=False,
                bar_sizes_supported=["1m", "5m"],
                description="Day of week (0=Monday, 6=Sunday)",
            )
        )

    # -------------------------------------------------------------------------
    # 6. META (Flags and masks)
    # -------------------------------------------------------------------------
    if config.get("output", {}).get("include_synthetic_flag", True):
        registry.append(
            FeatureSpec(
                name="is_synthetic",
                lookback_bars=0,
                uses_rolling_stats=False,
                requires_scaling=False,
                fit_on_train_only=False,
                bar_sizes_supported=["1m", "5m"],
                description="Flag marking synthetic/missing bars from reindexing",
            )
        )

    # Usable for training mask (computed in build.py)
    if config.get("computation", {}).get("nan_policy") == "keep_with_mask":
        registry.append(
            FeatureSpec(
                name="usable_for_training",
                lookback_bars=0,
                uses_rolling_stats=False,
                requires_scaling=False,
                fit_on_train_only=False,
                bar_sizes_supported=["1m", "5m"],
                description="Mask: True if all required features are non-NaN",
            )
        )

    return registry


def filter_registry_for_bar_size(
    registry: List[FeatureSpec], bar_size: Literal["1m", "5m"]
) -> List[FeatureSpec]:
    """
    Filter registry to only features supported for given bar size.

    Args:
        registry: Full feature registry
        bar_size: Target bar size ("1m" or "5m")

    Returns:
        Filtered registry containing only features for this bar size
    """
    return [spec for spec in registry if bar_size in spec.bar_sizes_supported]
