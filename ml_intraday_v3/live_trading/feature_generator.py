"""
Real-time feature generator for live trading.

Computes features from streaming bars using the same logic
as the training pipeline.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LiveFeatureGenerator:
    """
    Generates features in real-time from streaming bars.

    Uses the same feature calculation logic as the training pipeline
    to ensure consistency.
    """

    def __init__(self, feature_columns: List[str]):
        """
        Initialize the feature generator.

        Args:
            feature_columns: List of feature column names (from model bundle)
        """
        self.feature_columns = feature_columns
        logger.info(f"LiveFeatureGenerator initialized with {len(feature_columns)} features")

    def generate_features(self, bars_df: pd.DataFrame) -> pd.Series:
        """
        Generate features from a rolling window of bars.

        Args:
            bars_df: DataFrame with OHLCV bars (sorted by timestamp)

        Returns:
            Series with computed features for the latest bar
        """
        if bars_df.empty:
            logger.warning("Empty bars DataFrame")
            return pd.Series(dtype=float)

        if len(bars_df) < 30:
            logger.warning(f"Insufficient bars for feature calculation: {len(bars_df)} < 30")
            return pd.Series(dtype=float)

        # Compute all features
        features = {}

        # Price-based features
        close = bars_df['close'].values
        high = bars_df['high'].values
        low = bars_df['low'].values
        open_ = bars_df['open'].values

        # Log returns at various lags
        for lag in [1, 3, 6, 12, 24]:
            col = f'log_return_{lag}'
            if col in self.feature_columns:
                if len(close) > lag:
                    features[col] = np.log(close[-1] / close[-1-lag])
                else:
                    features[col] = 0.0

        # True Range and ATR
        if 'true_range' in self.feature_columns or 'atr_14' in self.feature_columns:
            tr = np.maximum(high - low,
                           np.maximum(abs(high - np.roll(close, 1)),
                                    abs(low - np.roll(close, 1))))
            tr[0] = high[0] - low[0]  # First bar has no previous close

            if 'true_range' in self.feature_columns:
                features['true_range'] = tr[-1]

            if 'atr_14' in self.feature_columns and len(tr) >= 14:
                # FIXED: Exponential moving average of TR matching training
                atr_ema = self._ema(tr, period=14)
                features['atr_14'] = atr_ema[-1]

        # Volatility features
        if 'vol_20' in self.feature_columns and len(close) >= 20:
            returns = np.diff(np.log(close))
            vol = np.std(returns[-20:])  # FIXED: Removed annualization to match training
            features['vol_20'] = vol

        if 'vol_regime' in self.feature_columns and len(close) >= 70:
            # FIXED: Match training - rolling median of vol_20 values
            # Compute vol_20 for last 50 bars (overlapping windows)
            returns = np.diff(np.log(close))
            vol_current = np.std(returns[-20:]) if len(returns) >= 20 else 0.0

            # Compute vol_20 for multiple historical windows to get median
            vol_history = []
            for i in range(50):  # 50-bar lookback for median calculation
                start_idx = -(20 + i)
                end_idx = -i if i > 0 else None
                if abs(start_idx) <= len(returns):
                    vol_history.append(np.std(returns[start_idx:end_idx]))

            vol_median = np.median(vol_history) if vol_history else vol_current
            features['vol_regime'] = vol_current / (vol_median + 1e-8)

        if 'parkinson_vol' in self.feature_columns and len(high) >= 20:
            # Parkinson volatility estimator
            hl_ratio = np.log(high / low)
            park_vol = np.sqrt(np.mean(hl_ratio[-20:]**2) / (4 * np.log(2)))  # FIXED: Removed annualization
            features['parkinson_vol'] = park_vol

        if 'vol_forecast' in self.feature_columns and len(close) >= 20:
            # FIXED: EWM std with alpha=0.06 matching training
            returns = np.diff(np.log(close))
            # Compute EWM std manually (matches pandas ewm(alpha=0.06).std())
            alpha = 0.06
            mean_ewm = returns[0]
            var_ewm = 0.0
            for ret in returns:
                mean_ewm = alpha * ret + (1 - alpha) * mean_ewm
                var_ewm = alpha * (ret - mean_ewm)**2 + (1 - alpha) * var_ewm
            features['vol_forecast'] = np.sqrt(var_ewm) if var_ewm > 0 else 0.0

        # EMAs
        for period, col in [(13, 'ema_13'), (21, 'ema_21')]:
            if col in self.feature_columns and len(close) >= period:
                ema = self._ema(close, period)
                features[col] = ema[-1]

        # EMA-based features
        if 'ema_spread' in self.feature_columns:
            if 'ema_13' in features and 'ema_21' in features:
                features['ema_spread'] = features['ema_13'] - features['ema_21']

        if 'ema_ratio' in self.feature_columns:
            if 'ema_13' in features and 'ema_21' in features:
                features['ema_ratio'] = features['ema_13'] / (features['ema_21'] + 1e-8)

        # SMAs
        for period, col in [(20, 'sma_20'), (30, 'sma_30')]:
            if col in self.feature_columns and len(close) >= period:
                sma = np.mean(close[-period:])
                features[col] = sma

        # Trend strength
        if 'trend_strength' in self.feature_columns and len(close) >= 30:
            # FIXED: Normalized distance from SMA-30 matching training
            sma_30 = np.mean(close[-30:])
            features['trend_strength'] = (close[-1] - sma_30) / (sma_30 + 1e-8)

        # Autocorrelation
        if 'autocorr_5' in self.feature_columns and len(close) >= 20:
            # FIXED: Use close prices not returns, matching training
            close_vals = close[-20:]
            if len(close_vals) > 5:
                autocorr = np.corrcoef(close_vals[:-5], close_vals[5:])[0, 1]
                features['autocorr_5'] = autocorr if not np.isnan(autocorr) else 0.0

        # Bollinger Bands
        if 'bb_position' in self.feature_columns and len(close) >= 20:
            sma = np.mean(close[-20:])
            std = np.std(close[-20:])
            upper_bb = sma + 2 * std
            lower_bb = sma - 2 * std
            bb_position = (close[-1] - lower_bb) / (upper_bb - lower_bb + 1e-8)
            features['bb_position'] = bb_position

        # Volume features
        volume = bars_df['volume'].values

        if 'volume_imbalance' in self.feature_columns:
            # FIXED: Price direction ratio matching training
            features['volume_imbalance'] = (close[-1] - open_[-1]) / (high[-1] - low[-1] + 1e-8)

        if 'relative_volume' in self.feature_columns and len(volume) >= 20:
            vol_ma = np.mean(volume[-20:])
            features['relative_volume'] = volume[-1] / (vol_ma + 1.0)

        # VWAP
        if 'price_vs_vwap' in self.feature_columns and len(bars_df) >= 50:
            # FIXED: 50-bar lookback matching training
            typical_price = (high + low + close) / 3.0
            vwap = np.sum(typical_price[-50:] * volume[-50:]) / np.sum(volume[-50:])
            features['price_vs_vwap'] = (close[-1] - vwap) / (vwap + 1e-8)

        # Candle patterns
        if 'large_move' in self.feature_columns:
            # FIXED: Return vs 2× volatility matching training
            log_return_1 = features.get('log_return_1', 0.0)
            vol_20 = features.get('vol_20', 0.0)
            features['large_move'] = float(abs(log_return_1) > 2 * vol_20)

        if 'candle_body' in self.feature_columns:
            features['candle_body'] = abs(close[-1] - open_[-1])

        if 'candle_range' in self.feature_columns:
            features['candle_range'] = high[-1] - low[-1]

        if 'body_pct' in self.feature_columns:
            body = abs(close[-1] - open_[-1])
            candle_range = high[-1] - low[-1]
            features['body_pct'] = body / (candle_range + 1e-8)

        if 'upper_wick' in self.feature_columns:
            features['upper_wick'] = high[-1] - max(open_[-1], close[-1])

        if 'lower_wick' in self.feature_columns:
            features['lower_wick'] = min(open_[-1], close[-1]) - low[-1]

        # Time features
        if len(bars_df) > 0:
            latest_ts = bars_df.index[-1]

            # Minute of day (sin/cos encoding)
            if 'minute_of_day_sin' in self.feature_columns:
                minute_of_day = latest_ts.hour * 60 + latest_ts.minute
                features['minute_of_day_sin'] = np.sin(2 * np.pi * minute_of_day / (24 * 60))

            if 'minute_of_day_cos' in self.feature_columns:
                minute_of_day = latest_ts.hour * 60 + latest_ts.minute
                features['minute_of_day_cos'] = np.cos(2 * np.pi * minute_of_day / (24 * 60))

            # Day of week
            if 'day_of_week' in self.feature_columns:
                features['day_of_week'] = float(latest_ts.dayofweek)

        # Create Series with all features (fill missing with NaN)
        feature_series = pd.Series(index=self.feature_columns, dtype=float)
        for col in self.feature_columns:
            feature_series[col] = features.get(col, np.nan)

        logger.debug(f"Generated {len(feature_series)} features, "
                    f"{feature_series.isna().sum()} NaN values")

        return feature_series

    def _ema(self, values: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate exponential moving average.

        Args:
            values: Array of values
            period: EMA period

        Returns:
            Array of EMA values
        """
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(values, dtype=float)
        ema[0] = values[0]

        for i in range(1, len(values)):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i-1]

        return ema

    def check_feature_quality(self, features: pd.Series) -> Dict[str, bool]:
        """
        Check quality of generated features.

        Args:
            features: Series of feature values

        Returns:
            Dictionary of quality checks
        """
        checks = {}

        # Check for NaN values
        nan_count = features.isna().sum()
        checks['has_nan'] = nan_count > 0
        checks['nan_count'] = int(nan_count)

        # Check for infinite values
        inf_count = np.isinf(features).sum()
        checks['has_inf'] = inf_count > 0
        checks['inf_count'] = int(inf_count)

        # Check for extreme values (> 100 std from mean)
        # This is a rough heuristic
        numeric_features = features.dropna()
        if len(numeric_features) > 0:
            z_scores = np.abs((numeric_features - numeric_features.mean()) /
                             (numeric_features.std() + 1e-8))
            extreme_count = (z_scores > 100).sum()
            checks['has_extreme'] = extreme_count > 0
            checks['extreme_count'] = int(extreme_count)
        else:
            checks['has_extreme'] = False
            checks['extreme_count'] = 0

        checks['healthy'] = not (checks['has_nan'] or checks['has_inf'] or checks['has_extreme'])

        if not checks['healthy']:
            logger.warning(f"Feature quality issues: {checks}")

        return checks
