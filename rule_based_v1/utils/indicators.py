"""Technical indicator calculations for rule-based trading system.

All functions operate on pandas Series/DataFrames and return Series.
NaN handling: indicators return NaN for insufficient lookback periods.
"""

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands. Returns (upper, middle, lower)."""
    middle = sma(close, period)
    std = close.rolling(window=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def bollinger_position(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """Position within Bollinger Bands (0=lower, 1=upper)."""
    upper, _, lower = bollinger_bands(close, period, num_std)
    band_width = upper - lower
    return (close - lower) / band_width.replace(0, np.nan)


def volume_ratio(volume: pd.Series, lookback: int = 20) -> pd.Series:
    """Current volume as ratio of rolling average volume."""
    avg_vol = sma(volume, lookback)
    return volume / avg_vol.replace(0, np.nan)


def ema_slope(series: pd.Series, period: int, lookback: int = 3) -> pd.Series:
    """Slope of EMA over lookback bars (positive = rising)."""
    ema_vals = ema(series, period)
    return ema_vals.diff(lookback) / lookback


def candle_body(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Absolute candle body size."""
    return (close - open_).abs()


def upper_wick(high: pd.Series, open_: pd.Series, close: pd.Series) -> pd.Series:
    """Upper wick size."""
    return high - pd.concat([open_, close], axis=1).max(axis=1)


def lower_wick(low: pd.Series, open_: pd.Series, close: pd.Series) -> pd.Series:
    """Lower wick size."""
    return pd.concat([open_, close], axis=1).min(axis=1) - low
