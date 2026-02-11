"""
Time-of-Day Filter

Rationale:
- First hour (9:30-10:30 CT): High volatility, wide spreads, news-driven moves
- Last hour (3:00-4:00 CT): Position squaring, liquidity withdrawal, volatility spikes
- Midday (10:30-3:00 CT): More stable, predictable price action

Implementation:
- Only trade during core hours (9:30 AM - 1:30 PM CT by default)
- Avoid toxic first/last hour periods
"""

import pandas as pd
import numpy as np


def apply_time_filter(
    signals_df: pd.DataFrame,
    start_hour: int = 9,
    start_minute: int = 30,
    end_hour: int = 13,
    end_minute: int = 30,
    timezone: str = 'America/Chicago'
) -> pd.DataFrame:
    """
    Filter signals to only trade during specified time window.

    Args:
        signals_df: DataFrame with trading signals (index = timestamps)
        start_hour: Start hour (0-23) in specified timezone
        start_minute: Start minute (0-59)
        end_hour: End hour (0-23) in specified timezone
        end_minute: End minute (0-59)
        timezone: Timezone name (default: 'America/Chicago' for CT)

    Returns:
        Filtered signals DataFrame (subset of signals_df)

    Expected Impact:
        - Avoid toxic first/last hour periods
        - Reduce slippage and news-driven whipsaws
        - More predictable price action
    """

    # Convert index to specified timezone
    signals_df_tz = signals_df.copy()
    if signals_df_tz.index.tz is None:
        signals_df_tz.index = signals_df_tz.index.tz_localize('UTC')
    signals_df_tz.index = signals_df_tz.index.tz_convert(timezone)

    # Create time mask
    valid_times = (
        # After or at start time
        (
            (signals_df_tz.index.hour > start_hour) |
            ((signals_df_tz.index.hour == start_hour) & (signals_df_tz.index.minute >= start_minute))
        ) &
        # Before or at end time
        (
            (signals_df_tz.index.hour < end_hour) |
            ((signals_df_tz.index.hour == end_hour) & (signals_df_tz.index.minute <= end_minute))
        )
    )

    # Filter signals
    signals_filtered = signals_df[valid_times].copy()

    # Log filtering stats
    n_before = len(signals_df)
    n_after = len(signals_filtered)
    pct_kept = 100 * n_after / n_before if n_before > 0 else 0

    print(f"Time filter ({start_hour:02d}:{start_minute:02d} - {end_hour:02d}:{end_minute:02d} {timezone}): "
          f"{n_before} → {n_after} signals ({pct_kept:.1f}% kept)")

    return signals_filtered


def analyze_hourly_performance(
    signals_df: pd.DataFrame,
    returns_column: str = 'ret_net',
    timezone: str = 'America/Chicago'
) -> pd.DataFrame:
    """
    Analyze signal performance by hour of day.

    Args:
        signals_df: DataFrame with signals and returns
        returns_column: Name of returns column
        timezone: Timezone for hour calculation

    Returns:
        DataFrame with hourly statistics (hour, count, mean_return, win_rate, etc.)
    """

    if returns_column not in signals_df.columns:
        raise ValueError(f"Returns column '{returns_column}' not found in signals_df")

    # Convert to timezone
    signals_tz = signals_df.copy()
    if signals_tz.index.tz is None:
        signals_tz.index = signals_tz.index.tz_localize('UTC')
    signals_tz.index = signals_tz.index.tz_convert(timezone)

    # Extract hour
    signals_tz['hour'] = signals_tz.index.hour

    # Calculate hourly stats
    hourly_stats = signals_tz.groupby('hour')[returns_column].agg([
        ('count', 'count'),
        ('mean_return', 'mean'),
        ('std_return', 'std'),
        ('win_rate', lambda x: (x > 0).mean()),
        ('total_return', 'sum')
    ]).round(4)

    return hourly_stats
