"""
Grid reindexing module.

Reindexes data to a full bar grid (e.g., every 1m or 5m) and marks
synthetic/missing bars. Tracks missing bar statistics per day.
"""

from typing import Literal, Dict, List, Optional, Any
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def _date_range_left_closed(
    start: pd.Timestamp, end: pd.Timestamp, freq: str, tz: str
) -> pd.DatetimeIndex:
    try:
        return pd.date_range(start=start, end=end, freq=freq, tz=tz, inclusive="left")
    except TypeError:
        delta = pd.Timedelta(freq)
        end_adj = end - delta
        if end_adj < start:
            return pd.DatetimeIndex([], tz=tz)
        return pd.date_range(start=start, end=end_adj, freq=freq, tz=tz)


def _session_from_definitions(
    sessions: List[Any], session_name: str
) -> Dict[str, str]:
    for session in sessions:
        if hasattr(session, "name"):
            name = getattr(session, "name")
            start_time = getattr(session, "start_time")
            end_time = getattr(session, "end_time")
        else:
            name = session.get("name")
            start_time = session.get("start_time")
            end_time = session.get("end_time")
        if name == session_name:
            return {
                "name": name,
                "start_time": start_time,
                "end_time": end_time,
            }
    raise ValueError(f"Session '{session_name}' not found in session definitions")


def build_session_grid(
    start_ts_utc: pd.Timestamp,
    end_ts_utc: pd.Timestamp,
    tz: str,
    sessions: List[Any],
    session_name: str,
    freq: str,
    exclude_weekends: bool = True,
    day_selection_mode: Literal["all_days_in_range", "data_present"] = "all_days_in_range",
    included_dates: Optional[List[pd.Timestamp]] = None,
    raw_data_index: Optional[pd.DatetimeIndex] = None,
    drop_sparse_days: bool = False,
    min_day_coverage_pct: float = 0.90,
    coverage_session: Optional[str] = None,
) -> tuple[pd.DatetimeIndex, List[Dict[str, any]]]:
    if start_ts_utc.tz is None:
        start_ts_utc = start_ts_utc.tz_localize("UTC")
    if end_ts_utc.tz is None:
        end_ts_utc = end_ts_utc.tz_localize("UTC")

    session = _session_from_definitions(sessions, session_name)
    start_hour, start_minute = map(int, session["start_time"].split(":"))
    end_hour, end_minute = map(int, session["end_time"].split(":"))

    start_local = start_ts_utc.tz_convert(tz)
    end_local = end_ts_utc.tz_convert(tz)
    start_date = start_local.date()
    end_date = end_local.date()

    if day_selection_mode == "data_present":
        date_range = included_dates or []
    else:
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    grids = []
    for day in date_range:
        day_date = day.date() if hasattr(day, "date") else day
        if exclude_weekends and day_date.weekday() >= 5:
            continue
        day_str = day_date.isoformat()
        session_start = pd.Timestamp(
            f"{day_str} {start_hour:02d}:{start_minute:02d}", tz=tz
        )
        session_end = pd.Timestamp(
            f"{day_str} {end_hour:02d}:{end_minute:02d}", tz=tz
        )
        if session_end <= session_start:
            session_end = session_end + pd.Timedelta(days=1)

        grids.append(
            _date_range_left_closed(session_start, session_end, freq=freq, tz=tz)
        )

    if not grids:
        return pd.DatetimeIndex([], tz="UTC"), []

    full_local = grids[0].append(grids[1:])
    full_utc = full_local.tz_convert("UTC")
    full_utc = pd.DatetimeIndex(full_utc.unique()).sort_values()

    # Sparse day filtering
    excluded_days = []
    if drop_sparse_days and raw_data_index is not None:
        # Use coverage_session if specified, otherwise use session_name
        coverage_sess_name = coverage_session if coverage_session is not None else session_name
        coverage_sess = _session_from_definitions(sessions, coverage_sess_name)
        coverage_start_hour, coverage_start_minute = map(int, coverage_sess["start_time"].split(":"))
        coverage_end_hour, coverage_end_minute = map(int, coverage_sess["end_time"].split(":"))

        # Group full_utc by session date to compute per-day coverage
        full_local_df = pd.DataFrame(index=full_utc.tz_convert(tz))
        full_local_df["session_date"] = full_local_df.index.date

        days_to_keep = []
        raw_local = raw_data_index.tz_convert(tz)

        for session_date in full_local_df["session_date"].unique():
            session_date_ts = pd.Timestamp(session_date)

            # Define session window for this date
            session_start_local = pd.Timestamp(
                f"{session_date} {coverage_start_hour:02d}:{coverage_start_minute:02d}", tz=tz
            )
            session_end_local = pd.Timestamp(
                f"{session_date} {coverage_end_hour:02d}:{coverage_end_minute:02d}", tz=tz
            )
            if session_end_local <= session_start_local:
                session_end_local = session_end_local + pd.Timedelta(days=1)

            session_start_utc = session_start_local.tz_convert("UTC")
            session_end_utc = session_end_local.tz_convert("UTC")

            # Count expected bars for this session day
            day_grid = full_local_df[full_local_df["session_date"] == session_date]
            expected = len(day_grid)

            if expected == 0:
                continue

            # Count observed bars in raw data for this session window
            raw_in_window = raw_local[(raw_local >= session_start_local) & (raw_local < session_end_local)]
            observed = len(raw_in_window)

            # Compute coverage
            coverage = observed / expected if expected > 0 else 0.0

            if coverage < min_day_coverage_pct:
                excluded_days.append({
                    "date": str(session_date),
                    "coverage": round(coverage, 4),
                    "observed": int(observed),
                    "expected": int(expected),
                })
            else:
                days_to_keep.append(session_date)

        # Filter full_utc to only include kept days
        if excluded_days:
            full_local_df_filtered = full_local_df[full_local_df["session_date"].isin(days_to_keep)]
            full_utc = pd.DatetimeIndex(full_local_df_filtered.index.tz_convert("UTC")).sort_values()

            logger.info(
                "Sparse day exclusion: total_candidates=%d included=%d excluded=%d",
                len(full_local_df["session_date"].unique()),
                len(days_to_keep),
                len(excluded_days),
            )

            # Log worst few excluded days
            excluded_sorted = sorted(excluded_days, key=lambda x: x["coverage"])
            for exc in excluded_sorted[:5]:
                logger.info(
                    "  Excluded: date=%s coverage=%.2f%% observed=%d expected=%d",
                    exc["date"],
                    exc["coverage"] * 100,
                    exc["observed"],
                    exc["expected"],
                )

    return full_utc, excluded_days


def reindex_to_grid(
    df: pd.DataFrame,
    bar_size: Literal["1m", "5m"],
    missing_fill_mode: Literal["nan", "forward_fill"] = "nan",
    forward_fill_max_consecutive: int = 0,
    add_synthetic_flag: bool = True,
    grid_mode: Literal["full_range", "session"] = "full_range",
    session_grid: str = "rth",
    session_timezone: str = "America/Chicago",
    sessions: Optional[List[Any]] = None,
    exclude_weekends: bool = True,
    day_selection_mode: Literal["all_days_in_range", "data_present"] = "all_days_in_range",
    min_rows_per_day: int = 1,
    drop_sparse_days: bool = False,
    min_day_coverage_pct: float = 0.90,
    coverage_session: Optional[str] = None,
) -> tuple[pd.DataFrame, Dict[str, any]]:
    """
    Reindex DataFrame to a full bar grid.

    Creates a complete time grid at the specified bar_size frequency and
    marks any missing/synthetic bars.

    IMPORTANT DEFAULTS (research-grade):
    - missing_fill_mode="nan": Missing bars have NaN OHLCV (no synthetic prices)
    - forward_fill_max_consecutive=0: No forward-filling by default
    - add_synthetic_flag=True: Always mark synthetic rows

    Args:
        df: Standardized OHLCV DataFrame with UTC DatetimeIndex
        bar_size: Bar size ("1m" or "5m")
        missing_fill_mode: How to handle missing bars:
            - "nan": Keep as NaN (DEFAULT - research-grade safe)
            - "forward_fill": Forward fill OHLCV from prior bar (RISKY - creates synthetic prices)
        forward_fill_max_consecutive: Max consecutive missing bars to fill (default 0)
            Only used if missing_fill_mode="forward_fill"
        add_synthetic_flag: Add 'is_synthetic' boolean column

    Returns:
        Tuple of (reindexed_df, metadata_dict) where metadata includes:
        - total_bars: Total bars in grid
        - original_bars: Number of original (non-synthetic) bars
        - synthetic_bars: Number of synthetic/filled bars
        - forward_filled_bars: Number of bars filled via forward-fill (if applicable)
        - missing_pct_per_day: Dict mapping date -> missing %
        - max_gap_bars: Maximum consecutive missing bars
        - missing_fill_mode: Fill mode used
        - forward_fill_max_consecutive: Max consecutive fill limit used

    Raises:
        ValueError: If index is not UTC DatetimeIndex
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")

    if df.index.tz is None or str(df.index.tz) != "UTC":
        raise ValueError("DataFrame index must be UTC timezone-aware")

    logger.info(f"Reindexing to {bar_size} grid (missing_fill_mode: {missing_fill_mode})")

    # Determine frequency
    freq = "1min" if bar_size == "1m" else "5min"

    # Create full date range from first to last timestamp
    start = df.index.min()
    end = df.index.max()

    # Round to bar boundaries
    if bar_size == "1m":
        start = start.floor("1min")
        end = end.ceil("1min")
    else:  # 5m
        start = start.floor("5min")
        end = end.ceil("5min")

    if grid_mode == "session":
        if sessions is None:
            raise ValueError("sessions must be provided for grid_mode='session'")
        start_local = start.tz_convert(session_timezone)
        end_local = end.tz_convert(session_timezone)
        candidate_days = pd.date_range(
            start=start_local.date(), end=end_local.date(), freq="D"
        )
        included_dates = None
        if day_selection_mode == "data_present":
            local_idx = df.index.tz_convert(session_timezone)
            local_dates = pd.Series(local_idx.date)
            counts = local_dates.value_counts()
            included_dates = [
                pd.Timestamp(d)
                for d, count in counts.items()
                if int(count) >= int(min_rows_per_day)
            ]
            included_dates = sorted(included_dates)
            if included_dates:
                logger.info(
                    "Session grid day selection: candidates=%d included=%d "
                    "first=%s last=%s",
                    len(candidate_days),
                    len(included_dates),
                    included_dates[0].date(),
                    included_dates[-1].date(),
                )
            else:
                logger.info(
                    "Session grid day selection: candidates=%d included=0",
                    len(candidate_days),
                )
        else:
            logger.info(
                "Session grid day selection: candidates=%d mode=all_days_in_range",
                len(candidate_days),
            )
        full_index, excluded_sparse_days = build_session_grid(
            start_ts_utc=start,
            end_ts_utc=end,
            tz=session_timezone,
            sessions=sessions,
            session_name=session_grid,
            freq=freq,
            exclude_weekends=exclude_weekends,
            day_selection_mode=day_selection_mode,
            included_dates=included_dates,
            raw_data_index=df.index,
            drop_sparse_days=drop_sparse_days,
            min_day_coverage_pct=min_day_coverage_pct,
            coverage_session=coverage_session,
        )
        logger.info(
            f"Session grid ({session_grid}) in {session_timezone}: "
            f"{len(full_index)} bars from {start} to {end}"
        )
    elif grid_mode == "full_range":
        full_index = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
        excluded_sparse_days = []  # Not applicable for full_range mode
        logger.info(
            f"Full grid: {len(full_index)} bars from {start} to {end}"
        )
    else:
        raise ValueError(f"Unknown grid_mode: {grid_mode}")
    logger.info(f"Original data: {len(df)} bars")

    # Reindex to full grid
    df_reindexed = df.reindex(full_index)

    # Track which bars are synthetic (missing in original)
    is_synthetic = df_reindexed["close"].isna()
    n_synthetic = is_synthetic.sum()
    n_forward_filled = 0

    logger.info(f"Synthetic/missing bars: {n_synthetic} ({n_synthetic/len(full_index)*100:.2f}%)")

    # Apply filling strategy
    if missing_fill_mode == "forward_fill":
        # Forward fill with max consecutive limit
        logger.warning(
            f"Forward-filling enabled with limit={forward_fill_max_consecutive}. "
            "This creates synthetic price paths - use with caution!"
        )

        df_filled = df_reindexed.copy()

        # Track which bars were filled
        pre_fill_na = df_filled["close"].isna().copy()

        for col in ["open", "high", "low", "close"]:
            if col in df_filled.columns:
                # Forward fill with limit
                df_filled[col] = df_filled[col].ffill(limit=forward_fill_max_consecutive)

        # Volume: set to 0 for filled bars (no actual trading occurred)
        if "volume" in df_filled.columns:
            df_filled.loc[pre_fill_na, "volume"] = 0

        # Count how many were actually filled
        post_fill_na = df_filled["close"].isna()
        n_forward_filled = (pre_fill_na & ~post_fill_na).sum()

        logger.info(f"Forward-filled {n_forward_filled} bars (volume set to 0)")

        df_reindexed = df_filled

    elif missing_fill_mode == "nan":
        # Keep NaNs as is (default, research-grade safe)
        logger.info("Missing bars will have NaN OHLCV (no synthetic prices)")
        pass

    else:
        raise ValueError(f"Unknown missing_fill_mode: {missing_fill_mode}")

    # Add synthetic flag column
    if add_synthetic_flag:
        df_reindexed["is_synthetic"] = is_synthetic

    # Compute per-day missing statistics
    missing_pct_per_day = {}

    if len(df_reindexed) > 0:
        # Group by date (in UTC)
        df_reindexed["_date"] = df_reindexed.index.date

        for date, group in df_reindexed.groupby("_date"):
            if add_synthetic_flag:
                n_synthetic_day = group["is_synthetic"].sum()
            else:
                n_synthetic_day = group["close"].isna().sum()

            total_day = len(group)
            pct = (n_synthetic_day / total_day * 100) if total_day > 0 else 0.0
            missing_pct_per_day[str(date)] = round(pct, 2)

        # Drop temp column
        df_reindexed = df_reindexed.drop(columns=["_date"])

    # Compute max gap (consecutive synthetic bars)
    max_gap_bars = 0
    if add_synthetic_flag and n_synthetic > 0:
        # Find runs of True in is_synthetic
        synthetic_series = df_reindexed["is_synthetic"].astype(int)
        gaps = synthetic_series.groupby(
            (synthetic_series != synthetic_series.shift()).cumsum()
        ).sum()
        max_gap_bars = int(gaps.max()) if len(gaps) > 0 else 0

    # Build metadata
    metadata = {
        "total_bars": len(df_reindexed),
        "original_bars": len(df),
        "synthetic_bars": int(n_synthetic),
        "forward_filled_bars": int(n_forward_filled),
        "synthetic_pct": round(n_synthetic / len(df_reindexed) * 100, 2),
        "missing_pct_per_day": missing_pct_per_day,
        "max_gap_bars": max_gap_bars,
        "missing_fill_mode": missing_fill_mode,
        "forward_fill_max_consecutive": forward_fill_max_consecutive,
        "excluded_sparse_days": excluded_sparse_days,
    }

    logger.info(f"Reindexing complete: {metadata['total_bars']} total bars")
    logger.info(f"Synthetic bars: {n_synthetic}, Forward-filled: {n_forward_filled}")
    logger.info(f"Max gap: {max_gap_bars} consecutive bars")

    return df_reindexed, metadata
