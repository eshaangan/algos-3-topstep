"""
Event generation for triple-barrier labeling.

Creates events for each feasible bar under the configured event policy.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _compute_atr(bars_df: pd.DataFrame, atr_period: int) -> pd.Series:
    """Compute ATR using the same true-range definition as features."""
    df = bars_df.sort_index()
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(span=atr_period, adjust=False).mean()


def _get_fill_offsets(fill_price: str) -> Dict[str, int]:
    """
    Determine entry and barrier-start offsets relative to t0.

    - next_bar_open: entry on t0+1 open; barriers start at t0+1
    - next_bar_close: entry on t0+1 close; barriers start at t0+2
    """
    if fill_price == "next_bar_open":
        return {"entry_offset": 1, "barrier_start_offset": 1, "entry_col": "open"}
    if fill_price == "next_bar_close":
        return {"entry_offset": 1, "barrier_start_offset": 2, "entry_col": "close"}
    raise ValueError(f"Unsupported fill_price: {fill_price}")


def _cusum_event_indices(price: pd.Series, threshold: pd.Series) -> np.ndarray:
    """
    Symmetric CUSUM filter (Lopez de Prado style) on price differences.

    Triggers an event when the positive cumulative sum exceeds +threshold or
    the negative cumulative sum exceeds -threshold (with per-bar thresholds).

    Returns integer positions into `price` (0-based).
    """
    if len(price) != len(threshold):
        raise ValueError("CUSUM threshold length must match price length")

    price = price.astype(float)
    threshold = threshold.astype(float)

    s_pos = 0.0
    s_neg = 0.0
    event_idx: List[int] = []

    for i in range(1, len(price)):
        if not np.isfinite(price.iat[i]) or not np.isfinite(price.iat[i - 1]):
            s_pos = 0.0
            s_neg = 0.0
            continue

        h = threshold.iat[i]
        if not np.isfinite(h) or h <= 0.0:
            continue

        diff = price.iat[i] - price.iat[i - 1]
        s_pos = max(0.0, s_pos + diff)
        s_neg = min(0.0, s_neg + diff)

        if s_pos > h:
            s_pos = 0.0
            event_idx.append(i)
        elif s_neg < -h:
            s_neg = 0.0
            event_idx.append(i)

    return np.asarray(event_idx, dtype=int)


def _trend_scan_tstats(
    y: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute rolling t-statistics of linear trend slope for fixed window length.

    Uses x = 0..window-1 and computes t-stat(beta) for OLS slope beta over each
    rolling window y[i:i+window].

    Returns:
        t_stats: array of length len(y) - window + 1 (NaN where insufficient)
        betas:   array of same length
    """
    y = np.asarray(y, dtype=float)
    n = int(window)
    if n < 3:
        raise ValueError("trend scan window must be >= 3")
    if y.size < n:
        return np.array([], dtype=float), np.array([], dtype=float)

    x = np.arange(n, dtype=float)
    x_bar = float(x.mean())
    sxx = float(((x - x_bar) ** 2).sum())
    sum_x = float(x.sum())
    sum_x2 = float((x ** 2).sum())

    ones = np.ones(n, dtype=float)
    sum_y = np.convolve(y, ones, mode="valid")
    sum_y2 = np.convolve(y ** 2, ones, mode="valid")
    # sum_{k=0}^{n-1} k * y[i+k]
    sum_xy = np.convolve(y, x[::-1], mode="valid")

    y_bar = sum_y / n
    sxy = sum_xy - x_bar * sum_y
    beta = sxy / sxx
    alpha = y_bar - beta * x_bar

    # SSE via sums
    sse = (
        sum_y2
        - 2.0 * alpha * sum_y
        - 2.0 * beta * sum_xy
        + (alpha ** 2) * n
        + 2.0 * alpha * beta * sum_x
        + (beta ** 2) * sum_x2
    )
    sse = np.maximum(sse, 0.0)
    denom = (n - 2) * sxx
    with np.errstate(divide="ignore", invalid="ignore"):
        se_beta = np.sqrt(sse / denom)
        t_stat = beta / se_beta
    t_stat = np.where(np.isfinite(t_stat), t_stat, np.nan)
    beta = np.where(np.isfinite(beta), beta, np.nan)
    return t_stat, beta


def generate_events(
    bars_df: pd.DataFrame,
    bar_size: str,
    labeling_config: dict,
    execution_spec: dict,
) -> pd.DataFrame:
    """
    Generate events for triple-barrier labeling.

    Feasibility rules:
    - close at t0 must be non-NaN
    - do not start events on synthetic bars by default (if is_synthetic exists)
    - must have enough future bars to reach vertical barrier t1
    - entry fill price at t0+1 must be non-NaN
    """
    df = bars_df.sort_index()

    required_cols = ["open", "high", "low", "close"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    primary_cfg = labeling_config.get("primary_labeling", {})
    event_policy = primary_cfg.get("event_policy", "every_bar")
    if event_policy not in {"every_bar", "cusum", "trend_scanning"}:
        raise NotImplementedError(f"event_policy not supported: {event_policy}")

    tb_cfg = primary_cfg.get("triple_barrier", {})
    pt_multipliers = tb_cfg.get("pt_multipliers", [1.0])
    sl_multipliers = tb_cfg.get("sl_multipliers", [1.0])
    horizon_map = tb_cfg.get("horizon_bars", {})
    if bar_size not in horizon_map:
        raise ValueError(f"Missing horizon_bars for bar_size={bar_size}")
    horizon_bars_list = horizon_map[bar_size]

    # Validate against execution spec holding constraints
    max_holding = (
        execution_spec.get("holding_constraints", {})
        .get("max_holding_bars", {})
        .get(bar_size)
    )
    if max_holding is not None and max(horizon_bars_list) > max_holding:
        raise ValueError(
            f"horizon_bars max {max(horizon_bars_list)} exceeds execution_spec "
            f"max_holding_bars {max_holding} for {bar_size}"
        )

    atr_period = tb_cfg.get("volatility_params", {}).get("atr_period", 14)
    sigma = _compute_atr(df, atr_period)

    fill_price = (
        execution_spec.get("fill_model", {}).get("fill_price", "next_bar_open")
    )
    offsets = _get_fill_offsets(fill_price)
    entry_offset = offsets["entry_offset"]
    barrier_start_offset = offsets["barrier_start_offset"]
    entry_col = offsets["entry_col"]

    close_ok = df["close"].notna()
    sigma_ok = sigma.notna()
    base_ok = close_ok & sigma_ok

    if "is_synthetic" in df.columns:
        synth = df["is_synthetic"].astype(bool)
        base_ok = base_ok & (~synth)

    entry_price_ok = df[entry_col].shift(-entry_offset).notna()
    base_ok = base_ok & entry_price_ok

    if "is_synthetic" in df.columns:
        entry_synth = df["is_synthetic"].shift(
            -entry_offset, fill_value=False
        ).astype(bool)
        base_ok = base_ok & (~entry_synth)

    n = len(df)
    if event_policy == "every_bar":
        t0_idx_all = np.arange(n, dtype=int)
        sides_all = np.ones(n, dtype=int)
        horizons_all = np.full(n, -1, dtype=int)
    elif event_policy == "cusum":
        cusum_cfg = primary_cfg.get("cusum", {})
        threshold_atr_mult = float(cusum_cfg.get("threshold_atr_mult", 1.0))
        if threshold_atr_mult <= 0.0:
            raise ValueError("primary_labeling.cusum.threshold_atr_mult must be > 0")
        t0_idx_all = _cusum_event_indices(df["close"], sigma * threshold_atr_mult)
        sides_all = np.ones(len(t0_idx_all), dtype=int)
        horizons_all = np.full(len(t0_idx_all), -1, dtype=int)
    else:
        ts_cfg = primary_cfg.get("trend_scanning", {}) or {}
        tstat_threshold = float(ts_cfg.get("tstat_threshold", 2.0))
        if tstat_threshold <= 0.0:
            raise ValueError("primary_labeling.trend_scanning.tstat_threshold must be > 0")

        use_cusum_prefilter = bool(ts_cfg.get("use_cusum_prefilter", True))
        cusum_mult = float(ts_cfg.get("cusum_threshold_atr_mult", 1.0))
        if use_cusum_prefilter and cusum_mult <= 0.0:
            raise ValueError("trend_scanning.cusum_threshold_atr_mult must be > 0")

        close = df["close"].to_numpy(dtype=float)
        start_idx_all = np.arange(n, dtype=int)
        if use_cusum_prefilter:
            start_idx_all = _cusum_event_indices(df["close"], sigma * cusum_mult)

        # Trend windows start where barriers start (execution-aligned)
        trend_start = start_idx_all + barrier_start_offset

        best_abs_t = np.full(len(start_idx_all), np.nan, dtype=float)
        best_beta = np.full(len(start_idx_all), np.nan, dtype=float)
        best_h = np.full(len(start_idx_all), -1, dtype=int)

        for h in horizon_bars_list:
            h = int(h)
            if h < 3:
                continue
            # For each i, window = close[trend_start[i] : trend_start[i] + h]
            # Build via direct convolution on the full series then index-select.
            t_all, beta_all = _trend_scan_tstats(close, window=h)
            valid = (trend_start >= 0) & (trend_start + h <= n) & (
                trend_start < len(t_all)
            )
            if not valid.any():
                continue

            t_by_start = np.full(len(start_idx_all), np.nan, dtype=float)
            beta_by_start = np.full(len(start_idx_all), np.nan, dtype=float)
            idx = trend_start[valid]
            t_by_start[valid] = t_all[idx]
            beta_by_start[valid] = beta_all[idx]
            abs_t_by_start = np.abs(t_by_start)

            improve = np.isfinite(abs_t_by_start) & (
                np.isnan(best_abs_t) | (abs_t_by_start > best_abs_t)
            )
            best_abs_t[improve] = abs_t_by_start[improve]
            best_beta[improve] = beta_by_start[improve]
            best_h[improve] = h

        keep = np.isfinite(best_abs_t) & (best_abs_t >= tstat_threshold) & (best_h > 0)
        t0_idx_all = start_idx_all[keep]
        horizons_all = best_h[keep]
        betas = best_beta[keep]
        sides_all = np.where(betas >= 0.0, 1, -1).astype(int)
    events = []
    base_ok_arr = base_ok.to_numpy()

    for horizon_bars in horizon_bars_list:
        if event_policy == "trend_scanning":
            select = horizons_all == int(horizon_bars)
            t0_pool = t0_idx_all[select]
            side_pool = sides_all[select]
        else:
            t0_pool = t0_idx_all
            side_pool = None

        if len(t0_pool) == 0:
            continue

        t1_idx = t0_pool + barrier_start_offset + int(horizon_bars) - 1
        feasible = (t1_idx < n) & base_ok_arr[t0_pool]

        if not feasible.any():
            continue

        t0_idx = t0_pool[feasible]
        t1_idx = t1_idx[feasible]
        if side_pool is not None:
            side_vals = side_pool[feasible]
        else:
            side_vals = np.ones(len(t0_idx), dtype=int)

        base_df = pd.DataFrame(
            {
                "t0": df.index[t0_idx],
                "t1": df.index[t1_idx],
                "bar_size": bar_size,
                "side": side_vals,
                "sigma": sigma.iloc[t0_idx].to_numpy(),
                "horizon_bars": horizon_bars,
            }
        )

        for pt_mult in pt_multipliers:
            for sl_mult in sl_multipliers:
                df_events = base_df.copy()
                df_events["pt_mult"] = pt_mult
                df_events["sl_mult"] = sl_mult
                events.append(df_events)

    if not events:
        return pd.DataFrame(
            columns=[
                "event_id",
                "t0",
                "t1",
                "bar_size",
                "side",
                "sigma",
                "pt_mult",
                "sl_mult",
                "horizon_bars",
            ]
        )

    events_df = pd.concat(events, axis=0, ignore_index=True)
    events_df.insert(0, "event_id", np.arange(len(events_df), dtype=int))

    return events_df
