"""Trend-scanning labeling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class TrendScanResult:
    t0_idx: np.ndarray
    horizon_bars: np.ndarray
    side: np.ndarray
    t_value: np.ndarray


def rolling_slope_tstat(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return rolling OLS slope t-statistics and slopes for fixed window."""
    y = np.asarray(values, dtype=float)
    n = int(window)
    if n < 3:
        raise ValueError("window must be >= 3")
    if y.size < n:
        return np.array([], dtype=float), np.array([], dtype=float)

    x = np.arange(n, dtype=float)
    x_bar = float(x.mean())
    sxx = float(((x - x_bar) ** 2).sum())
    sum_x = float(x.sum())
    sum_x2 = float((x**2).sum())

    ones = np.ones(n, dtype=float)
    sum_y = np.convolve(y, ones, mode="valid")
    sum_y2 = np.convolve(y**2, ones, mode="valid")
    sum_xy = np.convolve(y, x[::-1], mode="valid")

    y_bar = sum_y / n
    sxy = sum_xy - x_bar * sum_y
    beta = sxy / sxx
    alpha = y_bar - beta * x_bar

    sse = (
        sum_y2
        - 2.0 * alpha * sum_y
        - 2.0 * beta * sum_xy
        + (alpha**2) * n
        + 2.0 * alpha * beta * sum_x
        + (beta**2) * sum_x2
    )
    sse = np.maximum(sse, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        se_beta = np.sqrt(sse / ((n - 2) * sxx))
        tstat = beta / se_beta

    return np.where(np.isfinite(tstat), tstat, np.nan), np.where(np.isfinite(beta), beta, np.nan)


def scan_best_trend_horizon(
    close: pd.Series,
    start_idx: np.ndarray,
    horizons: Iterable[int],
    min_t_value: float = 2.0,
    barrier_start_offset: int = 1,
) -> TrendScanResult:
    """Find per-event best horizon and side from maximal absolute trend t-value."""
    close_np = close.to_numpy(dtype=float)
    n = len(close_np)
    starts = np.asarray(start_idx, dtype=int)

    trend_start = starts + int(barrier_start_offset)
    best_abs_t = np.full(len(starts), np.nan, dtype=float)
    best_t = np.full(len(starts), np.nan, dtype=float)
    best_beta = np.full(len(starts), np.nan, dtype=float)
    best_h = np.full(len(starts), -1, dtype=int)

    for h in horizons:
        h = int(h)
        if h < 3:
            continue
        t_all, beta_all = rolling_slope_tstat(close_np, window=h)
        valid = (trend_start >= 0) & (trend_start + h <= n) & (trend_start < len(t_all))
        if not valid.any():
            continue

        t_this = np.full(len(starts), np.nan, dtype=float)
        b_this = np.full(len(starts), np.nan, dtype=float)
        idx = trend_start[valid]
        t_this[valid] = t_all[idx]
        b_this[valid] = beta_all[idx]

        abs_t = np.abs(t_this)
        improve = np.isfinite(abs_t) & (np.isnan(best_abs_t) | (abs_t > best_abs_t))
        best_abs_t[improve] = abs_t[improve]
        best_t[improve] = t_this[improve]
        best_beta[improve] = b_this[improve]
        best_h[improve] = h

    keep = np.isfinite(best_abs_t) & (best_abs_t >= float(min_t_value)) & (best_h > 0)
    kept_starts = starts[keep]
    kept_h = best_h[keep]
    kept_t = best_t[keep]
    kept_side = np.where(best_beta[keep] >= 0.0, 1, -1).astype(int)

    return TrendScanResult(t0_idx=kept_starts, horizon_bars=kept_h, side=kept_side, t_value=kept_t)
