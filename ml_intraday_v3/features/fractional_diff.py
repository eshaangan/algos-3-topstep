"""Fractional differentiation helpers for non-stationary time series."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import adfuller
except Exception:  # pragma: no cover - optional dependency
    adfuller = None


def get_fracdiff_weights(d: float, threshold: float = 1e-5, max_size: int = 5000) -> np.ndarray:
    """Compute finite fractional differencing weights."""
    if d < 0:
        raise ValueError("d must be >= 0")
    w = [1.0]
    for k in range(1, int(max_size)):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
    return np.asarray(w, dtype=float)


def fracdiff_series(series: pd.Series, d: float, threshold: float = 1e-5) -> pd.Series:
    """Apply fixed-width fractional differencing to a series."""
    s = pd.to_numeric(series, errors="coerce").astype(float)
    w = get_fracdiff_weights(float(d), threshold=float(threshold))
    out = pd.Series(np.nan, index=s.index, dtype=float)

    arr = s.to_numpy(dtype=float)
    width = len(w)
    for i in range(width - 1, len(arr)):
        window = arr[i - width + 1 : i + 1]
        if np.isnan(window).any():
            continue
        out.iat[i] = float(np.dot(w[::-1], window))
    return out


def find_min_stationary_d(
    series: pd.Series,
    d_candidates: Iterable[float] = (0.2, 0.3, 0.4, 0.5, 0.6),
    pvalue_threshold: float = 0.01,
    threshold: float = 1e-5,
) -> float:
    """Pick smallest d that passes ADF p-value threshold."""
    if adfuller is None:
        return float(next(iter(d_candidates), 0.4))

    for d in d_candidates:
        fd = fracdiff_series(series, d=float(d), threshold=threshold).dropna()
        if len(fd) < 100:
            continue
        pval = adfuller(fd.values, autolag="AIC")[1]
        if pval <= pvalue_threshold:
            return float(d)
    return float(list(d_candidates)[-1])


def apply_fractional_diff(df: pd.DataFrame, columns: Iterable[str], d: float, threshold: float = 1e-5) -> pd.DataFrame:
    """Apply fractional differencing to selected DataFrame columns in-place copy."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = fracdiff_series(out[col], d=d, threshold=threshold)
    return out
