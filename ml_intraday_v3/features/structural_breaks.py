"""Structural break feature proxies (SADF-like + CUSUM)."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import adfuller
except Exception:  # pragma: no cover
    adfuller = None


def rolling_adf_pvalue(series: pd.Series, window: int = 100) -> pd.Series:
    """Rolling ADF p-value proxy for local stationarity shifts."""
    s = pd.to_numeric(series, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if adfuller is None:
        return out
    for i in range(window, len(s) + 1):
        w = s.iloc[i - window : i].dropna()
        if len(w) < max(20, window // 2):
            continue
        out.iloc[i - 1] = float(adfuller(w.values, autolag="AIC")[1])
    return out


def cusum_break_score(series: pd.Series, lookback: int = 100) -> pd.Series:
    """CUSUM-style change score over rolling windows."""
    s = pd.to_numeric(series, errors="coerce").astype(float)
    mean = s.rolling(lookback, min_periods=max(10, lookback // 2)).mean()
    std = s.rolling(lookback, min_periods=max(10, lookback // 2)).std()
    z = (s - mean) / (std + 1e-8)
    return z.cumsum().diff().abs().fillna(0.0)


def build_structural_break_features(close: pd.Series, returns: pd.Series | None = None) -> pd.DataFrame:
    """Build break-detection features for model input."""
    ret = returns if returns is not None else pd.to_numeric(close, errors="coerce").pct_change()
    adf_p = rolling_adf_pvalue(ret, window=120)
    cusum_score = cusum_break_score(ret, lookback=80)
    near_break = ((adf_p > 0.10) | (cusum_score > cusum_score.rolling(200, min_periods=50).quantile(0.9))).astype(int)
    return pd.DataFrame(
        {
            "sb_adf_pvalue": adf_p,
            "sb_cusum_score": cusum_score,
            "sb_near_break": near_break,
        },
        index=close.index,
    )
