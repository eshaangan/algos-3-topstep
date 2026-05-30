"""
Per-bar microstructure feature computation from raw trade ticks.

Shared by:
  - ml_intraday_v3/scripts/build_microstructure_features.py  (batch, from CSV)
  - core/rithmic_client.py                                    (live, from tick stream)

Input DataFrame columns: ts_recv (ns int), price_f (float), size (float), side ('B'/'A'/'N')
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LARGE_TRADE_THRESH = 5   # contracts — trades >= this are "institutional"


def compute_bar_features(grp: pd.DataFrame) -> dict:
    """Compute microstructure features for one bar from its ticks.

    Parameters
    ----------
    grp : DataFrame with columns ts_recv (ns), price_f, size, side (B/A/N)

    Returns
    -------
    dict of feature_name → value, or {} if grp is empty.
    """
    if grp.empty:
        return {}

    price  = grp["price_f"].values
    size   = grp["size"].values
    side   = grp["side"].values
    ts_ns  = grp["ts_recv"].values   # nanoseconds

    n = len(grp)
    buy_mask  = side == "B"
    sell_mask = side == "A"

    buy_vol   = size[buy_mask].sum()
    sell_vol  = size[sell_mask].sum()
    total_vol = size.sum()
    ofi       = buy_vol - sell_vol

    # Large vs small trade split
    large_mask = size >= LARGE_TRADE_THRESH
    small_mask = ~large_mask

    lg_buy  = size[buy_mask  & large_mask].sum()
    lg_sell = size[sell_mask & large_mask].sum()
    sm_buy  = size[buy_mask  & small_mask].sum()
    sm_sell = size[sell_mask & small_mask].sum()

    lg_vol = size[large_mask].sum()
    sm_vol = size[small_mask].sum()

    lg_ofi = lg_buy - lg_sell
    sm_ofi = sm_buy - sm_sell

    # Price return over bar
    p_open  = price[0]
    p_close = price[-1]
    bar_ret = p_close - p_open

    # Kyle's lambda: price impact per unit signed flow
    denom       = abs(ofi) if abs(ofi) > 0 else np.nan
    kyles_lambda = abs(bar_ret) / denom if denom else np.nan

    # Roll spread (effective spread proxy)
    dp = np.diff(price)
    roll_spread = np.nan
    if len(dp) >= 2:
        cov = np.cov(dp[:-1], dp[1:])[0, 1]
        roll_spread = 2 * np.sqrt(max(-cov, 0))

    trade_rate = n / 300.0
    avg_size   = size.mean()
    max_size   = size.max()
    size_std   = size.std() if n > 1 else 0
    large_frac = lg_vol / total_vol if total_vol > 0 else 0

    # Sub-bar OFI slices
    ofi_q      = [0.0] * 5
    ofi_early  = ofi_late = ofi_accel = 0.0
    trade_rate_accel = 0.0

    if ts_ns[-1] > ts_ns[0]:
        bar_dur = ts_ns[-1] - ts_ns[0]

        early_cutoff = ts_ns[0] + bar_dur * 0.1
        late_start   = ts_ns[0] + bar_dur * 0.9
        early_mask   = ts_ns <= early_cutoff
        late_mask    = ts_ns >= late_start

        def _ofi(mask: np.ndarray) -> float:
            b = size[mask & buy_mask].sum()
            s = size[mask & sell_mask].sum()
            return float(b - s)

        ofi_early = _ofi(early_mask)
        ofi_late  = _ofi(late_mask)
        ofi_accel = ofi_late - ofi_early

        for q in range(5):
            q_lo = ts_ns[0] + bar_dur * (q * 0.2)
            q_hi = ts_ns[0] + bar_dur * ((q + 1) * 0.2)
            ofi_q[q] = float(_ofi((ts_ns >= q_lo) & (ts_ns < q_hi)))

        _60s_ns = int(6e10)
        early60_mask = ts_ns <= (ts_ns[0] + min(_60s_ns, int(bar_dur * 0.2)))
        late60_mask  = ts_ns >= (ts_ns[-1] - min(_60s_ns, int(bar_dur * 0.2)))
        early60 = int(early60_mask.sum())
        late60  = int(late60_mask.sum())
        trade_rate_accel = float(late60 / max(early60, 1)) - 1.0

    # Longest consecutive same-side run
    max_run = 1; cur_run = 1
    for i in range(1, len(side)):
        if side[i] == side[i - 1] and side[i] in ("B", "A"):
            cur_run += 1
            max_run  = max(max_run, cur_run)
        else:
            cur_run = 1

    lg_sm_diverge = float((lg_ofi > 0) != (sm_ofi > 0)) if (lg_vol > 0 and sm_vol > 0) else 0.0

    dv   = (price * size).sum()
    vwap = dv / total_vol if total_vol > 0 else p_close

    ofi_imb    = ofi    / total_vol if total_vol > 0 else 0.0
    lg_ofi_imb = lg_ofi / lg_vol   if lg_vol   > 0 else 0.0

    return {
        "buy_vol":          float(buy_vol),
        "sell_vol":         float(sell_vol),
        "total_vol":        float(total_vol),
        "large_vol":        float(lg_vol),
        "large_frac":       float(large_frac),
        "ofi":              float(ofi),
        "ofi_imb":          float(ofi_imb),
        "lg_ofi":           float(lg_ofi),
        "lg_ofi_imb":       float(lg_ofi_imb),
        "sm_ofi":           float(sm_ofi),
        "ofi_accel":        float(ofi_accel),
        "ofi_early":        float(ofi_early),
        "ofi_late":         float(ofi_late),
        "ofi_q1":           ofi_q[0],
        "ofi_q2":           ofi_q[1],
        "ofi_q3":           ofi_q[2],
        "ofi_q4":           ofi_q[3],
        "ofi_q5":           ofi_q[4],
        "lg_sm_diverge":    float(lg_sm_diverge),
        "kyles_lambda":     float(kyles_lambda) if not np.isnan(kyles_lambda) else np.nan,
        "roll_spread":      float(roll_spread)  if not np.isnan(roll_spread)  else np.nan,
        "bar_ret":          float(bar_ret),
        "n_trades":         n,
        "trade_rate":       float(trade_rate),
        "trade_rate_accel": float(trade_rate_accel),
        "avg_size":         float(avg_size),
        "max_size":         float(max_size),
        "size_std":         float(size_std),
        "max_run":          float(max_run),
        "open":             float(p_open),
        "high":             float(price.max()),
        "low":              float(price.min()),
        "close":            float(p_close),
        "vwap":             float(vwap),
    }


def compute_vpin(buy_vol_series: pd.Series, sell_vol_series: pd.Series,
                 windows: tuple = (5, 20)) -> dict[str, pd.Series]:
    """Compute rolling VPIN from bar-level buy/sell volumes."""
    total = (buy_vol_series + sell_vol_series).replace(0, np.nan)
    V = total.mean()
    if np.isnan(V) or V <= 0:
        V = 1.0
    imbalance = (buy_vol_series - sell_vol_series).abs()
    return {f"vpin_{w}": imbalance.rolling(w, min_periods=w).sum() / (w * V)
            for w in windows}
