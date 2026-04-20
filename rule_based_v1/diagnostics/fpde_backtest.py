"""FPDE (First-Passage Distortion Engine) Backtest.

Computes the empirical distortion D_t = p_emp - p_null for each 1-min bar,
then uses a walk-forward surface estimate to generate long/short signals on MES
and MNQ 1-min RTH data.

Run:
    python rule_based_v1/diagnostics/fpde_backtest.py
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "processed"
DIAG_DIR = ROOT / "rule_based_v1" / "diagnostics"

MES_PATH = DATA_DIR / "mes_1m_bars_cache.h5"
MNQ_PATH = DATA_DIR / "mnq_2026ytd_1min.h5"
RESULTS_PATH = DIAG_DIR / "fpde_results.json"

# ---------------------------------------------------------------------------
# Instrument constants
# ---------------------------------------------------------------------------
MES_TICK = 0.25
MES_PV   = 5.0     # $ per point
MES_SLIP = 0.25    # 1 tick per side

MNQ_TICK = 0.25
MNQ_PV   = 2.0
MNQ_SLIP = 0.25

N_CONTRACTS = 1

# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------
DELTA_MIN       = 1.5          # minimum barrier extension (points)
SIGMA_WINDOW    = 20           # bars for sigma_hat
CHI_WINDOW      = 15           # bars for choppiness
OMEGA_WINDOW    = 15
R_TILDE_BINS    = 20           # u buckets
TRAIN_DAYS      = 60           # initial training period
LABEL_HORIZON   = 45           # bars for first-passage label
RECOMPUTE_EVERY = 10           # recompute R_tilde every N days

TIME_STOP_BARS  = 15
MAX_TRADES_DAY  = 2
FLAT_MINUTE     = 957          # 15:57 ET = 9*60+57 = 957 min from midnight

# V1 bins
Q_BINS   = [0, 0.3, 0.5, 0.7, 1.0]
B_BINS   = [0, 0.7, 1.0, 1.3, np.inf]
U_BINS   = [0, 0.25, 0.5, 0.75, 1.0]
# chi and omega: median split (2 bins each)

MIN_SAMPLES_V1  = 10
MIN_SAMPLES_V2  = 15

SWEEP_THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
BEST_D_STAR_IDX  = 1   # 0.08 default, updated after sweep


# ===========================================================================
# 1. DATA LOADING
# ===========================================================================

def load_mes() -> pd.DataFrame:
    """Load MES 1-min bars, convert to ET DatetimeIndex, RTH only."""
    with pd.HDFStore(str(MES_PATH), mode="r") as store:
        df = store["/bars_1m"].copy()

    # timestamp column is datetime64[ns, UTC]
    df = df.set_index("timestamp")
    df.index = df.index.tz_convert("US/Eastern")
    df.index.name = "ts"
    df = df.sort_index()

    # RTH only: 09:30–15:59 = min_of_day 570–959
    mod = df.index.hour * 60 + df.index.minute
    df = df[(mod >= 570) & (mod <= 959)].copy()
    df = df[["open", "high", "low", "close", "volume"]]
    return df


def load_mnq() -> pd.DataFrame:
    """Load MNQ 1-min bars (already ET RTH)."""
    with pd.HDFStore(str(MNQ_PATH), mode="r") as store:
        df = store["/bars_1min"].copy()
    df = df.sort_index()
    df.index.name = "ts"
    return df


# ===========================================================================
# 2. SESSION-LEVEL FEATURE COMPUTATION (vectorized per day)
# ===========================================================================

def compute_session_features(day_df: pd.DataFrame) -> pd.DataFrame:
    """Compute all FPDE features for a single RTH session."""
    n = len(day_df)
    if n < 2:
        return pd.DataFrame()

    hi  = day_df["high"].values
    lo  = day_df["low"].values
    cl  = day_df["close"].values
    op  = day_df["open"].values

    # Running session high/low
    H_t = np.maximum.accumulate(hi)
    L_t = np.minimum.accumulate(lo)

    # Session open price
    O_session = op[0]

    # u_t: bar index / (n-1), clipped to [0,1]
    idx = np.arange(n, dtype=float)
    u_t = idx / max(n - 1, 1)

    # sigma_hat: 20-bar mean abs range
    bar_range = hi - lo
    sigma_hat = np.empty(n)
    for i in range(n):
        start = max(0, i - SIGMA_WINDOW + 1)
        window = bar_range[start : i + 1]
        sigma_hat[i] = window.mean() if len(window) > 0 else bar_range[:i+1].mean()

    # delta_t
    delta_t = np.maximum(DELTA_MIN, 1.0 * sigma_hat)

    # Barriers
    U_t = H_t + delta_t
    D_t_barrier = L_t - delta_t

    # Distances
    a_t = U_t - cl
    b_t = cl - D_t_barrier

    # Null first-passage probability
    p_null = b_t / (a_t + b_t + 1e-9)

    # q_t: close location in running range
    q_t = (cl - L_t) / (H_t - L_t + 1e-9)

    # Running range
    running_range = H_t - L_t

    # chi_t: last 15 bars choppiness
    chi_t = np.empty(n)
    for i in range(n):
        start = max(0, i - CHI_WINDOW + 1)
        w_range = bar_range[start : i + 1]
        cum_rr = H_t[i] - L_t[i]
        chi_t[i] = w_range.sum() / (cum_rr + 1e-9)

    # One-sided excursion scores
    wick_up   = hi - np.maximum(op, cl)
    wick_down = np.minimum(op, cl) - lo
    bar_range_sq = bar_range ** 2

    omega_plus  = np.empty(n)
    omega_minus = np.empty(n)
    for i in range(n):
        start = max(0, i - OMEGA_WINDOW + 1)
        wu  = wick_up[start : i + 1]
        wd  = wick_down[start : i + 1]
        brs = bar_range_sq[start : i + 1]
        denom = brs.sum() + 1e-9
        omega_plus[i]  = (wu ** 2).sum() / denom
        omega_minus[i] = (wd ** 2).sum() / denom

    # u bucket index (0..19)
    u_bucket = np.clip((u_t * R_TILDE_BINS).astype(int), 0, R_TILDE_BINS - 1)

    result = pd.DataFrame({
        "open": op,
        "high": hi,
        "low":  lo,
        "close": cl,
        "volume": day_df["volume"].values,
        "H_t": H_t,
        "L_t": L_t,
        "delta_t": delta_t,
        "U_barrier": U_t,
        "D_barrier": D_t_barrier,
        "a_t": a_t,
        "b_t": b_t,
        "p_null": p_null,
        "q_t": q_t,
        "running_range": running_range,
        "chi_t": chi_t,
        "omega_plus": omega_plus,
        "omega_minus": omega_minus,
        "u_t": u_t,
        "u_bucket": u_bucket,
        "bar_range": bar_range,
    }, index=day_df.index)

    return result


# ===========================================================================
# 3. FIRST-PASSAGE LABELS (vectorized per day)
# ===========================================================================

def compute_labels(feat: pd.DataFrame) -> np.ndarray:
    """For each bar i, scan forward up to LABEL_HORIZON bars.
    Returns array of {0, 1, NaN}."""
    n = len(feat)
    hi  = feat["high"].values
    lo  = feat["low"].values
    U   = feat["U_barrier"].values
    D   = feat["D_barrier"].values

    labels = np.full(n, np.nan)

    for i in range(n - 1):
        end = min(i + 1 + LABEL_HORIZON, n)
        fwd_hi = hi[i + 1 : end]
        fwd_lo = lo[i + 1 : end]

        hit_up   = fwd_hi >= U[i]
        hit_down = fwd_lo <= D[i]

        # Find first occurrence of each
        idx_up   = np.argmax(hit_up)   if hit_up.any()   else LABEL_HORIZON
        idx_down = np.argmax(hit_down) if hit_down.any() else LABEL_HORIZON

        if not hit_up.any() and not hit_down.any():
            labels[i] = np.nan
        elif hit_up.any() and (not hit_down.any() or idx_up <= idx_down):
            labels[i] = 1.0
        else:
            labels[i] = 0.0

    return labels


# ===========================================================================
# 4. R_TILDE CURVE (training data only)
# ===========================================================================

def compute_r_tilde(all_feats: List[pd.DataFrame]) -> np.ndarray:
    """Compute median running range per u-bucket from training feature frames."""
    buckets = [[] for _ in range(R_TILDE_BINS)]
    for feat in all_feats:
        for ub, rr in zip(feat["u_bucket"].values, feat["running_range"].values):
            buckets[int(ub)].append(rr)

    r_tilde = np.zeros(R_TILDE_BINS)
    for k, vals in enumerate(buckets):
        r_tilde[k] = np.median(vals) if vals else 1.0
    # fill zeros
    r_tilde = np.where(r_tilde > 0, r_tilde, 1.0)
    return r_tilde


def add_B_t(feat: pd.DataFrame, r_tilde: np.ndarray) -> pd.DataFrame:
    """Add budget ratio B_t column to feature frame."""
    feat = feat.copy()
    r_vals = r_tilde[feat["u_bucket"].values]
    feat["B_t"] = feat["running_range"].values / (r_vals + 1e-9)
    return feat


# ===========================================================================
# 5. SURFACE ESTIMATION (V1 and V2)
# ===========================================================================

def _digitize_safe(x: np.ndarray, bins: list) -> np.ndarray:
    """0-indexed bin assignment, last bin = len(bins)-2."""
    out = np.digitize(x, bins[1:-1])  # returns 0..len(bins)-2
    return out.clip(0, len(bins) - 2)


def build_surface_v1(
    all_feats: List[pd.DataFrame],
    all_labels: List[np.ndarray],
) -> Dict[Tuple, float]:
    """Build empirical p(upper hit first) for each (q_bin, B_bin, chi_bin, u_bin)."""
    # Collect chi median for split
    chi_vals = np.concatenate([f["chi_t"].values for f in all_feats])
    chi_median = np.median(chi_vals)

    counts  = {}
    pos     = {}

    for feat, labels in zip(all_feats, all_labels):
        valid = ~np.isnan(labels)
        if not valid.any():
            continue

        q   = feat["q_t"].values[valid]
        B   = feat["B_t"].values[valid]
        chi = feat["chi_t"].values[valid]
        u   = feat["u_t"].values[valid]
        lbl = labels[valid]

        q_bin   = _digitize_safe(q, Q_BINS)
        B_bin   = _digitize_safe(B, B_BINS)
        chi_bin = (chi > chi_median).astype(int)
        u_bin   = _digitize_safe(u, U_BINS)

        for qb, bb, cb, ub, lb in zip(q_bin, B_bin, chi_bin, u_bin, lbl):
            key = (int(qb), int(bb), int(cb), int(ub))
            counts[key] = counts.get(key, 0) + 1
            pos[key]    = pos.get(key, 0) + lb

    surface = {}
    for key in counts:
        if counts[key] >= MIN_SAMPLES_V1:
            surface[key] = pos[key] / counts[key]
    return surface, chi_median


def build_surface_v2(
    all_feats: List[pd.DataFrame],
    all_labels: List[np.ndarray],
) -> Dict[Tuple, float]:
    """V2: adds omega_plus and omega_minus as 2-bin splits (512 bins total)."""
    chi_vals        = np.concatenate([f["chi_t"].values for f in all_feats])
    op_vals         = np.concatenate([f["omega_plus"].values for f in all_feats])
    om_vals         = np.concatenate([f["omega_minus"].values for f in all_feats])

    chi_median = np.median(chi_vals)
    op_median  = np.median(op_vals)
    om_median  = np.median(om_vals)

    counts = {}
    pos    = {}

    for feat, labels in zip(all_feats, all_labels):
        valid = ~np.isnan(labels)
        if not valid.any():
            continue

        q   = feat["q_t"].values[valid]
        B   = feat["B_t"].values[valid]
        chi = feat["chi_t"].values[valid]
        u   = feat["u_t"].values[valid]
        op  = feat["omega_plus"].values[valid]
        om  = feat["omega_minus"].values[valid]
        lbl = labels[valid]

        q_bin   = _digitize_safe(q, Q_BINS)
        B_bin   = _digitize_safe(B, B_BINS)
        chi_bin = (chi > chi_median).astype(int)
        u_bin   = _digitize_safe(u, U_BINS)
        op_bin  = (op > op_median).astype(int)
        om_bin  = (om > om_median).astype(int)

        for qb, bb, cb, ub, opb, omb, lb in zip(
            q_bin, B_bin, chi_bin, u_bin, op_bin, om_bin, lbl
        ):
            key = (int(qb), int(bb), int(cb), int(ub), int(opb), int(omb))
            counts[key] = counts.get(key, 0) + 1
            pos[key]    = pos.get(key, 0) + lb

    surface = {}
    for key in counts:
        if counts[key] >= MIN_SAMPLES_V2:
            surface[key] = pos[key] / counts[key]
    return surface, chi_median, op_median, om_median


def lookup_distortion_v1(
    feat: pd.DataFrame,
    surface: Dict,
    chi_median: float,
) -> np.ndarray:
    """Compute D_t = p_emp - p_null for each bar."""
    n = len(feat)
    q   = feat["q_t"].values
    B   = feat["B_t"].values
    chi = feat["chi_t"].values
    u   = feat["u_t"].values
    p_null = feat["p_null"].values

    q_bin   = _digitize_safe(q, Q_BINS)
    B_bin   = _digitize_safe(B, B_BINS)
    chi_bin = (chi > chi_median).astype(int)
    u_bin   = _digitize_safe(u, U_BINS)

    D = np.zeros(n)
    for i in range(n):
        key = (int(q_bin[i]), int(B_bin[i]), int(chi_bin[i]), int(u_bin[i]))
        p_emp = surface.get(key, p_null[i])
        D[i] = p_emp - p_null[i]
    return D


def lookup_distortion_v2(
    feat: pd.DataFrame,
    surface_v2: Dict,
    surface_v1: Dict,
    chi_median: float,
    op_median: float,
    om_median: float,
) -> np.ndarray:
    """V2 distortion with V1 fallback."""
    n = len(feat)
    q   = feat["q_t"].values
    B   = feat["B_t"].values
    chi = feat["chi_t"].values
    u   = feat["u_t"].values
    op  = feat["omega_plus"].values
    om  = feat["omega_minus"].values
    p_null = feat["p_null"].values

    q_bin   = _digitize_safe(q, Q_BINS)
    B_bin   = _digitize_safe(B, B_BINS)
    chi_bin = (chi > chi_median).astype(int)
    u_bin   = _digitize_safe(u, U_BINS)
    op_bin  = (op > op_median).astype(int)
    om_bin  = (om > om_median).astype(int)

    D = np.zeros(n)
    for i in range(n):
        key_v2 = (
            int(q_bin[i]), int(B_bin[i]), int(chi_bin[i]),
            int(u_bin[i]), int(op_bin[i]), int(om_bin[i])
        )
        key_v1 = (int(q_bin[i]), int(B_bin[i]), int(chi_bin[i]), int(u_bin[i]))

        if key_v2 in surface_v2:
            p_emp = surface_v2[key_v2]
        elif key_v1 in surface_v1:
            p_emp = surface_v1[key_v1]
        else:
            p_emp = p_null[i]
        D[i] = p_emp - p_null[i]
    return D


# ===========================================================================
# 6. MONOTONICITY TABLE
# ===========================================================================

def print_monotonicity_table(
    all_test_feats: List[pd.DataFrame],
    all_test_labels: List[np.ndarray],
    all_D: List[np.ndarray],
) -> List[dict]:
    """Bin D_t into 10 buckets [-0.5, 0.5], report p_actual, p_null, mean_D."""
    edges = np.linspace(-0.5, 0.5, 11)

    bucket_n       = np.zeros(10, dtype=int)
    bucket_hit     = np.zeros(10)
    bucket_p_null  = np.zeros(10)
    bucket_D_sum   = np.zeros(10)

    for feat, labels, D in zip(all_test_feats, all_test_labels, all_D):
        valid = ~np.isnan(labels)
        if not valid.any():
            continue
        lbl    = labels[valid]
        d      = D[valid]
        pn     = feat["p_null"].values[valid]

        bins = np.digitize(d, edges[1:-1])  # 0..9
        bins = bins.clip(0, 9)
        for i, (b, l, pnv) in enumerate(zip(bins, lbl, pn)):
            bucket_n[b]      += 1
            bucket_hit[b]    += l
            bucket_p_null[b] += pnv
            bucket_D_sum[b]  += d[i]

    rows = []
    print("\n=== MONOTONICITY TABLE ===")
    print(f"{'Bucket':>20}  {'N':>6}  {'p_actual':>9}  {'p_null':>7}  {'mean_D':>8}")
    print("-" * 60)
    for k in range(10):
        lo_edge = edges[k]
        hi_edge = edges[k + 1]
        n       = bucket_n[k]
        p_act   = (bucket_hit[k] / n) if n > 0 else float("nan")
        p_null  = (bucket_p_null[k] / n) if n > 0 else float("nan")
        mean_D  = (bucket_D_sum[k] / n) if n > 0 else float("nan")
        label   = f"[{lo_edge:+.2f}, {hi_edge:+.2f})"
        print(f"  {label:>18}  {n:>6}  {p_act:>9.4f}  {p_null:>7.4f}  {mean_D:>8.4f}")
        rows.append({
            "bucket": label,
            "N": int(n),
            "p_actual": round(float(p_act), 4) if not np.isnan(p_act) else None,
            "p_null":   round(float(p_null), 4) if not np.isnan(p_null) else None,
            "mean_D":   round(float(mean_D), 4) if not np.isnan(mean_D) else None,
        })
    return rows


# ===========================================================================
# 7. BACKTESTING ENGINE
# ===========================================================================

def run_backtest(
    days: List[pd.DataFrame],           # list of feature DataFrames (test set)
    D_arrays: List[np.ndarray],         # distortion array per day
    d_star: float,
    tick_size: float,
    point_value: float,
    slippage: float,
    chi_rolling: Dict[int, float],      # day_idx -> 20-day rolling chi median
    op_rolling: Dict[int, float],       # day_idx -> 20-day rolling omega_plus p75
    om_rolling: Dict[int, float],       # day_idx -> 20-day rolling omega_minus p75
) -> Tuple[List[dict], dict]:
    """Run single-threshold backtest. Returns (trades, metrics)."""
    trades       = []
    equity       = 0.0
    equity_curve = []
    daily_pnl    = []

    for day_idx, (feat, D) in enumerate(zip(days, D_arrays)):
        n = len(feat)
        if n < 5:
            continue

        hi    = feat["high"].values
        lo    = feat["low"].values
        cl    = feat["close"].values
        q     = feat["q_t"].values
        B     = feat["B_t"].values
        chi   = feat["chi_t"].values
        op_exc= feat["omega_plus"].values
        om_exc= feat["omega_minus"].values
        delta = feat["delta_t"].values
        U_bar = feat["U_barrier"].values
        D_bar = feat["D_barrier"].values
        ts    = feat.index

        chi_med  = chi_rolling.get(day_idx, np.median(chi))
        op_p75   = op_rolling.get(day_idx, np.percentile(op_exc, 75))
        om_p75   = om_rolling.get(day_idx, np.percentile(om_exc, 75))

        trades_today   = 0
        active         = None   # dict with entry info
        day_equity     = 0.0

        for i in range(3, n):
            bar_ts = ts[i]
            bar_mod = bar_ts.hour * 60 + bar_ts.minute

            # Hard flat at 15:57
            if bar_mod >= FLAT_MINUTE:
                if active is not None:
                    exit_px = cl[i]
                    pnl = (exit_px - active["entry"]) * active["direction"] * point_value * N_CONTRACTS
                    pnl -= 2 * slippage * point_value * N_CONTRACTS
                    day_equity += pnl
                    trades.append({**active, "exit": float(exit_px),
                                   "exit_reason": "flat", "pnl": round(pnl, 2),
                                   "bars_held": i - active["entry_bar"]})
                    active = None
                continue

            # --- Manage open trade ---
            if active is not None:
                d   = active["direction"]
                ep  = active["entry"]
                tp  = active["tp"]
                sl  = active["sl"]
                eb  = active["entry_bar"]
                dlt = active["delta"]

                # Check TP / SL using bar high/low
                hit_tp = (d ==  1 and hi[i] >= tp) or (d == -1 and lo[i] <= tp)
                hit_sl = (d ==  1 and lo[i] <= sl) or (d == -1 and hi[i] >= sl)
                time_stop = (i - eb) >= TIME_STOP_BARS
                sign_flip = (d == 1 and D[i] < 0) or (d == -1 and D[i] > 0)

                if hit_tp or hit_sl or time_stop or sign_flip:
                    if hit_tp:
                        exit_px = tp
                        reason  = "tp"
                    elif hit_sl:
                        exit_px = sl
                        reason  = "sl"
                    elif sign_flip:
                        exit_px = cl[i]
                        reason  = "sign_flip"
                    else:
                        exit_px = cl[i]
                        reason  = "time_stop"

                    pnl = (exit_px - ep) * d * point_value * N_CONTRACTS
                    pnl -= 2 * slippage * point_value * N_CONTRACTS
                    day_equity += pnl
                    trades.append({
                        **active,
                        "exit": float(exit_px),
                        "exit_reason": reason,
                        "pnl": round(pnl, 2),
                        "bars_held": i - eb,
                    })
                    active = None
                continue   # no new entry if just exited or still in trade

            # --- Entry conditions ---
            if trades_today >= MAX_TRADES_DAY:
                continue
            if active is not None:
                continue

            Dv = D[i]

            # Long
            if Dv > d_star:
                cond_q   = q[i] > 0.55
                cond_B   = B[i] < 1.00
                cond_chi = chi[i] < chi_med
                cond_op  = op_exc[i] < op_p75
                # 3-bar high breakout
                cond_bo  = cl[i] > np.max(hi[i-3:i])
                if cond_q and cond_B and cond_chi and cond_op and cond_bo:
                    entry_px = cl[i] + slippage
                    tp_px    = entry_px + delta[i]
                    sl_px    = entry_px - 0.65 * delta[i]
                    active   = {
                        "date": str(bar_ts.date()),
                        "entry_time": str(bar_ts),
                        "direction": 1,
                        "entry": float(entry_px),
                        "tp": float(tp_px),
                        "sl": float(sl_px),
                        "delta": float(delta[i]),
                        "D_at_entry": float(Dv),
                        "entry_bar": i,
                    }
                    trades_today += 1

            # Short
            elif Dv < -d_star:
                cond_q   = q[i] < 0.45
                cond_B   = B[i] < 1.00
                cond_chi = chi[i] < chi_med
                cond_om  = om_exc[i] < om_p75
                # 3-bar low breakout
                cond_bo  = cl[i] < np.min(lo[i-3:i])
                if cond_q and cond_B and cond_chi and cond_om and cond_bo:
                    entry_px = cl[i] - slippage
                    tp_px    = entry_px - delta[i]
                    sl_px    = entry_px + 0.65 * delta[i]
                    active   = {
                        "date": str(bar_ts.date()),
                        "entry_time": str(bar_ts),
                        "direction": -1,
                        "entry": float(entry_px),
                        "tp": float(tp_px),
                        "sl": float(sl_px),
                        "delta": float(delta[i]),
                        "D_at_entry": float(Dv),
                        "entry_bar": i,
                    }
                    trades_today += 1

        # Close any remaining trade at end of day
        if active is not None:
            exit_px = cl[-1]
            pnl = (exit_px - active["entry"]) * active["direction"] * point_value * N_CONTRACTS
            pnl -= 2 * slippage * point_value * N_CONTRACTS
            day_equity += pnl
            trades.append({
                **active,
                "exit": float(exit_px),
                "exit_reason": "eod",
                "pnl": round(pnl, 2),
                "bars_held": n - 1 - active["entry_bar"],
            })
            active = None

        equity += day_equity
        equity_curve.append(equity)
        daily_pnl.append(day_equity)

    metrics = compute_metrics(trades, daily_pnl, equity_curve, d_star)
    return trades, metrics


def compute_metrics(
    trades: List[dict],
    daily_pnl: List[float],
    equity_curve: List[float],
    d_star: float,
) -> dict:
    n = len(trades)
    if n == 0:
        return {
            "d_star": d_star, "N": 0, "WR": 0,
            "AvgPnL": 0, "Sharpe": 0, "MaxDD": 0,
            "AvgD_at_entry": 0, "TotalPnL": 0,
        }

    pnls    = np.array([t["pnl"] for t in trades])
    wins    = (pnls > 0).sum()
    wr      = wins / n
    avg_pnl = pnls.mean()

    daily   = np.array(daily_pnl)
    sharpe  = (daily.mean() / (daily.std() + 1e-9)) * np.sqrt(252) if len(daily) > 1 else 0.0

    # Max drawdown
    eq = np.array([0.0] + equity_curve)
    running_max = np.maximum.accumulate(eq)
    dd  = eq - running_max
    max_dd = dd.min()

    avg_D = np.mean([t["D_at_entry"] for t in trades])

    return {
        "d_star": d_star,
        "N": n,
        "WR": round(wr * 100, 1),
        "AvgPnL": round(avg_pnl, 2),
        "Sharpe": round(float(sharpe), 3),
        "MaxDD": round(float(max_dd), 2),
        "AvgD_at_entry": round(float(avg_D), 4),
        "TotalPnL": round(float(pnls.sum()), 2),
    }


# ===========================================================================
# 8. STRIPPED-MODE BACKTESTING ENGINE
# ===========================================================================

STRIPPED_THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15]

# Session start minute-of-day: 9:30 = 570
# 2 hours into session = 11:30 = 690, 3 hours = 12:30 = 750
U_T_LOW  = 0.20   # lower bound for S3 mid-session filter
U_T_HIGH = 0.70   # upper bound for S3 mid-session filter

# q_t thresholds for S2 price-location filter
Q_LONG_THRESH  = 0.55
Q_SHORT_THRESH = 0.45


def run_stripped_backtest(
    days: List[pd.DataFrame],
    D_arrays: List[np.ndarray],
    d_star: float,
    tick_size: float,
    point_value: float,
    slippage: float,
    version: str,   # "S1", "S2", "S3", "S4", or "S5"
    tp_frac: float = 1.0,  # TP as fraction of delta_t
    no_sl: bool = False,   # if True, skip hard SL (time stop only)
) -> Tuple[List[dict], dict]:
    """Stripped backtest: symmetric 1:1 RR, TP=SL=tp_frac*delta_t, time_stop=15 bars.

    S1: |D_t| > d* only, TP=SL=delta
    S2: S1 + price location (q_t > 0.55 long, q_t < 0.45 short)
    S3: S1 + u_t in [0.20, 0.70] (mid-session only)
    S4: S1 + TP=SL=0.5*delta_t (half-target symmetric)
    S5: S1 + TP=0.5*delta_t, NO hard SL (time stop only) — attempts to capture 81-83% MFE accuracy
    """
    trades       = []
    equity       = 0.0
    equity_curve = []
    daily_pnl    = []

    for day_idx, (feat, D) in enumerate(zip(days, D_arrays)):
        n = len(feat)
        if n < 5:
            continue

        hi    = feat["high"].values
        lo    = feat["low"].values
        cl    = feat["close"].values
        q     = feat["q_t"].values
        u     = feat["u_t"].values
        delta = feat["delta_t"].values
        ts    = feat.index

        trades_today = 0
        active       = None
        day_equity   = 0.0

        for i in range(1, n):
            bar_ts  = ts[i]
            bar_mod = bar_ts.hour * 60 + bar_ts.minute

            # Hard flat at 15:57
            if bar_mod >= FLAT_MINUTE:
                if active is not None:
                    exit_px = cl[i]
                    pnl = (exit_px - active["entry"]) * active["direction"] * point_value * N_CONTRACTS
                    pnl -= 2 * slippage * point_value * N_CONTRACTS
                    day_equity += pnl
                    trades.append({**active, "exit": float(exit_px),
                                   "exit_reason": "flat", "pnl": round(pnl, 2),
                                   "bars_held": i - active["entry_bar"]})
                    active = None
                continue

            # --- Manage open trade ---
            if active is not None:
                d   = active["direction"]
                ep  = active["entry"]
                tp  = active["tp"]
                sl  = active["sl"]
                eb  = active["entry_bar"]

                hit_tp    = (d ==  1 and hi[i] >= tp) or (d == -1 and lo[i] <= tp)
                hit_sl    = (not no_sl) and (
                    (d ==  1 and lo[i] <= sl) or (d == -1 and hi[i] >= sl)
                )
                time_stop = (i - eb) >= TIME_STOP_BARS

                if hit_tp or hit_sl or time_stop:
                    if hit_tp:
                        exit_px = tp
                        reason  = "tp"
                    elif hit_sl:
                        exit_px = sl
                        reason  = "sl"
                    else:
                        exit_px = cl[i]
                        reason  = "time_stop"

                    pnl = (exit_px - ep) * d * point_value * N_CONTRACTS
                    pnl -= 2 * slippage * point_value * N_CONTRACTS
                    day_equity += pnl
                    trades.append({
                        **active,
                        "exit": float(exit_px),
                        "exit_reason": reason,
                        "pnl": round(pnl, 2),
                        "bars_held": i - eb,
                    })
                    active = None
                continue   # no new entry while in trade or just exited

            # --- Entry conditions ---
            if trades_today >= MAX_TRADES_DAY:
                continue

            Dv = D[i]
            abs_D = abs(Dv)

            if abs_D <= d_star:
                continue

            direction = 1 if Dv > 0 else -1

            # Version-specific additional filters
            if version == "S2":
                if direction == 1 and q[i] <= Q_LONG_THRESH:
                    continue
                if direction == -1 and q[i] >= Q_SHORT_THRESH:
                    continue
            elif version == "S3":
                if not (U_T_LOW <= u[i] <= U_T_HIGH):
                    continue

            # Symmetric 1:1 RR: SL = TP = tp_frac * delta_t
            dlt     = delta[i]
            tgt_dlt = dlt * tp_frac
            if direction == 1:
                entry_px = cl[i] + slippage
                tp_px    = entry_px + tgt_dlt
                sl_px    = entry_px - tgt_dlt
            else:
                entry_px = cl[i] - slippage
                tp_px    = entry_px - tgt_dlt
                sl_px    = entry_px + tgt_dlt

            active = {
                "date":       str(bar_ts.date()),
                "entry_time": str(bar_ts),
                "direction":  direction,
                "entry":      float(entry_px),
                "tp":         float(tp_px),
                "sl":         float(sl_px),
                "delta":      float(dlt),
                "D_at_entry": float(Dv),
                "entry_bar":  i,
            }
            trades_today += 1

        # Close any remaining trade at end of day
        if active is not None:
            exit_px = cl[-1]
            pnl = (exit_px - active["entry"]) * active["direction"] * point_value * N_CONTRACTS
            pnl -= 2 * slippage * point_value * N_CONTRACTS
            day_equity += pnl
            trades.append({
                **active,
                "exit": float(exit_px),
                "exit_reason": "eod",
                "pnl": round(pnl, 2),
                "bars_held": n - 1 - active["entry_bar"],
            })
            active = None

        equity += day_equity
        equity_curve.append(equity)
        daily_pnl.append(day_equity)

    metrics = compute_metrics(trades, daily_pnl, equity_curve, d_star)
    return trades, metrics


def print_directional_accuracy_table(
    days: List[pd.DataFrame],
    D_arrays: List[np.ndarray],
    d_star_max_trades: float,
) -> List[dict]:
    """Bin bars by |D_t| and report directional accuracy.

    A bar is counted only where |D_t| >= d_star_max_trades.
    'Win' = price moved in direction of D_t by at least delta_t/2 within 15 bars.
    """
    bins = [
        (0.05, 0.10),
        (0.10, 0.15),
        (0.15, 0.20),
        (0.20, float("inf")),
    ]

    bin_n   = [0] * len(bins)
    bin_win = [0] * len(bins)

    for feat, D in zip(days, D_arrays):
        n   = len(feat)
        hi  = feat["high"].values
        lo  = feat["low"].values
        cl  = feat["close"].values
        dlt = feat["delta_t"].values

        for i in range(n - 1):
            Dv     = D[i]
            abs_D  = abs(Dv)
            if abs_D < 0.05:
                continue

            direction = 1 if Dv > 0 else -1
            target    = cl[i] + direction * dlt[i] * 0.5

            end  = min(i + 1 + TIME_STOP_BARS, n)
            won  = False
            for j in range(i + 1, end):
                if direction == 1 and hi[j] >= target:
                    won = True
                    break
                if direction == -1 and lo[j] <= target:
                    won = True
                    break

            for b_idx, (lo_b, hi_b) in enumerate(bins):
                if lo_b <= abs_D < hi_b:
                    bin_n[b_idx]   += 1
                    bin_win[b_idx] += int(won)
                    break

    rows = []
    print("\n=== DIRECTIONAL ACCURACY TABLE (|D_t| bins, 15-bar horizon, delta/2 target) ===")
    print(f"  {'Bin':>14}  {'N':>6}  {'WR%':>7}")
    print("  " + "-" * 32)
    bin_labels = ["[0.05, 0.10)", "[0.10, 0.15)", "[0.15, 0.20)", "[0.20, inf)"]
    for b_idx in range(len(bins)):
        n_b  = bin_n[b_idx]
        wr_b = (bin_win[b_idx] / n_b * 100) if n_b > 0 else float("nan")
        label = bin_labels[b_idx]
        print(f"  {label:>14}  {n_b:>6}  {wr_b:>6.1f}%")
        rows.append({
            "bin": label,
            "N":   n_b,
            "WR":  round(wr_b, 1) if n_b > 0 else None,
        })
    return rows


def run_raw_signal_backtest(
    days: List[pd.DataFrame],
    D_arrays: List[np.ndarray],
    point_value: float,
    slippage: float,
) -> List[dict]:
    """Trade every qualifying bar as an independent position (mirrors the directional
    accuracy table exactly but adds real PnL accounting).

    For each bar where |D_t| >= threshold:
      - Enter at close + slippage (direction = sign(D_t))
      - TP  = entry + direction * 0.5 * delta_t
      - Exit at TP if hit within 15 bars, else exit at bar-15 close (time stop)
      - No hard SL, no max-trades-per-day limit
      - Overlapping positions are treated as independent (each bar is a separate bet)
    """
    d_star_vals = [0.05, 0.08, 0.10, 0.15, 0.20]
    rows = []

    print("\n=== RAW SIGNAL BACKTEST (every qualifying bar = independent trade) ===")
    print(f"  {'d*':>5}  {'N':>7}  {'WR%':>6}  {'AvgPnL':>8}  {'TotalPnL':>10}  {'AvgWin':>8}  {'AvgLoss':>9}")
    print("  " + "-" * 70)

    for d_star in d_star_vals:
        trade_pnls = []
        wins = 0

        for feat, D in zip(days, D_arrays):
            n   = len(feat)
            hi  = feat["high"].values
            lo  = feat["low"].values
            cl  = feat["close"].values
            dlt = feat["delta_t"].values

            for i in range(n - 1):
                Dv = D[i]
                if abs(Dv) < d_star:
                    continue

                direction = 1 if Dv > 0 else -1
                entry_px  = cl[i] + direction * slippage
                target    = entry_px + direction * 0.5 * dlt[i]

                end  = min(i + 1 + TIME_STOP_BARS, n)
                exit_px = cl[end - 1]   # default: time stop at last bar close

                for j in range(i + 1, end):
                    if direction == 1 and hi[j] >= target:
                        exit_px = target
                        break
                    if direction == -1 and lo[j] <= target:
                        exit_px = target
                        break

                pnl = (exit_px - entry_px) * direction * point_value
                pnl -= 2 * slippage * point_value
                trade_pnls.append(pnl)
                if pnl > 0:
                    wins += 1

        n_trades = len(trade_pnls)
        if n_trades == 0:
            continue
        arr     = np.array(trade_pnls)
        wr      = wins / n_trades * 100
        avg_pnl = arr.mean()
        total   = arr.sum()
        avg_win  = arr[arr > 0].mean() if (arr > 0).any() else 0.0
        avg_loss = arr[arr <= 0].mean() if (arr <= 0).any() else 0.0

        print(
            f"  {d_star:.2f}  {n_trades:>7}  {wr:>5.1f}%  "
            f"${avg_pnl:>7.4f}  ${total:>9.2f}  ${avg_win:>7.4f}  ${avg_loss:>8.4f}"
        )
        rows.append({
            "d_star": d_star, "N": n_trades, "WR": round(wr, 1),
            "AvgPnL": round(avg_pnl, 4), "TotalPnL": round(total, 2),
            "AvgWin": round(avg_win, 4), "AvgLoss": round(avg_loss, 4),
        })

    return rows


def run_raw_optimization_sweep(
    days: List[pd.DataFrame],
    D_arrays: List[np.ndarray],
    point_value: float,
    slippage: float,
) -> List[dict]:
    """3D grid sweep over (tp_frac, sl_frac, time_stop_bars) using raw signal logic.

    For each combination:
      - d_star = 0.08 (fixed)
      - Every qualifying bar (|D_t| >= d_star) = independent trade
      - Entry at close + direction * slippage
      - TP = entry + direction * tp_frac * delta_t
      - SL = entry - direction * sl_frac * delta_t  (skipped if sl_frac is None)
      - Exit at TP or SL if hit within time_stop_bars bars (check high/low)
      - If both TP and SL hit in same bar, TP wins
      - Otherwise exit at close of the time_stop_bars-th bar
    """
    D_STAR = 0.08

    tp_fracs       = [0.25, 0.50, 0.75, 1.0]
    sl_fracs       = [0.5, 1.0, 1.5, 2.0, 3.0, None]
    time_stop_vals = [3, 5, 7, 10, 15]

    rows = []

    print("\n=== OPTIMIZATION SWEEP (tp_frac × sl_frac × time_stop_bars, d*=0.08) ===")
    print(
        f"  {'tp_frac':>7}  {'sl_frac':>7}  {'ts_bars':>7}  "
        f"{'N':>6}  {'WR%':>6}  {'AvgPnL':>9}  {'TotalPnL':>10}  "
        f"{'AvgWin':>8}  {'AvgLoss':>9}"
    )
    print("  " + "-" * 90)

    for tp_frac in tp_fracs:
        for sl_frac in sl_fracs:
            for ts_bars in time_stop_vals:
                trade_pnls = []
                wins = 0

                for feat, D in zip(days, D_arrays):
                    n   = len(feat)
                    hi  = feat["high"].values
                    lo  = feat["low"].values
                    cl  = feat["close"].values
                    dlt = feat["delta_t"].values

                    for i in range(n - 1):
                        Dv = D[i]
                        if abs(Dv) < D_STAR:
                            continue

                        direction = 1 if Dv > 0 else -1
                        entry_px  = cl[i] + direction * slippage
                        tp_px     = entry_px + direction * tp_frac * dlt[i]
                        sl_px     = (
                            entry_px - direction * sl_frac * dlt[i]
                            if sl_frac is not None else None
                        )

                        end = min(i + 1 + ts_bars, n)
                        exit_px = cl[end - 1]   # default: time stop

                        for j in range(i + 1, end):
                            hit_tp = (
                                (direction == 1 and hi[j] >= tp_px) or
                                (direction == -1 and lo[j] <= tp_px)
                            )
                            hit_sl = (sl_px is not None) and (
                                (direction == 1 and lo[j] <= sl_px) or
                                (direction == -1 and hi[j] >= sl_px)
                            )

                            if hit_tp:
                                exit_px = tp_px
                                break
                            if hit_sl:
                                exit_px = sl_px
                                break

                        pnl = (exit_px - entry_px) * direction * point_value
                        pnl -= 2 * slippage * point_value
                        trade_pnls.append(pnl)
                        if pnl > 0:
                            wins += 1

                n_trades = len(trade_pnls)
                if n_trades == 0:
                    continue

                arr      = np.array(trade_pnls)
                wr       = wins / n_trades * 100
                avg_pnl  = arr.mean()
                total    = arr.sum()
                avg_win  = arr[arr > 0].mean() if (arr > 0).any() else 0.0
                avg_loss = arr[arr <= 0].mean() if (arr <= 0).any() else 0.0
                sl_label = f"{sl_frac:.1f}" if sl_frac is not None else "None"

                print(
                    f"  {tp_frac:>7.2f}  {sl_label:>7}  {ts_bars:>7}  "
                    f"{n_trades:>6}  {wr:>5.1f}%  "
                    f"${avg_pnl:>8.4f}  ${total:>9.2f}  "
                    f"${avg_win:>7.4f}  ${avg_loss:>8.4f}"
                )
                rows.append({
                    "tp_frac":       tp_frac,
                    "sl_frac":       sl_frac,
                    "time_stop_bars": ts_bars,
                    "N":             n_trades,
                    "WR":            round(wr, 1),
                    "AvgPnL":        round(avg_pnl, 4),
                    "TotalPnL":      round(total, 2),
                    "AvgWin":        round(avg_win, 4),
                    "AvgLoss":       round(avg_loss, 4),
                })

    # Sort and print summary
    rows_sorted = sorted(rows, key=lambda r: r["AvgPnL"], reverse=True)

    print("\n--- TOP 20 COMBOS BY AvgPnL ---")
    print(
        f"  {'tp_frac':>7}  {'sl_frac':>7}  {'ts_bars':>7}  "
        f"{'N':>6}  {'WR%':>6}  {'AvgPnL':>9}  {'TotalPnL':>10}"
    )
    print("  " + "-" * 70)
    for r in rows_sorted[:20]:
        sl_label = f"{r['sl_frac']:.1f}" if r["sl_frac"] is not None else "None"
        print(
            f"  {r['tp_frac']:>7.2f}  {sl_label:>7}  {r['time_stop_bars']:>7}  "
            f"{r['N']:>6}  {r['WR']:>5.1f}%  "
            f"${r['AvgPnL']:>8.4f}  ${r['TotalPnL']:>9.2f}"
        )

    positive = [r for r in rows_sorted if r["AvgPnL"] > 0]
    if positive:
        print(f"\n--- ALL COMBOS WITH AvgPnL > 0 ({len(positive)} found) ---")
        print(
            f"  {'tp_frac':>7}  {'sl_frac':>7}  {'ts_bars':>7}  "
            f"{'N':>6}  {'WR%':>6}  {'AvgPnL':>9}  {'TotalPnL':>10}"
        )
        print("  " + "-" * 70)
        for r in positive:
            sl_label = f"{r['sl_frac']:.1f}" if r["sl_frac"] is not None else "None"
            print(
                f"  {r['tp_frac']:>7.2f}  {sl_label:>7}  {r['time_stop_bars']:>7}  "
                f"{r['N']:>6}  {r['WR']:>5.1f}%  "
                f"${r['AvgPnL']:>8.4f}  ${r['TotalPnL']:>9.2f}"
            )
    else:
        print("\n--- NO COMBOS WITH AvgPnL > 0 ---")

    return rows_sorted


def run_stripped_mode():
    """Run stripped progressive filter analysis and print/save results."""
    print("=" * 65)
    print("FPDE Stripped Mode — Progressive Filter Analysis")
    print("=" * 65)

    # --- Load MES ---
    print("\nLoading MES 1-min bars...", end="", flush=True)
    mes_raw = load_mes()
    print(f" {len(mes_raw):,} RTH bars loaded.")

    # --- Build session features ---
    print("Building features...", end="", flush=True)
    all_days_mes = build_all_features(mes_raw)
    n_days = len(all_days_mes)
    print(f" {n_days} days.")

    if n_days < TRAIN_DAYS + 10:
        print(f"ERROR: need at least {TRAIN_DAYS + 10} trading days, got {n_days}")
        sys.exit(1)

    # --- Labels ---
    print("Computing first-passage labels...", end="", flush=True)
    all_labels_mes = [compute_labels(f) for f in all_days_mes]
    print(f" done.")

    # --- Walk-forward D_t computation (same as main, V1 surface) ---
    print("Computing walk-forward D_t arrays...", end="", flush=True)

    r_tilde = compute_r_tilde(all_days_mes[:TRAIN_DAYS])
    for i in range(len(all_days_mes)):
        all_days_mes[i] = add_B_t(all_days_mes[i], r_tilde)

    test_feats = []
    test_D     = []

    surface_v1  = {}
    chi_median  = 0.5
    recompute_c = 0

    for t in range(TRAIN_DAYS, n_days):
        train_feats  = all_days_mes[:t]
        train_labels = all_labels_mes[:t]

        if recompute_c % RECOMPUTE_EVERY == 0:
            r_tilde = compute_r_tilde(train_feats)
            for j in range(t, n_days):
                all_days_mes[j] = add_B_t(all_days_mes[j], r_tilde)
        recompute_c += 1

        if recompute_c % RECOMPUTE_EVERY == 1 or t == TRAIN_DAYS:
            surface_v1, chi_median = build_surface_v1(train_feats, train_labels)

        feat_t = all_days_mes[t]
        D_v1   = lookup_distortion_v1(feat_t, surface_v1, chi_median)

        test_feats.append(feat_t)
        test_D.append(D_v1)

    print(f" {len(test_feats)} test days.")

    # --- Sweep ---
    print(
        f"\n{'Version':<9}  {'d*':>5}  {'N':>5}  {'WR%':>6}  "
        f"{'AvgPnL':>8}  {'Sharpe':>7}  {'MaxDD':>9}  Notes"
    )
    print("-" * 75)

    stripped_rows = []

    version_notes = {
        "S1": "D only, 1:1 RR (TP=SL=delta)",
        "S2": "D + price loc, 1:1 RR",
        "S3": "D + u_t mid-session, 1:1 RR",
        "S4": "D only, 1:1 RR (TP=SL=0.5*delta)",
        "S5": "D only, TP=0.5*delta, NO hard SL (time-stop only)",
    }
    version_tp_frac = {"S1": 1.0, "S2": 1.0, "S3": 1.0, "S4": 0.5, "S5": 0.5}
    version_no_sl   = {"S1": False, "S2": False, "S3": False, "S4": False, "S5": True}

    for version in ("S1", "S2", "S3", "S4", "S5"):
        for d_star in STRIPPED_THRESHOLDS:
            trades, metrics = run_stripped_backtest(
                test_feats, test_D, d_star,
                MES_TICK, MES_PV, MES_SLIP,
                version,
                tp_frac=version_tp_frac[version],
                no_sl=version_no_sl[version],
            )
            note = version_notes[version]
            print(
                f"{version:<9}  {d_star:.2f}  {metrics['N']:>5}  {metrics['WR']:>5.1f}%  "
                f"${metrics['AvgPnL']:>7.2f}  {metrics['Sharpe']:>7.3f}  "
                f"${metrics['MaxDD']:>8.2f}  {note}"
            )
            stripped_rows.append({
                "version": version,
                "d_star":  d_star,
                "N":       metrics["N"],
                "WR":      metrics["WR"],
                "AvgPnL":  metrics["AvgPnL"],
                "Sharpe":  metrics["Sharpe"],
                "MaxDD":   metrics["MaxDD"],
                "TotalPnL": metrics["TotalPnL"],
                "notes":   note,
            })

    # --- Directional accuracy table using the d* with most S1 trades ---
    s1_rows = [r for r in stripped_rows if r["version"] == "S1"]
    best_s1 = max(s1_rows, key=lambda r: r["N"]) if s1_rows else None
    best_d_dir = best_s1["d_star"] if best_s1 else STRIPPED_THRESHOLDS[0]

    dir_rows = print_directional_accuracy_table(test_feats, test_D, best_d_dir)

    # --- Raw signal backtest: every qualifying bar = independent trade ---
    raw_rows = run_raw_signal_backtest(test_feats, test_D, MES_PV, MES_SLIP)

    # --- Optimization sweep: 3D grid (tp_frac × sl_frac × time_stop_bars) ---
    opt_rows = run_raw_optimization_sweep(test_feats, test_D, MES_PV, MES_SLIP)

    # --- Merge into existing results file ---
    existing = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as fh:
            try:
                existing = json.load(fh)
            except json.JSONDecodeError:
                existing = {}

    existing["stripped"] = {
        "sweep": stripped_rows,
        "directional_accuracy": dir_rows,
        "best_d_star_for_dir_table": best_d_dir,
        "raw_signal": raw_rows,
        "optimization": opt_rows,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as fh:
        json.dump(existing, fh, indent=2)
    print(f"\nResults saved to {RESULTS_PATH} (key: 'stripped').")
    print("Done.")


# ===========================================================================
# 8. MAIN PIPELINE
# ===========================================================================

def build_all_features(df: pd.DataFrame) -> List[pd.DataFrame]:
    """Build feature frames for all RTH sessions, return list sorted by date."""
    days = []
    for date, day_df in df.groupby(df.index.date):
        feat = compute_session_features(day_df)
        if len(feat) >= 10:
            days.append(feat)
    return days


def compute_rolling_stats(
    train_feats: List[pd.DataFrame],
    lookback: int = 20,
) -> Tuple[float, float, float]:
    """Compute chi median, omega_plus p75, omega_minus p75 from last N days."""
    recent = train_feats[-lookback:] if len(train_feats) >= lookback else train_feats
    chi_all = np.concatenate([f["chi_t"].values for f in recent])
    op_all  = np.concatenate([f["omega_plus"].values for f in recent])
    om_all  = np.concatenate([f["omega_minus"].values for f in recent])
    return np.median(chi_all), np.percentile(op_all, 75), np.percentile(om_all, 75)


def main():
    print("=" * 65)
    print("FPDE (First-Passage Distortion Engine) Backtest")
    print("=" * 65)

    # --- Load MES ---
    print("\nLoading MES 1-min bars...", end="", flush=True)
    mes_raw = load_mes()
    print(f" {len(mes_raw):,} RTH bars loaded.")

    # --- Build all session features ---
    print("Building features...", end="", flush=True)
    all_days_mes = build_all_features(mes_raw)
    n_days = len(all_days_mes)
    print(f" {n_days} days.")

    if n_days < TRAIN_DAYS + 10:
        print(f"ERROR: need at least {TRAIN_DAYS + 10} trading days, got {n_days}")
        sys.exit(1)

    # --- Compute first-passage labels for all days ---
    print("Computing first-passage labels...", end="", flush=True)
    all_labels_mes = [compute_labels(f) for f in all_days_mes]
    labeled_count = sum(np.sum(~np.isnan(l)) for l in all_labels_mes)
    print(f" {labeled_count:,} labeled bars.")

    # -----------------------------------------------------------------------
    # Walk-forward loop
    # -----------------------------------------------------------------------
    print("Computing surfaces (walk-forward)...", end="", flush=True)

    # Initial R_tilde from first TRAIN_DAYS
    r_tilde = compute_r_tilde(all_days_mes[:TRAIN_DAYS])

    # Add B_t to all days using current r_tilde
    for i in range(len(all_days_mes)):
        all_days_mes[i] = add_B_t(all_days_mes[i], r_tilde)

    # Pre-compute V1 surface for entire test period using walk-forward
    # For monotonicity table and backtest
    test_feats_v1  = []
    test_labels_v1 = []
    test_D_v1      = []

    # For V2
    test_D_v2      = []

    # Rolling stats per day
    chi_rolling = {}
    op_rolling  = {}
    om_rolling  = {}

    surface_v1    = {}
    chi_median    = 0.5
    surface_v2    = {}
    op_median_v2  = 0.5
    om_median_v2  = 0.5

    r_tilde_recompute_counter = 0

    for t in range(TRAIN_DAYS, n_days):
        train_feats  = all_days_mes[:t]
        train_labels = all_labels_mes[:t]

        # Recompute R_tilde every RECOMPUTE_EVERY days
        if r_tilde_recompute_counter % RECOMPUTE_EVERY == 0:
            r_tilde = compute_r_tilde(train_feats)
            # Re-add B_t for remaining days using updated curve
            for j in range(t, n_days):
                all_days_mes[j] = add_B_t(all_days_mes[j], r_tilde)
        r_tilde_recompute_counter += 1

        # Rebuild V1 surface every RECOMPUTE_EVERY days
        if r_tilde_recompute_counter % RECOMPUTE_EVERY == 1 or t == TRAIN_DAYS:
            surface_v1, chi_median = build_surface_v1(train_feats, train_labels)
            surface_v2, chi_median_v2, op_median_v2, om_median_v2 = build_surface_v2(
                train_feats, train_labels
            )

        # Rolling stats for this test day
        chi_m, op_p75, om_p75 = compute_rolling_stats(train_feats, lookback=20)
        day_idx_test = t - TRAIN_DAYS
        chi_rolling[day_idx_test] = chi_m
        op_rolling[day_idx_test]  = op_p75
        om_rolling[day_idx_test]  = om_p75

        # Distortion for this test day
        feat_t = all_days_mes[t]
        D_v1   = lookup_distortion_v1(feat_t, surface_v1, chi_median)
        D_v2   = lookup_distortion_v2(
            feat_t, surface_v2, surface_v1, chi_median_v2, op_median_v2, om_median_v2
        )

        test_feats_v1.append(feat_t)
        test_labels_v1.append(all_labels_mes[t])
        test_D_v1.append(D_v1)
        test_D_v2.append(D_v2)

    print(f" {len(test_feats_v1)} test days processed.")

    # -----------------------------------------------------------------------
    # Monotonicity table (V1)
    # -----------------------------------------------------------------------
    mono_rows = print_monotonicity_table(test_feats_v1, test_labels_v1, test_D_v1)

    # -----------------------------------------------------------------------
    # Threshold sweep (V1, MES)
    # -----------------------------------------------------------------------
    print("\n=== VERSION 1 DISTORTION THRESHOLD SWEEP (MES) ===")
    print(
        f"{'d*':>6}  {'N':>5}  {'WR%':>6}  {'AvgPnL':>8}  "
        f"{'Sharpe':>7}  {'MaxDD':>9}  {'AvgD':>7}  {'TotalPnL':>10}"
    )
    print("-" * 70)

    sweep_results = []
    best_sharpe   = -999.0
    best_d_star   = 0.08
    best_result   = None

    for d_star in SWEEP_THRESHOLDS:
        trades, metrics = run_backtest(
            test_feats_v1, test_D_v1, d_star,
            MES_TICK, MES_PV, MES_SLIP,
            chi_rolling, op_rolling, om_rolling,
        )
        print(
            f"  {d_star:.2f}  {metrics['N']:>5}  {metrics['WR']:>5.1f}%  "
            f"${metrics['AvgPnL']:>7.2f}  {metrics['Sharpe']:>7.3f}  "
            f"${metrics['MaxDD']:>8.2f}  {metrics['AvgD_at_entry']:>7.4f}  "
            f"${metrics['TotalPnL']:>9.2f}"
        )
        sweep_results.append({"d_star": d_star, "metrics": metrics})
        if metrics["Sharpe"] > best_sharpe and metrics["N"] >= 5:
            best_sharpe = metrics["Sharpe"]
            best_d_star = d_star
            best_result = (trades, metrics)

    # -----------------------------------------------------------------------
    # V2 backtest at best d*
    # -----------------------------------------------------------------------
    print(f"\n=== VERSION 2 BACKTEST (best d*={best_d_star}) (MES) ===")
    trades_v2, metrics_v2 = run_backtest(
        test_feats_v1, test_D_v2, best_d_star,
        MES_TICK, MES_PV, MES_SLIP,
        chi_rolling, op_rolling, om_rolling,
    )
    print(
        f"  N={metrics_v2['N']}  WR={metrics_v2['WR']}%  "
        f"AvgPnL=${metrics_v2['AvgPnL']}  Sharpe={metrics_v2['Sharpe']}  "
        f"MaxDD=${metrics_v2['MaxDD']}  TotalPnL=${metrics_v2['TotalPnL']}"
    )

    # -----------------------------------------------------------------------
    # MNQ cross-check at best d*
    # -----------------------------------------------------------------------
    print(f"\n=== MNQ CROSS-CHECK (d*={best_d_star}) ===")
    try:
        mnq_raw = load_mnq()
        all_days_mnq = build_all_features(mnq_raw)
        n_days_mnq   = len(all_days_mnq)
        print(f"  MNQ: {n_days_mnq} RTH sessions loaded.")

        if n_days_mnq >= TRAIN_DAYS + 5:
            all_labels_mnq = [compute_labels(f) for f in all_days_mnq]

            r_tilde_mnq = compute_r_tilde(all_days_mnq[:TRAIN_DAYS])
            for i in range(len(all_days_mnq)):
                all_days_mnq[i] = add_B_t(all_days_mnq[i], r_tilde_mnq)

            # Simple split: use first 60 days as train, rest as test
            surf_mnq_v1, chi_med_mnq = build_surface_v1(
                all_days_mnq[:TRAIN_DAYS], all_labels_mnq[:TRAIN_DAYS]
            )

            mnq_test_feats  = all_days_mnq[TRAIN_DAYS:]
            mnq_test_labels = all_labels_mnq[TRAIN_DAYS:]
            mnq_D = [
                lookup_distortion_v1(f, surf_mnq_v1, chi_med_mnq)
                for f in mnq_test_feats
            ]

            # Rolling stats
            chi_r_mnq, op_r_mnq, om_r_mnq = {}, {}, {}
            train_so_far = list(all_days_mnq[:TRAIN_DAYS])
            for di, _ in enumerate(mnq_test_feats):
                cm, op75, om75 = compute_rolling_stats(train_so_far, lookback=20)
                chi_r_mnq[di] = cm
                op_r_mnq[di]  = op75
                om_r_mnq[di]  = om75
                if TRAIN_DAYS + di < n_days_mnq:
                    train_so_far.append(all_days_mnq[TRAIN_DAYS + di])

            trades_mnq, metrics_mnq = run_backtest(
                mnq_test_feats, mnq_D, best_d_star,
                MNQ_TICK, MNQ_PV, MNQ_SLIP,
                chi_r_mnq, op_r_mnq, om_r_mnq,
            )
            print(
                f"  N={metrics_mnq['N']}  WR={metrics_mnq['WR']}%  "
                f"AvgPnL=${metrics_mnq['AvgPnL']}  Sharpe={metrics_mnq['Sharpe']}  "
                f"MaxDD=${metrics_mnq['MaxDD']}  TotalPnL=${metrics_mnq['TotalPnL']}"
            )
        else:
            print(f"  Insufficient MNQ data ({n_days_mnq} days < {TRAIN_DAYS + 5} required).")
            metrics_mnq = None
            trades_mnq  = []
    except Exception as exc:
        print(f"  MNQ cross-check failed: {exc}")
        metrics_mnq = None
        trades_mnq  = []

    # -----------------------------------------------------------------------
    # Per-trade log for best V1 config
    # -----------------------------------------------------------------------
    best_trades, best_metrics = best_result if best_result else ([], {})
    print(f"\n=== PER-TRADE LOG (V1, d*={best_d_star}, first 30) ===")
    print(f"  {'Date':>12}  {'Dir':>5}  {'Entry':>8}  {'Exit':>8}  {'PnL':>8}  {'Reason':>12}  {'D_entry':>8}")
    print("  " + "-" * 72)
    for t in best_trades[:30]:
        direction = "LONG" if t["direction"] == 1 else "SHORT"
        print(
            f"  {t['date']:>12}  {direction:>5}  "
            f"{t['entry']:>8.2f}  {t['exit']:>8.2f}  "
            f"${t['pnl']:>7.2f}  {t['exit_reason']:>12}  "
            f"{t['D_at_entry']:>8.4f}"
        )
    if len(best_trades) > 30:
        print(f"  ... ({len(best_trades) - 30} more trades not shown)")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    results = {
        "monotonicity_table": mono_rows,
        "sweep_v1_mes": sweep_results,
        "best_d_star": best_d_star,
        "v2_mes": {"metrics": metrics_v2, "n_trades": len(trades_v2)},
        "mnq_crosscheck": {
            "metrics": metrics_mnq,
            "n_trades": len(trades_mnq) if trades_mnq else 0,
        },
        "per_trade_log_v1_best": best_trades,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FPDE Backtest")
    parser.add_argument(
        "--mode",
        choices=["full", "stripped"],
        default="full",
        help="Run mode: 'full' (default) runs original pipeline; 'stripped' runs progressive filter analysis.",
    )
    args = parser.parse_args()

    if args.mode == "stripped":
        run_stripped_mode()
    else:
        main()
