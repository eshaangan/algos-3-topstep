"""FPDE ORB Filter Diagnostic.

Tests whether the FPDE distortion score D, computed at the ORB entry bar on
MES 1-min data, improves ORB trade quality when used as a qualifying filter.

Hypothesis:
    ORB entries where D > threshold (market favors upside extension) produce
    higher win rates than ORB entries where D is neutral or negative.

Data:
    MES 1-min: data/processed/mes_1m_bars_cache.h5  key=/bars_1m  (Jun–Dec 2025)
    MNQ 5-min: data/processed/mnq_5min_aug25_mar26.h5  key=bars_5min (Aug 2025–Mar 2026)
    Overlap window: Aug–Dec 2025

Run:
    python rule_based_v1/diagnostics/fpde_orb_filter.py
"""

from __future__ import annotations

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
MNQ_PATH = DATA_DIR / "mnq_5min_aug25_mar26.h5"
RESULTS_PATH = DIAG_DIR / "fpde_orb_results.json"

sys.path.insert(0, str(ROOT / "rule_based_v1"))

# ---------------------------------------------------------------------------
# Instrument constants (MNQ)
# ---------------------------------------------------------------------------
MNQ_TICK = 0.25
MNQ_PV = 2.0        # $ per point
MNQ_SLIP = 0.25     # 1 tick per side (one-way)
N_CONTRACTS = 2

# ---------------------------------------------------------------------------
# ORB parameters (match live system)
# ---------------------------------------------------------------------------
OR_END = "10:04"          # opening range closes at 10:04 ET
OR_END_HOUR, OR_END_MIN = 10, 4
ENTRY_CUTOFF_HOUR, ENTRY_CUTOFF_MIN = 12, 0
MIN_OR_BARS = 7
ATR_PERIOD = 14
PT_ATR = 3.0
SL_ATR = 1.5
TIME_STOP_BARS = 24       # 120 min on 5-min bars

# ---------------------------------------------------------------------------
# FPDE parameters (match fpde_backtest.py exactly)
# ---------------------------------------------------------------------------
DELTA_MIN = 1.5
SIGMA_WINDOW = 20
CHI_WINDOW = 15
OMEGA_WINDOW = 15
R_TILDE_BINS = 20
TRAIN_DAYS = 40           # first 40 days of MES used to fit surface

Q_BINS = [0, 0.3, 0.5, 0.7, 1.0]
B_BINS = [0, 0.7, 1.0, 1.3, np.inf]
U_BINS = [0, 0.25, 0.5, 0.75, 1.0]
MIN_SAMPLES_V1 = 10
LABEL_HORIZON = 45        # bars for first-passage label (used during training)


# ===========================================================================
# 1. DATA LOADING
# ===========================================================================

def load_mes() -> pd.DataFrame:
    """Load MES 1-min bars, convert to ET DatetimeIndex, RTH only."""
    with pd.HDFStore(str(MES_PATH), mode="r") as store:
        df = store["/bars_1m"].copy()

    df = df.set_index("timestamp")
    df.index = df.index.tz_convert("US/Eastern")
    df.index.name = "ts"
    df = df.sort_index()

    mod = df.index.hour * 60 + df.index.minute
    df = df[(mod >= 570) & (mod <= 959)].copy()
    df = df[["open", "high", "low", "close", "volume"]]
    return df


def load_mnq() -> pd.DataFrame:
    """Load MNQ 5-min bars (already ET RTH, index is ts_event in ET)."""
    df = pd.read_hdf(str(MNQ_PATH), key="bars_5min")
    df = df.sort_index()
    # File may name the index ts_event; normalise to ts
    df.index.name = "ts"
    # Ensure timezone-aware (file is already US/Eastern)
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("US/Eastern")
    return df


# ===========================================================================
# 2. ATR (manual, no ta library dependency)
# ===========================================================================

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """EWM ATR matching utils/indicators.atr()."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ===========================================================================
# 3. FPDE FEATURE COMPUTATION (1-min session)
# ===========================================================================

def _digitize_safe(x: np.ndarray, bins: list) -> np.ndarray:
    """0-indexed bin assignment."""
    out = np.digitize(x, bins[1:-1])
    return out.clip(0, len(bins) - 2)


def compute_session_features_mes(day_df: pd.DataFrame) -> pd.DataFrame:
    """Compute FPDE features for a single MES 1-min RTH session."""
    n = len(day_df)
    if n < 2:
        return pd.DataFrame()

    hi = day_df["high"].values
    lo = day_df["low"].values
    cl = day_df["close"].values
    op = day_df["open"].values

    H_t = np.maximum.accumulate(hi)
    L_t = np.minimum.accumulate(lo)

    idx = np.arange(n, dtype=float)
    u_t = idx / max(n - 1, 1)

    bar_range = hi - lo
    sigma_hat = np.empty(n)
    for i in range(n):
        start = max(0, i - SIGMA_WINDOW + 1)
        window = bar_range[start: i + 1]
        sigma_hat[i] = window.mean() if len(window) > 0 else bar_range[: i + 1].mean()

    delta_t = np.maximum(DELTA_MIN, sigma_hat)

    U_t = H_t + delta_t
    D_barrier = L_t - delta_t

    a_t = U_t - cl
    b_t = cl - D_barrier
    p_null = b_t / (a_t + b_t + 1e-9)

    q_t = (cl - L_t) / (H_t - L_t + 1e-9)
    running_range = H_t - L_t

    chi_t = np.empty(n)
    for i in range(n):
        start = max(0, i - CHI_WINDOW + 1)
        w_range = bar_range[start: i + 1]
        cum_rr = H_t[i] - L_t[i]
        chi_t[i] = w_range.sum() / (cum_rr + 1e-9)

    wick_up = hi - np.maximum(op, cl)
    wick_down = np.minimum(op, cl) - lo
    bar_range_sq = bar_range ** 2
    omega_plus = np.empty(n)
    omega_minus = np.empty(n)
    for i in range(n):
        start = max(0, i - OMEGA_WINDOW + 1)
        wu = wick_up[start: i + 1]
        wd = wick_down[start: i + 1]
        brs = bar_range_sq[start: i + 1]
        denom = brs.sum() + 1e-9
        omega_plus[i] = (wu ** 2).sum() / denom
        omega_minus[i] = (wd ** 2).sum() / denom

    u_bucket = np.clip((u_t * R_TILDE_BINS).astype(int), 0, R_TILDE_BINS - 1)

    return pd.DataFrame(
        {
            "open": op,
            "high": hi,
            "low": lo,
            "close": cl,
            "H_t": H_t,
            "L_t": L_t,
            "delta_t": delta_t,
            "U_barrier": U_t,
            "D_barrier": D_barrier,
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
        },
        index=day_df.index,
    )


def compute_labels(feat: pd.DataFrame) -> np.ndarray:
    """First-passage labels for training the FPDE surface."""
    n = len(feat)
    hi = feat["high"].values
    lo = feat["low"].values
    U = feat["U_barrier"].values
    D = feat["D_barrier"].values

    labels = np.full(n, np.nan)
    for i in range(n - 1):
        end = min(i + 1 + LABEL_HORIZON, n)
        fwd_hi = hi[i + 1: end]
        fwd_lo = lo[i + 1: end]

        hit_up = fwd_hi >= U[i]
        hit_down = fwd_lo <= D[i]

        idx_up = np.argmax(hit_up) if hit_up.any() else LABEL_HORIZON
        idx_down = np.argmax(hit_down) if hit_down.any() else LABEL_HORIZON

        if not hit_up.any() and not hit_down.any():
            labels[i] = np.nan
        elif hit_up.any() and (not hit_down.any() or idx_up <= idx_down):
            labels[i] = 1.0
        else:
            labels[i] = 0.0

    return labels


# ===========================================================================
# 4. R_TILDE CURVE
# ===========================================================================

def compute_r_tilde(all_feats: List[pd.DataFrame]) -> np.ndarray:
    """Median running range per u-bucket from training features."""
    buckets: List[List[float]] = [[] for _ in range(R_TILDE_BINS)]
    for feat in all_feats:
        for ub, rr in zip(feat["u_bucket"].values, feat["running_range"].values):
            buckets[int(ub)].append(rr)

    r_tilde = np.zeros(R_TILDE_BINS)
    for k, vals in enumerate(buckets):
        r_tilde[k] = np.median(vals) if vals else 1.0
    return np.where(r_tilde > 0, r_tilde, 1.0)


def add_B_t(feat: pd.DataFrame, r_tilde: np.ndarray) -> pd.DataFrame:
    """Add budget ratio B_t = running_range / R_tilde[u_bucket]."""
    feat = feat.copy()
    r_vals = r_tilde[feat["u_bucket"].values]
    feat["B_t"] = feat["running_range"].values / (r_vals + 1e-9)
    return feat


# ===========================================================================
# 5. SURFACE ESTIMATION (V1)
# ===========================================================================

def build_surface_v1(
    all_feats: List[pd.DataFrame],
    all_labels: List[np.ndarray],
) -> Tuple[Dict[Tuple, float], float]:
    """Build empirical p(upper hit first) for each (q_bin, B_bin, chi_bin, u_bin)."""
    chi_vals = np.concatenate([f["chi_t"].values for f in all_feats])
    chi_median = np.median(chi_vals)

    counts: Dict[Tuple, int] = {}
    pos: Dict[Tuple, float] = {}

    for feat, labels in zip(all_feats, all_labels):
        valid = ~np.isnan(labels)
        if not valid.any():
            continue

        q = feat["q_t"].values[valid]
        B = feat["B_t"].values[valid]
        chi = feat["chi_t"].values[valid]
        u = feat["u_t"].values[valid]
        lbl = labels[valid]

        q_bin = _digitize_safe(q, Q_BINS)
        B_bin = _digitize_safe(B, B_BINS)
        chi_bin = (chi > chi_median).astype(int)
        u_bin = _digitize_safe(u, U_BINS)

        for qb, bb, cb, ub, lb in zip(q_bin, B_bin, chi_bin, u_bin, lbl):
            key = (int(qb), int(bb), int(cb), int(ub))
            counts[key] = counts.get(key, 0) + 1
            pos[key] = pos.get(key, 0) + lb

    surface = {}
    for key in counts:
        if counts[key] >= MIN_SAMPLES_V1:
            surface[key] = pos[key] / counts[key]

    return surface, chi_median


def lookup_distortion_v1(
    feat: pd.DataFrame,
    surface: Dict[Tuple, float],
    chi_median: float,
) -> np.ndarray:
    """Compute D_t = p_emp - p_null for each bar in feat."""
    n = len(feat)
    q = feat["q_t"].values
    B = feat["B_t"].values
    chi = feat["chi_t"].values
    u = feat["u_t"].values
    p_null = feat["p_null"].values

    q_bin = _digitize_safe(q, Q_BINS)
    B_bin = _digitize_safe(B, B_BINS)
    chi_bin = (chi > chi_median).astype(int)
    u_bin = _digitize_safe(u, U_BINS)

    D = np.zeros(n)
    for i in range(n):
        key = (int(q_bin[i]), int(B_bin[i]), int(chi_bin[i]), int(u_bin[i]))
        p_emp = surface.get(key, p_null[i])
        D[i] = p_emp - p_null[i]
    return D


def heuristic_distortion(feat: pd.DataFrame) -> np.ndarray:
    """
    Simplified distortion without binned surface.

    D_heuristic = (q_t - 0.5) * 2 * (1.5 - B_t)

    Positive when price is high in session range AND range budget is low
    (session range still has room to expand upward).
    """
    q = feat["q_t"].values
    B = feat["B_t"].values
    return (q - 0.5) * 2.0 * (1.5 - B)


# ===========================================================================
# 6. FPDE D LOOKUP AT ENTRY TIMESTAMP
# ===========================================================================

def get_d_at_timestamp(
    ts: pd.Timestamp,
    mes_day_feats: Dict[str, pd.DataFrame],
    mes_day_D: Dict[str, np.ndarray],
    mes_day_D_heur: Dict[str, np.ndarray],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Return (D_surface, D_heuristic) for the MES 1-min bar at or before ts.

    Uses the last 1-min bar that is <= ts (inclusive).
    Returns (None, None) if no MES data available for that day.
    """
    date_key = str(ts.date())
    if date_key not in mes_day_feats:
        return None, None

    feat = mes_day_feats[date_key]
    D_arr = mes_day_D[date_key]
    D_heur_arr = mes_day_D_heur[date_key]

    # Find last 1-min bar <= ts
    # ts is at e.g. 10:05 ET (5-min bar close). Use bars up to 10:05.
    mask = feat.index <= ts
    if not mask.any():
        return None, None

    idx = int(np.where(mask)[0][-1])
    return float(D_arr[idx]), float(D_heur_arr[idx])


# ===========================================================================
# 7. ORB BACKTEST (per-day, returns list of trade dicts)
# ===========================================================================

def run_orb_day(
    day_bars: pd.DataFrame,
    atr_series: pd.Series,
) -> Optional[dict]:
    """
    Run ORB logic for a single day on 5-min MNQ bars.

    Returns a trade dict (or None if no signal fired).
    Only LONG trades (long_only=True).
    """
    date = day_bars.index[0].date()

    # Opening range window: 09:30 <= ts <= 10:04
    or_mask = (
        (day_bars.index.hour * 60 + day_bars.index.minute >= 570) &
        (day_bars.index.hour * 60 + day_bars.index.minute <= OR_END_HOUR * 60 + OR_END_MIN)
    )
    or_bars = day_bars[or_mask]

    if len(or_bars) < MIN_OR_BARS:
        return None

    range_high = float(or_bars["high"].max())
    range_low = float(or_bars["low"].min())

    # Filter bars after OR window and before entry cutoff
    post_or = day_bars[~or_mask].copy()
    cutoff_minutes = ENTRY_CUTOFF_HOUR * 60 + ENTRY_CUTOFF_MIN
    bar_minutes = post_or.index.hour * 60 + post_or.index.minute
    post_or = post_or[bar_minutes <= cutoff_minutes]

    if post_or.empty:
        return None

    prev_close = range_high  # last OR bar close approximated by range_high boundary check

    for i, (ts, row) in enumerate(post_or.iterrows()):
        bar_atr = float(atr_series.loc[ts]) if ts in atr_series.index else float(atr_series.iloc[-1])
        if np.isnan(bar_atr) or bar_atr <= 0:
            continue

        curr_close = float(row["close"])
        # Previous close: use the bar before this one in day_bars
        bar_loc = day_bars.index.get_loc(ts)
        if bar_loc > 0:
            prev_c = float(day_bars.iloc[bar_loc - 1]["close"])
        else:
            prev_c = curr_close

        # LONG signal: first bar where close > range_high AND prev_close <= range_high
        if curr_close > range_high and prev_c <= range_high:
            entry_px = curr_close + MNQ_SLIP
            pt = entry_px + PT_ATR * bar_atr
            sl = entry_px - SL_ATR * bar_atr
            entry_bar_loc = day_bars.index.get_loc(ts)

            # Simulate exit on subsequent bars
            remaining = day_bars.iloc[entry_bar_loc + 1:]
            exit_px = None
            exit_reason = "time_stop"
            bars_held = 0

            for j, (exit_ts, exit_row) in enumerate(remaining.iterrows()):
                if j >= TIME_STOP_BARS:
                    exit_px = float(exit_row["close"])
                    exit_reason = "time_stop"
                    bars_held = j + 1
                    break

                hi = float(exit_row["high"])
                lo = float(exit_row["low"])

                hit_pt = hi >= pt
                hit_sl = lo <= sl

                if hit_pt and hit_sl:
                    # Assume SL hit first (conservative)
                    exit_px = sl
                    exit_reason = "sl"
                    bars_held = j + 1
                    break
                elif hit_pt:
                    exit_px = pt
                    exit_reason = "pt"
                    bars_held = j + 1
                    break
                elif hit_sl:
                    exit_px = sl
                    exit_reason = "sl"
                    bars_held = j + 1
                    break
            else:
                if exit_px is None:
                    exit_px = float(remaining.iloc[-1]["close"]) if not remaining.empty else entry_px
                    exit_reason = "time_stop"
                    bars_held = len(remaining)

            if exit_px is None:
                exit_px = entry_px
                exit_reason = "time_stop"
                bars_held = 0

            pnl_raw = (exit_px - entry_px) * N_CONTRACTS * MNQ_PV
            # Slippage on both sides at exit
            pnl = pnl_raw - 2 * MNQ_SLIP * MNQ_PV * N_CONTRACTS

            return {
                "date": str(date),
                "entry_ts": str(ts),
                "entry_px": round(entry_px, 4),
                "exit_px": round(exit_px, 4),
                "pt": round(pt, 4),
                "sl": round(sl, 4),
                "atr": round(bar_atr, 4),
                "range_high": round(range_high, 4),
                "range_low": round(range_low, 4),
                "exit_reason": exit_reason,
                "bars_held": bars_held,
                "pnl": round(pnl, 2),
                "win": pnl > 0,
            }

    return None


# ===========================================================================
# 8. METRICS COMPUTATION
# ===========================================================================

def compute_metrics(trades: List[dict], n_base: int) -> dict:
    """Compute WR, AvgPnL, Sharpe, MaxDD, N, delta_N_vs_base."""
    if not trades:
        return {
            "N": 0,
            "WR_pct": 0.0,
            "AvgPnL": 0.0,
            "Sharpe": float("nan"),
            "MaxDD": 0.0,
            "delta_N_pct": -100.0 if n_base > 0 else 0.0,
        }

    pnls = np.array([t["pnl"] for t in trades])
    wins = np.array([t["win"] for t in trades])
    n = len(trades)
    wr = float(wins.mean()) * 100
    avg_pnl = float(pnls.mean())

    if pnls.std() > 0:
        sharpe = float(pnls.mean() / pnls.std() * np.sqrt(252))
    else:
        sharpe = float("nan")

    equity = np.cumsum(pnls)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max
    max_dd = float(drawdown.min())

    delta_n = ((n - n_base) / n_base * 100) if n_base > 0 else 0.0

    return {
        "N": n,
        "WR_pct": round(wr, 1),
        "AvgPnL": round(avg_pnl, 2),
        "Sharpe": round(sharpe, 2) if not np.isnan(sharpe) else float("nan"),
        "MaxDD": round(max_dd, 2),
        "delta_N_pct": round(delta_n, 1),
    }


# ===========================================================================
# 9. MAIN
# ===========================================================================

def main() -> None:
    print("Loading MES 1-min bars ...")
    mes = load_mes()
    print(f"  MES: {mes.index[0].date()} — {mes.index[-1].date()}, {len(mes):,} bars")

    print("Loading MNQ 5-min bars ...")
    mnq = load_mnq()
    print(f"  MNQ: {mnq.index[0].date()} — {mnq.index[-1].date()}, {len(mnq):,} bars")

    # Overlap: Aug 1 2025 – Dec 31 2025
    overlap_start = pd.Timestamp("2025-08-01", tz="US/Eastern")
    overlap_end = pd.Timestamp("2025-12-31", tz="US/Eastern")

    mes_overlap = mes[(mes.index >= overlap_start) & (mes.index <= overlap_end)]
    mnq_overlap = mnq[(mnq.index >= overlap_start) & (mnq.index <= overlap_end)]
    print(f"  Overlap MES: {mes_overlap.index[0].date()} — {mes_overlap.index[-1].date()}, {len(mes_overlap):,} bars")
    print(f"  Overlap MNQ: {mnq_overlap.index[0].date()} — {mnq_overlap.index[-1].date()}, {len(mnq_overlap):,} bars")

    # Use first TRAIN_DAYS calendar days of MES (before overlap) for surface training
    # MES starts Jun 2025; take all MES days before overlap_start
    train_cutoff = overlap_start
    mes_train_group = mes[mes.index < train_cutoff]
    mes_train_days = sorted(set(mes_train_group.index.date))[:TRAIN_DAYS]

    print(f"\nBuilding FPDE surface from first {len(mes_train_days)} MES training days ...")

    train_feats = []
    train_labels = []
    for d in mes_train_days:
        dstr = str(d)
        day_df = mes_train_group[mes_train_group.index.date == d]
        if len(day_df) < 10:
            continue
        feat = compute_session_features_mes(day_df)
        if feat.empty:
            continue
        train_feats.append(feat)
        labels = compute_labels(feat)
        train_labels.append(labels)

    if not train_feats:
        print("ERROR: No training data for FPDE surface. Aborting.")
        return

    r_tilde = compute_r_tilde(train_feats)

    # Add B_t to training feats
    train_feats_b = [add_B_t(f, r_tilde) for f in train_feats]
    surface_v1, chi_median = build_surface_v1(train_feats_b, train_labels)
    print(f"  Surface cells populated: {len(surface_v1)}")
    print(f"  chi_median: {chi_median:.4f}")

    # --- Compute FPDE features for each overlap day ---
    print("\nComputing FPDE D for MES overlap days ...")
    mes_day_feats: Dict[str, pd.DataFrame] = {}
    mes_day_D: Dict[str, np.ndarray] = {}
    mes_day_D_heur: Dict[str, np.ndarray] = {}

    mes_overlap_days = sorted(set(mes_overlap.index.date))
    for d in mes_overlap_days:
        day_df = mes_overlap[mes_overlap.index.date == d]
        if len(day_df) < 5:
            continue
        feat = compute_session_features_mes(day_df)
        if feat.empty:
            continue
        feat = add_B_t(feat, r_tilde)
        D_arr = lookup_distortion_v1(feat, surface_v1, chi_median)
        D_heur_arr = heuristic_distortion(feat)
        dkey = str(d)
        mes_day_feats[dkey] = feat
        mes_day_D[dkey] = D_arr
        mes_day_D_heur[dkey] = D_heur_arr

    print(f"  FPDE computed for {len(mes_day_feats)} overlap days")

    # --- Compute ATR on MNQ ---
    mnq_atr = compute_atr(mnq_overlap["high"], mnq_overlap["low"], mnq_overlap["close"], ATR_PERIOD)

    # --- Run ORB backtest for each day and collect FPDE D at entry ---
    print("\nRunning ORB backtest on MNQ overlap days ...")
    all_trades = []
    mnq_overlap_days = sorted(set(mnq_overlap.index.date))

    for d in mnq_overlap_days:
        day_bars = mnq_overlap[mnq_overlap.index.date == d]
        if len(day_bars) < MIN_OR_BARS + 2:
            continue

        trade = run_orb_day(day_bars, mnq_atr)
        if trade is None:
            continue

        # Look up FPDE D at entry timestamp (convert to ET if needed)
        raw_ts = pd.Timestamp(trade["entry_ts"])
        if raw_ts.tzinfo is None:
            entry_ts = raw_ts.tz_localize("US/Eastern")
        else:
            entry_ts = raw_ts.tz_convert("US/Eastern")

        d_surf, d_heur = get_d_at_timestamp(
            entry_ts, mes_day_feats, mes_day_D, mes_day_D_heur
        )

        trade["D_surface"] = round(d_surf, 5) if d_surf is not None else None
        trade["D_heuristic"] = round(d_heur, 5) if d_heur is not None else None
        all_trades.append(trade)

    print(f"  Total ORB trades in overlap period: {len(all_trades)}")

    # Separate trades with available D
    trades_with_d = [t for t in all_trades if t["D_surface"] is not None]
    trades_without_d = [t for t in all_trades if t["D_surface"] is None]
    print(f"  Trades with D_surface: {len(trades_with_d)}, without: {len(trades_without_d)}")

    # ---------------------------------------------------------------------------
    # CONFIGURATIONS
    # ---------------------------------------------------------------------------
    configs = {
        "Baseline":     all_trades,
        "D_filter_low": [t for t in trades_with_d if t["D_surface"] > 0.03],
        "D_filter_med": [t for t in trades_with_d if t["D_surface"] > 0.06],
        "D_filter_high":[t for t in trades_with_d if t["D_surface"] > 0.10],
        "H_filter_low": [t for t in trades_with_d if t["D_heuristic"] > 0.10],
        "H_filter_med": [t for t in trades_with_d if t["D_heuristic"] > 0.20],
    }

    n_base = len(all_trades)

    # ---------------------------------------------------------------------------
    # RESULTS TABLE
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 82)
    print(f"{'Config':<16}  {'N':>4}  {'WR%':>6}  {'AvgPnL':>8}  {'Sharpe':>7}  {'MaxDD':>8}  {'ΔN_vs_base':>11}")
    print("-" * 82)

    results_summary = {}
    for cfg_name, trades in configs.items():
        m = compute_metrics(trades, n_base)
        sharpe_str = f"{m['Sharpe']:>7.2f}" if not np.isnan(m["Sharpe"]) else "    nan"
        delta_str = f"{m['delta_N_pct']:>+.0f}%" if cfg_name != "Baseline" else "—"
        print(
            f"{cfg_name:<16}  {m['N']:>4}  {m['WR_pct']:>5.1f}%  "
            f"${m['AvgPnL']:>7.2f}  {sharpe_str}  ${m['MaxDD']:>7.2f}  {delta_str:>11}"
        )
        results_summary[cfg_name] = m

    print("=" * 82)

    # ---------------------------------------------------------------------------
    # FILTERED-OUT DAYS ANALYSIS (per filter config)
    # ---------------------------------------------------------------------------
    print("\n--- Days filtered OUT by D_filter_low (D_surface <= 0.03) ---")
    print(f"{'Date':<12}  {'D_surf':>8}  {'D_heur':>8}  {'Outcome':>8}  {'PnL':>8}")
    print("-" * 52)
    filtered_out_low = [t for t in trades_with_d if t["D_surface"] <= 0.03]
    filtered_analysis = []
    for t in filtered_out_low:
        outcome = "WIN" if t["win"] else "LOSS"
        print(f"{t['date']:<12}  {t['D_surface']:>8.4f}  {t['D_heuristic']:>8.4f}  {outcome:>8}  ${t['pnl']:>7.2f}")
        filtered_analysis.append({
            "date": t["date"],
            "D_surface": t["D_surface"],
            "D_heuristic": t["D_heuristic"],
            "outcome": outcome,
            "pnl": t["pnl"],
        })

    print(f"\n--- Days filtered OUT by H_filter_med (D_heuristic <= 0.20) ---")
    print(f"{'Date':<12}  {'D_surf':>8}  {'D_heur':>8}  {'Outcome':>8}  {'PnL':>8}")
    print("-" * 52)
    filtered_out_hmed = [t for t in trades_with_d if t["D_heuristic"] <= 0.20]
    for t in filtered_out_hmed:
        outcome = "WIN" if t["win"] else "LOSS"
        print(f"{t['date']:<12}  {t['D_surface']:>8.4f}  {t['D_heuristic']:>8.4f}  {outcome:>8}  ${t['pnl']:>7.2f}")

    # ---------------------------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------------------------
    output = {
        "overlap_period": {"start": "2025-08-01", "end": "2025-12-31"},
        "surface_training_days": len(mes_train_days),
        "surface_cells": len(surface_v1),
        "chi_median": round(chi_median, 5),
        "total_orb_trades": len(all_trades),
        "trades_with_fpde": len(trades_with_d),
        "configs": results_summary,
        "filtered_out_by_D_low": filtered_analysis,
        "per_trade_log": all_trades,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
