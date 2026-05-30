"""
Build rich 5-min microstructure features from raw MNQ trade ticks.

Instead of simple OFI, extract features that capture:
  - Institutional vs retail flow (large vs small trades)
  - Price impact (Kyle's lambda)
  - Trade clustering and burstiness
  - VPIN-style informed trading probability
  - Sequential flow dynamics
  - Sub-bar OFI velocity (first/last 30s of bar vs full bar)

Output: data/processed/mnq_microstructure_5min.parquet
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).parents[2]
sys.path.insert(0, str(BASE))
DATA = BASE / "data" / "processed"
OUT  = DATA / "mnq_microstructure_5min.parquet"

PRICE_SCALE = 1e9
CHUNK = 2_000_000

from core.microstructure import compute_bar_features, LARGE_TRADE_THRESH  # noqa: E402

# RTH filter: 14:30-21:00 UTC = 9:30-16:00 ET (standard time)
RTH_START_H, RTH_START_M = 14, 30
RTH_END_H = 21

# Core Databento files (always included)
_CORE_FILES = [
    DATA / "mnq_trades_dec2025.csv.gz",
    DATA / "mnq_trades_jan_feb9_2026.csv.gz",
    DATA / "mnq_trades_mar_may_2026.csv.gz",
]

# MotiveWave converted files (auto-discovered — added by fetch_motivewave_ticks.py)
_MW_FILES = sorted(DATA.glob("mnq_trades_mw_*.csv.gz"))

# Rithmic download files (auto-discovered — added by fetch_rithmic_ticks.py)
_RITHMIC_FILES = sorted(DATA.glob("mnq_trades_rithmic_*.csv.gz"))

FILES = [f for f in _CORE_FILES + _MW_FILES + _RITHMIC_FILES if f.exists()]


def is_rth(ts_series: pd.Series) -> pd.Series:
    h = ts_series.dt.hour
    m = ts_series.dt.minute
    after_open = (h > RTH_START_H) | ((h == RTH_START_H) & (m >= RTH_START_M))
    before_close = h < RTH_END_H
    return after_open & before_close


# compute_bar_features is imported from core.microstructure above


def _compute_bar_features_unused(grp: pd.DataFrame) -> dict:
    """Kept for reference only — use core.microstructure.compute_bar_features."""
    if grp.empty:
        return {}

    price  = grp["price_f"].values
    size   = grp["size"].values
    side   = grp["side"].values   # 'B'=buy aggressor, 'A'=sell aggressor, 'N'=neutral
    ts_ns  = grp["ts_recv"].values  # nanoseconds

    n = len(grp)
    buy_mask  = side == "B"
    sell_mask = side == "A"

    buy_vol  = size[buy_mask].sum()
    sell_vol = size[sell_mask].sum()
    total_vol = size.sum()
    ofi = buy_vol - sell_vol

    # Large vs small trade split
    large_mask = size >= LARGE_TRADE_THRESH
    small_mask = ~large_mask

    lg_buy  = size[buy_mask  & large_mask].sum()
    lg_sell = size[sell_mask & large_mask].sum()
    sm_buy  = size[buy_mask  & small_mask].sum()
    sm_sell = size[sell_mask & small_mask].sum()

    lg_vol = size[large_mask].sum()
    sm_vol = size[small_mask].sum()

    lg_ofi  = lg_buy  - lg_sell
    sm_ofi  = sm_buy  - sm_sell

    # Price return over bar
    p_open  = price[0]
    p_close = price[-1]
    bar_ret = p_close - p_open  # in MNQ points

    # Kyle's lambda: price impact per unit signed flow
    denom = abs(ofi) if abs(ofi) > 0 else np.nan
    kyles_lambda = abs(bar_ret) / denom if denom else np.nan

    # Effective spread proxy (Roll's measure): -2 * cov(dp_t, dp_{t-1})
    dp = np.diff(price)
    roll_spread = np.nan
    if len(dp) >= 2:
        cov = np.cov(dp[:-1], dp[1:])[0, 1]
        roll_spread = 2 * np.sqrt(max(-cov, 0))

    # Trade rate (trades per second over 5-min bar = 300s)
    trade_rate = n / 300.0

    # Average and max trade size
    avg_size = size.mean()
    max_size = size.max()
    size_std = size.std() if n > 1 else 0

    # Large trade fraction
    large_frac = lg_vol / total_vol if total_vol > 0 else 0

    # Sub-bar OFI: first/last 10% + 5 quintile slices (each 20% of bar duration)
    ofi_q = [0.0] * 5
    trade_rate_accel = 0.0
    if ts_ns[-1] > ts_ns[0]:
        bar_duration_ns = ts_ns[-1] - ts_ns[0]
        early_cutoff = ts_ns[0] + bar_duration_ns * 0.1  # first 10%
        late_start   = ts_ns[0] + bar_duration_ns * 0.9  # last 10%
        early = (ts_ns <= early_cutoff)
        late  = (ts_ns >= late_start)
        def _ofi(mask):
            b = size[mask & buy_mask].sum()
            s = size[mask & sell_mask].sum()
            return b - s
        ofi_early = _ofi(early)
        ofi_late  = _ofi(late)
        ofi_accel = ofi_late - ofi_early   # flow acceleration

        # 5 quintile OFI slices (each 20% of bar = ~60s of a 5-min bar)
        for q in range(5):
            q_lo = ts_ns[0] + bar_duration_ns * (q * 0.2)
            q_hi = ts_ns[0] + bar_duration_ns * ((q + 1) * 0.2)
            q_mask = (ts_ns >= q_lo) & (ts_ns < q_hi)
            ofi_q[q] = float(_ofi(q_mask))

        # Trade arrival acceleration: late-60s rate vs early-60s rate
        _60s_ns = int(6e10)  # 60 seconds in nanoseconds
        early60_mask = ts_ns <= (ts_ns[0] + min(_60s_ns, int(bar_duration_ns * 0.2)))
        late60_mask  = ts_ns >= (ts_ns[-1] - min(_60s_ns, int(bar_duration_ns * 0.2)))
        early60_count = int(early60_mask.sum())
        late60_count  = int(late60_mask.sum())
        trade_rate_accel = float(late60_count / max(early60_count, 1)) - 1.0
    else:
        ofi_early = ofi_late = ofi_accel = 0.0

    # Longest consecutive same-side run (flow persistence)
    max_run = 1; cur_run = 1
    for i in range(1, len(side)):
        if side[i] == side[i-1] and side[i] in ("B", "A"):
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1

    # Informed flow divergence: large and small flow in opposite directions?
    lg_sm_diverge = float((lg_ofi > 0) != (sm_ofi > 0)) if (lg_vol > 0 and sm_vol > 0) else 0.0

    # VWAP
    dv = (price * size).sum()
    vwap = dv / total_vol if total_vol > 0 else p_close

    # OFI imbalance (normalized)
    ofi_imb = ofi / total_vol if total_vol > 0 else 0.0
    lg_ofi_imb = lg_ofi / lg_vol if lg_vol > 0 else 0.0

    return {
        # Volume
        "buy_vol":      buy_vol,
        "sell_vol":     sell_vol,
        "total_vol":    total_vol,
        "large_vol":    lg_vol,
        "large_frac":   large_frac,
        # OFI family
        "ofi":          ofi,
        "ofi_imb":      ofi_imb,
        "lg_ofi":       lg_ofi,
        "lg_ofi_imb":   lg_ofi_imb,
        "sm_ofi":       sm_ofi,
        "ofi_accel":    ofi_accel,
        "ofi_early":    ofi_early,
        "ofi_late":     ofi_late,
        "lg_sm_diverge": lg_sm_diverge,
        # Price impact
        "kyles_lambda": kyles_lambda,
        "roll_spread":  roll_spread,
        "bar_ret":      bar_ret,
        # Sub-bar quintile OFI slices
        "ofi_q1": ofi_q[0], "ofi_q2": ofi_q[1], "ofi_q3": ofi_q[2],
        "ofi_q4": ofi_q[3], "ofi_q5": ofi_q[4],
        # Trade activity
        "n_trades":          n,
        "trade_rate":        trade_rate,
        "trade_rate_accel":  trade_rate_accel,
        "avg_size":          avg_size,
        "max_size":          max_size,
        "size_std":          size_std,
        "max_run":           float(max_run),
        # Price
        "open":  p_open,
        "high":  price.max(),
        "low":   price.min(),
        "close": p_close,
        "vwap":  vwap,
    }


def process_file(path: Path) -> pd.DataFrame:
    print(f"\nProcessing {path.name} ...", flush=True)
    bar_data: dict = {}   # ts -> list of tick dicts

    for i, chunk in enumerate(pd.read_csv(
        path,
        usecols=["ts_recv", "side", "price", "size"],
        dtype={"price": "float64", "size": "float64"},
        compression=None,
        chunksize=CHUNK,
    )):
        chunk = chunk.dropna(subset=["price", "size"])
        chunk["ts_dt"] = pd.to_datetime(chunk["ts_recv"], unit="ns", utc=True)
        chunk = chunk[is_rth(chunk["ts_dt"])]
        if chunk.empty:
            continue

        chunk["bar_5min"] = chunk["ts_dt"].dt.floor("5min")
        chunk["price_f"]  = chunk["price"] / PRICE_SCALE

        for bar_ts, grp in chunk.groupby("bar_5min"):
            key = bar_ts
            if key not in bar_data:
                bar_data[key] = []
            bar_data[key].append(grp[["ts_recv", "price_f", "size", "side"]].copy())

        if (i + 1) % 5 == 0:
            print(f"  chunk {i+1} ({(i+1)*CHUNK/1e6:.0f}M rows), bars so far: {len(bar_data)}", flush=True)

    print(f"  Computing features for {len(bar_data)} bars...", flush=True)
    records = {}
    for ts, parts in bar_data.items():
        grp = pd.concat(parts).sort_values("ts_recv")
        feat = compute_bar_features(grp)
        if feat:
            records[ts] = feat

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "ts"
    df = df.sort_index()
    print(f"  -> {len(df)} 5-min bars, {df.index[0].date()} to {df.index[-1].date()}", flush=True)
    return df


def add_vpin(df: pd.DataFrame, windows=(5, 20)) -> pd.DataFrame:
    """Rolling VPIN approximation from bar-level buy/sell volumes.

    VPIN = rolling mean of |buy_vol - sell_vol| / avg_volume.
    Bounded in [0, 1]; high values signal elevated informed-flow toxicity.
    """
    V = df["total_vol"].replace(0, np.nan).mean()
    if np.isnan(V) or V <= 0:
        V = 1.0
    imbalance = (df["buy_vol"] - df["sell_vol"]).abs()
    for w in windows:
        df[f"vpin_{w}"] = imbalance.rolling(w, min_periods=w).sum() / (w * V)
    return df


def add_cross_timeframe_ofi(df_5min: pd.DataFrame) -> pd.DataFrame:
    """Join 1-minute OFI parquets to add cross-timeframe OFI features.

    ofi_1m_first : OFI of the opening 1-min sub-bar of each 5-min bar
    ofi_1m_last  : OFI of the closing 1-min sub-bar (T+4 min)
    ofi_15m      : lagged rolling sum of previous 3 bars' OFI (≈15-min lookback)
    """
    ofi_dir = DATA / "ofi_1min"
    parts = [pd.read_parquet(p) for p in sorted(ofi_dir.glob("*.parquet")) if p.exists()]
    if not parts:
        print("  WARNING: no 1-min OFI parquets found; skipping cross-timeframe features.", flush=True)
        for col in ("ofi_1m_first", "ofi_1m_last"):
            df_5min[col] = np.nan
    else:
        ofi1 = pd.concat(parts).sort_index()
        # Ensure UTC timezone alignment
        if ofi1.index.tz is None:
            ofi1.index = ofi1.index.tz_localize("UTC")
        else:
            ofi1.index = ofi1.index.tz_convert("UTC")
        ofi1_series = ofi1["ofi"].astype(float)

        idx_utc = df_5min.index
        if idx_utc.tz is None:
            idx_utc = idx_utc.tz_localize("UTC")

        df_5min["ofi_1m_first"] = ofi1_series.reindex(idx_utc).values
        last_idx = idx_utc + pd.Timedelta(minutes=4)
        df_5min["ofi_1m_last"]  = ofi1_series.reindex(last_idx).values

    # 15-min lagged OFI: shift by 1 bar then rolling sum of 3 (no lookahead)
    df_5min["ofi_15m"] = df_5min["ofi"].shift(1).rolling(3, min_periods=3).sum()
    return df_5min


def main():
    all_parts = []
    for f in FILES:
        if not f.exists():
            print(f"SKIP (not found): {f.name}", flush=True)
            continue
        df = process_file(f)
        all_parts.append(df)

    combined = pd.concat(all_parts).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    print("\nAdding VPIN features...", flush=True)
    combined = add_vpin(combined, windows=(5, 20))

    print("Adding cross-timeframe OFI features...", flush=True)
    combined = add_cross_timeframe_ofi(combined)

    combined.to_parquet(OUT)
    print(f"\nSaved {len(combined)} bars to {OUT}", flush=True)
    print(f"Columns ({len(combined.columns)}): {list(combined.columns)}", flush=True)
    new_cols = ["ofi_q1","ofi_q2","ofi_q3","ofi_q4","ofi_q5",
                "trade_rate_accel","vpin_5","vpin_20",
                "ofi_1m_first","ofi_1m_last","ofi_15m"]
    print("\nNew feature preview:", flush=True)
    print(combined[new_cols].describe().to_string(), flush=True)


if __name__ == "__main__":
    main()
