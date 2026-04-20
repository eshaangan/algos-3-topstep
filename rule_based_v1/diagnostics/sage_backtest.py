"""SAGE (Synthetic Amplification Gamma Engine) Backtest — MES 1-min RTH bars.

4-stage backtest:
  Stage 1: Morning ORB-pullback trade (10:01–12:00 ET)
  Stage 2: Closing echo trade (15:35–15:57 ET)
  Validation: monotonicity table, threshold sweep, feature importance

Instrument: MES, $5/pt, 1 contract, 0.25 pt slippage per side.
Data: data/processed/mes_1m_bars_cache.h5, key /bars_1m

Run:
    python rule_based_v1/diagnostics/sage_backtest.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "processed"
DIAG_DIR = ROOT / "rule_based_v1" / "diagnostics"
MES_PATH = DATA_DIR / "mes_1m_bars_cache.h5"
RESULTS_PATH = DIAG_DIR / "sage_results.json"

for p in [str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POINT_VALUE = 5.0   # $/pt MES
SLIP = 0.25         # slippage per side (pts)
N_CONTRACTS = 1
BURN_IN = 60        # days before trading begins
OR_MED_WINDOW = 30  # rolling OR median window
PCTL_WINDOW = 60    # rolling S percentile window


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_bars() -> pd.DataFrame:
    with pd.HDFStore(str(MES_PATH), mode="r") as store:
        df = store["/bars_1m"].copy()
    df = df.set_index("timestamp")
    df.index = df.index.tz_convert("US/Eastern")
    df = df.sort_index()
    # RTH only: 09:30–15:59 ET
    mod = df.index.hour * 60 + df.index.minute
    df = df[(mod >= 570) & (mod <= 959)].copy()
    return df


# ---------------------------------------------------------------------------
# Per-day feature computation
# ---------------------------------------------------------------------------

def compute_day_features(
    date: pd.Timestamp,
    day_bars: pd.DataFrame,
    prior_rth_close: float,
    or_median: float,
) -> Optional[dict]:
    """Compute SAGE features for a single day from its 1-min RTH bars."""
    # Opening drive window: 9:30–10:00 ET (bars where minute <= 10:00)
    od_mask = (day_bars.index.hour == 9) & (day_bars.index.minute >= 30)
    od_mask |= (day_bars.index.hour == 10) & (day_bars.index.minute == 0)
    od = day_bars[od_mask].copy()

    if len(od) < 5:
        return None

    O_open = od.iloc[0]["open"]      # 9:30 open
    C_drive = od.iloc[-1]["close"]   # 10:00 close
    H_or = od["high"].max()
    L_or = od["low"].min()
    OR = H_or - L_or

    if OR < 1e-6:
        return None

    s = 1 if C_drive >= O_open else -1

    # I — impulse efficiency
    I = abs(C_drive - O_open) / (OR + 1e-9)

    # R — range surprise
    R = OR / (or_median + 1e-9)

    # P — max adverse retrace during drive
    if s == 1:
        P_raw = max((max(O_open - row.low, 0.0) for row in od.itertuples()), default=0.0)
    else:
        P_raw = max((max(row.high - O_open, 0.0) for row in od.itertuples()), default=0.0)
    P = P_raw / (OR + 1e-9)

    # CL — signed close-location value
    clv_vals = []
    for _, row in od.iterrows():
        bar_range = row["high"] - row["low"] + 1e-9
        clv = s * (2.0 * (row["close"] - row["low"]) / bar_range - 1.0)
        clv_vals.append(clv)
    CL = float(np.mean(clv_vals))

    # Q — return-sign concentration
    closes = od["close"].values
    if len(closes) > 1:
        signs = [np.sign(closes[j] - closes[j - 1]) for j in range(1, len(closes))]
        Q = abs(sum(signs)) / len(signs)
    else:
        Q = 0.0

    # G — gap non-repair
    gap = abs(O_open - prior_rth_close)
    remaining = abs(C_drive - prior_rth_close)
    G_raw = 1.0 - remaining / (gap + 1e-9)
    G = float(np.clip(G_raw, 0.0, 1.0))
    G_neg = max(0.0, G - 0.5)

    # Score
    S = I + R + CL + Q - P - G_neg

    return {
        "date": date,
        "O_open": O_open,
        "C_drive": C_drive,
        "H_or": H_or,
        "L_or": L_or,
        "OR": OR,
        "s": s,
        "I": I,
        "R": R,
        "P": P,
        "CL": CL,
        "Q": Q,
        "G": G,
        "G_neg": G_neg,
        "S": S,
    }


# ---------------------------------------------------------------------------
# Label computation
# ---------------------------------------------------------------------------

def compute_label(
    date: pd.Timestamp,
    day_bars: pd.DataFrame,
    feat: dict,
) -> int:
    """TP=1.25*OR, SL=0.60*OR, scan 10:01–12:00 ET."""
    s = feat["s"]
    C_drive = feat["C_drive"]
    OR = feat["OR"]

    TP_level = C_drive + s * 1.25 * OR
    SL_level = C_drive - s * 0.60 * OR

    # Post-drive bars: after 10:00, until 12:00
    post_mask = (
        (day_bars.index.hour == 10) & (day_bars.index.minute >= 1)
        | (day_bars.index.hour == 11)
        | (day_bars.index.hour == 12) & (day_bars.index.minute == 0)
    )
    post = day_bars[post_mask]

    y = 0
    for row in post.itertuples():
        if s == 1:
            if row.high >= TP_level:
                y = 1
                break
            if row.low <= SL_level:
                y = 0
                break
        else:
            if row.low <= TP_level:
                y = 1
                break
            if row.high >= SL_level:
                y = 0
                break
    return y


# ---------------------------------------------------------------------------
# Stage 1: Morning pullback entry
# ---------------------------------------------------------------------------

def run_stage1_trade(
    day_bars: pd.DataFrame,
    feat: dict,
) -> dict:
    """Attempt morning pullback entry 10:01–10:25 ET.

    Returns a trade dict with keys: traded, pnl, win, entry, sl, tp,
    exit_reason, entry_time.
    """
    s = feat["s"]
    C_drive = feat["C_drive"]
    OR = feat["OR"]
    null_result = {"traded": False, "pnl": 0.0, "win": False}

    # Scan window: 10:01–10:25 ET
    scan_mask = (day_bars.index.hour == 10) & (day_bars.index.minute >= 1) & (day_bars.index.minute <= 25)
    scan_bars = day_bars[scan_mask]
    if len(scan_bars) == 0:
        return null_result

    pullback_bar = None
    pullback_idx = None

    for i, (ts, row) in enumerate(scan_bars.iterrows()):
        # Check pullback touch
        if s == 1:
            touched = row["low"] <= C_drive - 0.15 * OR
        else:
            touched = row["high"] >= C_drive + 0.15 * OR

        if not touched:
            continue

        # Validate depth <= 35% of OR
        if s == 1:
            retrace = C_drive - row["low"]
        else:
            retrace = row["high"] - C_drive

        if retrace > 0.35 * OR:
            # Too deep — skip and keep looking
            continue

        pullback_bar = row
        pullback_idx = i
        break

    if pullback_bar is None:
        return null_result

    # Now look for trigger bar (break of pullback extreme back in drive direction)
    trigger_bars = scan_bars.iloc[pullback_idx + 1:]
    entry_price = None
    entry_time = None

    for ts, row in trigger_bars.iterrows():
        if s == 1:
            if row["high"] > pullback_bar["high"]:
                entry_price = pullback_bar["high"] + SLIP
                entry_time = ts
                break
        else:
            if row["low"] < pullback_bar["low"]:
                entry_price = pullback_bar["low"] - SLIP
                entry_time = ts
                break

    if entry_price is None:
        return null_result

    stop_dist = 0.45 * OR
    sl = entry_price - s * stop_dist
    tp = entry_price + s * stop_dist  # 1R TP

    # Simulate trade: scan from entry_time until 12:00 ET
    post_entry_mask = day_bars.index > entry_time
    hard_exit_mask = (day_bars.index.hour < 12) | (
        (day_bars.index.hour == 12) & (day_bars.index.minute == 0)
    )
    trade_bars = day_bars[post_entry_mask & hard_exit_mask]

    exit_price = None
    exit_reason = "hard_exit_1200"

    for ts, row in trade_bars.iterrows():
        if s == 1:
            if row["high"] >= tp:
                exit_price = tp
                exit_reason = "tp"
                break
            if row["low"] <= sl:
                exit_price = sl
                exit_reason = "sl"
                break
        else:
            if row["low"] <= tp:
                exit_price = tp
                exit_reason = "tp"
                break
            if row["high"] >= sl:
                exit_price = sl
                exit_reason = "sl"
                break

    # Hard exit at 12:00
    if exit_price is None:
        # Find the 12:00 bar or last bar before it
        hard_bars = day_bars[
            (day_bars.index.hour == 12) & (day_bars.index.minute == 0)
        ]
        if len(hard_bars) > 0:
            exit_price = hard_bars.iloc[0]["close"]
        elif len(trade_bars) > 0:
            exit_price = trade_bars.iloc[-1]["close"]
        else:
            exit_price = entry_price
        exit_reason = "hard_exit_1200"

    gross_pts = s * (exit_price - entry_price)
    net_pts = gross_pts - 2 * SLIP
    pnl = net_pts * POINT_VALUE * N_CONTRACTS
    win = pnl > 0.0

    return {
        "traded": True,
        "pnl": round(pnl, 2),
        "win": win,
        "entry": entry_price,
        "sl": sl,
        "tp": tp,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "entry_time": str(entry_time),
        "stop_dist": stop_dist,
    }


# ---------------------------------------------------------------------------
# Stage 2: Closing echo trade
# ---------------------------------------------------------------------------

def run_stage2_trade(
    day_bars: pd.DataFrame,
    feat: dict,
    s1_result: dict,
    s_pct: float,
) -> dict:
    """Closing echo: 15:35–15:57 ET.

    Fires only if:
      - Stage 1 hit TP OR no Stage 1 trade fired
      - S_pct > 65
      - Day return at 15:30 still in drive direction
    """
    null_result = {"traded": False, "pnl": 0.0, "win": False}

    # Eligibility
    s1_traded = s1_result["traded"]
    s1_tp = s1_traded and s1_result.get("exit_reason") == "tp"
    if s1_traded and not s1_tp:
        return null_result

    if s_pct <= 65.0:
        return null_result

    s = feat["s"]
    C_drive = feat["C_drive"]
    OR = feat["OR"]

    # Check 15:30 bar still in direction
    bar_1530 = day_bars[
        (day_bars.index.hour == 15) & (day_bars.index.minute == 30)
    ]
    if len(bar_1530) == 0:
        return null_result
    C_1530 = bar_1530.iloc[0]["close"]

    if s == 1 and C_1530 <= C_drive:
        return null_result
    if s == -1 and C_1530 >= C_drive:
        return null_result

    # Find 15:30–15:34 range
    ref_mask = (
        (day_bars.index.hour == 15)
        & (day_bars.index.minute >= 30)
        & (day_bars.index.minute <= 34)
    )
    ref_bars = day_bars[ref_mask]
    if len(ref_bars) == 0:
        return null_result

    ref_high = ref_bars["high"].max()
    ref_low = ref_bars["low"].min()

    # Entry scan: 15:35–15:50 ET
    entry_scan_mask = (
        (day_bars.index.hour == 15)
        & (day_bars.index.minute >= 35)
        & (day_bars.index.minute <= 50)
    )
    entry_scan = day_bars[entry_scan_mask]

    entry_price = None
    entry_time = None

    for ts, row in entry_scan.iterrows():
        if s == 1:
            if row["high"] > ref_high:
                entry_price = ref_high + SLIP
                entry_time = ts
                break
        else:
            if row["low"] < ref_low:
                entry_price = ref_low - SLIP
                entry_time = ts
                break

    if entry_price is None:
        return null_result

    stop_dist = 0.30 * OR
    tp_dist = 0.50 * OR
    sl = entry_price - s * stop_dist
    tp = entry_price + s * tp_dist

    # Simulate until 15:57 ET
    post_entry_mask = day_bars.index > entry_time
    hard_exit_mask = (
        (day_bars.index.hour == 15) & (day_bars.index.minute <= 57)
    )
    trade_bars = day_bars[post_entry_mask & hard_exit_mask]

    exit_price = None
    exit_reason = "hard_exit_1557"

    for ts, row in trade_bars.iterrows():
        if s == 1:
            if row["high"] >= tp:
                exit_price = tp
                exit_reason = "tp"
                break
            if row["low"] <= sl:
                exit_price = sl
                exit_reason = "sl"
                break
        else:
            if row["low"] <= tp:
                exit_price = tp
                exit_reason = "tp"
                break
            if row["high"] >= sl:
                exit_price = sl
                exit_reason = "sl"
                break

    if exit_price is None:
        hard_bars = day_bars[
            (day_bars.index.hour == 15) & (day_bars.index.minute == 57)
        ]
        if len(hard_bars) > 0:
            exit_price = hard_bars.iloc[0]["close"]
        elif len(trade_bars) > 0:
            exit_price = trade_bars.iloc[-1]["close"]
        else:
            exit_price = entry_price
        exit_reason = "hard_exit_1557"

    gross_pts = s * (exit_price - entry_price)
    net_pts = gross_pts - 2 * SLIP
    pnl = net_pts * POINT_VALUE * N_CONTRACTS
    win = pnl > 0.0

    return {
        "traded": True,
        "pnl": round(pnl, 2),
        "win": win,
        "entry": entry_price,
        "sl": sl,
        "tp": tp,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "entry_time": str(entry_time),
    }


# ---------------------------------------------------------------------------
# Percentile rank utility
# ---------------------------------------------------------------------------

def percentile_rank(val: float, history: list) -> float:
    """Fraction of history values strictly below val, scaled to [0,100]."""
    if len(history) == 0:
        return 50.0
    return 100.0 * sum(1 for x in history if x < val) / len(history)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_metrics(pnl_series: list) -> dict:
    if len(pnl_series) == 0:
        return {"n": 0, "wr": 0.0, "avg_pnl": 0.0, "sharpe": 0.0, "max_dd": 0.0, "total_pnl": 0.0}
    arr = np.array(pnl_series)
    n = len(arr)
    wr = float(np.mean(arr > 0))
    avg_pnl = float(np.mean(arr))
    total_pnl = float(np.sum(arr))
    std = float(np.std(arr))
    sharpe = (avg_pnl / std * np.sqrt(252)) if std > 0 else 0.0
    # Max drawdown on cumulative
    cum = np.cumsum(arr)
    running_max = np.maximum.accumulate(cum)
    dd = cum - running_max
    max_dd = float(np.min(dd))
    return {
        "n": n,
        "wr": round(wr, 4),
        "avg_pnl": round(avg_pnl, 2),
        "total_pnl": round(total_pnl, 2),
        "sharpe": round(sharpe, 4),
        "max_dd": round(max_dd, 2),
    }


# ---------------------------------------------------------------------------
# Main backtest engine
# ---------------------------------------------------------------------------

def run_backtest(df: pd.DataFrame) -> dict:
    # Group by trading date
    dates = sorted(df.index.normalize().unique())
    print(f"Total RTH trading days: {len(dates)}")

    # Pass 1: compute features and labels for all days
    all_features = []
    or_history = []

    for i, date in enumerate(dates):
        day_str = str(date.date())
        day_bars = df[df.index.normalize() == date]

        # Prior RTH close: last close of prior RTH day
        if i == 0:
            prior_rth_close = None
        else:
            prev_date = dates[i - 1]
            prev_bars = df[df.index.normalize() == prev_date]
            prior_rth_close = float(prev_bars.iloc[-1]["close"]) if len(prev_bars) > 0 else None

        if prior_rth_close is None:
            or_history.append(None)
            all_features.append(None)
            continue

        # Rolling 30-day OR median (using prior days only — no lookahead)
        valid_ors = [f["OR"] for f in all_features if f is not None]
        if len(valid_ors) >= 5:
            window_ors = valid_ors[-OR_MED_WINDOW:]
            or_median = float(np.median(window_ors))
        elif len(valid_ors) > 0:
            or_median = float(np.median(valid_ors))
        else:
            or_median = 5.0  # fallback

        feat = compute_day_features(date, day_bars, prior_rth_close, or_median)

        if feat is None:
            all_features.append(None)
            or_history.append(None)
            continue

        feat["prior_rth_close"] = prior_rth_close
        feat["or_median"] = or_median

        # Compute label
        y = compute_label(date, day_bars, feat)
        feat["y"] = y

        all_features.append(feat)
        or_history.append(feat["OR"])

    # Pass 2: compute S_pct for each day using prior 60-day S distribution
    s_history = []
    for i, feat in enumerate(all_features):
        if feat is None:
            s_history.append(None)
            continue
        if i < BURN_IN:
            feat["S_pct"] = None
            s_history.append(feat["S"])
        else:
            prior_s = [s_history[j] for j in range(max(0, i - PCTL_WINDOW), i) if s_history[j] is not None]
            if len(prior_s) >= 10:
                feat["S_pct"] = percentile_rank(feat["S"], prior_s)
            else:
                feat["S_pct"] = None
            s_history.append(feat["S"])

    # Pass 3: run Stage 1 and Stage 2 trades on days 61+
    day_records = []
    THRESHOLDS = [40, 50, 55, 60, 65, 70, 75, 80]

    for i, (date, feat) in enumerate(zip(dates, all_features)):
        if feat is None:
            continue
        if i < BURN_IN:
            continue
        if feat["S_pct"] is None:
            continue

        day_bars = df[df.index.normalize() == date]
        s_pct = feat["S_pct"]

        # Run Stage 1 for every day (threshold filtering done in analysis)
        s1 = run_stage1_trade(day_bars, feat)
        s2 = run_stage2_trade(day_bars, feat, s1, s_pct)

        rec = {
            "date": str(date.date()),
            "s": feat["s"],
            "OR": feat["OR"],
            "I": feat["I"],
            "R": feat["R"],
            "P": feat["P"],
            "CL": feat["CL"],
            "Q": feat["Q"],
            "G_neg": feat["G_neg"],
            "S": feat["S"],
            "S_pct": s_pct,
            "y": feat["y"],
            # Stage 1
            "s1_traded": s1["traded"],
            "s1_pnl": s1["pnl"],
            "s1_win": s1["win"],
            "s1_exit_reason": s1.get("exit_reason", ""),
            # Stage 2
            "s2_traded": s2["traded"],
            "s2_pnl": s2["pnl"],
            "s2_win": s2["win"],
        }
        day_records.append(rec)

    results_df = pd.DataFrame(day_records)
    print(f"\nTradeable days (after burn-in): {len(results_df)}")

    # -----------------------------------------------------------------------
    # Analysis: monotonicity table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MONOTONICITY TABLE — S_pct deciles vs label hit-rate and Stage-1 WR%")
    print("=" * 70)

    bins = list(range(0, 101, 10))
    labels_bin = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
    results_df["decile"] = pd.cut(
        results_df["S_pct"], bins=bins, labels=labels_bin, include_lowest=True
    )

    mono_rows = []
    for decile in labels_bin:
        sub = results_df[results_df["decile"] == decile]
        n = len(sub)
        if n == 0:
            mono_rows.append({"decile": decile, "N": 0, "label_hr": 0.0, "s1_wr": 0.0, "s1_n": 0})
            continue
        label_hr = float(sub["y"].mean()) if n > 0 else 0.0
        s1_sub = sub[sub["s1_traded"]]
        s1_wr = float(s1_sub["s1_win"].mean()) if len(s1_sub) > 0 else 0.0
        mono_rows.append({
            "decile": decile,
            "N": n,
            "label_hr": round(label_hr, 3),
            "s1_wr": round(s1_wr, 3),
            "s1_n": len(s1_sub),
        })

    mono_df = pd.DataFrame(mono_rows)
    header = f"{'Decile':>10} {'N':>5} {'LabelHR%':>10} {'S1 N':>6} {'S1 WR%':>8}"
    print(header)
    print("-" * len(header))
    for _, row in mono_df.iterrows():
        print(
            f"{row['decile']:>10} {row['N']:>5} "
            f"{row['label_hr']*100:>9.1f}% "
            f"{row['s1_n']:>6} "
            f"{row['s1_wr']*100:>7.1f}%"
        )

    # ASCII bar chart of label hit-rate by decile
    print("\nLabel hit-rate by S_pct decile (ASCII):")
    max_hr = max(r["label_hr"] for r in mono_rows) or 1.0
    bar_width = 40
    for row in mono_rows:
        bar_len = int(row["label_hr"] / max_hr * bar_width) if max_hr > 0 else 0
        bar = "#" * bar_len
        print(f"  {row['decile']:>6}: {bar:<{bar_width}} {row['label_hr']*100:.1f}%")

    # -----------------------------------------------------------------------
    # Analysis: threshold sweep (Stage 1)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("THRESHOLD SWEEP — Stage 1 only (S_pct threshold)")
    print("=" * 70)
    header2 = f"{'Thresh':>7} {'N':>5} {'WR%':>7} {'AvgPnL':>8} {'TotalPnL':>10} {'Sharpe':>8} {'MaxDD':>8}"
    print(header2)
    print("-" * len(header2))

    sweep_rows = []
    for thresh in THRESHOLDS:
        subset = results_df[(results_df["S_pct"] > thresh) & results_df["s1_traded"]]
        m = compute_metrics(list(subset["s1_pnl"]))
        print(
            f"{thresh:>7} {m['n']:>5} {m['wr']*100:>6.1f}% "
            f"${m['avg_pnl']:>7.2f} ${m['total_pnl']:>9.2f} "
            f"{m['sharpe']:>7.3f} ${m['max_dd']:>7.2f}"
        )
        sweep_rows.append({"threshold": thresh, **m})

    # -----------------------------------------------------------------------
    # Analysis: Stage 1 vs Stage 1+2
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE COMPARISON (baseline: S_pct > 50)")
    print("=" * 70)

    base_thresh = 50
    base_sub = results_df[results_df["S_pct"] > base_thresh]

    s1_pnls = list(base_sub[base_sub["s1_traded"]]["s1_pnl"])
    s2_pnls = list(base_sub[base_sub["s2_traded"]]["s2_pnl"])

    # Combined: s1 + s2 pnl per day
    combined_by_day = base_sub.copy()
    combined_by_day["combined_pnl"] = combined_by_day["s1_pnl"] + combined_by_day["s2_pnl"]
    combined_pnls = list(combined_by_day["combined_pnl"])

    m1 = compute_metrics(s1_pnls)
    m2 = compute_metrics(s2_pnls)
    mc = compute_metrics(combined_pnls)

    print(f"{'Stage':>15} {'N':>5} {'WR%':>7} {'AvgPnL':>8} {'TotalPnL':>10} {'Sharpe':>8} {'MaxDD':>8}")
    print("-" * 60)
    for label, m in [("Stage 1 only", m1), ("Stage 2 only", m2), ("S1 + S2 combined", mc)]:
        print(
            f"{label:>15} {m['n']:>5} {m['wr']*100:>6.1f}% "
            f"${m['avg_pnl']:>7.2f} ${m['total_pnl']:>9.2f} "
            f"{m['sharpe']:>7.3f} ${m['max_dd']:>7.2f}"
        )

    # -----------------------------------------------------------------------
    # Analysis: feature importance (correlation with label y and S1 PnL)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE — Correlation with label y and Stage-1 PnL")
    print("=" * 70)
    features = ["I", "R", "CL", "Q", "P", "G_neg"]
    fi_rows = []
    s1_traded_df = results_df[results_df["s1_traded"]].copy()

    header3 = f"{'Feature':>8} {'Corr(y)':>10} {'Corr(S1_PnL)':>14}"
    print(header3)
    print("-" * len(header3))
    for f in features:
        corr_y = float(results_df[f].corr(results_df["y"].astype(float)))
        corr_pnl = float(s1_traded_df[f].corr(s1_traded_df["s1_pnl"])) if len(s1_traded_df) > 0 else 0.0
        print(f"{f:>8} {corr_y:>10.4f} {corr_pnl:>14.4f}")
        fi_rows.append({"feature": f, "corr_label": round(corr_y, 4), "corr_s1_pnl": round(corr_pnl, 4)})

    # -----------------------------------------------------------------------
    # Overall summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY (all Stage-1 trades, no S_pct filter)")
    print("=" * 70)
    all_s1 = results_df[results_df["s1_traded"]]
    m_all = compute_metrics(list(all_s1["s1_pnl"]))
    print(f"  Stage-1 trades: {m_all['n']}")
    print(f"  Win rate:       {m_all['wr']*100:.1f}%")
    print(f"  Avg PnL:        ${m_all['avg_pnl']:.2f}")
    print(f"  Total PnL:      ${m_all['total_pnl']:.2f}")
    print(f"  Sharpe:         {m_all['sharpe']:.3f}")
    print(f"  Max DD:         ${m_all['max_dd']:.2f}")

    # Exit reason breakdown
    print("\nStage-1 exit reasons:")
    for reason, grp in all_s1.groupby("s1_exit_reason"):
        m_r = compute_metrics(list(grp["s1_pnl"]))
        print(f"  {reason:<20} N={m_r['n']:>3}  WR={m_r['wr']*100:.1f}%  AvgPnL=${m_r['avg_pnl']:.2f}")

    # -----------------------------------------------------------------------
    # Build results JSON
    # -----------------------------------------------------------------------
    results = {
        "meta": {
            "strategy": "SAGE",
            "instrument": "MES",
            "point_value": POINT_VALUE,
            "slippage_per_side": SLIP,
            "n_contracts": N_CONTRACTS,
            "burn_in_days": BURN_IN,
            "total_days": len(dates),
            "tradeable_days": len(results_df),
        },
        "monotonicity_table": mono_rows,
        "threshold_sweep": sweep_rows,
        "stage_comparison": {
            "threshold_used": base_thresh,
            "stage1": m1,
            "stage2": m2,
            "combined": mc,
        },
        "feature_importance": fi_rows,
        "overall_stage1": m_all,
        "day_records": day_records,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {RESULTS_PATH}")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading MES 1-min RTH bars...")
    df = load_bars()
    print(f"Loaded {len(df):,} bars  |  {df.index[0]}  ->  {df.index[-1]}")
    run_backtest(df)


if __name__ == "__main__":
    main()
