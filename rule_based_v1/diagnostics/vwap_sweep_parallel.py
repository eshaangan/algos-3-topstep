"""VWAP MR parameter sweep — parallelized across all CPU cores.

Uses multiprocessing.Pool to run grid configs simultaneously.
All data pre-converted to numpy arrays (no pandas per iteration).
MPS GPU available for future ML work — not needed here (sequential sim).

Usage:
    cd "algos 3 topstep"
    python rule_based_v1/diagnostics/vwap_sweep_parallel.py
    python rule_based_v1/diagnostics/vwap_sweep_parallel.py --top 30
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_PATH = ROOT / "rule_based_v1" / "diagnostics" / "vwap_sweep_results.json"

_DATA_CANDIDATES = [
    ROOT / "data" / "processed" / "mnq_2026ytd_databento_5min_rth.h5",
    ROOT / "data" / "processed" / "mnq_2026ytd_5min.h5",
]

POINT_VALUE = 2.0
TICK_SIZE   = 0.25
COMMISSION  = 0.62
N_CONTRACTS = 5
MAX_DAILY_LOSS = -700.0
MAX_PER_DAY    = 2
TIME_STOP_BARS = 12
MIN_TRADES     = 15

# ---------------------------------------------------------------------------
# Pre-compute everything as numpy arrays — done once, shared across workers
# ---------------------------------------------------------------------------
def prepare_data():
    for p in _DATA_CANDIDATES:
        if p.exists():
            bars = pd.read_hdf(str(p), key="bars_5min")
            if bars.index.tz is None:
                bars.index = bars.index.tz_localize("US/Eastern")
            else:
                bars.index = bars.index.tz_convert("US/Eastern")
            break
    else:
        raise FileNotFoundError("No 2026 YTD data found")

    print(f"Loaded {len(bars):,} bars [{bars.index[0].date()} → {bars.index[-1].date()}]")

    # ATR (EWM span=14)
    prev = bars["close"].shift(1)
    tr = pd.concat([bars["high"]-bars["low"],
                    (bars["high"]-prev).abs(),
                    (bars["low"]-prev).abs()], axis=1).max(axis=1)
    atr_s = tr.ewm(span=14, adjust=False).mean()

    # Session VWAP
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3
    date_idx = bars.index.map(lambda t: t.date())
    vwap_s = (tp * bars["volume"]).groupby(date_idx).cumsum() / \
             bars["volume"].groupby(date_idx).cumsum().replace(0, np.nan)

    # Day-open price per bar
    day_opens = bars.groupby(bars.index.map(lambda t: t.date()))["open"].transform("first")

    # Bar times as integer minutes since midnight (fast comparison)
    bar_minutes = bars.index.hour * 60 + bars.index.minute

    # Date integers for day boundary detection
    bar_dates = np.array([t.toordinal() for t in bars.index])

    return {
        "close":       bars["close"].values.astype(np.float64),
        "high":        bars["high"].values.astype(np.float64),
        "low":         bars["low"].values.astype(np.float64),
        "atr":         atr_s.values.astype(np.float64),
        "vwap":        vwap_s.values.astype(np.float64),
        "day_open":    day_opens.values.astype(np.float64),
        "bar_minutes": bar_minutes.values.astype(np.int32),
        "bar_dates":   bar_dates,
        "n_bars":      len(bars),
    }

# ---------------------------------------------------------------------------
# Single-config simulation — runs in worker process
# ---------------------------------------------------------------------------
def _sim(args):
    cfg, arrays = args
    ed, pt, sl, tstart_m, tend_m, max_move = cfg

    close      = arrays["close"]
    high       = arrays["high"]
    low        = arrays["low"]
    atr        = arrays["atr"]
    vwap       = arrays["vwap"]
    day_open   = arrays["day_open"]
    minutes    = arrays["bar_minutes"]
    dates      = arrays["bar_dates"]
    n          = arrays["n_bars"]

    TICK = TICK_SIZE
    PV   = POINT_VALUE
    COMM = COMMISSION
    NC   = N_CONTRACTS

    pos_active    = False
    pos_dir       = 0
    pos_entry     = 0.0
    pos_sl        = 0.0
    pos_pt        = 0.0
    pos_bar_stop  = 0

    trades     = []
    cur_date   = dates[16]
    daily_pnl  = 0.0
    day_map    = {}
    vday       = 0
    equity     = 50_000.0
    eq         = [equity]

    for i in range(16, n):
        d = dates[i]

        # Day boundary
        if d != cur_date:
            day_map[cur_date] = daily_pnl
            daily_pnl = 0.0
            vday = 0
            cur_date = d

        bt_m = minutes[i]
        # Session close: >=955 (15:55)
        sess_close = (bt_m >= 955)
        is_last = (i + 1 >= n) or (dates[i+1] != d)
        sess_close = sess_close or is_last

        # ---- Check open position ----
        if pos_active:
            h = high[i]; l = low[i]; c = close[i]
            exited = False; ep = 0.0; reason = ""
            if sess_close or i >= pos_bar_stop:
                ep = c - pos_dir * TICK
                reason = "time_stop"
                exited = True
            elif pos_dir == 1:
                if l <= pos_sl:
                    ep = pos_sl - TICK; reason = "stop_loss"; exited = True
                elif h >= pos_pt:
                    ep = pos_pt - TICK; reason = "profit_target"; exited = True
            if exited:
                p = (ep - pos_entry) * pos_dir * NC * PV - 2 * COMM * NC
                trades.append(p)
                daily_pnl += p
                equity += p
                eq.append(equity)
                pos_active = False

        if sess_close:
            continue
        if pos_active or daily_pnl <= MAX_DAILY_LOSS or vday >= MAX_PER_DAY:
            continue
        if not (tstart_m <= bt_m <= tend_m):
            continue

        atr_val = atr[i]
        if np.isnan(atr_val) or atr_val <= 0:
            continue
        vwap_val = vwap[i]
        if np.isnan(vwap_val):
            continue

        c = close[i]
        prev_c = close[i-1]
        dev = (c - vwap_val) / atr_val

        if abs(dev) > 3.0:
            continue

        # Regime gate
        if max_move > 0:
            do = day_open[i]
            mv = (c - do) / atr_val
            if dev < 0 and mv < -max_move:
                continue

        if dev <= -ed and c > prev_c:
            entry = c + TICK
            pos_active   = True
            pos_dir      = 1
            pos_entry    = entry
            pos_sl       = entry - sl * atr_val
            pos_pt       = entry + pt * atr_val
            pos_bar_stop = i + TIME_STOP_BARS
            vday += 1

    if cur_date not in day_map:
        day_map[cur_date] = daily_pnl

    if len(trades) < MIN_TRADES:
        return None

    trades_arr = np.array(trades)
    wins  = (trades_arr > 0).sum()
    total = trades_arr.sum()
    gp    = trades_arr[trades_arr > 0].sum() if wins > 0 else 0
    gl    = abs(trades_arr[trades_arr <= 0].sum())

    eq_arr = np.array(eq)
    max_dd = float((eq_arr - np.maximum.accumulate(eq_arr)).min())

    daily_arr = np.array(list(day_map.values()))
    active    = daily_arr[daily_arr != 0]
    n_wks     = max(1, len(daily_arr) / 5)
    sharpe    = float(active.mean() / active.std() * np.sqrt(252)) if len(active) > 1 and active.std() > 0 else 0

    return {
        "entry_dist": ed, "pt": pt, "sl": sl,
        "tstart_m": tstart_m, "tend_m": tend_m, "max_move": max_move,
        "n": int(len(trades)), "wr": round(float(wins / len(trades)), 3),
        "weekly_pnl": round(float(total / n_wks), 1),
        "max_dd": round(max_dd, 1),
        "sharpe": round(sharpe, 3),
        "pf": round(float(gp / gl), 3) if gl > 0 else 99.0,
    }

# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=20, help="Print top N results")
    parser.add_argument("--cores", type=int, default=None,
                        help="Number of CPU cores (default: all available)")
    args = parser.parse_args()

    n_cores = args.cores or os.cpu_count()
    print(f"Using {n_cores} CPU cores")

    arrays = prepare_data()

    # Build grid — time in minutes since midnight
    entry_dists = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    pt_mults    = [1.0, 1.5, 2.0, 2.5, 3.0]
    sl_mults    = [0.75, 1.0, 1.25, 1.5]
    starts_m    = [570, 600, 630, 660]    # 9:30, 10:00, 10:30, 11:00
    ends_m      = [750, 780, 810, 840]    # 12:30, 13:00, 13:30, 14:00
    max_moves   = [1.0, 1.5, 2.0, 3.0, 9999.0]  # 9999 = disabled

    grid = [(ed, pt, sl, ts, te, mm)
            for ed, pt, sl, ts, te, mm in product(entry_dists, pt_mults, sl_mults, starts_m, ends_m, max_moves)
            if pt > sl]

    print(f"Grid size: {len(grid):,} configurations")
    print(f"Distributing across {n_cores} cores...\n")

    tasks = [(cfg, arrays) for cfg in grid]

    with mp.Pool(n_cores) as pool:
        raw = pool.map(_sim, tasks, chunksize=max(1, len(tasks) // (n_cores * 4)))

    results = [r for r in raw if r is not None]
    results.sort(key=lambda x: x["sharpe"] * x["wr"], reverse=True)

    print(f"Valid configs (≥{MIN_TRADES} trades): {len(results)} / {len(grid)}")

    # Decode minutes back to HH:MM for display
    def m2t(m): return f"{m//60:02d}:{m%60:02d}"

    print(f"\nTop {args.top} by Sharpe×WR:\n")
    print(f"  {'Dist':>5} {'PT':>4} {'SL':>5} {'Start':>6} {'End':>6} {'MaxMv':>6} "
          f"{'N':>5} {'WR':>7} {'$/wk':>9} {'MaxDD':>9} {'Sharpe':>8} {'PF':>6}")
    print(f"  {'-'*80}")
    for r in results[:args.top]:
        mm = f"{r['max_move']:.1f}" if r['max_move'] < 999 else "off"
        print(f"  {r['entry_dist']:>5.2f} {r['pt']:>4.1f} {r['sl']:>5.2f} "
              f"{m2t(r['tstart_m']):>6} {m2t(r['tend_m']):>6} {mm:>6} "
              f"{r['n']:>5} {r['wr']:>7.1%} {r['weekly_pnl']:>9,.0f} "
              f"{r['max_dd']:>9,.0f} {r['sharpe']:>8.3f} {r['pf']:>6.2f}")

    # Baseline — default config
    baseline_cfg = (1.0, 2.0, 1.5, 630, 810, 2.0)
    b = _sim((baseline_cfg, arrays))
    if b:
        print(f"\nBaseline (dist=1.0, PT=2.0, SL=1.5, 10:30-13:30, mv=2.0):")
        print(f"  n={b['n']}  WR={b['wr']:.1%}  $/wk=${b['weekly_pnl']:,.0f}  "
              f"DD=${b['max_dd']:,.0f}  Sharpe={b['sharpe']:.3f}  PF={b['pf']:.2f}")

    # Save results
    save = []
    for r in results[:50]:
        save.append({**r, "tstart": m2t(r["tstart_m"]), "tend": m2t(r["tend_m"])})
    with open(RESULTS_PATH, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved top 50 → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
