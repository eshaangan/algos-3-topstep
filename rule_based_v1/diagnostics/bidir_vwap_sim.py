"""
Bidirectional ATR-based VWAP MR simulation — pure numpy, fast.

Strategy: Enter LONG when price is X ATR below VWAP (bounce up expected)
          Enter SHORT when price is X ATR above VWAP (reversion down expected)
Exit: PT=2.0x ATR, SL=1.5x ATR, time_stop=12 bars
Regime gate: don't take LONG entries if price has fallen >2.5x ATR from open (trending down)
             don't take SHORT entries if price has risen >2.5x ATR from open (trending up)

Tests across 10, 15, 20, 25, 30 MNQ contracts (Topstep max = 50).
Sweeps entry distance: 0.5, 0.75, 1.0, 1.25, 1.5
Compares: long-only vs bidirectional
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

DATA_CANDIDATES = [
    ROOT / "data" / "processed" / "mnq_2026ytd_databento_5min_rth.h5",
    ROOT / "data" / "processed" / "mnq_2026ytd_5min.h5",
]

POINT_VALUE   = 2.0
TICK_SIZE     = 0.25
COMMISSION    = 0.62
MAX_DAILY_PCT = 0.032   # $1,600 / $50,000
TSTART_M      = 630     # 10:30
TEND_M        = 810     # 13:30
MAX_PER_DAY   = 3
TIME_STOP     = 12      # bars
PT_ATR        = 2.0
SL_ATR        = 1.5
MAX_MOVE_ATR  = 2.5


def load_bars():
    for p in DATA_CANDIDATES:
        if p.exists():
            bars = pd.read_hdf(str(p), key="bars_5min")
            if bars.index.tz is None:
                bars.index = bars.index.tz_localize("US/Eastern")
            else:
                bars.index = bars.index.tz_convert("US/Eastern")
            return bars
    raise FileNotFoundError("No MNQ data found")


def prepare_arrays(bars: pd.DataFrame) -> dict:
    prev = bars["close"].shift(1)
    tr = pd.concat([bars["high"]-bars["low"],
                    (bars["high"]-prev).abs(),
                    (bars["low"]-prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()

    tp = (bars["high"] + bars["low"] + bars["close"]) / 3
    date_idx = bars.index.map(lambda t: t.date())
    vwap = (tp * bars["volume"]).groupby(date_idx).cumsum() / \
           bars["volume"].groupby(date_idx).cumsum().replace(0, np.nan)

    day_opens = bars.groupby(bars.index.map(lambda t: t.date()))["open"].transform("first")

    minutes = bars.index.hour * 60 + bars.index.minute
    dates   = np.array([t.toordinal() for t in bars.index])

    return {
        "close":    bars["close"].values.astype(np.float64),
        "high":     bars["high"].values.astype(np.float64),
        "low":      bars["low"].values.astype(np.float64),
        "atr":      atr.values.astype(np.float64),
        "vwap":     vwap.values.astype(np.float64),
        "day_open": day_opens.values.astype(np.float64),
        "minutes":  minutes.values.astype(np.int32),
        "dates":    dates,
        "n":        len(bars),
    }


def sim(arrays: dict, entry_dist: float, n_contracts: int, bidirectional: bool) -> dict:
    close  = arrays["close"]
    high   = arrays["high"]
    low    = arrays["low"]
    atr    = arrays["atr"]
    vwap   = arrays["vwap"]
    dopen  = arrays["day_open"]
    mins   = arrays["minutes"]
    dates  = arrays["dates"]
    n      = arrays["n"]

    NC     = n_contracts
    comm_rt = 2 * COMMISSION * NC
    max_dl  = -1600.0 * (NC / 10)   # scale loss limit with contracts

    trades     = []
    equity     = 50_000.0
    eq         = [equity]
    cur_date   = dates[20]
    daily_pnl  = 0.0
    day_count  = 0

    pos_active  = False
    pos_dir     = 0
    pos_entry   = 0.0
    pos_sl      = 0.0
    pos_pt      = 0.0
    pos_stop_i  = 0

    for i in range(20, n):
        d = dates[i]
        if d != cur_date:
            daily_pnl = 0.0
            day_count = 0
            cur_date  = d

        m  = mins[i]
        is_close = (m >= 955) or (i+1 >= n) or (dates[i+1] != d)

        # Exit
        if pos_active:
            h = high[i]; l = low[i]; c = close[i]
            exited = False; ep = 0.0
            if is_close or i >= pos_stop_i:
                ep = c + (-pos_dir * TICK_SIZE)  # cross spread on time exit
                exited = True
            elif pos_dir == 1:
                if l <= pos_sl: ep = pos_sl - TICK_SIZE; exited = True
                elif h >= pos_pt: ep = pos_pt - TICK_SIZE; exited = True
            elif pos_dir == -1:
                if h >= pos_sl: ep = pos_sl + TICK_SIZE; exited = True
                elif l <= pos_pt: ep = pos_pt + TICK_SIZE; exited = True
            if exited:
                pnl = (ep - pos_entry) * pos_dir * NC * POINT_VALUE - comm_rt
                trades.append(pnl)
                daily_pnl += pnl
                equity    += pnl
                eq.append(equity)
                pos_active = False

        if is_close:
            continue
        if pos_active or daily_pnl <= max_dl or day_count >= MAX_PER_DAY:
            continue
        if not (TSTART_M <= m <= TEND_M):
            continue

        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue
        v = vwap[i]
        if np.isnan(v):
            continue

        c      = close[i]
        pc     = close[i-1]
        dev    = (c - v) / a
        mov    = (c - dopen[i]) / a

        if abs(dev) > 3.0:
            continue

        # LONG entry: below VWAP, last bar up
        if dev <= -entry_dist and c > pc:
            if mov > -MAX_MOVE_ATR:   # not in strong downtrend
                pos_active  = True
                pos_dir     = 1
                pos_entry   = c + TICK_SIZE
                pos_sl      = pos_entry - SL_ATR * a
                pos_pt      = pos_entry + PT_ATR * a
                pos_stop_i  = i + TIME_STOP
                day_count  += 1
                continue

        # SHORT entry: above VWAP, last bar down
        if bidirectional and dev >= entry_dist and c < pc:
            if mov < MAX_MOVE_ATR:    # not in strong uptrend
                pos_active  = True
                pos_dir     = -1
                pos_entry   = c - TICK_SIZE
                pos_sl      = pos_entry + SL_ATR * a
                pos_pt      = pos_entry - PT_ATR * a
                pos_stop_i  = i + TIME_STOP
                day_count  += 1

    if len(trades) < 5:
        return None

    arr     = np.array(trades)
    eq_arr  = np.array(eq)
    wins    = (arr > 0).sum()
    total   = arr.sum()
    gp      = arr[arr > 0].sum() if wins > 0 else 0.0
    gl      = abs(arr[arr <= 0].sum())
    max_dd  = float((eq_arr - np.maximum.accumulate(eq_arr)).min())
    n_days  = len(np.unique(dates[20:]))
    n_weeks = max(1, n_days / 5)

    return {
        "entry_dist":  entry_dist,
        "contracts":   NC,
        "bidir":       bidirectional,
        "n":           len(arr),
        "wr":          wins / len(arr),
        "weekly":      total / n_weeks,
        "max_dd":      max_dd,
        "pf":          gp/gl if gl > 0 else 99.0,
        "sharpe":      float(arr.mean() / arr.std() * np.sqrt(252)) if arr.std() > 0 else 0,
    }


def main():
    print("Loading data...")
    bars = load_bars()
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("US/Eastern")

    print(f"  {len(bars):,} bars [{bars.index[0].date()} → {bars.index[-1].date()}]")
    arrays = prepare_arrays(bars)

    entry_dists = [0.5, 0.75, 1.0, 1.25, 1.5]
    contract_sizes = [5, 10, 15, 20, 25, 30]

    print(f"\n{'='*90}")
    print(f"LONG-ONLY VWAP MR (2026 YTD)")
    print(f"{'='*90}")
    print(f"{'Dist':>5} {'NC':>4} {'N':>5} {'WR':>7} {'$/wk':>10} {'MaxDD':>10} {'Sharpe':>8} {'PF':>6}")
    print(f"  {'-'*75}")

    best_long = None
    for ed in entry_dists:
        for nc in contract_sizes:
            r = sim(arrays, ed, nc, bidirectional=False)
            if r is None:
                continue
            if best_long is None or (r["sharpe"] * r["wr"]) > (best_long["sharpe"] * best_long["wr"]):
                best_long = r
            flag = ""
            if r["weekly"] >= 6000 and r["max_dd"] >= -2000:
                flag = "  *** TARGET ***"
            print(f"  {ed:>4.2f} {nc:>4} {r['n']:>5} {r['wr']:>7.1%} "
                  f"{r['weekly']:>10,.0f} {r['max_dd']:>10,.0f} "
                  f"{r['sharpe']:>8.3f} {r['pf']:>6.2f}{flag}")

    print(f"\n{'='*90}")
    print(f"BIDIRECTIONAL VWAP MR (2026 YTD)")
    print(f"{'='*90}")
    print(f"{'Dist':>5} {'NC':>4} {'N':>5} {'WR':>7} {'$/wk':>10} {'MaxDD':>10} {'Sharpe':>8} {'PF':>6}")
    print(f"  {'-'*75}")

    best_bidir = None
    for ed in entry_dists:
        for nc in contract_sizes:
            r = sim(arrays, ed, nc, bidirectional=True)
            if r is None:
                continue
            if best_bidir is None or (r["sharpe"] * r["wr"]) > (best_bidir["sharpe"] * best_bidir["wr"]):
                best_bidir = r
            flag = ""
            if r["weekly"] >= 6000 and r["max_dd"] >= -2000:
                flag = "  *** TARGET ***"
            print(f"  {ed:>4.2f} {nc:>4} {r['n']:>5} {r['wr']:>7.1%} "
                  f"{r['weekly']:>10,.0f} {r['max_dd']:>10,.0f} "
                  f"{r['sharpe']:>8.3f} {r['pf']:>6.2f}{flag}")

    print(f"\n{'='*90}")
    print("SUMMARY: Best configs at key contract sizes")
    print(f"{'='*90}")

    for nc_target in [10, 20, 30, 50]:
        configs_at_nc = []
        for ed in entry_dists:
            for bidir in [False, True]:
                r = sim(arrays, ed, nc_target, bidirectional=bidir)
                if r:
                    r["mode"] = "bidir" if bidir else "long"
                    configs_at_nc.append(r)
        if not configs_at_nc:
            continue
        best = max(configs_at_nc, key=lambda x: x["sharpe"] * x["wr"])
        mode_str = "BIDIR" if best["bidir"] else "LONG "
        print(f"  {nc_target:>3} MNQ [{mode_str}] dist={best['entry_dist']:.2f}  "
              f"n={best['n']}  WR={best['wr']:.1%}  "
              f"$/wk=${best['weekly']:,.0f}  DD=${best['max_dd']:,.0f}  "
              f"Sharpe={best['sharpe']:.3f}")

    # Combined ORB + VWAP estimate
    print(f"\n{'='*90}")
    print("COMBINED ESTIMATE: ORB (from prior backtest) + Best VWAP at 10 MNQ")
    print(f"{'='*90}")
    orb_weekly = 2138.0  # from prior combined_strategy_backtest at 10 MNQ
    orb_dd     = -3286.0
    best_v_10 = max(
        [sim(arrays, ed, 10, bidir) for ed in entry_dists for bidir in [False,True]
         if sim(arrays, ed, 10, bidir)],
        key=lambda x: x["sharpe"] * x["wr"]
    )
    comb_weekly = orb_weekly + best_v_10["weekly"]
    comb_dd     = orb_dd + best_v_10["max_dd"]   # conservative (not correlated perfectly)
    print(f"  ORB:       ${orb_weekly:,.0f}/wk  DD=${orb_dd:,.0f}")
    print(f"  VWAP:      ${best_v_10['weekly']:,.0f}/wk  DD=${best_v_10['max_dd']:,.0f}  "
          f"(dist={best_v_10['entry_dist']}, {'bidir' if best_v_10['bidir'] else 'long'}, "
          f"WR={best_v_10['wr']:.1%})")
    print(f"  Combined:  ${comb_weekly:,.0f}/wk  DD=${comb_dd:,.0f} (additive estimate)")

if __name__ == "__main__":
    main()
