"""
Gap Analysis Backtest — MNQ Gap Direction Strategies
=====================================================
Tests four gap-based approaches on MNQ 2026 YTD + extended dataset:

A. Gap Continuation: enter at open in gap direction, exit via ATR targets
B. Gap Fade: enter opposite gap direction, target 50% gap fill
C. ORB alignment: split ORB trades by gap alignment (same dir vs counter)
D. Intraday gap fill timing: how often does MNQ fill the gap by when?

Data: mnq_5min_aug25_mar26.h5 (Aug 2025 – Mar 2026, 162 days)
"""
from __future__ import annotations
import sys, json, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
RBV1 = ROOT / "rule_based_v1"
for p in [str(ROOT), str(RBV1)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.indicators import atr as compute_atr

DATA_PATH    = ROOT / "data" / "processed" / "mnq_5min_aug25_mar26.h5"
RESULTS_PATH = RBV1 / "diagnostics" / "gap_analysis_results.json"

POINT_VALUE = 2.0
TICK_SIZE   = 0.25
COMMISSION  = 0.62
SLIPPAGE    = 1
ATR_PERIOD  = 14
N_CONTRACTS = 3
MAX_DAILY_LOSS = -950.0
DRAWDOWN_BUF   = 1_950.0
STARTING_EQ    = 50_000.0


def slip(p, d, is_entry):
    s = SLIPPAGE * TICK_SIZE
    return p + s * d if is_entry else p - s * d


def calc_pnl(entry, exit_, direction, n):
    return (exit_ - entry) * direction * n * POINT_VALUE - 2 * COMMISSION * n


def build_daily_gap_features(bars: pd.DataFrame) -> pd.DataFrame:
    atr_5m = compute_atr(bars["high"], bars["low"], bars["close"], ATR_PERIOD)
    rows = []
    for date, grp in bars.groupby(bars.index.date):
        et = grp.index.tz_convert("US/Eastern")
        open_bars = grp[(et.hour == 9) & (et.minute == 30)]
        if open_bars.empty:
            continue
        open_930  = float(open_bars["open"].iloc[0])
        prev_close_series = bars[bars.index < grp.index[0]]["close"]
        if prev_close_series.empty:
            continue
        prev_close = float(prev_close_series.iloc[-1])
        gap_pts  = open_930 - prev_close
        gap_pct  = gap_pts / prev_close

        sig_idx = bars.index.get_loc(grp.index[-1])
        atr_now = float(atr_5m.iloc[sig_idx])

        rows.append({
            "date":       date,
            "open_930":   open_930,
            "prev_close": prev_close,
            "gap_pts":    gap_pts,
            "gap_pct":    gap_pct,
            "atr":        atr_now,
        })

    df = pd.DataFrame(rows).set_index("date")
    df["gap_dir"] = np.sign(df["gap_pts"])  # +1 = gap up, -1 = gap down
    df["gap_atr_ratio"] = df["gap_pts"].abs() / df["atr"]
    return df


def run_gap_continuation(
    bars: pd.DataFrame,
    feat: pd.DataFrame,
    gap_thresh_pct: float = 0.001,
    pt_mult: float = 2.0,
    sl_mult: float = 1.5,
) -> dict:
    """Trade in gap direction at 9:30 open bar. Exit PT/SL/session close."""
    trades = []
    equity = STARTING_EQ; peak = STARTING_EQ; max_dd = 0.0
    daily_pnl = {}

    for date, frow in feat.iterrows():
        if abs(frow["gap_pct"]) < gap_thresh_pct:
            continue
        if np.isnan(frow["atr"]) or frow["atr"] <= 0:
            continue

        day_bars = bars[bars.index.date == date]
        et = day_bars.index.tz_convert("US/Eastern")
        entry_bars = day_bars[(et.hour == 9) & (et.minute == 30)]
        if entry_bars.empty:
            continue

        direction = int(frow["gap_dir"])
        atr = frow["atr"]
        entry = slip(float(entry_bars["close"].iloc[0]), direction, True)
        stop  = entry - direction * sl_mult * atr
        tgt   = entry + direction * pt_mult * atr

        after = day_bars[day_bars.index > entry_bars.index[-1]]
        exit_price = None; exit_reason = "session_close"

        for bar_ts, bar in after.iterrows():
            bar_et = bar_ts.tz_convert("US/Eastern")
            is_close = bar_et.hour >= 16

            if direction == 1:
                if bar["low"] <= stop:
                    exit_price = slip(stop, 1, False); exit_reason = "stop_loss"; break
                if bar["high"] >= tgt:
                    exit_price = slip(tgt, 1, False); exit_reason = "profit_target"; break
            else:
                if bar["high"] >= stop:
                    exit_price = slip(stop, -1, False); exit_reason = "stop_loss"; break
                if bar["low"] <= tgt:
                    exit_price = slip(tgt, -1, False); exit_reason = "profit_target"; break
            if is_close:
                exit_price = slip(float(bar["close"]), direction, False); exit_reason = "session_close"; break

        if exit_price is None:
            last = day_bars.iloc[-1]
            exit_price = slip(float(last["close"]), direction, False)

        pnl = calc_pnl(entry, exit_price, direction, N_CONTRACTS)
        trades.append({"date": str(date), "dir": direction, "gap_pct": round(frow["gap_pct"]*100, 3),
                       "pnl": round(pnl, 2), "reason": exit_reason})
        daily_pnl[str(date)] = round(pnl, 2)
        equity += pnl; peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)

    wins  = [t for t in trades if t["pnl"] > 0]
    total = sum(t["pnl"] for t in trades)
    gp    = sum(t["pnl"] for t in wins)
    gl    = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    dp    = pd.Series(list(daily_pnl.values()))
    sharpe = dp.mean() / dp.std() * np.sqrt(252) if len(dp) > 1 and dp.std() > 0 else 0
    return {
        "n": len(trades), "wr": len(wins)/max(len(trades),1),
        "pnl": round(total,2), "sharpe": round(sharpe,3), "dd": round(max_dd,2),
        "mll": max_dd > -2000,
        "pf": round(gp/gl,3) if gl > 0 else float("inf"),
        "exit_reasons": {r: sum(1 for t in trades if t["reason"]==r) for r in ["stop_loss","profit_target","session_close"]},
    }


def run_gap_fade(
    bars: pd.DataFrame,
    feat: pd.DataFrame,
    gap_thresh_pct: float = 0.002,
    fill_target_frac: float = 0.5,
    sl_mult: float = 1.5,
) -> dict:
    """Fade the gap — enter opposite direction, target partial fill."""
    trades = []
    equity = STARTING_EQ; peak = STARTING_EQ; max_dd = 0.0
    daily_pnl = {}

    for date, frow in feat.iterrows():
        if abs(frow["gap_pct"]) < gap_thresh_pct:
            continue
        if np.isnan(frow["atr"]) or frow["atr"] <= 0:
            continue

        day_bars = bars[bars.index.date == date]
        et = day_bars.index.tz_convert("US/Eastern")
        entry_bars = day_bars[(et.hour == 9) & (et.minute == 30)]
        if entry_bars.empty:
            continue

        gap_dir   = int(frow["gap_dir"])
        direction = -gap_dir   # fade = opposite
        atr       = frow["atr"]
        gap_pts   = frow["gap_pts"]

        entry = slip(float(entry_bars["close"].iloc[0]), direction, True)
        stop  = entry - direction * sl_mult * atr
        tgt   = frow["prev_close"] + gap_pts * (1 - fill_target_frac)  # partial fill

        after = day_bars[day_bars.index > entry_bars.index[-1]]
        exit_price = None; exit_reason = "session_close"

        for bar_ts, bar in after.iterrows():
            bar_et = bar_ts.tz_convert("US/Eastern")
            is_close = bar_et.hour >= 16

            if direction == 1:
                if bar["low"] <= stop:
                    exit_price = slip(stop, 1, False); exit_reason = "stop_loss"; break
                if bar["high"] >= tgt:
                    exit_price = slip(tgt, 1, False); exit_reason = "profit_target"; break
            else:
                if bar["high"] >= stop:
                    exit_price = slip(stop, -1, False); exit_reason = "stop_loss"; break
                if bar["low"] <= tgt:
                    exit_price = slip(tgt, -1, False); exit_reason = "profit_target"; break
            if is_close:
                exit_price = slip(float(bar["close"]), direction, False); exit_reason = "session_close"; break

        if exit_price is None:
            last = day_bars.iloc[-1]
            exit_price = slip(float(last["close"]), direction, False)

        pnl = calc_pnl(entry, exit_price, direction, N_CONTRACTS)
        trades.append({"date": str(date), "dir": direction, "gap_pct": round(frow["gap_pct"]*100, 3),
                       "pnl": round(pnl, 2), "reason": exit_reason})
        daily_pnl[str(date)] = round(pnl, 2)
        equity += pnl; peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)

    wins  = [t for t in trades if t["pnl"] > 0]
    total = sum(t["pnl"] for t in trades)
    gp    = sum(t["pnl"] for t in wins)
    gl    = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    dp    = pd.Series(list(daily_pnl.values()))
    sharpe = dp.mean() / dp.std() * np.sqrt(252) if len(dp) > 1 and dp.std() > 0 else 0
    return {
        "n": len(trades), "wr": len(wins)/max(len(trades),1),
        "pnl": round(total,2), "sharpe": round(sharpe,3), "dd": round(max_dd,2),
        "mll": max_dd > -2000,
        "pf": round(gp/gl,3) if gl > 0 else float("inf"),
        "exit_reasons": {r: sum(1 for t in trades if t["reason"]==r) for r in ["stop_loss","profit_target","session_close"]},
    }


def analyze_gap_fill_timing(bars: pd.DataFrame, feat: pd.DataFrame) -> dict:
    """How often does MNQ fill the gap by 10 AM, 11 AM, noon, 3 PM?"""
    checkpoints = [("10:00", 10, 0), ("11:00", 11, 0), ("12:00", 12, 0), ("15:00", 15, 0)]
    results = defaultdict(list)

    for date, frow in feat.iterrows():
        if abs(frow["gap_pct"]) < 0.001:
            continue
        if np.isnan(frow["atr"]):
            continue

        day_bars = bars[bars.index.date == date]
        et = day_bars.index.tz_convert("US/Eastern")
        gap_dir  = int(frow["gap_dir"])
        prev_close = frow["prev_close"]

        for label, h, m in checkpoints:
            subset = day_bars[(et.hour < h) | ((et.hour == h) & (et.minute <= m))]
            if subset.empty:
                continue
            filled = False
            for _, bar in subset.iterrows():
                if gap_dir > 0:  # gap-up: fill = price goes below prev_close
                    if bar["low"] <= prev_close:
                        filled = True; break
                else:            # gap-down: fill = price goes above prev_close
                    if bar["high"] >= prev_close:
                        filled = True; break
            results[label].append(int(filled))

    return {
        label: {
            "pct_filled": round(np.mean(v)*100, 1),
            "n": len(v)
        }
        for label, v in results.items()
    }


if __name__ == "__main__":
    print(f"Loading {DATA_PATH} ...")
    bars = pd.read_hdf(str(DATA_PATH), key="bars_5min")
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("US/Eastern")
    else:
        bars.index = bars.index.tz_convert("US/Eastern")

    bars = bars[
        ((bars.index.hour == 9) & (bars.index.minute >= 30)) |
        ((bars.index.hour > 9) & (bars.index.hour < 16))
    ].copy()

    print(f"Bars: {len(bars):,} | Days: {len(set(bars.index.date))} | {bars.index[0].date()} → {bars.index[-1].date()}")
    feat = build_daily_gap_features(bars)
    print(f"Days with gap data: {len(feat)}")

    # Gap distribution
    gap_pcts = feat["gap_pct"] * 100
    print(f"\nGap distribution:")
    print(f"  |gap| > 0.1%: {(gap_pcts.abs() > 0.1).sum()} days ({(gap_pcts.abs() > 0.1).mean()*100:.0f}%)")
    print(f"  |gap| > 0.2%: {(gap_pcts.abs() > 0.2).sum()} days ({(gap_pcts.abs() > 0.2).mean()*100:.0f}%)")
    print(f"  |gap| > 0.3%: {(gap_pcts.abs() > 0.3).sum()} days ({(gap_pcts.abs() > 0.3).mean()*100:.0f}%)")
    print(f"  |gap| > 0.5%: {(gap_pcts.abs() > 0.5).sum()} days ({(gap_pcts.abs() > 0.5).mean()*100:.0f}%)")
    print(f"  Gap up  days: {(gap_pcts > 0.1).sum()}")
    print(f"  Gap down days: {(gap_pcts < -0.1).sum()}")

    print(f"\nGap fill timing analysis:")
    fill_stats = analyze_gap_fill_timing(bars, feat)
    for label, v in fill_stats.items():
        print(f"  By {label}: {v['pct_filled']:.1f}% fill ({v['n']} events)")

    print(f"\n{'='*80}")
    print("STRATEGY A — Gap Continuation")
    W = 40
    print(f"\n  {'Config':<{W}} {'N':>4} {'WR':>6} {'PnL':>9} {'Sharpe':>7} {'MaxDD':>9}")
    print(f"  {'-'*75}")
    cont_results = {}
    for thresh in [0.001, 0.002, 0.003, 0.005]:
        for pt in [1.5, 2.0, 3.0]:
            r = run_gap_continuation(bars, feat, thresh, pt, 1.5)
            label = f"Cont thresh={thresh*100:.1f}% PT={pt}x"
            cont_results[label] = r
            mll = "OK" if r["mll"] else "XX"
            print(f"  {label:<{W}} {r['n']:>4}  {r['wr']:>5.1%}  ${r['pnl']:>8,.0f}  {r['sharpe']:>6.2f}  ${r['dd']:>7,.0f}  {mll}")

    print(f"\n{'='*80}")
    print("STRATEGY B — Gap Fade")
    print(f"\n  {'Config':<{W}} {'N':>4} {'WR':>6} {'PnL':>9} {'Sharpe':>7} {'MaxDD':>9}")
    print(f"  {'-'*75}")
    fade_results = {}
    for thresh in [0.001, 0.002, 0.003, 0.005]:
        for fill_frac in [0.3, 0.5, 0.8]:
            r = run_gap_fade(bars, feat, thresh, fill_frac, 1.5)
            label = f"Fade thresh={thresh*100:.1f}% fill={fill_frac:.0%}"
            fade_results[label] = r
            mll = "OK" if r["mll"] else "XX"
            print(f"  {label:<{W}} {r['n']:>4}  {r['wr']:>5.1%}  ${r['pnl']:>8,.0f}  {r['sharpe']:>6.2f}  ${r['dd']:>7,.0f}  {mll}")

    all_results = {
        "gap_fill_timing": fill_stats,
        "gap_distribution": {
            "gt01pct": int((gap_pcts.abs() > 0.1).sum()),
            "gt02pct": int((gap_pcts.abs() > 0.2).sum()),
            "gt03pct": int((gap_pcts.abs() > 0.3).sum()),
            "gt05pct": int((gap_pcts.abs() > 0.5).sum()),
            "gap_up":   int((gap_pcts > 0.1).sum()),
            "gap_down": int((gap_pcts < -0.1).sum()),
        },
        "continuation": cont_results,
        "fade": fade_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved → {RESULTS_PATH}")
