"""
RCM — Regime-Conditioned Close-Window Momentum
===============================================
Based on Baltussen et al. (2021): rest-of-day return predicts last 30-min return.
Options market makers delta-hedge throughout the day, amplifying intraday trends into close.

Signal (computed at 2:30 PM ET):
  r_rod          = (close_14:30 - open_9:30) / open_9:30
  path_efficiency = |close_14:30 - open_9:30| / (day_high - day_low + eps)
  atr_ratio      = today_atr14 / rolling_20d_atr
  overnight_gap  = |open_9:30 - prev_close| / prev_close

Entry: 2:30 PM ET bar close, direction = sign(r_rod)
Exit:  3:00 PM ET hard exit OR stop = entry ± 0.5*ATR
Long-only variant also tested.
"""
from __future__ import annotations
import sys, json, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parent.parent.parent
RBV1 = ROOT / "rule_based_v1"
for p in [str(ROOT), str(RBV1)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.indicators import atr as compute_atr

RESULTS_PATH = RBV1 / "diagnostics" / "rcm_results.json"

POINT_VALUE  = 2.0
TICK_SIZE    = 0.25
COMMISSION   = 0.62
SLIPPAGE     = 1          # ticks
ATR_PERIOD   = 14
MAX_DLL      = -950.0
DRAWDOWN_BUF = 1_950.0
START_EQ     = 50_000.0

ENTRY_HOUR, ENTRY_MIN = 14, 30   # 2:30 PM ET
EXIT_HOUR,  EXIT_MIN  = 15,  0   # 3:00 PM ET
STOP_MULT    = 0.5               # stop = 0.5 * ATR from entry
ATR_ROLL     = 20                # days for rolling ATR mean


@dataclass
class Trade:
    date: object; direction: int; entry: float; exit_: float
    pnl: float; reason: str; n: int


def slip(p, d, entry):
    s = SLIPPAGE * TICK_SIZE
    return p + s * d if entry else p - s * d


def calc_pnl(entry, exit_, direction, n):
    return (exit_ - entry) * direction * n * POINT_VALUE - 2 * COMMISSION * n


def build_daily_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Build per-day feature frame for RCM signal."""
    atr_5m = compute_atr(bars["high"], bars["low"], bars["close"], ATR_PERIOD)
    rows = []

    for date, grp in bars.groupby(bars.index.date):
        et = grp.index.tz_convert("US/Eastern")

        # 9:30 open bar
        open_bars = grp[(et.hour == 9) & (et.minute == 30)]
        # 14:30 signal bar
        sig_bars  = grp[(et.hour == 14) & (et.minute == 30)]
        # all bars up to 14:30
        intraday  = grp[(et.hour < 14) | ((et.hour == 14) & (et.minute <= 30))]

        if open_bars.empty or sig_bars.empty or len(intraday) < 5:
            continue

        open_930  = float(open_bars["open"].iloc[0])
        close_1430 = float(sig_bars["close"].iloc[-1])
        high_14   = float(intraday["high"].max())
        low_14    = float(intraday["low"].min())
        day_range = high_14 - low_14

        # daily ATR from the 14:30 bar
        sig_idx  = bars.index.get_loc(sig_bars.index[-1]) if not sig_bars.empty else -1
        atr_now  = float(atr_5m.iloc[sig_idx]) if sig_idx >= 0 else np.nan

        r_rod   = (close_1430 - open_930) / open_930 if open_930 != 0 else 0
        path_eff = abs(close_1430 - open_930) / (day_range + 0.01)

        rows.append({
            "date":       date,
            "open_930":   open_930,
            "close_1430": close_1430,
            "high_14":    high_14,
            "low_14":     low_14,
            "day_range":  day_range,
            "r_rod":      r_rod,
            "path_eff":   path_eff,
            "atr_now":    atr_now,
            "sig_idx":    sig_idx,
        })

    df = pd.DataFrame(rows).set_index("date")
    df["atr_20d_mean"] = df["atr_now"].rolling(ATR_ROLL, min_periods=5).mean()
    df["atr_ratio"]    = df["atr_now"] / df["atr_20d_mean"]
    df["prev_close"]   = df["close_1430"].shift(1)
    df["gap"]          = (df["open_930"] - df["prev_close"]).abs() / df["prev_close"].replace(0, np.nan)
    return df


def run_rcm(
    bars: pd.DataFrame,
    feat: pd.DataFrame,
    n_contracts: int = 3,
    rod_thresh: float = 0.002,
    path_thresh: float = 0.30,
    atr_ratio_min: float = 0.80,
    long_only: bool = False,
) -> dict:
    equity = START_EQ; peak = START_EQ; max_dd = 0.0
    daily_loss = 0.0; cur_date = None; daily_pnl = {}
    trades: list[Trade] = []

    for date, frow in feat.iterrows():
        if pd.isna(frow["r_rod"]) or pd.isna(frow["atr_ratio"]):
            continue

        # Reset daily
        if cur_date != date:
            if cur_date is not None:
                daily_pnl[cur_date] = daily_loss
            daily_loss = 0.0
            cur_date   = date

        # Filter conditions
        r_rod    = frow["r_rod"]
        path_eff = frow["path_eff"]
        atr_rat  = frow["atr_ratio"]

        if path_eff < path_thresh:
            continue
        if atr_rat < atr_ratio_min:
            continue
        if abs(r_rod) < rod_thresh:
            continue

        direction = 1 if r_rod > 0 else -1
        if long_only and direction == -1:
            continue

        # Risk checks
        if daily_loss <= MAX_DLL:
            continue
        if (equity - peak) <= -DRAWDOWN_BUF:
            continue

        # Find 14:30 and 15:00 bars
        day_bars = bars[bars.index.date == date]
        et_idx   = day_bars.index.tz_convert("US/Eastern")
        sig_bars = day_bars[(et_idx.hour == 14) & (et_idx.minute == 30)]
        exit_bars = day_bars[(et_idx.hour == 15) & (et_idx.minute == 0)]

        if sig_bars.empty:
            continue

        entry_price = slip(float(sig_bars["close"].iloc[-1]), direction, True)
        atr_now     = frow["atr_now"]
        if np.isnan(atr_now) or atr_now <= 0:
            continue
        stop_price  = entry_price - direction * STOP_MULT * atr_now

        # Simulate bar-by-bar from 14:35 to 15:00
        after_entry = day_bars[day_bars.index > sig_bars.index[-1]]
        exit_price  = None
        exit_reason = "session_close"

        for bar_ts, bar in after_entry.iterrows():
            bar_et = bar_ts.tz_convert("US/Eastern")
            is_exit_time = (bar_et.hour == 15 and bar_et.minute >= 0)

            if direction == 1:
                if bar["low"] <= stop_price:
                    exit_price  = slip(stop_price, 1, False)
                    exit_reason = "stop_loss"
                    break
            else:
                if bar["high"] >= stop_price:
                    exit_price  = slip(stop_price, -1, False)
                    exit_reason = "stop_loss"
                    break

            if is_exit_time:
                exit_price  = slip(float(bar["close"]), direction, False)
                exit_reason = "time_exit"
                break

        if exit_price is None:
            # Use last bar
            last = day_bars.iloc[-1]
            exit_price  = slip(float(last["close"]), direction, False)
            exit_reason = "session_close"

        pnl = calc_pnl(entry_price, exit_price, direction, n_contracts)
        trades.append(Trade(date, direction, entry_price, exit_price, pnl, exit_reason, n_contracts))
        daily_loss += pnl
        equity     += pnl
        peak        = max(peak, equity)
        max_dd      = min(max_dd, equity - peak)

    if cur_date and cur_date not in daily_pnl:
        daily_pnl[cur_date] = daily_loss

    wins  = [t for t in trades if t.pnl > 0]
    total = sum(t.pnl for t in trades)
    gp    = sum(t.pnl for t in wins)
    gl    = abs(sum(t.pnl for t in trades if t.pnl <= 0))
    dp    = pd.Series(daily_pnl); dp = dp[dp != 0]
    sharpe = dp.mean() / dp.std() * np.sqrt(252) if len(dp) > 1 and dp.std() > 0 else 0

    longs  = [t for t in trades if t.direction ==  1]
    shorts = [t for t in trades if t.direction == -1]

    return {
        "n": len(trades), "wr": len(wins) / max(len(trades), 1),
        "pnl": total, "sharpe": sharpe, "dd": max_dd, "mll": max_dd > -2000,
        "pf": gp / gl if gl > 0 else float("inf"),
        "long_n": len(longs), "long_wr": sum(1 for t in longs if t.pnl > 0) / max(len(longs), 1),
        "short_n": len(shorts), "short_wr": sum(1 for t in shorts if t.pnl > 0) / max(len(shorts), 1),
        "trades": [{"date": str(t.date), "dir": t.direction, "entry": t.entry,
                    "exit": t.exit_, "pnl": round(t.pnl, 2), "reason": t.reason}
                   for t in trades],
        "daily_pnl": {str(k): round(v, 2) for k, v in daily_pnl.items()},
    }


def monthly_breakdown(r):
    m = defaultdict(list)
    for t in r["trades"]:
        m[t["date"][:7]].append(t)
    return {ym: (len(ts), sum(1 for t in ts if t["pnl"] > 0) / len(ts), sum(t["pnl"] for t in ts))
            for ym, ts in m.items()}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ytd",
                        choices=["ytd", "extended", "both"],
                        help="ytd=2026, extended=2024-2025, both=all")
    args = parser.parse_args()

    datasets = {}
    ytd_path = ROOT / "data/processed/mnq_2026ytd_5min.h5"
    ext_path = ROOT / "data/processed/mnq_2024_2025_5min.h5"

    if ytd_path.exists():
        b = pd.read_hdf(str(ytd_path), key="bars_5min")
        if b.index.tz is None:
            b.index = b.index.tz_localize("US/Eastern")
        datasets["2026_ytd"] = b

    if ext_path.exists() and args.data in ("extended", "both"):
        b = pd.read_hdf(str(ext_path), key="bars_5min")
        if b.index.tz is None:
            b.index = b.index.tz_localize("US/Eastern")
        datasets["2024_2025"] = b

    if args.data == "both" and "2026_ytd" in datasets and "2024_2025" in datasets:
        combined = pd.concat([datasets["2024_2025"], datasets["2026_ytd"]]).sort_index()
        combined = combined[~combined.index.duplicated()]
        datasets["full"] = combined

    all_results = {}

    for dataset_name, bars in datasets.items():
        print(f"\n{'='*72}")
        print(f"  RCM Backtest — {dataset_name}  ({bars.index[0].date()} → {bars.index[-1].date()})")
        print(f"{'='*72}")

        feat = build_daily_features(bars)
        print(f"  Feature days built: {len(feat)}")

        W = 46
        print(f"\n  {'Config':<{W}} {'N':>4} {'WR':>6} {'PnL':>10} {'Sharpe':>7} {'MaxDD':>9} {'L_wr':>6} {'S_wr':>6}  MLL?")
        print(f"  {'-'*82}")

        dataset_results = {}

        # Sweep configurations
        for n_c in [2, 3]:
            for rod_thresh in [0.001, 0.002, 0.003, 0.005]:
                for path_thresh in [0.25, 0.35]:
                    for long_only in [True, False]:
                        label = (f"{'LONG' if long_only else 'L+S'} "
                                 f"n={n_c} rod>{rod_thresh*100:.1f}% path>{path_thresh:.2f}")
                        r = run_rcm(bars, feat, n_c, rod_thresh, path_thresh,
                                    long_only=long_only)
                        dataset_results[label] = r
                        if r["n"] >= 5:
                            mll = "✅" if r["mll"] else "❌"
                            print(f"  {label:<{W}} {r['n']:>4}  {r['wr']:>5.1%} "
                                  f"${r['pnl']:>8,.0f} {r['sharpe']:>7.2f} "
                                  f"${r['dd']:>7,.0f} {r['long_wr']:>6.1%} "
                                  f"{r['short_wr']:>6.1%}  {mll}")

        # Top performers
        top = sorted(
            [(k, v) for k, v in dataset_results.items() if v["mll"] and v["n"] >= 8],
            key=lambda x: x[1]["sharpe"], reverse=True
        )[:3]

        print(f"\n  TOP 3 (MLL-safe, ≥8 trades):")
        for label, r in top:
            ms = monthly_breakdown(r)
            print(f"\n  [{label}]  Sharpe={r['sharpe']:.2f} DD=${r['dd']:,.0f} PF={r['pf']:.2f}")
            cum = 0.0
            for ym in sorted(ms):
                n, wr, pnl = ms[ym]; cum += pnl
                print(f"    {ym}  {n:>3}t  {wr:>5.1%}  ${pnl:>8,.0f}  cumul=${cum:>8,.0f}")

        all_results[dataset_name] = {
            k: {kk: vv for kk, vv in v.items() if kk != "trades"}
            for k, v in dataset_results.items()
        }

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved → {RESULTS_PATH}")
