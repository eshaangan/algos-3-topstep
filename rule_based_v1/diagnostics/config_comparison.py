"""
Live config comparison — current vs new
========================================
Current live: require_prev_vwap=True, PT=2.0x, no gap filter
New config:   require_prev_vwap=True, skip_gap_up>0.3%, PT=3.0x

Runs both on:
  - 2026 YTD   (Jan 02 – Mar 19 2026, 55 trading days)
  - Extended   (Aug 01 2025 – Mar 23 2026, 162 trading days)
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

RESULTS_PATH = RBV1 / "diagnostics" / "config_comparison_results.json"

POINT_VALUE  = 2.0
TICK_SIZE    = 0.25
COMMISSION   = 0.62
SLIPPAGE     = 1
ATR_PERIOD   = 14
N_CONTRACTS  = 3
MAX_TRADES_DAY  = 2
MAX_DAILY_LOSS  = -950.0
DRAWDOWN_BUF    = 1_950.0
STARTING_EQ     = 50_000.0
OR_END_HOUR, OR_END_MIN = 10, 4
MIN_OR_BARS   = 7
ENTRY_CUTOFF_H = 12
TIME_STOP_BARS = 24
EMA_PERIOD     = 20


def slip(p, d, entry):
    return p + SLIPPAGE * TICK_SIZE * d if entry else p - SLIPPAGE * TICK_SIZE * d


def calc_pnl(entry, exit_, direction, n):
    return (exit_ - entry) * direction * n * POINT_VALUE - 2 * COMMISSION * n


def compute_vwap_last(grp):
    tp = (grp["high"] + grp["low"] + grp["close"]) / 3
    cum_vol = grp["volume"].cumsum().replace(0, np.nan)
    return float((tp * grp["volume"]).cumsum().iloc[-1] / cum_vol.iloc[-1])


def build_day_meta(bars: pd.DataFrame) -> pd.DataFrame:
    atr_5m = compute_atr(bars["high"], bars["low"], bars["close"], ATR_PERIOD)
    rows = []
    for date, grp in bars.groupby(bars.index.date):
        et = grp.index.tz_convert("US/Eastern")
        open_bars = grp[(et.hour == 9) & (et.minute == 30)]
        if open_bars.empty:
            continue
        open_930 = float(open_bars["open"].iloc[0])
        prev_close_s = bars[bars.index < grp.index[0]]["close"]
        prev_close = float(prev_close_s.iloc[-1]) if not prev_close_s.empty else np.nan
        gap_pct = (open_930 - prev_close) / prev_close if (not np.isnan(prev_close) and prev_close != 0) else np.nan

        sig_idx = bars.index.get_loc(grp.index[-1])
        atr_now = float(atr_5m.iloc[sig_idx])

        rows.append({
            "date":           date,
            "open_930":       open_930,
            "close_day":      float(grp["close"].iloc[-1]),
            "vwap_day":       compute_vwap_last(grp),
            "atr":            atr_now,
            "gap_pct":        gap_pct,
        })

    df = pd.DataFrame(rows).set_index("date")
    df["prev_close"] = df["close_day"].shift(1)
    df["prev_vwap"]  = df["vwap_day"].shift(1)
    df["above_prev_vwap"] = df["open_930"] > df["prev_vwap"]
    df["ema20_daily"] = df["close_day"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df["above_ema20"] = df["close_day"].shift(1) > df["ema20_daily"].shift(1)
    return df


def run_backtest(bars, meta, n_contracts, pt_mult, require_above_prevvwap, skip_gap_up):
    equity = STARTING_EQ; peak = STARTING_EQ; max_dd = 0.0
    trades = []; daily_pnl = {}

    for date, grp in bars.groupby(bars.index.date):
        if date not in meta.index:
            continue
        m = meta.loc[date]
        if pd.isna(m["atr"]) or m["atr"] <= 0:
            continue

        # Filters
        if require_above_prevvwap and not m.get("above_prev_vwap", True):
            continue
        if skip_gap_up is not None and not pd.isna(m["gap_pct"]) and m["gap_pct"] > skip_gap_up:
            continue

        # OR window
        et = grp.index.tz_convert("US/Eastern")
        or_mask = (((et.hour == 9) & (et.minute >= 30)) |
                   ((et.hour == 10) & (et.minute <= OR_END_MIN)))
        or_bars = grp[or_mask]
        if len(or_bars) < MIN_OR_BARS:
            continue

        or_high  = float(or_bars["high"].max())
        or_low   = float(or_bars["low"].min())
        atr_now  = m["atr"]
        if (or_high - or_low) / atr_now < 0.3:
            continue

        day_loss = 0.0; pos = None; trades_today = 0
        bar_list = list(grp.index)

        for bar_ts in bar_list:
            bar    = grp.loc[bar_ts]
            bar_et = bar_ts.tz_convert("US/Eastern")
            is_close = bar_et.hour >= 16
            bidx = bar_list.index(bar_ts)

            if pos is not None:
                if is_close or bidx >= pos["ts_bar"]:
                    ep  = slip(float(bar["close"]), pos["dir"], False)
                    pnl = calc_pnl(pos["entry"], ep, pos["dir"], n_contracts)
                    rsn = "session_close" if is_close else "time_stop"
                    trades.append({"date": str(date), "dir": pos["dir"], "entry": pos["entry"],
                                   "exit": ep, "pnl": round(pnl,2), "reason": rsn})
                    day_loss += pnl; equity += pnl; peak = max(peak, equity)
                    max_dd = min(max_dd, equity - peak); pos = None; continue

                if pos["dir"] == 1:
                    if bar["low"] <= pos["stop"]:
                        ep = slip(pos["stop"], 1, False)
                        pnl = calc_pnl(pos["entry"], ep, 1, n_contracts)
                        trades.append({"date": str(date), "dir": 1, "entry": pos["entry"],
                                       "exit": ep, "pnl": round(pnl,2), "reason": "stop_loss"})
                        day_loss += pnl; equity += pnl; peak = max(peak, equity)
                        max_dd = min(max_dd, equity - peak); pos = None; continue
                    if bar["high"] >= pos["tgt"]:
                        ep = slip(pos["tgt"], 1, False)
                        pnl = calc_pnl(pos["entry"], ep, 1, n_contracts)
                        trades.append({"date": str(date), "dir": 1, "entry": pos["entry"],
                                       "exit": ep, "pnl": round(pnl,2), "reason": "profit_target"})
                        day_loss += pnl; equity += pnl; peak = max(peak, equity)
                        max_dd = min(max_dd, equity - peak); pos = None; continue

            if pos is None and trades_today < MAX_TRADES_DAY:
                if day_loss <= MAX_DAILY_LOSS or (equity - peak) <= -DRAWDOWN_BUF:
                    continue
                if (bar_et.hour < ENTRY_CUTOFF_H and
                    (bar_et.hour > OR_END_HOUR or (bar_et.hour == OR_END_HOUR and bar_et.minute > OR_END_MIN))):
                    if float(bar["high"]) > or_high:
                        entry = slip(or_high, 1, True)
                        pos = {"dir": 1, "entry": entry,
                               "stop": entry - 1.5 * atr_now,
                               "tgt":  entry + pt_mult * atr_now,
                               "ts_bar": bidx + TIME_STOP_BARS}
                        trades_today += 1

        if pos is not None:
            last = grp.iloc[-1]
            ep = slip(float(last["close"]), pos["dir"], False)
            pnl = calc_pnl(pos["entry"], ep, pos["dir"], n_contracts)
            trades.append({"date": str(date), "dir": pos["dir"], "entry": pos["entry"],
                           "exit": ep, "pnl": round(pnl,2), "reason": "eod_close"})
            day_loss += pnl; equity += pnl; peak = max(peak, equity)
            max_dd = min(max_dd, equity - peak)

        if day_loss != 0:
            daily_pnl[str(date)] = round(day_loss, 2)

    wins  = [t for t in trades if t["pnl"] > 0]
    total = sum(t["pnl"] for t in trades)
    gp    = sum(t["pnl"] for t in wins)
    gl    = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    dp    = pd.Series(list(daily_pnl.values()))
    dp_nonzero = dp[dp != 0]
    sharpe  = dp_nonzero.mean() / dp_nonzero.std() * np.sqrt(252) if len(dp_nonzero) > 1 and dp_nonzero.std() > 0 else 0
    sortino = (dp_nonzero.mean() / dp_nonzero[dp_nonzero < 0].std() * np.sqrt(252)
               if len(dp_nonzero[dp_nonzero < 0]) > 1 else float("inf"))
    avg_win  = np.mean([t["pnl"] for t in trades if t["pnl"] > 0]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in trades if t["pnl"] <= 0]) if gl > 0 else 0

    monthly = defaultdict(list)
    for t in trades:
        monthly[t["date"][:7]].append(t["pnl"])

    exit_reasons = defaultdict(int)
    for t in trades:
        exit_reasons[t["reason"]] += 1

    return {
        "n_trades":       len(trades),
        "wr":             round(len(wins) / max(len(trades),1), 4),
        "pnl":            round(total, 2),
        "avg_pnl":        round(total / max(len(trades),1), 2),
        "avg_win":        round(avg_win, 2),
        "avg_loss":       round(avg_loss, 2),
        "pf":             round(gp / gl, 3) if gl > 0 else float("inf"),
        "sharpe":         round(sharpe, 3),
        "sortino":        round(sortino, 3),
        "max_dd":         round(max_dd, 2),
        "mll_safe":       max_dd > -2000,
        "trades_per_day": round(len(trades) / max(len(daily_pnl),1), 3),
        "exit_reasons":   dict(exit_reasons),
        "monthly":        {
            ym: {
                "n":   len(ps),
                "wr":  round(sum(1 for p in ps if p > 0) / len(ps), 3),
                "pnl": round(sum(ps), 2),
                "avg": round(sum(ps)/len(ps), 2),
            }
            for ym, ps in sorted(monthly.items())
        },
        "daily_pnl": daily_pnl,
    }


def monte_carlo(daily_pnls, n_sims=20_000, profit_target=3_000, max_dd_limit=2_000):
    arr = np.array([v for v in daily_pnls.values() if v != 0])
    if len(arr) < 5:
        return {}
    rng = np.random.default_rng(42)
    pass30 = pass60 = 0
    dds = []
    for _ in range(n_sims):
        # Simulate 30 days
        path30 = rng.choice(arr, size=30, replace=True)
        cum30  = np.cumsum(path30)
        peak30 = np.maximum.accumulate(cum30)
        dd30   = float((cum30 - peak30).min())
        dds.append(dd30)
        if float(cum30[-1]) >= profit_target and dd30 > -max_dd_limit:
            pass30 += 1
        # Simulate 60 days
        path60 = rng.choice(arr, size=60, replace=True)
        cum60  = np.cumsum(path60)
        peak60 = np.maximum.accumulate(cum60)
        dd60   = float((cum60 - peak60).min())
        if float(cum60[-1]) >= profit_target and dd60 > -max_dd_limit:
            pass60 += 1
    return {
        "p_pass_30d":   round(pass30 / n_sims, 4),
        "p_pass_60d":   round(pass60 / n_sims, 4),
        "p95_dd":       round(float(np.percentile(dds, 5)), 2),
        "median_dd":    round(float(np.median(dds)), 2),
        "worst_dd":     round(float(np.min(dds)), 2),
    }


def print_comparison(label_a, a, label_b, b, period):
    W = 28
    print(f"\n{'='*70}")
    print(f"  {period}")
    print(f"{'='*70}")
    print(f"  {'Metric':<{W}} {'Current live':>16}  {'New config':>16}  {'Delta':>10}")
    print(f"  {'-'*70}")

    def row(name, ka, kb, fmt=".2f", is_pct=False, higher_better=True):
        va = a.get(ka, 0)
        vb = b.get(kb if kb else ka, 0)
        va = int(va) if isinstance(va, (bool, np.bool_)) else va
        vb = int(vb) if isinstance(vb, (bool, np.bool_)) else vb
        delta = vb - va
        arrow = ("↑" if delta > 0 else "↓") if delta != 0 else " "
        good  = (delta > 0) == higher_better
        flag  = " ✅" if good and abs(delta) > 0.01 else (" ❌" if not good and abs(delta) > 0.01 else "")
        if is_pct:
            print(f"  {name:<{W}} {va:>15.1%}  {vb:>15.1%}  {arrow}{abs(delta):>8.1%}{flag}")
        elif fmt == "$":
            print(f"  {name:<{W}} ${va:>14,.0f}  ${vb:>14,.0f}  {arrow}${abs(delta):>7,.0f}{flag}")
        else:
            print(f"  {name:<{W}} {va:>{15}{fmt}}  {vb:>{15}{fmt}}  {arrow}{abs(delta):>{9}{fmt}}{flag}")

    row("N trades",          "n_trades",     "n_trades",     fmt="d",    higher_better=False)
    row("Win rate",          "wr",           "wr",           is_pct=True)
    row("Total PnL",         "pnl",          "pnl",          fmt="$")
    row("Avg PnL/trade",     "avg_pnl",      "avg_pnl",      fmt="$")
    row("Avg win",           "avg_win",      "avg_win",      fmt="$")
    row("Avg loss",          "avg_loss",     "avg_loss",     fmt="$",    higher_better=False)
    row("Profit factor",     "pf",           "pf",           fmt=".3f")
    row("Sharpe",            "sharpe",       "sharpe",       fmt=".3f")
    row("Sortino",           "sortino",      "sortino",      fmt=".3f")
    row("Max drawdown",      "max_dd",       "max_dd",       fmt="$",    higher_better=False)
    row("MLL safe (<$2K)",   "mll_safe",     "mll_safe",     fmt="d")
    row("Trades/day",        "trades_per_day","trades_per_day",fmt=".3f", higher_better=False)

    # Exit reasons
    print(f"\n  Exit reasons:")
    all_reasons = set(a.get("exit_reasons",{}).keys()) | set(b.get("exit_reasons",{}).keys())
    for r in sorted(all_reasons):
        va = a.get("exit_reasons",{}).get(r, 0)
        vb = b.get("exit_reasons",{}).get(r, 0)
        print(f"    {r:<22} {va:>6}  →  {vb:>6}")

    # Monthly breakdown
    all_months = sorted(set(a.get("monthly",{}).keys()) | set(b.get("monthly",{}).keys()))
    if all_months:
        print(f"\n  Monthly PnL:")
        print(f"    {'Month':<10} {'Curr N':>6} {'Curr WR':>8} {'Curr PnL':>10}  {'New N':>6} {'New WR':>8} {'New PnL':>10}")
        print(f"    {'-'*65}")
        cum_a = 0.0; cum_b = 0.0
        for ym in all_months:
            ma = a.get("monthly",{}).get(ym, {})
            mb = b.get("monthly",{}).get(ym, {})
            na = ma.get("n",0); nb = mb.get("n",0)
            wra = ma.get("wr",0); wrb = mb.get("wr",0)
            pa = ma.get("pnl",0); pb = mb.get("pnl",0)
            cum_a += pa; cum_b += pb
            print(f"    {ym:<10} {na:>6}  {wra:>7.1%}  ${pa:>8,.0f}  {nb:>6}  {wrb:>7.1%}  ${pb:>8,.0f}")
        print(f"    {'CUMUL':<10} {'':>6}  {'':>8}  ${cum_a:>8,.0f}  {'':>6}  {'':>8}  ${cum_b:>8,.0f}")


if __name__ == "__main__":
    datasets = {
        "2026 YTD (Jan–Mar 19, 55d)": (
            ROOT / "data/processed/mnq_2026ytd_5min.h5", "bars_5min"
        ),
        "Extended (Aug 2025–Mar 2026, 162d)": (
            ROOT / "data/processed/mnq_5min_aug25_mar26.h5", "bars_5min"
        ),
    }

    CURRENT_LIVE = dict(
        n_contracts=N_CONTRACTS,
        pt_mult=2.0,
        require_above_prevvwap=True,
        skip_gap_up=None,
        label="Current live (PrevVWAP PT=2.0x)"
    )
    NEW_CONFIG = dict(
        n_contracts=N_CONTRACTS,
        pt_mult=3.0,
        require_above_prevvwap=True,
        skip_gap_up=0.003,
        label="New config (PrevVWAP+SkipGapUp PT=3.0x)"
    )

    all_results = {}

    for period, (path, key) in datasets.items():
        print(f"\nLoading {path.name} ...")
        bars = pd.read_hdf(str(path), key=key)
        if bars.index.tz is None:
            bars.index = bars.index.tz_localize("US/Eastern")
        else:
            bars.index = bars.index.tz_convert("US/Eastern")
        bars = bars[
            ((bars.index.hour == 9) & (bars.index.minute >= 30)) |
            ((bars.index.hour > 9) & (bars.index.hour < 16))
        ].copy()
        print(f"  {bars.index[0].date()} → {bars.index[-1].date()}, {len(set(bars.index.date))} days")

        meta = build_day_meta(bars)

        cfg = CURRENT_LIVE
        r_curr = run_backtest(bars, meta, cfg["n_contracts"], cfg["pt_mult"],
                              cfg["require_above_prevvwap"], cfg["skip_gap_up"])

        cfg = NEW_CONFIG
        r_new  = run_backtest(bars, meta, cfg["n_contracts"], cfg["pt_mult"],
                              cfg["require_above_prevvwap"], cfg["skip_gap_up"])

        # Monte Carlo on daily PnL series
        mc_curr = monte_carlo(r_curr["daily_pnl"])
        mc_new  = monte_carlo(r_new["daily_pnl"])
        r_curr["monte_carlo"] = mc_curr
        r_new["monte_carlo"]  = mc_new

        print_comparison(CURRENT_LIVE["label"], r_curr, NEW_CONFIG["label"], r_new, period)

        print(f"\n  Monte Carlo (20k paths, $3K target, $2K DD limit):")
        print(f"    {'':30} {'Current live':>14}  {'New config':>14}")
        print(f"    P(pass by 30d):              {mc_curr.get('p_pass_30d',0):>13.1%}  {mc_new.get('p_pass_30d',0):>13.1%}")
        print(f"    P(pass by 60d):              {mc_curr.get('p_pass_60d',0):>13.1%}  {mc_new.get('p_pass_60d',0):>13.1%}")
        print(f"    p95 drawdown:            ${mc_curr.get('p95_dd',0):>12,.0f}  ${mc_new.get('p95_dd',0):>12,.0f}")
        print(f"    Median drawdown:         ${mc_curr.get('median_dd',0):>12,.0f}  ${mc_new.get('median_dd',0):>12,.0f}")

        all_results[period] = {
            "current_live": {k:v for k,v in r_curr.items() if k != "daily_pnl"},
            "new_config":   {k:v for k,v in r_new.items()  if k != "daily_pnl"},
        }

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nSaved → {RESULTS_PATH}")
