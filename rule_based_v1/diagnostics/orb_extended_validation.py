"""
ORB Extended Validation — Aug 2025 – Mar 2026 (162 days)
=========================================================
Validates the MNQ ORB strategy and key filters on the FULL extended dataset.

Tests:
  1. Baseline ORB (3c, LONG-only, or_end=10:04, PT=3.0x, SL=1.5x)
  2. PrevVWAP filter (trade only when prev-day close > prev-day VWAP)
  3. PT=2.0x (regime quality finding)
  4. PrevVWAP + PT=2.0x combined
  5. Skip gap-up >0.3% filter
  6. EMA20 regime scaler (3c above, 2c below)
  7. Skip large gap (>0.5% either direction)
  8. Best combo: PrevVWAP + EMA20 scaler

Data: data/processed/mnq_5min_aug25_mar26.h5 (Aug 2025 – Mar 23 2026, 162 days)
"""
from __future__ import annotations
import sys, json, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

ROOT = Path(__file__).resolve().parent.parent.parent
RBV1 = ROOT / "rule_based_v1"
for p in [str(ROOT), str(RBV1)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.indicators import atr as compute_atr

DATA_PATH    = ROOT / "data" / "processed" / "mnq_5min_aug25_mar26.h5"
RESULTS_PATH = RBV1 / "diagnostics" / "orb_extended_results.json"

# ── Strategy parameters (live config) ────────────────────────────────────────
OR_END_HOUR, OR_END_MIN = 10, 4
MIN_OR_BARS     = 7
PT_MULT_DEFAULT = 3.0
SL_MULT         = 1.5
ENTRY_CUTOFF_H  = 12
ATR_PERIOD      = 14
TIME_STOP_BARS  = 24
TRAILING_ACT    = 999.0

POINT_VALUE  = 2.0
TICK_SIZE    = 0.25
COMMISSION   = 0.62
SLIPPAGE     = 1        # ticks

MAX_TRADES_DAY  = 2
MAX_DAILY_LOSS  = -950.0
DRAWDOWN_BUF    = 1_950.0
STARTING_EQ     = 50_000.0
EMA_PERIOD      = 20    # days for EMA20 regime

@dataclass
class Pos:
    direction: int
    entry: float
    stop_loss: float
    profit_target: float
    time_stop_bar: int
    n: int


def slip(p, d, is_entry):
    s = SLIPPAGE * TICK_SIZE
    return p + s * d if is_entry else p - s * d


def calc_pnl(entry, exit_, direction, n):
    return (exit_ - entry) * direction * n * POINT_VALUE - 2 * COMMISSION * n


def compute_vwap(grp):
    typical = (grp["high"] + grp["low"] + grp["close"]) / 3
    pv = typical * grp["volume"]
    cum_pv  = pv.cumsum()
    cum_vol = grp["volume"].cumsum().replace(0, np.nan)
    return (cum_pv / cum_vol).iloc[-1]


def build_day_meta(bars: pd.DataFrame) -> pd.DataFrame:
    """Per-day: vwap, close, open_930, atr, gap."""
    atr_5m = compute_atr(bars["high"], bars["low"], bars["close"], ATR_PERIOD)
    rows = []
    for date, grp in bars.groupby(bars.index.date):
        et = grp.index.tz_convert("US/Eastern")
        open_bars = grp[(et.hour == 9) & (et.minute == 30)]
        if open_bars.empty:
            continue
        rows.append({
            "date":      date,
            "open_930":  float(open_bars["open"].iloc[0]),
            "close_day": float(grp["close"].iloc[-1]),
            "vwap_day":  compute_vwap(grp),
            "day_high":  float(grp["high"].max()),
            "day_low":   float(grp["low"].min()),
            "atr":       float(atr_5m.reindex(grp.index).dropna().iloc[-1]) if len(atr_5m.reindex(grp.index).dropna()) > 0 else np.nan,
        })
    df = pd.DataFrame(rows).set_index("date")
    df["prev_close"] = df["close_day"].shift(1)
    df["prev_vwap"]  = df["vwap_day"].shift(1)
    df["gap_pct"]    = (df["open_930"] - df["prev_close"]) / df["prev_close"].replace(0, np.nan)
    df["above_prev_vwap"] = df["open_930"] > df["prev_vwap"]
    # EMA20 of close
    df["ema20_daily"] = df["close_day"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df["above_ema20"] = df["close_day"].shift(1) > df["ema20_daily"].shift(1)
    return df


def run_backtest(
    bars: pd.DataFrame,
    day_meta: pd.DataFrame,
    n_contracts: int = 3,
    pt_mult: float = 3.0,
    # Filters
    require_above_prevvwap: bool = False,
    skip_large_gap: float | None = None,   # skip if |gap_pct| > this
    skip_gap_up: float | None = None,      # skip if gap_pct > this (gap-up filter)
    ema20_scaler: bool = False,            # use 2c when below EMA20
    long_only: bool = True,
    label: str = "",
) -> dict:
    equity = STARTING_EQ
    peak   = STARTING_EQ
    max_dd = 0.0
    trades = []
    daily_pnl: dict = {}

    for date, grp in bars.groupby(bars.index.date):
        if date not in day_meta.index:
            continue
        meta = day_meta.loc[date]
        if pd.isna(meta.get("atr", np.nan)):
            continue

        # ── Filters ──────────────────────────────────────────────────────────
        if require_above_prevvwap and not meta.get("above_prev_vwap", True):
            continue
        gap = meta.get("gap_pct", 0.0)
        if skip_large_gap is not None and abs(gap) > skip_large_gap:
            continue
        if skip_gap_up is not None and gap > skip_gap_up:
            continue

        # Effective contracts
        n = n_contracts
        if ema20_scaler and not meta.get("above_ema20", True):
            n = max(1, n - 1)

        # ── Find OR high/low ──────────────────────────────────────────────────
        et     = grp.index.tz_convert("US/Eastern")
        or_mask = ((et.hour == 9) & (et.minute >= 30)) | \
                  ((et.hour == 10) & (et.minute <= OR_END_MIN)) | \
                  ((et.hour < 10))
        or_mask &= (et.hour > 0)   # remove midnight bars
        or_bars = grp[or_mask]
        post_or = grp[
            (et.hour > OR_END_HOUR) |
            ((et.hour == OR_END_HOUR) & (et.minute > OR_END_MIN))
        ]
        entry_mask = post_or.index[
            post_or.index.tz_convert("US/Eastern").hour < ENTRY_CUTOFF_H
        ]

        if len(or_bars) < MIN_OR_BARS or entry_mask.empty:
            continue

        or_high = float(or_bars["high"].max())
        or_low  = float(or_bars["low"].min())
        or_range = or_high - or_low
        atr_now  = meta["atr"]
        if atr_now <= 0 or np.isnan(atr_now):
            continue

        # ── Check OR width ────────────────────────────────────────────────────
        if or_range / atr_now < 0.3:
            continue

        day_loss  = 0.0
        pos       = None
        trades_today = 0

        for bar_ts in grp.index:
            bar    = grp.loc[bar_ts]
            bar_et = bar_ts.tz_convert("US/Eastern")

            # Session close at 16:00
            is_close = (bar_et.hour >= 16)

            # Check existing position exit
            if pos is not None:
                bar_idx = list(grp.index).index(bar_ts)
                if is_close:
                    ep = slip(float(bar["close"]), pos.direction, False)
                    pnl = calc_pnl(pos.entry, ep, pos.direction, pos.n)
                    trades.append({"date": str(date), "dir": pos.direction,
                                   "entry": pos.entry, "exit": ep,
                                   "pnl": round(pnl, 2), "reason": "session_close", "n": pos.n})
                    day_loss += pnl; equity += pnl; peak = max(peak, equity)
                    max_dd = min(max_dd, equity - peak)
                    pos = None
                    continue

                if bar_idx >= pos.time_stop_bar:
                    ep = slip(float(bar["close"]), pos.direction, False)
                    pnl = calc_pnl(pos.entry, ep, pos.direction, pos.n)
                    trades.append({"date": str(date), "dir": pos.direction,
                                   "entry": pos.entry, "exit": ep,
                                   "pnl": round(pnl, 2), "reason": "time_stop", "n": pos.n})
                    day_loss += pnl; equity += pnl; peak = max(peak, equity)
                    max_dd = min(max_dd, equity - peak)
                    pos = None
                    continue

                if pos.direction == 1:
                    if bar["low"] <= pos.stop_loss:
                        ep = slip(pos.stop_loss, 1, False)
                        pnl = calc_pnl(pos.entry, ep, 1, pos.n)
                        trades.append({"date": str(date), "dir": 1,
                                       "entry": pos.entry, "exit": ep,
                                       "pnl": round(pnl, 2), "reason": "stop_loss", "n": pos.n})
                        day_loss += pnl; equity += pnl; peak = max(peak, equity)
                        max_dd = min(max_dd, equity - peak)
                        pos = None
                        continue
                    if bar["high"] >= pos.profit_target:
                        ep = slip(pos.profit_target, 1, False)
                        pnl = calc_pnl(pos.entry, ep, 1, pos.n)
                        trades.append({"date": str(date), "dir": 1,
                                       "entry": pos.entry, "exit": ep,
                                       "pnl": round(pnl, 2), "reason": "profit_target", "n": pos.n})
                        day_loss += pnl; equity += pnl; peak = max(peak, equity)
                        max_dd = min(max_dd, equity - peak)
                        pos = None
                        continue
                else:
                    if bar["high"] >= pos.stop_loss:
                        ep = slip(pos.stop_loss, -1, False)
                        pnl = calc_pnl(pos.entry, ep, -1, pos.n)
                        trades.append({"date": str(date), "dir": -1,
                                       "entry": pos.entry, "exit": ep,
                                       "pnl": round(pnl, 2), "reason": "stop_loss", "n": pos.n})
                        day_loss += pnl; equity += pnl; peak = max(peak, equity)
                        max_dd = min(max_dd, equity - peak)
                        pos = None
                        continue
                    if bar["low"] <= pos.profit_target:
                        ep = slip(pos.profit_target, -1, False)
                        pnl = calc_pnl(pos.entry, ep, -1, pos.n)
                        trades.append({"date": str(date), "dir": -1,
                                       "entry": pos.entry, "exit": ep,
                                       "pnl": round(pnl, 2), "reason": "profit_target", "n": pos.n})
                        day_loss += pnl; equity += pnl; peak = max(peak, equity)
                        max_dd = min(max_dd, equity - peak)
                        pos = None
                        continue

            # ── Entry logic ──────────────────────────────────────────────────
            if pos is None and trades_today < MAX_TRADES_DAY:
                if day_loss <= MAX_DAILY_LOSS:
                    continue
                if (equity - peak) <= -DRAWDOWN_BUF:
                    continue
                if bar_et.hour < ENTRY_CUTOFF_H and (
                    (bar_et.hour > OR_END_HOUR) or
                    (bar_et.hour == OR_END_HOUR and bar_et.minute > OR_END_MIN)
                ):
                    h, l = float(bar["high"]), float(bar["low"])
                    if h > or_high:  # LONG breakout
                        entry = slip(or_high, 1, True)
                        stop  = entry - SL_MULT * atr_now
                        tgt   = entry + pt_mult * atr_now
                        bar_idx = list(grp.index).index(bar_ts)
                        ts_bar  = bar_idx + TIME_STOP_BARS
                        pos = Pos(1, entry, stop, tgt, ts_bar, n)
                        trades_today += 1
                    elif not long_only and l < or_low:  # SHORT breakdown
                        entry = slip(or_low, -1, True)
                        stop  = entry + SL_MULT * atr_now
                        tgt   = entry - pt_mult * atr_now
                        bar_idx = list(grp.index).index(bar_ts)
                        ts_bar  = bar_idx + TIME_STOP_BARS
                        pos = Pos(-1, entry, stop, tgt, ts_bar, n)
                        trades_today += 1

        # End of day: force close any open position
        if pos is not None and grp.shape[0] > 0:
            last = grp.iloc[-1]
            ep = slip(float(last["close"]), pos.direction, False)
            pnl = calc_pnl(pos.entry, ep, pos.direction, pos.n)
            trades.append({"date": str(date), "dir": pos.direction,
                           "entry": pos.entry, "exit": ep,
                           "pnl": round(pnl, 2), "reason": "eod_close", "n": pos.n})
            day_loss += pnl; equity += pnl; peak = max(peak, equity)
            max_dd = min(max_dd, equity - peak)

        if day_loss != 0:
            daily_pnl[str(date)] = round(day_loss, 2)

    wins  = [t for t in trades if t["pnl"] > 0]
    total = sum(t["pnl"] for t in trades)
    gp    = sum(t["pnl"] for t in wins)
    gl    = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    dp    = pd.Series(list(daily_pnl.values()))
    dp    = dp[dp != 0]
    sharpe = dp.mean() / dp.std() * np.sqrt(252) if len(dp) > 1 and dp.std() > 0 else 0

    monthly: dict = defaultdict(list)
    for t in trades:
        monthly[t["date"][:7]].append(t["pnl"])

    return {
        "label": label,
        "n": len(trades),
        "wr": len(wins) / max(len(trades), 1),
        "pnl": round(total, 2),
        "sharpe": round(sharpe, 3),
        "dd": round(max_dd, 2),
        "mll": max_dd > -2000,
        "pf": round(gp / gl, 3) if gl > 0 else float("inf"),
        "avg_pnl": round(total / max(len(trades), 1), 2),
        "trades_per_day": round(len(trades) / max(len(daily_pnl), 1), 3),
        "exit_reasons": {
            r: sum(1 for t in trades if t["reason"] == r)
            for r in ["stop_loss", "time_stop", "profit_target", "session_close", "eod_close"]
        },
        "monthly": {
            ym: {
                "n": len(pnls),
                "wr": sum(1 for p in pnls if p > 0) / len(pnls),
                "pnl": round(sum(pnls), 2)
            }
            for ym, pnls in sorted(monthly.items())
        },
        "daily_pnl": daily_pnl,
    }


def monte_carlo(trades_pnl, n_sims=10_000, target=3_000, max_dd_limit=2_000, n_trades_needed=50):
    if len(trades_pnl) < 5:
        return {}
    rng = np.random.default_rng(42)
    arr = np.array(trades_pnl)
    pf  = sum(1 for p in arr if p > 0) / len(arr)
    pass_count = 0
    drawdowns  = []
    for _ in range(n_sims):
        path = rng.choice(arr, size=min(n_trades_needed, len(arr)*2), replace=True)
        cum  = np.cumsum(path)
        peak = np.maximum.accumulate(cum)
        dd   = float((cum - peak).min())
        final = float(cum[-1])
        if final >= target and dd > -max_dd_limit:
            pass_count += 1
        drawdowns.append(dd)
    return {
        "p_pass": round(pass_count / n_sims, 4),
        "p95_dd": round(np.percentile(drawdowns, 5), 2),
        "median_dd": round(np.median(drawdowns), 2),
    }


if __name__ == "__main__":
    print(f"Loading {DATA_PATH} ...")
    bars = pd.read_hdf(str(DATA_PATH), key="bars_5min")
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("US/Eastern")
    else:
        bars.index = bars.index.tz_convert("US/Eastern")

    # RTH only
    bars = bars[
        ((bars.index.hour == 9) & (bars.index.minute >= 30)) |
        ((bars.index.hour > 9) & (bars.index.hour < 16))
    ].copy()

    print(f"Bars: {len(bars):,} | Days: {len(set(bars.index.date))} | {bars.index[0].date()} → {bars.index[-1].date()}")

    print("Building day meta...")
    meta = build_day_meta(bars)
    print(f"Day meta: {len(meta)} rows")

    W = 55
    print(f"\n{'Config':<{W}} {'N':>4} {'WR':>6} {'PnL':>9} {'Sharpe':>7} {'MaxDD':>9} {'MLL':>4}")
    print("-" * 95)

    configs = [
        dict(n_contracts=3, pt_mult=3.0, label="Baseline 3c PT=3.0x"),
        dict(n_contracts=2, pt_mult=3.0, label="Baseline 2c PT=3.0x"),
        dict(n_contracts=3, pt_mult=3.0, require_above_prevvwap=True, label="PrevVWAP 3c PT=3.0x"),
        dict(n_contracts=3, pt_mult=2.0, require_above_prevvwap=True, label="PrevVWAP 3c PT=2.0x"),
        dict(n_contracts=3, pt_mult=2.0, label="Baseline 3c PT=2.0x"),
        dict(n_contracts=3, pt_mult=3.0, skip_gap_up=0.003, label="Skip gap-up>0.3% 3c"),
        dict(n_contracts=3, pt_mult=3.0, ema20_scaler=True, label="EMA20 scaler 3c→2c"),
        dict(n_contracts=3, pt_mult=3.0, require_above_prevvwap=True, ema20_scaler=True, label="PrevVWAP+EMA20 3c→2c"),
        dict(n_contracts=3, pt_mult=2.0, require_above_prevvwap=True, ema20_scaler=True, label="PrevVWAP+EMA20 PT=2.0x"),
        dict(n_contracts=3, pt_mult=3.0, skip_large_gap=0.005, label="Skip large gap>0.5%"),
        dict(n_contracts=3, pt_mult=3.0, long_only=False, label="L+S 3c PT=3.0x"),
        dict(n_contracts=2, pt_mult=3.0, long_only=False, label="L+S 2c PT=3.0x"),
        dict(n_contracts=3, pt_mult=3.0, require_above_prevvwap=True, skip_gap_up=0.003, label="PrevVWAP+SkipGapUp"),
        dict(n_contracts=3, pt_mult=3.0, long_only=False, skip_large_gap=0.003, label="L+S 3c SkipGap>0.3%"),
    ]

    all_results = {}
    for cfg in configs:
        r = run_backtest(bars, meta, **cfg)
        all_results[r["label"]] = r
        mll = "OK" if r["mll"] else "XX"
        print(f"  {r['label']:<{W}} {r['n']:>4}  {r['wr']:>5.1%}  ${r['pnl']:>8,.0f}  {r['sharpe']:>6.2f}  ${r['dd']:>7,.0f}  {mll}")

    # Best config Monte Carlo
    by_sharpe = sorted(
        [(k, v) for k, v in all_results.items() if v["mll"] and v["n"] >= 10],
        key=lambda x: x[1]["sharpe"], reverse=True
    )
    print(f"\n{'='*80}")
    print("TOP CONFIGS (MLL-safe, n>=10):")
    for label, r in by_sharpe[:4]:
        mc = monte_carlo([v for v in r.get("daily_pnl", {}).values() if v != 0],
                         n_trades_needed=r["n"])
        r["monte_carlo"] = mc
        print(f"\n  [{label}]")
        print(f"    N={r['n']} WR={r['wr']:.1%} PnL=${r['pnl']:,.0f} Sharpe={r['sharpe']:.2f} DD=${r['dd']:,.0f}")
        print(f"    Exits: {r['exit_reasons']}")
        print(f"    MonteCarlo: {mc}")
        print(f"    Monthly breakdown:")
        for ym, m in sorted(r["monthly"].items()):
            print(f"      {ym}: n={m['n']} wr={m['wr']:.1%} pnl=${m['pnl']:,.0f}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved → {RESULTS_PATH}")
