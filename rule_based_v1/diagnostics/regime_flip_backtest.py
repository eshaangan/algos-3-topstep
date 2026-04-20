"""
Regime-Flip SHORT ORB Backtest — "Turn Blocked Days Into Edge"
==============================================================
Tests whether PrevVWAP-bearish days (currently blocked by live filter) have a SHORT ORB
edge in historical bear-market regimes analogous to 2026 tariff-fear.

Hypothesis:
  PrevVWAP bullish  →  LONG ORB  (trend continuation upward)
  PrevVWAP bearish  →  SHORT ORB (trend continuation downward)

Historical analogues to 2026 tariff-fear tested:
  - 2018-Q4   (Oct–Dec 2018): Trump trade war, NQ -23%
  - 2020-Q1   (Jan–Mar 2020): COVID crash, extreme vol
  - 2022 full (Jan–Dec 2022): Fed tightening, NQ -33%  ← strongest bear analogue
  - 2024–2025             :  Recent control period

Three strategies per period:
  1. LONG-only baseline          (ignores bearish days)
  2. SHORT-only on bearish days  (novel hypothesis)
  3. Regime-flip                 (LONG bullish + SHORT bearish — double frequency)

Also tests: VXN-elevated SHORT gate (VXN/VIX spread > 0.08 + PrevVWAP bearish)
            Gap-fade SHORT (PrevVWAP bearish + gap-up today → SHORT the fade)

Usage:
    cd "algos 3 topstep"
    python rule_based_v1/diagnostics/regime_flip_backtest.py             # all periods
    python rule_based_v1/diagnostics/regime_flip_backtest.py --year 2022 # single year
    python rule_based_v1/diagnostics/regime_flip_backtest.py --save      # write JSON

Output: regime_flip_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import datetime
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RBV1 = ROOT / "rule_based_v1"
for p in [str(ROOT), str(RBV1)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.indicators import atr as compute_atr

DATA_PATH    = ROOT / "data" / "processed" / "es_bars_2010_2025.h5"
RESULTS_PATH = ROOT / "rule_based_v1" / "diagnostics" / "regime_flip_results.json"

# ── Execution config (ES-calibrated) ─────────────────────────────────────────
N_CONTRACTS    = 1
POINT_VALUE    = 50.0
TICK_SIZE      = 0.25
COMMISSION     = 2.02
SLIPPAGE_TICKS = 1
PT_MULT        = 3.0
SL_MULT        = 1.5
ATR_PERIOD     = 14
TIME_STOP_BARS = 24
MAX_DAILY_LOSS = -2_000.0
MAX_TRADES_DAY = 1
STARTING_EQUITY  = 50_000.0
DRAWDOWN_BUFFER  = 4_000.0

# ── Historical bear-market periods to test ────────────────────────────────────
PERIODS = {
    "2018-Q4 Trade War":   ("2018-10-01", "2018-12-31"),
    "2020-Q1 COVID":       ("2020-01-01", "2020-03-31"),
    "2022 Fed Bear":       ("2022-01-01", "2022-12-31"),
    "2023 Recovery":       ("2023-01-01", "2023-12-31"),
    "2024-2025 Recent":    ("2024-01-01", "2025-12-31"),
    "Full 2010-2025":      ("2010-01-01", "2025-12-31"),
}


@dataclass
class Pos:
    entry_price:   float
    entry_bar_idx: int
    stop_loss:     float
    profit_target: float
    time_stop_bar: int
    n_contracts:   int
    direction:     int    # +1 = LONG, -1 = SHORT
    atr:           float


def _slip(price: float, direction: int, is_entry: bool) -> float:
    s = SLIPPAGE_TICKS * TICK_SIZE
    return price + s * direction if is_entry else price - s * direction


def _calc_pnl(entry: float, exit_: float, n: int, direction: int) -> float:
    return (exit_ - entry) * direction * n * POINT_VALUE - 2 * COMMISSION * n


def _check_exit(pos: Pos, bar, idx: int, sess_close: bool):
    h, l, c = bar["high"], bar["low"], bar["close"]
    if sess_close:
        ep = _slip(c, pos.direction, False)
        return True, ep, "session_close"
    if idx >= pos.time_stop_bar:
        ep = _slip(c, pos.direction, False)
        return True, ep, "time_stop"
    if pos.direction == 1:          # LONG
        if l <= pos.stop_loss:
            return True, _slip(pos.stop_loss, pos.direction, False), "stop_loss"
        if h >= pos.profit_target:
            return True, _slip(pos.profit_target, pos.direction, False), "profit_target"
    else:                           # SHORT
        if h >= pos.stop_loss:
            return True, _slip(pos.stop_loss, pos.direction, False), "stop_loss"
        if l <= pos.profit_target:
            return True, _slip(pos.profit_target, pos.direction, False), "profit_target"
    return False, 0.0, ""


def _build_day_meta(bars: pd.DataFrame) -> dict:
    """ATR, gap%, prev_vwap_bullish per day."""
    daily = bars.resample("1D").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),   close=("close", "last"),
    ).dropna()
    if daily.index.tz is None:
        daily.index = daily.index.tz_localize("US/Eastern")
    daily["atr14"]   = compute_atr(daily["high"], daily["low"], daily["close"], 14)
    daily["gap_pct"] = (daily["open"] - daily["close"].shift(1)) / daily["close"].shift(1)

    # Intraday VWAP per session → prev_vwap_bullish
    typical  = (bars["high"] + bars["low"] + bars["close"]) / 3
    cum_tp   = (typical * bars["volume"]).groupby(bars.index.date).cumsum()
    cum_vol  = bars["volume"].groupby(bars.index.date).cumsum().replace(0, float("nan"))
    vwap_ser = cum_tp / cum_vol
    vwap_day  = vwap_ser.groupby(vwap_ser.index.map(lambda t: t.date())).last()
    close_day = bars["close"].groupby(bars.index.map(lambda t: t.date())).last()

    meta = {}
    dates_sorted = []
    for ts, row in daily.iterrows():
        d = ts.tz_convert("US/Eastern").date() if hasattr(ts, "tz_convert") else ts.date()
        dates_sorted.append(d)
        meta[d] = {
            "atr14":   float(row["atr14"])   if not np.isnan(row["atr14"])   else 10.0,
            "gap_pct": float(row["gap_pct"]) if not np.isnan(row["gap_pct"]) else 0.0,
        }
    dates_sorted.sort()
    for i, d in enumerate(dates_sorted):
        if i == 0:
            meta[d]["prev_vwap_bullish"] = None
        else:
            prev_d = dates_sorted[i - 1]
            if prev_d in vwap_day.index and prev_d in close_day.index:
                meta[d]["prev_vwap_bullish"] = bool(close_day[prev_d] > vwap_day[prev_d])
            else:
                meta[d]["prev_vwap_bullish"] = None
    return meta


def _enrich_vix_vxn(meta: dict, start: str, end: str, verbose: bool = True) -> dict:
    try:
        import yfinance as yf
    except ImportError:
        for d in meta:
            meta[d]["tech_fear_spread"] = None
        return meta

    try:
        vix_df = yf.download("^VIX", start=start, end=end, interval="1d", progress=False, auto_adjust=True)
        vxn_df = yf.download("^VXN", start=start, end=end, interval="1d", progress=False, auto_adjust=True)
    except Exception:
        for d in meta:
            meta[d]["tech_fear_spread"] = None
        return meta

    def to_series(df, label):
        c = df["Close"] if hasattr(df, "columns") else df
        if hasattr(c, "squeeze"):
            c = c.squeeze()
        idx = c.index
        dates = [pd.to_datetime(t).date() for t in idx]
        return pd.Series(c.values.flatten().astype(float), index=dates, name=label)

    vix = to_series(vix_df, "VIX")
    vxn = to_series(vxn_df, "VXN")
    spread = (vxn / vix - 1.0)

    dates_sorted = sorted(meta.keys())
    hits = 0
    for i, d in enumerate(dates_sorted):
        if i == 0:
            meta[d]["tech_fear_spread"] = None
        else:
            prev_d = dates_sorted[i - 1]
            val = spread.get(prev_d)
            if val is not None and not np.isnan(float(val)):
                meta[d]["tech_fear_spread"] = float(val)
                hits += 1
            else:
                meta[d]["tech_fear_spread"] = None

    if verbose:
        print(f"    VXN/VIX spread: {hits}/{len(dates_sorted)} days populated")
    return meta


# ── Core backtest: direction-aware ────────────────────────────────────────────
def run_flip(
    bars: pd.DataFrame,
    day_meta: dict,
    mode: str = "long_only",   # "long_only" | "short_only" | "flip" | "short_bearish" | "gap_fade_short"
    vxn_gate: bool = False,    # if True, SHORT only when vxn_spread > 0.08
    label: str = "",
) -> dict:
    """
    mode:
      long_only       — LONG on bullish days only (baseline)
      short_only      — SHORT on all days (control)
      short_bearish   — SHORT only on PrevVWAP-bearish days
      flip            — LONG on bullish + SHORT on bearish (double frequency)
      gap_fade_short  — SHORT only when PrevVWAP bearish AND today gap > 0.3%
    """
    atr_s    = compute_atr(bars["high"], bars["low"], bars["close"], ATR_PERIOD)
    min_bars = ATR_PERIOD + 1

    pos = None; trades = []; equity = STARTING_EQUITY; peak = STARTING_EQUITY
    max_dd = 0.0; cur_date = None; daily_pnl: dict = {}
    trades_today = 0; daily_loss = 0.0

    or_high: dict[datetime.date, float] = {}
    or_low:  dict[datetime.date, float] = {}

    OR_START = 9 * 60 + 30
    OR_END   = 10 * 60 + 0
    ENTRY_CUTOFF = 12 * 60

    for i in range(min_bars, len(bars)):
        bar   = bars.iloc[i]
        bt_et = bars.index[i].tz_convert("US/Eastern")
        bdate = bt_et.date()
        bh, bm = bt_et.hour, bt_et.minute
        btime  = bh * 60 + bm

        if cur_date is not None and bdate != cur_date:
            daily_pnl[cur_date] = daily_loss
            daily_loss = 0.0; trades_today = 0
        cur_date = bdate

        meta    = day_meta.get(bdate, {})
        atr_now = atr_s.iloc[i]
        if np.isnan(atr_now) or atr_now <= 0:
            continue

        is_last    = (i + 1 >= len(bars)) or bars.index[i + 1].tz_convert("US/Eastern").date() != bdate
        sess_close = is_last or (bh == 15 and bm >= 55)

        # Build OR
        if OR_START <= btime <= OR_END:
            if bdate not in or_high:
                or_high[bdate] = bar["high"]
                or_low[bdate]  = bar["low"]
            else:
                or_high[bdate] = max(or_high[bdate], bar["high"])
                or_low[bdate]  = min(or_low[bdate],  bar["low"])

        # Check exit
        if pos is not None:
            exited, exit_p, reason = _check_exit(pos, bar, i, sess_close)
            if exited:
                p = _calc_pnl(pos.entry_price, exit_p, pos.n_contracts, pos.direction)
                trades.append({
                    "date": bdate, "entry": pos.entry_price, "exit": exit_p,
                    "pnl": p, "reason": reason, "direction": pos.direction,
                    "prev_vwap_bullish": meta.get("prev_vwap_bullish"),
                    "gap_pct": meta.get("gap_pct", 0.0),
                })
                equity += p; daily_loss += p
                peak   = max(peak, equity)
                max_dd = min(max_dd, equity - peak)
                pos = None

        can_enter = (
            pos is None and not sess_close
            and trades_today < MAX_TRADES_DAY
            and daily_loss > MAX_DAILY_LOSS
            and (equity - peak) > -DRAWDOWN_BUFFER
            and bdate in or_high
            and btime > OR_END
            and btime <= ENTRY_CUTOFF
        )

        if not can_enter:
            continue

        pv  = meta.get("prev_vwap_bullish")
        gap = meta.get("gap_pct", 0.0)
        vxn = meta.get("tech_fear_spread")

        # Determine direction to take based on mode
        direction = 0  # 0 = no trade

        if mode == "long_only":
            if pv is True:
                direction = 1
        elif mode == "short_only":
            direction = -1
        elif mode == "short_bearish":
            if pv is False:
                direction = -1
                if vxn_gate:
                    # Require elevated tech fear for SHORT
                    if vxn is None or vxn <= 0.08:
                        direction = 0
        elif mode == "flip":
            if pv is True:
                direction = 1
            elif pv is False:
                direction = -1
                if vxn_gate and (vxn is None or vxn <= 0.08):
                    direction = 0
        elif mode == "gap_fade_short":
            # SHORT only: bearish prev day AND gap up today
            if pv is False and gap > 0.003:   # gap > 0.3%
                direction = -1

        if direction == 0:
            continue

        orh = or_high[bdate]
        orl = or_low[bdate]
        or_range = orh - orl
        if or_range < 0.3 * atr_now:
            continue   # OR too narrow — no signal

        if direction == 1:
            # LONG: need close above OR high
            if bar["close"] > orh:
                ep = _slip(bar["close"], 1, True)
                sl = ep - SL_MULT * atr_now
                pt = ep + PT_MULT * atr_now
                pos = Pos(ep, i, sl, pt, i + TIME_STOP_BARS, N_CONTRACTS, 1, atr_now)
                trades_today += 1
        else:
            # SHORT: need close below OR low
            if bar["close"] < orl:
                ep = _slip(bar["close"], -1, True)
                sl = ep + SL_MULT * atr_now
                pt = ep - PT_MULT * atr_now
                pos = Pos(ep, i, sl, pt, i + TIME_STOP_BARS, N_CONTRACTS, -1, atr_now)
                trades_today += 1

    if cur_date and cur_date not in daily_pnl:
        daily_pnl[cur_date] = daily_loss

    wins  = [t for t in trades if t["pnl"] > 0]
    longs = [t for t in trades if t["direction"] == 1]
    shorts = [t for t in trades if t["direction"] == -1]
    total = sum(t["pnl"] for t in trades)
    gp    = sum(t["pnl"] for t in wins)
    gl    = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    daily = pd.Series(daily_pnl)
    daily = daily[daily != 0]
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if len(daily) > 1 and daily.std() > 0 else 0.0

    long_wins  = [t for t in longs  if t["pnl"] > 0]
    short_wins = [t for t in shorts if t["pnl"] > 0]

    return {
        "label":       label,
        "mode":        mode,
        "n":           len(trades),
        "wr":          len(wins) / max(len(trades), 1),
        "pnl":         total,
        "sharpe":      sharpe,
        "dd":          max_dd,
        "pf":          gp / gl if gl > 0 else float("inf"),
        "n_long":      len(longs),
        "wr_long":     len(long_wins) / max(len(longs), 1),
        "n_short":     len(shorts),
        "wr_short":    len(short_wins) / max(len(shorts), 1),
        "trades":      trades,
        "daily_pnl":   daily_pnl,
    }


def yearly_stats(r: dict) -> dict:
    by_year: dict[int, list] = defaultdict(list)
    for t in r["trades"]:
        by_year[t["date"].year].append(t)
    out = {}
    for yr, ts in sorted(by_year.items()):
        wins = [t for t in ts if t["pnl"] > 0]
        out[yr] = {"n": len(ts), "wr": len(wins) / max(len(ts), 1), "pnl": sum(t["pnl"] for t in ts)}
    return out


def print_row(label: str, r: dict, width: int = 40) -> None:
    print(f"  {label:<{width}}  {r['n']:>4}  {r['wr']:>5.1%}  ${r['pnl']:>8,.0f}"
          f"  {r['sharpe']:>6.2f}  ${r['dd']:>7,.0f}"
          f"  L:{r['n_long']:>3}/{r['wr_long']:>4.0%}  S:{r['n_short']:>3}/{r['wr_short']:>4.0%}")


def run_period(bars: pd.DataFrame, meta: dict, period_name: str, start: str, end: str) -> dict:
    """Run all strategy variants on a specific time period."""
    mask = (bars.index >= pd.Timestamp(start, tz="US/Eastern")) & \
           (bars.index <= pd.Timestamp(end + " 23:59", tz="US/Eastern"))
    pb = bars[mask]
    if len(pb) < 100:
        return {}

    n_days = len(set(pb.index.date))
    print(f"\n  {'─'*80}")
    print(f"  Period: {period_name}  [{start} → {end}]  ({n_days} trading days, {len(pb):,} bars)")
    print(f"  {'─'*80}")
    print(f"  {'Strategy':<40}  {'N':>4}  {'WR':>5}  {'PnL':>9}  {'Sharpe':>6}  {'MaxDD':>8}  LONG       SHORT")
    print(f"  {'-'*95}")

    configs = [
        ("LONG-only baseline",           "long_only",       False),
        ("SHORT-only (all days)",         "short_only",      False),
        ("SHORT on bearish days only",    "short_bearish",   False),
        ("SHORT on bearish + VXN>0.08",   "short_bearish",   True),
        ("Regime-Flip (LONG + SHORT)",    "flip",            False),
        ("Regime-Flip + VXN gate",        "flip",            True),
        ("Gap-fade SHORT (bearish+gap)",  "gap_fade_short",  False),
    ]

    results = {}
    for lbl, mode, vxn_gate in configs:
        r = run_flip(pb, meta, mode=mode, vxn_gate=vxn_gate, label=lbl)
        results[lbl] = r
        print_row(lbl, r)

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=None, help="Test single year only")
    parser.add_argument("--save", action="store_true", help="Save results JSON")
    args = parser.parse_args()

    if not DATA_PATH.exists():
        print(f"  ES data not found: {DATA_PATH}")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"  REGIME-FLIP SHORT ORB BACKTEST")
    print(f"  Hypothesis: PrevVWAP-bearish days have SHORT ORB edge in bear regimes")
    print(f"{'='*80}")

    print(f"\nLoading ES 5-min bars ...")
    bars = pd.read_hdf(str(DATA_PATH), key="bars_5min")
    if "timestamp" in bars.columns:
        bars.index = pd.to_datetime(bars["timestamp"])
        bars = bars.drop(columns=["timestamp"])
    if not isinstance(bars.index, pd.DatetimeIndex):
        bars.index = pd.to_datetime(bars.index)
    if getattr(bars.index, "tz", None) is None:
        bars.index = bars.index.tz_localize("UTC")
    bars.index = bars.index.tz_convert("US/Eastern")

    # RTH only
    rth = (
        ((bars.index.hour > 9) | ((bars.index.hour == 9) & (bars.index.minute >= 30)))
        & (bars.index.hour < 16)
    )
    bars = bars[rth]
    print(f"  {len(bars):,} RTH bars  {bars.index[0].date()} → {bars.index[-1].date()}")

    # Build metadata once for full range
    print("\nBuilding day metadata (ATR, gap, prevVWAP) ...")
    meta = _build_day_meta(bars)

    # Enrich VXN/VIX
    full_start = (bars.index[0] - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    full_end   = (bars.index[-1] + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    print("Fetching VXN/VIX regime data ...")
    meta = _enrich_vix_vxn(meta, full_start, full_end, verbose=True)

    # ── Single year mode ──────────────────────────────────────────────────────
    if args.year:
        periods_to_run = {
            f"{args.year}": (f"{args.year}-01-01", f"{args.year}-12-31")
        }
    else:
        periods_to_run = PERIODS

    # ── Run each period ───────────────────────────────────────────────────────
    all_results = {}
    for period_name, (start, end) in periods_to_run.items():
        r = run_period(bars, meta, period_name, start, end)
        if r:
            all_results[period_name] = r

    # ── Cross-period summary: SHORT WR on bearish days by year ────────────────
    print(f"\n\n{'='*80}")
    print(f"  YEAR-BY-YEAR: SHORT WR on PrevVWAP-Bearish Days (full 2010-2025)")
    print(f"{'='*80}")

    # Run full period once for year breakdown
    full_short = run_flip(bars, meta, mode="short_bearish", label="SHORT-bearish full")
    full_long  = run_flip(bars, meta, mode="long_only",     label="LONG-only full")
    full_flip  = run_flip(bars, meta, mode="flip",          label="Flip full")

    ys_short = yearly_stats(full_short)
    ys_long  = yearly_stats(full_long)
    ys_flip  = yearly_stats(full_flip)

    print(f"  {'Year':>4}  {'L_N':>4} {'L_WR':>5}  {'S_N':>4} {'S_WR':>5}  "
          f"{'Flip_N':>6} {'Flip_WR':>6} {'Flip_PnL':>10}")
    print(f"  {'-'*70}")

    all_years = sorted(set(list(ys_short.keys()) + list(ys_long.keys())))
    for yr in all_years:
        ls = ys_long.get(yr,  {"n": 0, "wr": 0.0, "pnl": 0.0})
        ss = ys_short.get(yr, {"n": 0, "wr": 0.0, "pnl": 0.0})
        fs = ys_flip.get(yr,  {"n": 0, "wr": 0.0, "pnl": 0.0})
        # Highlight bear years
        flag = " ◀ BEAR" if yr in (2018, 2020, 2022) else ""
        print(f"  {yr:>4}  {ls['n']:>4} {ls['wr']:>5.0%}  {ss['n']:>4} {ss['wr']:>5.0%}  "
              f"{fs['n']:>6} {fs['wr']:>6.0%} ${fs['pnl']:>9,.0f}{flag}")

    # ── Summary: does the flip strategy beat LONG-only? ───────────────────────
    print(f"\n{'='*80}")
    print(f"  COMBINED STRATEGY COMPARISON (full 2010-2025)")
    print(f"{'='*80}")
    print(f"  {'Strategy':<40}  {'N':>4}  {'WR':>5}  {'PnL':>9}  {'Sharpe':>6}  {'MaxDD':>8}")
    print(f"  {'-'*75}")
    for lbl, r in [
        ("LONG-only",                    full_long),
        ("SHORT on bearish days",        full_short),
        ("Regime-Flip (LONG + SHORT)",   full_flip),
    ]:
        print(f"  {lbl:<40}  {r['n']:>4}  {r['wr']:>5.1%}  ${r['pnl']:>8,.0f}"
              f"  {r['sharpe']:>6.2f}  ${r['dd']:>7,.0f}")

    # ── Decision guidance ─────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  DEPLOYMENT DECISION GATES")
    print(f"{'='*80}")

    bear_years = [2018, 2020, 2022]
    bear_short_wrs = [ys_short.get(yr, {}).get("wr", 0.0) for yr in bear_years]
    bear_short_ns  = [ys_short.get(yr, {}).get("n", 0)   for yr in bear_years]

    for yr, wr, n in zip(bear_years, bear_short_wrs, bear_short_ns):
        gate_wr  = "✅" if wr >= 0.55 else "❌"
        gate_n   = "✅" if n  >= 20   else "❌"
        print(f"  {yr}: SHORT WR={wr:.0%} {gate_wr} (need ≥55%)   N={n} {gate_n} (need ≥20)")

    flip_sharpe = full_flip["sharpe"]
    long_sharpe = full_long["sharpe"]
    gate_flip   = "✅" if flip_sharpe > long_sharpe else "❌"
    print(f"\n  Flip Sharpe ({flip_sharpe:.2f}) > LONG-only Sharpe ({long_sharpe:.2f}): {gate_flip}")

    all_gates_pass = (
        all(wr >= 0.55 for wr in bear_short_wrs if wr > 0) and
        any(n >= 20 for n in bear_short_ns) and
        flip_sharpe > long_sharpe
    )

    print(f"\n  Verdict: {'DEPLOY SHORT ORB ON BEARISH DAYS ✅' if all_gates_pass else 'NOT YET — needs more validation ❌'}")
    print(f"\n  If deploying: change rules.yaml → long_only: false")
    print(f"  Live logic: PrevVWAP bearish → flip direction to SHORT in runner.py")

    # ── Save ─────────────────────────────────────────────────────────────────
    if args.save:
        serializable = {}
        for period, configs in all_results.items():
            serializable[period] = {}
            for lbl, r in configs.items():
                sv = {k: v for k, v in r.items() if k not in ("trades", "daily_pnl")}
                sv["yearly"] = {str(yr): s for yr, s in yearly_stats(r).items()}
                serializable[period][lbl] = sv
        # Add full-range summary
        for lbl, r in [("LONG-only", full_long), ("SHORT-bearish", full_short), ("Flip", full_flip)]:
            sv = {k: v for k, v in r.items() if k not in ("trades", "daily_pnl")}
            sv["yearly"] = {str(yr): s for yr, s in (yearly_stats(r) if r["n"] > 0 else {}).items()}
            serializable.setdefault("Full 2010-2025", {})[lbl] = sv

        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(serializable, indent=2, default=str))
        print(f"\n  Results saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
