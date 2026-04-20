"""Edge Research Diagnostic — OR Compression + Best Filters Combined.

Tests three genuine edge sources not previously combined:

1. OR Compression Filter (NEW — not tested anywhere)
   Hypothesis: narrow Opening Range (< max_range_atr × ATR) = coiled energy.
   Wide ORs mean the market has already moved; narrow ORs precede explosive breaks.
   Academic: Parkinson (1980), Rogers & Satchell (1991) — low intraday H-L range
   variance predicts subsequent volatility. Bollinger squeeze applied to ORB.

2. Prev-Day VWAP Filter (from vwap_sentiment_sweep results)
   Require yesterday's close > yesterday's VWAP for LONG entries.
   Result: OOS WR=75% (vs 70.4% baseline), DD=-$796 (vs -$1,516), Sharpe=4.21.
   Mechanism: funds that closed long above VWAP yesterday continue adding longs.

3. Skip Gap-Up Filter (from research_filter_sweep results)
   Skip LONG ORB when pre-market gap is >0.3% UP in the same direction.
   Result: OOS Sharpe=6.71 (vs 6.26 baseline), WR=72.7%.
   Mechanism: institutions already positioned in gap-up; retail FOMO entry reverses.

4. CDP-Anchor (best filter from v2/v3 — included for combination testing)

Config grid (2 contracts, 2/day, IS + OOS):
  0.  Baseline
  1.  CDP-Anchor
  2.  OR Compression max=1.0
  3.  OR Compression max=0.8
  4.  PrevVWAP filter
  5.  Skip gap-up >0.3%
  6.  CDP + PrevVWAP
  7.  CDP + Skip gap-up
  8.  CDP + max_range=1.0
  9.  CDP + max_range=0.8
  10. CDP + PrevVWAP + Skip gap-up
  11. CDP + PrevVWAP + max_range=1.0
  12. CDP + PrevVWAP + Skip gap-up + max_range=1.0  (kitchen sink)

Run:
    cd rule_based_v1
    python diagnostics/edge_research_diagnostic.py

Output: diagnostics/edge_research_results.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RBV1 = ROOT / "rule_based_v1"
for _p in [str(ROOT), str(RBV1)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.signal_aggregator import SignalAggregator
from rules.cumulative_delta_filter import CumulativeDeltaFilter
from rules.opening_range import OpeningRangeBreakoutRule
from utils.indicators import atr as compute_atr

# ── Constants ─────────────────────────────────────────────────────────────────
POINT_VALUE    = 2.0
TICK_SIZE      = 0.25
COMMISSION     = 0.62
SLIPPAGE_TICKS = 1
PT_MULT        = 3.0
SL_MULT        = 1.5
ATR_PERIOD     = 14
TIME_STOP_BARS = 24
MAX_DAILY_LOSS = -950.0
DRAWDOWN_BUFFER = 1950.0
STARTING_EQUITY = 50_000.0
N_CONTRACTS    = 2
MAX_TRADES_PER_DAY = 2

ORB_PARAMS = dict(
    or_end_time="10:04", min_or_bars=7, min_range_atr=0.3,
    entry_cutoff_time="12:00", atr_period=ATR_PERIOD, long_only=True,
)


@dataclass
class _Pos:
    entry_price: float
    entry_bar_idx: int
    stop_loss: float
    profit_target: float
    time_stop_bar: int


def _slip(price: float, direction: int, is_entry: bool) -> float:
    offset = SLIPPAGE_TICKS * TICK_SIZE
    return price + offset * direction if is_entry else price - offset * direction


def _calc_pnl(entry: float, exit_: float, n: int) -> float:
    return (exit_ - entry) * n * POINT_VALUE - 2 * COMMISSION * n


def _check_exit(pos: _Pos, bar: pd.Series, idx: int, sess_close: bool):
    h, l, c = bar["high"], bar["low"], bar["close"]
    if sess_close:
        return True, _slip(c, 1, False), "session_close"
    if idx >= pos.time_stop_bar:
        return True, _slip(c, 1, False), "time_stop"
    if l <= pos.stop_loss:
        return True, _slip(pos.stop_loss, 1, False), "stop_loss"
    if h >= pos.profit_target:
        return True, _slip(pos.profit_target, 1, False), "profit_target"
    return False, 0.0, ""


def _build_daily_meta(bars: pd.DataFrame) -> dict:
    """Pre-compute per-day: prev-VWAP signal, gap direction."""
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    cum_tp_vol = (typical * bars["volume"]).groupby(bars.index.date).cumsum()
    cum_vol    = bars["volume"].groupby(bars.index.date).cumsum()
    vwap_series = cum_tp_vol / cum_vol.replace(0, np.nan)

    dates = sorted(set(bars.index.date))
    session_close_above_vwap: dict = {}
    session_last_close: dict = {}

    for d in dates:
        day_mask  = bars.index.date == d
        day_bars  = bars[day_mask]
        day_vwap  = vwap_series[day_mask]
        if len(day_bars) == 0:
            continue
        last_close = float(day_bars["close"].iloc[-1])
        last_vwap  = float(day_vwap.iloc[-1]) if not day_vwap.empty else last_close
        session_close_above_vwap[d] = (last_close > last_vwap)
        session_last_close[d] = last_close

    meta: dict = {}
    for i, d in enumerate(dates):
        prev_vwap_bullish = None
        gap_pct = 0.0
        if i > 0:
            prev_d = dates[i - 1]
            prev_vwap_bullish = session_close_above_vwap.get(prev_d)
            prev_close = session_last_close.get(prev_d)
            # Today's opening price (first 5-min bar open)
            day_mask = bars.index.date == d
            day_bars = bars[day_mask]
            if prev_close and len(day_bars) > 0:
                today_open = float(day_bars["open"].iloc[0])
                gap_pct = (today_open - prev_close) / prev_close

        meta[d] = {
            "prev_vwap_bullish": prev_vwap_bullish,
            "gap_pct": gap_pct,
        }

    return meta


def run_backtest(
    bars: pd.DataFrame,
    daily_meta: dict,
    use_cdp: bool = False,
    max_range_atr: float | None = None,   # OR compression upper bound
    require_prev_vwap: bool = False,       # prev-day VWAP filter
    skip_gap_up_threshold: float | None = None,  # skip if gap > X%
    label: str = "variant",
) -> dict:
    """Run one backtest variant."""
    orb = OpeningRangeBreakoutRule(**ORB_PARAMS)

    if use_cdp:
        cd = CumulativeDeltaFilter(cdp_required=True, min_other_score=1, allow_cdp_shorts=False)
        agg = SignalAggregator(primary_rule=orb, filter_rules=[cd],
                               confirmation_rules=[], min_confirmations=0)
    else:
        agg = SignalAggregator(primary_rule=orb, filter_rules=[],
                               confirmation_rules=[], min_confirmations=0)

    atr_series = compute_atr(bars["high"], bars["low"], bars["close"], ATR_PERIOD)
    min_bars = agg.required_bars()

    pos: _Pos | None = None
    trades: list[dict] = []
    equity = STARTING_EQUITY
    peak = STARTING_EQUITY
    max_dd = 0.0
    cur_date = None
    daily_pnl: dict = {}
    trades_today = 0
    daily_loss = 0.0
    exit_counts: dict[str, int] = {}
    filtered_counts = {"prev_vwap": 0, "gap_up": 0, "compression": 0}

    for i in range(min_bars, len(bars)):
        bar = bars.iloc[i]
        ts_et = bars.index[i].tz_convert("US/Eastern")
        bdate = ts_et.date()
        bh, bm = ts_et.hour, ts_et.minute

        if cur_date is not None and bdate != cur_date:
            daily_pnl[cur_date] = daily_loss
            daily_loss = 0.0
            trades_today = 0
        cur_date = bdate

        atr_now = atr_series.iloc[i]
        if np.isnan(atr_now) or atr_now <= 0:
            continue

        is_last = (
            i + 1 >= len(bars)
            or bars.index[i + 1].tz_convert("US/Eastern").date() != bdate
        )
        sess_close = is_last or (bh == 15 and bm >= 55)

        # ── Exit ─────────────────────────────────────────────────────────────
        if pos is not None:
            exited, exit_p, reason = _check_exit(pos, bar, i, sess_close)
            if exited:
                pnl = _calc_pnl(pos.entry_price, exit_p, N_CONTRACTS)
                trades.append({"date": str(bdate), "pnl": round(pnl, 2), "reason": reason})
                equity += pnl; daily_loss += pnl
                peak = max(peak, equity)
                max_dd = min(max_dd, equity - peak)
                exit_counts[reason] = exit_counts.get(reason, 0) + 1
                pos = None

        # ── Entry ─────────────────────────────────────────────────────────────
        can_enter = (
            pos is None and not sess_close
            and trades_today < MAX_TRADES_PER_DAY
            and daily_loss > MAX_DAILY_LOSS
            and (equity - peak) > -DRAWDOWN_BUFFER
        )
        if not can_enter:
            continue

        # ── Day-level filters ─────────────────────────────────────────────────
        meta = daily_meta.get(bdate, {})

        if require_prev_vwap:
            pv = meta.get("prev_vwap_bullish")
            if pv is False:  # explicitly bearish (not None/unknown)
                filtered_counts["prev_vwap"] += 1
                continue

        gap_pct = meta.get("gap_pct", 0.0)
        if skip_gap_up_threshold is not None and gap_pct > skip_gap_up_threshold:
            filtered_counts["gap_up"] += 1
            continue

        # ── Signal evaluation ─────────────────────────────────────────────────
        lookback = bars.iloc[max(0, i - min_bars + 1): i + 1]
        dec = agg.evaluate(lookback)

        if not dec.should_trade:
            continue

        # ── OR Compression check (post signal, uses OR range from primary) ───
        if max_range_atr is not None:
            # Recompute OR range for today
            or_end_ts = pd.Timestamp(f"{bdate} 10:04:00", tz="US/Eastern")
            sess_open  = pd.Timestamp(f"{bdate} 09:30:00", tz="US/Eastern")
            or_mask = (bars.index >= sess_open) & (bars.index <= or_end_ts)
            or_bars = bars.loc[or_mask]
            if len(or_bars) >= 2:
                or_range = float(or_bars["high"].max()) - float(or_bars["low"].min())
                if or_range > max_range_atr * atr_now:
                    filtered_counts["compression"] += 1
                    continue

        # ── Enter ─────────────────────────────────────────────────────────────
        ep = _slip(bar["close"], 1, True)
        sl = ep - SL_MULT * atr_now
        pt = ep + PT_MULT * atr_now
        pos = _Pos(ep, i, sl, pt, i + TIME_STOP_BARS)
        trades_today += 1

    if cur_date and cur_date not in daily_pnl:
        daily_pnl[cur_date] = daily_loss

    wins = [t for t in trades if t["pnl"] > 0]
    total_pnl = sum(t["pnl"] for t in trades)
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    daily_series = pd.Series({k: v for k, v in daily_pnl.items() if v != 0})
    sharpe = 0.0
    if len(daily_series) > 1 and daily_series.std() > 0:
        sharpe = daily_series.mean() / daily_series.std() * np.sqrt(252)

    return {
        "label": label,
        "n": len(trades),
        "wr": round(len(wins) / max(len(trades), 1), 4),
        "pnl": round(total_pnl, 2),
        "sharpe": round(sharpe, 3),
        "dd": round(max_dd, 2),
        "pf": round(gross_win / gross_loss, 3) if gross_loss > 0 else float("inf"),
        "avg_pnl": round(total_pnl / max(len(trades), 1), 2),
        "exit_reasons": exit_counts,
        "trades_per_day": round(len(trades) / max(len(daily_pnl), 1), 3),
        "mll_ok": max_dd > -2000,
        "filtered": filtered_counts,
        "trades": trades,
    }


def monte_carlo_no_limit(
    trades: list[dict],
    n_paths: int = 10_000,
    max_days: int = 60,
    trades_per_day: float = 1.0,
    rng_seed: int = 42,
) -> dict:
    if not trades:
        return {"p_pass": 0.0, "p_pass_30d": 0.0, "p_pass_60d": 0.0,
                "p95_dd": 0.0, "median_days": max_days}
    rng = np.random.default_rng(rng_seed)
    pnls = np.array([t["pnl"] for t in trades])
    n_per_day = max(1, round(trades_per_day))
    passes = 0; passes_30d = 0; passes_60d = 0
    drawdowns: list[float] = []
    days_to_pass: list[int] = []

    for _ in range(n_paths):
        bal = STARTING_EQUITY; peak = STARTING_EQUITY
        passed_day = None; halted = False
        for day in range(1, max_days + 1):
            if halted: break
            dt = 0.0
            for p in rng.choice(pnls, size=n_per_day, replace=True):
                dt += p; bal += p; peak = max(peak, bal)
                if (bal - peak) <= -2000: halted = True; break
                if dt <= -1000: halted = True; break
            if not halted and (bal - STARTING_EQUITY) >= 3000:
                passed_day = day; break
        drawdowns.append(min(bal - peak, 0.0))
        if passed_day:
            passes += 1; days_to_pass.append(passed_day)
            if passed_day <= 30: passes_30d += 1
            if passed_day <= 60: passes_60d += 1

    return {
        "p_pass": round(passes / n_paths, 4),
        "p_pass_30d": round(passes_30d / n_paths, 4),
        "p_pass_60d": round(passes_60d / n_paths, 4),
        "p95_dd": round(float(np.percentile(drawdowns, 5)), 2),
        "median_days": int(np.median(days_to_pass)) if days_to_pass else max_days,
    }


def _load_rth(path: Path) -> pd.DataFrame:
    bars = pd.read_hdf(str(path), key="bars_5min")
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("US/Eastern")
    et = bars.index.tz_convert("US/Eastern")
    rth = ((et.hour > 9) | ((et.hour == 9) & (et.minute >= 30))) & (et.hour < 16)
    return bars[rth].copy()


def _print_header(title: str) -> None:
    print(f"\n{'='*108}")
    print(f"  {title}")
    print(f"{'='*108}")
    print(
        f"  {'Config':<42} {'N':>4} {'WR':>6} {'PnL':>9} {'Sharpe':>7} "
        f"{'DD':>9} {'T/Day':>6} {'$/Trd':>7}  MLL"
    )
    print(f"  {'-'*104}")


def _print_row(r: dict) -> None:
    mll = "✅" if r["mll_ok"] else "❌"
    print(
        f"  {r['label']:<42} {r['n']:>4} {r['wr']:>5.1%} "
        f"${r['pnl']:>8,.0f} {r['sharpe']:>7.2f} "
        f"${r['dd']:>8,.0f} {r['trades_per_day']:>6.2f} "
        f"${r['avg_pnl']:>6.0f}  {mll}"
    )


if __name__ == "__main__":
    IS_DATA  = ROOT / "data" / "processed" / "mnq_bars_5min.h5"
    OOS_DATA = ROOT / "data" / "processed" / "mnq_2026ytd_5min.h5"

    # (label, use_cdp, max_range_atr, require_prev_vwap, skip_gap_up)
    configs = [
        ("0:  Baseline",                          False, None, False, None),
        ("1:  CDP-Anchor",                         True,  None, False, None),
        ("2:  OR Compress max=1.0",                False, 1.0,  False, None),
        ("3:  OR Compress max=0.8",                False, 0.8,  False, None),
        ("4:  PrevVWAP filter",                    False, None, True,  None),
        ("5:  Skip gap-up >0.3%",                  False, None, False, 0.003),
        ("6:  CDP + PrevVWAP",                     True,  None, True,  None),
        ("7:  CDP + Skip gap-up",                  True,  None, False, 0.003),
        ("8:  CDP + Compress max=1.0",             True,  1.0,  False, None),
        ("9:  CDP + Compress max=0.8",             True,  0.8,  False, None),
        ("10: CDP + PrevVWAP + Skip gap-up",       True,  None, True,  0.003),
        ("11: CDP + PrevVWAP + Compress max=1.0",  True,  1.0,  True,  None),
        ("12: CDP+PrevVWAP+Gap+Compress max=1.0",  True,  1.0,  True,  0.003),
    ]

    all_results: dict = {}

    for dataset_label, data_path in [
        ("in_sample", IS_DATA),
        ("oos",       OOS_DATA),
    ]:
        if not data_path.exists():
            print(f"\n  [{dataset_label}] Data not found: {data_path}")
            all_results[dataset_label] = []
            continue

        print(f"\nLoading {dataset_label}...")
        bars = _load_rth(data_path)
        n_days = bars.index.normalize().nunique()
        print(f"  {len(bars):,} bars | {bars.index[0].date()} → {bars.index[-1].date()} | {n_days} days")

        daily_meta = _build_daily_meta(bars)

        _print_header(
            f"EDGE RESEARCH  |  {dataset_label.upper()}  |  2c 2/day  |  PT=3.0x SL=1.5x"
        )

        dataset_results = []
        for label, cdp, max_r, prev_v, gap_skip in configs:
            r = run_backtest(bars, daily_meta,
                             use_cdp=cdp, max_range_atr=max_r,
                             require_prev_vwap=prev_v,
                             skip_gap_up_threshold=gap_skip,
                             label=label)
            dataset_results.append(r)
            _print_row(r)

        # Filter effectiveness
        print(f"\n  Filter effectiveness (trades blocked):")
        for r in dataset_results:
            f = r["filtered"]
            total_filtered = sum(f.values())
            if total_filtered > 0:
                parts = [f"  {k}={v}" for k, v in f.items() if v > 0]
                print(f"    {r['label']:<42} {','.join(parts)}")

        # Monte Carlo for promising configs
        print(f"\n  Monte Carlo P(pass combine) — 10,000 paths, max 60 days:")
        mc_idxs = [0, 1, 4, 5, 6, 10, 12]
        for idx in mc_idxs:
            if idx >= len(dataset_results): continue
            r = dataset_results[idx]
            tpd = max(r["trades_per_day"], 0.1)
            mc = monte_carlo_no_limit(r["trades"], n_paths=10_000, trades_per_day=tpd)
            r["monte_carlo"] = mc
            print(
                f"    [{r['label']:<42}]  "
                f"P={mc['p_pass']:.1%}  "
                f"P(≤30d)={mc['p_pass_30d']:.1%}  "
                f"p95_dd=${mc['p95_dd']:,.0f}  "
                f"median={mc['median_days']}d"
            )

        all_results[dataset_label] = [
            {k: v for k, v in r.items() if k != "trades"}
            for r in dataset_results
        ]

    out_path = RBV1 / "diagnostics" / "edge_research_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results → {out_path}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n{'='*108}")
    print("  VERDICT — Which filters show genuine OOS edge?")
    print(f"{'='*108}")
    if "oos" in all_results and all_results["oos"]:
        baseline_wr = all_results["oos"][0]["wr"]
        baseline_sharpe = all_results["oos"][0]["sharpe"]
        for r in all_results["oos"]:
            beat_wr     = r["wr"]     >= baseline_wr
            beat_sharpe = r["sharpe"] >= baseline_sharpe
            mll_ok      = r["mll_ok"]
            enough_n    = r["n"] >= 8
            if beat_wr and beat_sharpe and mll_ok and enough_n and r["label"] != "0:  Baseline":
                print(f"  ✅ EDGE: {r['label']}  WR={r['wr']:.1%}  Sharpe={r['sharpe']:.2f}  DD=${r['dd']:,.0f}")
        print(f"  Baseline: WR={baseline_wr:.1%}  Sharpe={baseline_sharpe:.2f}")
