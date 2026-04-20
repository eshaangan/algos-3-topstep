"""Afternoon ORB Diagnostic — validate PM session edge with CDP filter.

Tests whether an independent post-lunch ORB (13:00–13:30 range, entry 13:35–14:30)
has statistical edge on MNQ, both standalone and combined with the morning session.

Key question: Does afternoon ORB (Heston U-shape reversal microstructure) add
independently-decorrelated alpha to morning ORB, or is it noise?

Note: The existing OpeningRangeBreakoutRule anchors to 09:30. For a genuine
afternoon OR (13:00–13:30), we compute range and CDP inline here without the
class, so the AM and PM sessions stay completely independent.

Configurations tested:
  0. Morning-only    2c CDP-Anchor (matches cd_orb_v2 Config 2 as reference)
  1. Afternoon-only  2c CDP-Anchor (13:00–13:30 OR, entry 13:35–14:30)
  2. AM + PM dual    2c, up to 2 trades/day (1 per session)
  3. AM + PM dual    3c, up to 2 trades/day (1 per session)

Data:
  IS:  data/processed/mnq_bars_5min.h5      (Aug 2025 – Jan 2026)
  OOS: data/processed/mnq_2026ytd_5min.h5  (Jan–Mar 2026)

Run:
    cd rule_based_v1
    python diagnostics/afternoon_orb_diagnostic.py

Output: diagnostics/afternoon_orb_results.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

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

# AM session config
AM_OR_END      = "10:04"
AM_MIN_OR_BARS = 7
AM_ENTRY_CUTOFF = "12:00"

# PM session config
PM_OR_START    = "13:00"
PM_OR_END      = "13:30"
PM_ENTRY_CUTOFF = "14:30"
PM_MIN_OR_BARS = 3
PM_MIN_CDP     = 0.15          # same threshold as AM


@dataclass
class _Pos:
    entry_price: float
    entry_bar_idx: int
    stop_loss: float
    profit_target: float
    time_stop_bar: int
    n_contracts: int
    session: str   # "am" | "pm"


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


def _cdp_ratio(or_bars: pd.DataFrame) -> float:
    """Compute normalized CDP ratio for a set of OR bars."""
    if len(or_bars) == 0:
        return 0.0
    eps = 1e-10
    bar_range = (or_bars["high"] - or_bars["low"]).clip(lower=eps)
    cdp_per_bar = or_bars["volume"] * (
        2.0 * (or_bars["close"] - or_bars["low"]) / bar_range - 1.0
    )
    total_vol = float(or_bars["volume"].sum())
    if total_vol <= 0:
        return 0.0
    return float(cdp_per_bar.sum()) / total_vol


def _pm_signal(
    bars: pd.DataFrame,
    current_idx: int,
    atr_now: float,
    use_cdp: bool = True,
) -> tuple[bool, float]:
    """Evaluate PM ORB signal for current bar.

    Returns (should_enter, entry_price). Long-only.
    """
    bar = bars.iloc[current_idx]
    ts_et = bars.index[current_idx].tz_convert("US/Eastern")
    bdate = ts_et.date()
    bh, bm = ts_et.hour, ts_et.minute
    current_mins = bh * 60 + bm

    # Only trade in PM entry window: after PM_OR_END and before PM_ENTRY_CUTOFF
    pm_end_h, pm_end_m = map(int, PM_OR_END.split(":"))
    pm_cut_h, pm_cut_m = map(int, PM_ENTRY_CUTOFF.split(":"))
    pm_end_mins = pm_end_h * 60 + pm_end_m
    pm_cut_mins = pm_cut_h * 60 + pm_cut_m

    if current_mins <= pm_end_mins or current_mins > pm_cut_mins:
        return False, 0.0

    # Slice PM OR bars: 13:00–13:30
    pm_or_start_ts = pd.Timestamp(f"{bdate} {PM_OR_START}:00", tz="US/Eastern")
    pm_or_end_ts   = pd.Timestamp(f"{bdate} {PM_OR_END}:00",   tz="US/Eastern")
    pm_or_mask = (bars.index >= pm_or_start_ts) & (bars.index <= pm_or_end_ts)
    pm_or_bars = bars.loc[pm_or_mask]

    if len(pm_or_bars) < PM_MIN_OR_BARS:
        return False, 0.0

    range_high = float(pm_or_bars["high"].max())
    range_low  = float(pm_or_bars["low"].min())
    range_width = range_high - range_low

    # Range width filter (same as AM: 0.3 × ATR)
    if range_width < 0.3 * atr_now:
        return False, 0.0

    current_close = float(bar["close"])
    prev_close = float(bars["close"].iloc[current_idx - 1]) if current_idx > 0 else current_close

    # LONG breakout: close breaks above range_high for first time
    is_long_break = current_close > range_high and prev_close <= range_high
    if not is_long_break:
        return False, 0.0

    # CDP gate for PM session
    if use_cdp:
        cdp = _cdp_ratio(pm_or_bars)
        if cdp <= PM_MIN_CDP:
            return False, 0.0

    ep = _slip(current_close, 1, True)
    return True, ep


def run_backtest(
    bars: pd.DataFrame,
    n_contracts: int = 2,
    use_am: bool = True,
    use_pm: bool = False,
    use_cdp: bool = True,
    label: str = "variant",
) -> dict:
    """Run one backtest variant. Long-only for both sessions."""

    # AM aggregator (mirrors cd_orb_v2 Config 2)
    agg_am = None
    if use_am:
        orb_am = OpeningRangeBreakoutRule(
            or_end_time=AM_OR_END, min_or_bars=AM_MIN_OR_BARS,
            min_range_atr=0.3, entry_cutoff_time=AM_ENTRY_CUTOFF,
            atr_period=ATR_PERIOD, long_only=True,
        )
        if use_cdp:
            cd_am = CumulativeDeltaFilter(
                or_end_time=AM_OR_END, cdp_required=True, min_other_score=1,
                allow_cdp_shorts=False,
            )
            agg_am = SignalAggregator(
                primary_rule=orb_am, filter_rules=[cd_am],
                confirmation_rules=[], min_confirmations=0,
            )
        else:
            agg_am = SignalAggregator(
                primary_rule=orb_am, filter_rules=[],
                confirmation_rules=[], min_confirmations=0,
            )

    atr_series = compute_atr(bars["high"], bars["low"], bars["close"], ATR_PERIOD)
    min_bars = 90  # always use 90-bar lookback

    pos: _Pos | None = None
    trades: list[dict] = []
    equity = STARTING_EQUITY
    peak = STARTING_EQUITY
    max_dd = 0.0
    cur_date = None
    daily_pnl: dict = {}
    am_traded_today = False
    pm_traded_today = False
    daily_loss = 0.0
    exit_counts: dict[str, int] = {}
    session_trades: dict[str, int] = {"am": 0, "pm": 0}

    for i in range(min_bars, len(bars)):
        bar = bars.iloc[i]
        ts_et = bars.index[i].tz_convert("US/Eastern")
        bdate = ts_et.date()
        bh, bm = ts_et.hour, ts_et.minute

        if cur_date is not None and bdate != cur_date:
            daily_pnl[cur_date] = daily_loss
            daily_loss = 0.0
            am_traded_today = False
            pm_traded_today = False
        cur_date = bdate

        atr_now = atr_series.iloc[i]
        if np.isnan(atr_now) or atr_now <= 0:
            continue

        is_last = (
            i + 1 >= len(bars)
            or bars.index[i + 1].tz_convert("US/Eastern").date() != bdate
        )
        sess_close = is_last or (bh == 15 and bm >= 55)

        # ── Exit check ────────────────────────────────────────────────────────
        if pos is not None:
            exited, exit_p, reason = _check_exit(pos, bar, i, sess_close)
            if exited:
                pnl = _calc_pnl(pos.entry_price, exit_p, pos.n_contracts)
                trades.append({
                    "date": str(bdate),
                    "entry": pos.entry_price,
                    "exit": exit_p,
                    "pnl": round(pnl, 2),
                    "reason": reason,
                    "session": pos.session,
                })
                session_trades[pos.session] = session_trades.get(pos.session, 0) + 1
                equity += pnl
                daily_loss += pnl
                peak = max(peak, equity)
                max_dd = min(max_dd, equity - peak)
                exit_counts[reason] = exit_counts.get(reason, 0) + 1
                pos = None

        # ── Entry check ───────────────────────────────────────────────────────
        can_enter_base = (
            pos is None
            and not sess_close
            and daily_loss > MAX_DAILY_LOSS
            and (equity - peak) > -DRAWDOWN_BUFFER
        )

        if not can_enter_base:
            continue

        current_mins = bh * 60 + bm
        am_cut_mins = 12 * 60  # 12:00

        # ── Try AM entry ──────────────────────────────────────────────────────
        if use_am and not am_traded_today and current_mins <= am_cut_mins and agg_am:
            lookback = bars.iloc[max(0, i - min_bars + 1): i + 1]
            dec = agg_am.evaluate(lookback)
            if dec.should_trade:
                ep = _slip(bar["close"], 1, True)
                sl = ep - SL_MULT * atr_now
                pt = ep + PT_MULT * atr_now
                pos = _Pos(ep, i, sl, pt, i + TIME_STOP_BARS, n_contracts, "am")
                am_traded_today = True
                continue

        # ── Try PM entry ──────────────────────────────────────────────────────
        if use_pm and not pm_traded_today:
            entered, ep = _pm_signal(bars, i, atr_now, use_cdp=use_cdp)
            if entered:
                sl = ep - SL_MULT * atr_now
                pt = ep + PT_MULT * atr_now
                pos = _Pos(ep, i, sl, pt, i + TIME_STOP_BARS, n_contracts, "pm")
                pm_traded_today = True

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

    am_trades = [t for t in trades if t["session"] == "am"]
    pm_trades = [t for t in trades if t["session"] == "pm"]
    am_wr = sum(1 for t in am_trades if t["pnl"] > 0) / max(len(am_trades), 1)
    pm_wr = sum(1 for t in pm_trades if t["pnl"] > 0) / max(len(pm_trades), 1)

    return {
        "label": label,
        "n": len(trades),
        "wr": round(len(wins) / max(len(trades), 1), 4),
        "pnl": round(total_pnl, 2),
        "sharpe": round(sharpe, 3),
        "dd": round(max_dd, 2),
        "pf": round(gross_win / gross_loss, 3) if gross_loss > 0 else float("inf"),
        "exit_reasons": exit_counts,
        "trades_per_day": round(len(trades) / max(len(daily_pnl), 1), 3),
        "mll_ok": max_dd > -2000,
        "am_n": len(am_trades),
        "am_wr": round(am_wr, 4),
        "pm_n": len(pm_trades),
        "pm_wr": round(pm_wr, 4),
        "trades": trades,
    }


def monte_carlo_no_limit(
    trades: list[dict],
    n_paths: int = 10_000,
    max_days: int = 60,
    starting_balance: float = 50_000.0,
    daily_loss_limit: float = -1000.0,
    trailing_dd_limit: float = -2000.0,
    profit_target: float = 3000.0,
    trades_per_day: float = 1.0,
    rng_seed: int = 42,
) -> dict:
    """Realistic combine sim: run until PT hit, DD hit, or max_days."""
    if not trades:
        return {"p_pass": 0.0, "p95_dd": 0.0, "median_days": max_days,
                "p_pass_30d": 0.0, "p_pass_60d": 0.0, "n_paths": n_paths}

    rng = np.random.default_rng(rng_seed)
    pnls = np.array([t["pnl"] for t in trades])
    n_per_day = max(1, round(trades_per_day))
    passes = 0
    passes_30d = 0
    passes_60d = 0
    drawdowns: list[float] = []
    days_to_pass: list[int] = []

    for _ in range(n_paths):
        balance = starting_balance
        peak = starting_balance
        passed_day: int | None = None
        halted = False

        for day in range(1, max_days + 1):
            if halted:
                break
            daily_total = 0.0
            day_pnls = rng.choice(pnls, size=n_per_day, replace=True)
            for trade_pnl in day_pnls:
                daily_total += trade_pnl
                balance += trade_pnl
                peak = max(peak, balance)
                if (balance - peak) <= trailing_dd_limit:
                    halted = True
                    break
                if daily_total <= daily_loss_limit:
                    halted = True
                    break
            if not halted and (balance - starting_balance) >= profit_target:
                passed_day = day
                break

        drawdowns.append(min(balance - peak, 0.0))
        if passed_day is not None:
            passes += 1
            days_to_pass.append(passed_day)
            if passed_day <= 30:
                passes_30d += 1
            if passed_day <= 60:
                passes_60d += 1

    return {
        "p_pass": round(passes / n_paths, 4),
        "p_pass_30d": round(passes_30d / n_paths, 4),
        "p_pass_60d": round(passes_60d / n_paths, 4),
        "p95_dd": round(float(np.percentile(drawdowns, 5)), 2),
        "median_days": int(np.median(days_to_pass)) if days_to_pass else max_days,
        "n_paths": n_paths,
    }


def _load_rth(path: Path, key: str = "bars_5min") -> pd.DataFrame:
    bars = pd.read_hdf(str(path), key=key)
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("US/Eastern")
    et = bars.index.tz_convert("US/Eastern")
    rth = ((et.hour > 9) | ((et.hour == 9) & (et.minute >= 30))) & (et.hour < 16)
    return bars[rth].copy()


def _print_header(title: str) -> None:
    print(f"\n{'='*96}")
    print(f"  {title}")
    print(f"{'='*96}")
    print(
        f"  {'Config':<38} {'N':>5} {'WR':>6} {'PnL':>9} {'Sharpe':>7} "
        f"{'DD':>9} {'T/Day':>6} {'AM':>5}/{('PM'):>4}  MLL"
    )
    print(f"  {'-'*92}")


def _print_row(r: dict) -> None:
    mll = "✅" if r["mll_ok"] else "❌"
    print(
        f"  {r['label']:<38} {r['n']:>5} {r['wr']:>5.1%} "
        f"${r['pnl']:>8,.0f} {r['sharpe']:>7.2f} "
        f"${r['dd']:>8,.0f} {r['trades_per_day']:>6.2f} "
        f"{r['am_n']:>5}/{r['pm_n']:>4}  {mll}"
    )


if __name__ == "__main__":
    IS_DATA  = ROOT / "data" / "processed" / "mnq_bars_5min.h5"
    OOS_DATA = ROOT / "data" / "processed" / "mnq_2026ytd_5min.h5"

    # (label, n_contracts, use_am, use_pm, use_cdp)
    configs = [
        ("0: Morning-only 2c (baseline)",       2, True,  False, True),
        ("1: Afternoon-only 2c CDP",             2, False, True,  True),
        ("2: Afternoon-only 2c no-CDP",          2, False, True,  False),
        ("3: AM + PM dual 2c CDP",               2, True,  True,  True),
        ("4: AM + PM dual 3c CDP",               3, True,  True,  True),
    ]

    all_results: dict = {}

    for dataset_label, data_path in [
        ("in_sample",  IS_DATA),
        ("oos",        OOS_DATA),
    ]:
        if not data_path.exists():
            print(f"\n  [{dataset_label}] Data not found: {data_path}")
            all_results[dataset_label] = []
            continue

        print(f"\nLoading {dataset_label} ({data_path.name})...")
        bars = _load_rth(data_path)
        n_days = bars.index.normalize().nunique()
        print(f"  {len(bars):,} bars | {bars.index[0].date()} → {bars.index[-1].date()} | {n_days} trading days")

        _print_header(
            f"AFTERNOON ORB  |  {dataset_label.upper()}  |  PT=3.0x SL=1.5x"
        )

        dataset_results = []
        for label, nc, use_am, use_pm, use_cdp in configs:
            r = run_backtest(bars, n_contracts=nc, use_am=use_am, use_pm=use_pm,
                             use_cdp=use_cdp, label=label)
            dataset_results.append(r)
            _print_row(r)

        # Session breakdown for dual config
        dual = dataset_results[3]  # Config 3: AM + PM 2c
        if dual["am_n"] + dual["pm_n"] > 0:
            print(f"\n  Session breakdown — '{dual['label']}':")
            print(f"    AM: {dual['am_n']} trades  WR={dual['am_wr']:.1%}")
            print(f"    PM: {dual['pm_n']} trades  WR={dual['pm_wr']:.1%}")

        # PM-only exit reasons
        pm_only = dataset_results[1]
        if pm_only["n"] > 0:
            print(f"\n  Exit reasons — '{pm_only['label']}':")
            for reason, cnt in sorted(pm_only["exit_reasons"].items(), key=lambda x: -x[1]):
                pnls = [t["pnl"] for t in pm_only["trades"] if t["reason"] == reason]
                avg = sum(pnls) / len(pnls) if pnls else 0
                wr = sum(1 for p in pnls if p > 0) / max(len(pnls), 1)
                print(f"    {reason:<20} {cnt:>4}  WR={wr:.0%}  avg=${avg:>7.1f}")

        # Monte Carlo for main configs
        print(f"\n  Monte Carlo P(pass Topstep 50k combine) — 10,000 paths, max 60 days:")
        mc_configs = [0, 1, 3, 4]
        for idx in mc_configs:
            r = dataset_results[idx]
            mc = monte_carlo_no_limit(
                r["trades"],
                n_paths=10_000,
                trades_per_day=max(r["trades_per_day"], 0.1),
            )
            r["monte_carlo"] = mc
            print(
                f"    [{r['label']:<38}]  "
                f"P(pass)={mc['p_pass']:.1%}  "
                f"P(/30d)={mc['p_pass_30d']:.1%}  "
                f"P(/60d)={mc['p_pass_60d']:.1%}  "
                f"p95_dd=${mc['p95_dd']:,.0f}  "
                f"median={mc['median_days']}d"
            )

        all_results[dataset_label] = [
            {k: v for k, v in r.items() if k != "trades"}
            for r in dataset_results
        ]

    # Save results
    out_path = RBV1 / "diagnostics" / "afternoon_orb_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved → {out_path}")

    # Summary verdict
    print(f"\n{'='*96}")
    print("  VERDICT")
    print(f"{'='*96}")
    if "in_sample" in all_results and all_results["in_sample"]:
        pm_is = all_results["in_sample"][1]   # afternoon-only
        dual_is = all_results["in_sample"][3]  # dual session 2c
        wr_threshold = 0.50
        pm_ok = pm_is["wr"] >= wr_threshold and pm_is["n"] >= 5
        print(f"  PM WR (IS): {pm_is['wr']:.1%}  (need ≥ 50% with ≥5 trades)  → {'✅ PROCEED to v3' if pm_ok else '❌ WEAK — proceed with caution'}")
        print(f"  Dual 2c (IS): Sharpe={dual_is['sharpe']:.2f}  DD=${dual_is['dd']:,.0f}  T/Day={dual_is['trades_per_day']:.2f}")
