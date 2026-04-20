"""CD-ORB v3 Combine-Optimised Backtest.

Afternoon ORB was tested and REJECTED (IS WR=35.3%, OOS WR=20.0%) — the PM
session has no continuation edge. The three remaining improvements are tested here:

  1. Scale to 3 contracts with CDP safety (DD stays within Topstep -$2,000 limit)
  2. Extend time stop from 24 bars (2h) to 36 bars (3h) — time-stop winners average
     +$214 at 2h; giving them an extra hour captures further upside
  3. Adaptive Kelly sizing: start 2c, scale to 3c after +$500 cumulative PnL,
     drop back to 2c if drawdown > -$800

Sweep configurations:
  0. Baseline (live)       — 2c, 1/day, 24 bars, no CDP  [current live config]
  1. CDP-Anchor 2c 2/day   — 2c, 2/day, 24 bars, CDP     [best v2 config]
  2. CDP-Anchor 3c 2/day   — 3c, 2/day, 24 bars, CDP     [scale up]
  3. CDP-Anchor 3c ext-ts  — 3c, 2/day, 36 bars, CDP     [+ extended time stop]
  4. Adaptive Kelly        — 2→3c, 2/day, 36 bars, CDP   [Kelly-adaptive sizing]

Monte Carlo model:
  Realistic Topstep 50k: run until PnL ≥ +$3,000, OR DD ≤ -$2,000, OR 60 days.
  Reports P(pass), P(pass in ≤30d), P(pass in ≤60d), p95 drawdown, median days.
  10,000 paths.

Run:
    cd rule_based_v1
    python diagnostics/cd_orb_v3_backtest.py

Output: diagnostics/cd_orb_v3_results.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
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

# ── Execution constants ───────────────────────────────────────────────────────
POINT_VALUE    = 2.0
TICK_SIZE      = 0.25
COMMISSION     = 0.62
SLIPPAGE_TICKS = 1
PT_MULT        = 3.0
SL_MULT        = 1.5
ATR_PERIOD     = 14
MAX_DAILY_LOSS = -950.0
DRAWDOWN_BUFFER = 1950.0
STARTING_EQUITY = 50_000.0

# Adaptive Kelly thresholds
KELLY_SCALE_UP_PNL   = 500.0   # scale 2c→3c after +$500 cumulative PnL
KELLY_SCALE_DOWN_DD  = -800.0  # drop 3c→2c if drawdown > -$800

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
    n_contracts: int


@dataclass
class _VetoLog:
    cdp_required: list = field(default_factory=list)
    other_score_low: list = field(default_factory=list)
    shorts_disabled: list = field(default_factory=list)
    min_score: list = field(default_factory=list)

    def record(self, reason: str) -> None:
        getattr(self, reason.replace("-", "_"), self.min_score).append(1)


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


def run_backtest(
    bars: pd.DataFrame,
    n_contracts: int = 2,
    max_trades_per_day: int = 1,
    time_stop_bars: int = 24,
    use_cdp: bool = False,
    adaptive_kelly: bool = False,
    label: str = "variant",
) -> dict:
    """Run one backtest variant and return full metrics.

    adaptive_kelly : bool
        If True, start at 2 contracts; scale to 3 after +$500 cumulative PnL;
        drop back to 2 if drawdown exceeds -$800.  n_contracts is ignored.
    """
    orb = OpeningRangeBreakoutRule(**ORB_PARAMS)

    if use_cdp:
        cd = CumulativeDeltaFilter(cdp_required=True, min_other_score=1, allow_cdp_shorts=False)
        agg = SignalAggregator(
            primary_rule=orb, filter_rules=[cd],
            confirmation_rules=[], min_confirmations=0,
        )
    else:
        agg = SignalAggregator(
            primary_rule=orb, filter_rules=[],
            confirmation_rules=[], min_confirmations=0,
        )

    atr_series = compute_atr(bars["high"], bars["low"], bars["close"], ATR_PERIOD)
    min_bars = agg.required_bars()

    pos: _Pos | None = None
    trades: list[dict] = []
    veto_log = _VetoLog()
    equity = STARTING_EQUITY
    peak = STARTING_EQUITY
    max_dd = 0.0
    cur_date = None
    daily_pnl: dict = {}
    trades_today = 0
    daily_loss = 0.0
    exit_counts: dict[str, int] = {}

    # Adaptive Kelly state
    cur_contracts = 2 if adaptive_kelly else n_contracts

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

        # ── Adaptive Kelly: update contract size ──────────────────────────────
        if adaptive_kelly:
            cumul_pnl = equity - STARTING_EQUITY
            dd_from_peak = equity - peak
            if dd_from_peak <= KELLY_SCALE_DOWN_DD:
                cur_contracts = 2
            elif cumul_pnl >= KELLY_SCALE_UP_PNL:
                cur_contracts = 3
            else:
                cur_contracts = 2

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
                    "n_contracts": pos.n_contracts,
                })
                equity += pnl
                daily_loss += pnl
                peak = max(peak, equity)
                max_dd = min(max_dd, equity - peak)
                exit_counts[reason] = exit_counts.get(reason, 0) + 1
                pos = None

        # ── Entry check ───────────────────────────────────────────────────────
        can_enter = (
            pos is None
            and not sess_close
            and trades_today < max_trades_per_day
            and daily_loss > MAX_DAILY_LOSS
            and (equity - peak) > -DRAWDOWN_BUFFER
        )

        if can_enter:
            lookback = bars.iloc[max(0, i - min_bars + 1): i + 1]
            dec = agg.evaluate(lookback)

            if not dec.should_trade and dec.filter_signals:
                for fsig in dec.filter_signals:
                    vr = fsig.metadata.get("veto_reason", "min_score")
                    veto_log.record(vr)

            if dec.should_trade:
                nc = cur_contracts if adaptive_kelly else n_contracts
                ep = _slip(bar["close"], 1, True)
                sl = ep - SL_MULT * atr_now
                pt = ep + PT_MULT * atr_now
                pos = _Pos(ep, i, sl, pt, i + time_stop_bars, nc)
                trades_today += 1

    if cur_date and cur_date not in daily_pnl:
        daily_pnl[cur_date] = daily_loss

    # ── Metrics ───────────────────────────────────────────────────────────────
    wins = [t for t in trades if t["pnl"] > 0]
    total_pnl = sum(t["pnl"] for t in trades)
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))

    daily_series = pd.Series({k: v for k, v in daily_pnl.items() if v != 0})
    sharpe = 0.0
    if len(daily_series) > 1 and daily_series.std() > 0:
        sharpe = daily_series.mean() / daily_series.std() * np.sqrt(252)

    avg_pnl = total_pnl / max(len(trades), 1)

    # Exit reason detail
    exit_detail: dict = {}
    for reason in exit_counts:
        reason_pnls = [t["pnl"] for t in trades if t["reason"] == reason]
        exit_detail[reason] = {
            "n": len(reason_pnls),
            "wr": round(sum(1 for p in reason_pnls if p > 0) / max(len(reason_pnls), 1), 4),
            "avg_pnl": round(sum(reason_pnls) / max(len(reason_pnls), 1), 2),
        }

    return {
        "label": label,
        "n": len(trades),
        "wr": round(len(wins) / max(len(trades), 1), 4),
        "pnl": round(total_pnl, 2),
        "sharpe": round(sharpe, 3),
        "dd": round(max_dd, 2),
        "pf": round(gross_win / gross_loss, 3) if gross_loss > 0 else float("inf"),
        "avg_pnl": round(avg_pnl, 2),
        "exit_reasons": exit_counts,
        "exit_detail": exit_detail,
        "trades_per_day": round(len(trades) / max(len(daily_pnl), 1), 3),
        "mll_ok": max_dd > -2000,
        "veto_breakdown": {
            "cdp_required": len(veto_log.cdp_required),
            "other_score_low": len(veto_log.other_score_low),
        },
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
    """Realistic Topstep combine sim: no fixed time limit."""
    if not trades:
        return {
            "p_pass": 0.0, "p_pass_30d": 0.0, "p_pass_60d": 0.0,
            "p95_dd": 0.0, "median_days": max_days, "n_paths": n_paths,
        }

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
            for trade_pnl in rng.choice(pnls, size=n_per_day, replace=True):
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
    print(f"\n{'='*104}")
    print(f"  {title}")
    print(f"{'='*104}")
    print(
        f"  {'Config':<38} {'N':>5} {'WR':>6} {'PnL':>9} {'Sharpe':>7} "
        f"{'DD':>9} {'T/Day':>6} {'$/Trd':>7}  MLL"
    )
    print(f"  {'-'*100}")


def _print_row(r: dict) -> None:
    mll = "✅" if r["mll_ok"] else "❌"
    print(
        f"  {r['label']:<38} {r['n']:>5} {r['wr']:>5.1%} "
        f"${r['pnl']:>8,.0f} {r['sharpe']:>7.2f} "
        f"${r['dd']:>8,.0f} {r['trades_per_day']:>6.2f} "
        f"${r['avg_pnl']:>6.0f}  {mll}"
    )


if __name__ == "__main__":
    IS_DATA  = ROOT / "data" / "processed" / "mnq_bars_5min.h5"
    OOS_DATA = ROOT / "data" / "processed" / "mnq_2026ytd_5min.h5"

    # (label, n_contracts, max_trades_per_day, time_stop_bars, use_cdp, adaptive_kelly)
    configs = [
        ("0: Baseline 2c 1/day",         2, 1, 24, False, False),
        ("1: CDP-Anchor 2c 2/day",        2, 2, 24, True,  False),
        ("2: CDP-Anchor 3c 2/day",        3, 2, 24, True,  False),
        ("3: CDP-Anchor 3c 36-bar TS",    3, 2, 36, True,  False),
        ("4: Adaptive Kelly 2→3c 36-bar", 2, 2, 36, True,  True),
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
            f"CD-ORB v3  |  {dataset_label.upper()}  |  PT=3.0x SL=1.5x  |  MNQ"
        )

        dataset_results = []
        for label, nc, mtpd, ts_bars, cdp, kelly in configs:
            r = run_backtest(
                bars,
                n_contracts=nc,
                max_trades_per_day=mtpd,
                time_stop_bars=ts_bars,
                use_cdp=cdp,
                adaptive_kelly=kelly,
                label=label,
            )
            dataset_results.append(r)
            _print_row(r)

        # Exit reason detail for best CDP config (Config 3)
        best = dataset_results[3]  # 3c 36-bar
        if best["n"] > 0:
            print(f"\n  Exit reasons — '{best['label']}':")
            for reason, detail in sorted(best["exit_detail"].items(), key=lambda x: -x[1]["n"]):
                print(
                    f"    {reason:<20} {detail['n']:>4}  "
                    f"WR={detail['wr']:.0%}  avg=${detail['avg_pnl']:>7.1f}"
                )

        # Time-stop comparison: 24 vs 36 bars
        ts24 = dataset_results[2]  # 3c 24-bar
        ts36 = dataset_results[3]  # 3c 36-bar
        if ts24["n"] > 0 and ts36["n"] > 0:
            def _ts_stats(r):
                ts = r["exit_detail"].get("time_stop", {})
                return ts.get("n", 0), ts.get("wr", 0), ts.get("avg_pnl", 0)
            n24, wr24, avg24 = _ts_stats(ts24)
            n36, wr36, avg36 = _ts_stats(ts36)
            print(f"\n  Time-stop comparison (3c):")
            print(f"    24-bar TS: {n24} trades  WR={wr24:.0%}  avg=${avg24:.1f}")
            print(f"    36-bar TS: {n36} trades  WR={wr36:.0%}  avg=${avg36:.1f}")

        # Monte Carlo for all configs
        print(f"\n  Monte Carlo P(pass Topstep 50k combine) — 10,000 paths, max 60 days:")
        for r in dataset_results:
            tpd = max(r["trades_per_day"], 0.1)
            mc = monte_carlo_no_limit(r["trades"], n_paths=10_000, trades_per_day=tpd)
            r["monte_carlo"] = mc
            print(
                f"    [{r['label']:<38}]  "
                f"P(pass)={mc['p_pass']:.1%}  "
                f"P(≤30d)={mc['p_pass_30d']:.1%}  "
                f"P(≤60d)={mc['p_pass_60d']:.1%}  "
                f"p95_dd=${mc['p95_dd']:,.0f}  "
                f"median={mc['median_days']}d"
            )

        # Two-attempt probability for best config
        best_mc = dataset_results[3]["monte_carlo"]
        p1 = best_mc["p_pass"]
        p_two_attempts = 1 - (1 - p1) ** 2
        print(f"\n  Best config P(pass / 2 attempts): {p_two_attempts:.1%}")

        all_results[dataset_label] = [
            {k: v for k, v in r.items() if k != "trades"}
            for r in dataset_results
        ]

    # Save results
    out_path = RBV1 / "diagnostics" / "cd_orb_v3_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved → {out_path}")

    # ── Overall recommendation ────────────────────────────────────────────────
    print(f"\n{'='*104}")
    print("  RECOMMENDED CONFIG FOR COMBINE")
    print(f"{'='*104}")
    if "oos" in all_results and all_results["oos"]:
        configs_oos = all_results["oos"]
        best = max(configs_oos, key=lambda r: r.get("monte_carlo", {}).get("p_pass", 0))
        mc = best.get("monte_carlo", {})
        print(f"  Config: {best['label']}")
        print(f"  OOS stats: N={best['n']} WR={best['wr']:.1%} PnL=${best['pnl']:,.0f} "
              f"Sharpe={best['sharpe']:.2f} DD=${best['dd']:,.0f}")
        print(f"  Combine:   P(pass)={mc.get('p_pass', 0):.1%}  "
              f"P(≤30d)={mc.get('p_pass_30d', 0):.1%}  "
              f"median={mc.get('median_days', '?')}d")
        p1 = mc.get("p_pass", 0)
        print(f"  2-attempt: P={1-(1-p1)**2:.1%}")
