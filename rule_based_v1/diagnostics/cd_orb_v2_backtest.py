"""CD-ORB v2 Diagnostic Backtest — Three Targeted Improvements.

Tests the following improvements to the CD-ORB strategy discovered in v1:

  INSIGHT from v1: Q=2 is anti-signal (WR=42.9%, Sharpe=-1.16) because
  high-volume breakouts AGAINST CDP order flow are "institutional traps".
  Removing these via CDP-anchored gating is the core fix.

Improvement 1 — CDP-Anchored Filter:
  CDP direction must pass as a hard gate; then need ≥1 of {Bias, Vol, Momentum}.
  Eliminates the Q=2 trap pattern; keeps Q=1 profitable trades where CDP leads.

Improvement 2 — CDP-Confirmed SHORTs:
  Enable SHORT entries when cdp_ratio < -0.30 (stricter threshold than LONG 0.15).
  Adds frequency without degrading quality; order-flow-theory-backed.

Improvement 3 — 2 Trades Per Day:
  Allow a second entry after first trade exits; same quality gate applies.
  Live runner already supports max_trades_per_day (live/runner.py:71).

Sweep Configurations:
  0. Baseline          — No filter, LONG-only, 1 trade/day
  1. CDP-Anchor        — cdp_required=True, LONG-only, 1 trade/day
  2. CDP-Anchor+2/day  — cdp_required=True, LONG-only, 2 trades/day
  3. CDP-Anchor+Short  — cdp_required=True, SHORTs enabled, 1 trade/day
  4. CDP-Anchor+Short+2— cdp_required=True, SHORTs enabled, 2 trades/day
  5. Q≥3 reference     — original Q≥3 (legacy cdp_required=False), 1 trade/day

Run:
    cd rule_based_v1
    python diagnostics/cd_orb_v2_backtest.py

Output: diagnostics/cd_orb_v2_results.json
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

# ── Execution config (2 contracts, mirrors live) ─────────────────────────────
N_CONTRACTS    = 2
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

ORB_PARAMS = dict(
    or_end_time="10:04", min_or_bars=7, min_range_atr=0.3,
    entry_cutoff_time="12:00", atr_period=ATR_PERIOD,
)


@dataclass
class _Pos:
    entry_price: float
    entry_bar_idx: int
    stop_loss: float
    profit_target: float
    time_stop_bar: int
    n_contracts: int
    direction: int


@dataclass
class _VetoLog:
    cdp_required: list = field(default_factory=list)
    other_score_low: list = field(default_factory=list)
    shorts_disabled: list = field(default_factory=list)
    min_score: list = field(default_factory=list)

    def record(self, reason: str, hypothetical_pnl: float | None = None):
        entry = hypothetical_pnl if hypothetical_pnl is not None else 0.0
        getattr(self, reason.replace("-", "_"), self.min_score).append(entry)


def _slip(price: float, direction: int, is_entry: bool) -> float:
    offset = SLIPPAGE_TICKS * TICK_SIZE
    return price + offset * direction if is_entry else price - offset * direction


def _calc_pnl(entry: float, exit_: float, n: int, direction: int) -> float:
    return (exit_ - entry) * direction * n * POINT_VALUE - 2 * COMMISSION * n


def _check_exit(pos: _Pos, bar: pd.Series, idx: int, sess_close: bool):
    h, l, c = bar["high"], bar["low"], bar["close"]
    if sess_close:
        ep = _slip(c, pos.direction, False)
        return True, ep, "session_close"
    if idx >= pos.time_stop_bar:
        ep = _slip(c, pos.direction, False)
        return True, ep, "time_stop"
    if pos.direction == 1:
        if l <= pos.stop_loss:
            return True, _slip(pos.stop_loss, 1, False), "stop_loss"
        if h >= pos.profit_target:
            return True, _slip(pos.profit_target, 1, False), "profit_target"
    else:  # SHORT
        if h >= pos.stop_loss:
            return True, _slip(pos.stop_loss, -1, False), "stop_loss"
        if l <= pos.profit_target:
            return True, _slip(pos.profit_target, -1, False), "profit_target"
    return False, 0.0, ""


def run_backtest(
    bars: pd.DataFrame,
    long_only: bool = True,
    cdp_required: bool = False,
    min_other_score: int = 1,
    allow_cdp_shorts: bool = False,
    min_short_cdp_ratio: float = 0.30,
    use_legacy_q3: bool = False,
    max_trades_per_day: int = 1,
    label: str = "variant",
) -> dict:
    """Run one backtest variant and return metrics + veto breakdown."""
    orb = OpeningRangeBreakoutRule(**ORB_PARAMS, long_only=long_only)

    if use_legacy_q3:
        cd = CumulativeDeltaFilter(
            cdp_required=False, min_quality_score=3,
            allow_cdp_shorts=False,
        )
    else:
        cd = CumulativeDeltaFilter(
            cdp_required=cdp_required,
            min_other_score=min_other_score,
            allow_cdp_shorts=allow_cdp_shorts,
            min_short_cdp_ratio=min_short_cdp_ratio,
        )

    if cdp_required or use_legacy_q3:
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

        # ── Exit check ────────────────────────────────────────────────────────
        if pos is not None:
            exited, exit_p, reason = _check_exit(pos, bar, i, sess_close)
            if exited:
                pnl = _calc_pnl(pos.entry_price, exit_p, pos.n_contracts, pos.direction)
                trades.append({
                    "date": str(bdate),
                    "entry": pos.entry_price,
                    "exit": exit_p,
                    "pnl": round(pnl, 2),
                    "reason": reason,
                    "direction": pos.direction,
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
            lookback = bars.iloc[max(0, i - min_bars + 1) : i + 1]
            dec = agg.evaluate(lookback)

            # Track vetoed trades for component breakdown
            if not dec.should_trade and dec.filter_signals:
                for fsig in dec.filter_signals:
                    vr = fsig.metadata.get("veto_reason", "min_score")
                    veto_log.record(vr)

            if dec.should_trade:
                d = dec.direction
                ep = _slip(bar["close"], d, True)
                sl = ep - d * SL_MULT * atr_now
                pt = ep + d * PT_MULT * atr_now
                pos = _Pos(ep, i, sl, pt, i + TIME_STOP_BARS, N_CONTRACTS, d)
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

    long_trades = [t for t in trades if t["direction"] == 1]
    short_trades = [t for t in trades if t["direction"] == -1]

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
        "long_n": len(long_trades),
        "long_wr": round(
            sum(1 for t in long_trades if t["pnl"] > 0) / max(len(long_trades), 1), 4
        ),
        "short_n": len(short_trades),
        "short_wr": round(
            sum(1 for t in short_trades if t["pnl"] > 0) / max(len(short_trades), 1), 4
        ),
        "veto_breakdown": {
            "cdp_required": len(veto_log.cdp_required),
            "other_score_low": len(veto_log.other_score_low),
            "shorts_disabled": len(veto_log.shorts_disabled),
            "min_score": len(veto_log.min_score),
        },
        "trades": trades,
    }


def monte_carlo_pass_prob(
    trades: list[dict],
    n_paths: int = 5000,
    days_per_path: int = 14,
    trades_per_day: float = 1.0,
    starting_balance: float = 50_000.0,
    daily_loss_limit: float = -1000.0,
    trailing_dd_limit: float = -2000.0,
    profit_target: float = 3000.0,
    rng_seed: int = 42,
) -> dict:
    if not trades:
        return {"p_pass": 0.0, "p95_dd": 0.0, "median_days": 0, "n_paths": n_paths}

    rng = np.random.default_rng(rng_seed)
    pnls = np.array([t["pnl"] for t in trades])
    n_per_day = max(1, round(trades_per_day))
    passes = 0
    drawdowns: list[float] = []
    days_to_pass: list[int] = []

    for _ in range(n_paths):
        balance = starting_balance
        peak = starting_balance
        halted = False
        passed_day = None

        for day in range(days_per_path):
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
            if not halted and balance >= starting_balance + profit_target:
                passed_day = day + 1
                break

        drawdowns.append(min(balance - peak, 0.0))
        if passed_day is not None:
            passes += 1
            days_to_pass.append(passed_day)

    return {
        "p_pass": round(passes / n_paths, 4),
        "p95_dd": round(float(np.percentile(drawdowns, 5)), 2),
        "median_days": int(np.median(days_to_pass)) if days_to_pass else days_per_path,
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
        f"{'DD':>9} {'T/Day':>6} {'L':>4}/{'{S}':>4}  MLL"
    )
    print(f"  {'-'*92}")


def _print_row(r: dict) -> None:
    mll = "✅" if r["mll_ok"] else "❌"
    print(
        f"  {r['label']:<38} {r['n']:>5} {r['wr']:>5.1%} "
        f"${r['pnl']:>8,.0f} {r['sharpe']:>7.2f} "
        f"${r['dd']:>8,.0f} {r['trades_per_day']:>6.2f} "
        f"{r['long_n']:>4}/{r['short_n']:>4}  {mll}"
    )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    IS_DATA  = ROOT / "data" / "processed" / "mnq_bars_5min.h5"
    OOS_DATA = ROOT / "data" / "processed" / "jan_feb_2026_oos_test.h5"

    configs = [
        # (label,                    long_only, cdp_req, min_other, shorts, min_short_cdp, legacy_q3, max_tpd)
        ("0: Baseline",              True,  False, 1, False, 0.30, False, 1),
        ("1: CDP-Anchor",            True,  True,  1, False, 0.30, False, 1),
        ("2: CDP-Anchor + 2/day",    True,  True,  1, False, 0.30, False, 2),
        ("3: CDP-Anchor + Shorts",   False, True,  1, True,  0.30, False, 1),
        ("4: CDP-Anch+Short+2/day",  False, True,  1, True,  0.30, False, 2),
        ("5: Q≥3 reference",         True,  False, 1, False, 0.30, True,  1),
    ]

    for dataset_label, data_path in [
        ("IN-SAMPLE (mnq_bars_5min)", IS_DATA),
        ("OOS Jan-Feb 2026",          OOS_DATA),
    ]:
        if not data_path.exists():
            print(f"\n  [{dataset_label}] Data not found at {data_path}, skipping.")
            continue

        print(f"\nLoading {dataset_label}...")
        bars = _load_rth(data_path)
        print(f"  {len(bars):,} bars  |  {bars.index[0].date()} → {bars.index[-1].date()}")

        _print_header(
            f"CD-ORB v2  |  {dataset_label}  |  2 contracts  |  PT=3.0x SL=1.5x"
        )

        results = []
        for label, lo, cr, mos, sht, mscdp, lq3, mtpd in configs:
            r = run_backtest(
                bars, long_only=lo, cdp_required=cr, min_other_score=mos,
                allow_cdp_shorts=sht, min_short_cdp_ratio=mscdp,
                use_legacy_q3=lq3, max_trades_per_day=mtpd, label=label,
            )
            results.append(r)
            _print_row(r)

        # Veto breakdown for CDP-Anchor (config 1)
        cdp_anchor = results[1]
        print(f"\n  Veto breakdown — '{cdp_anchor['label']}':")
        vb = cdp_anchor["veto_breakdown"]
        for reason, cnt in sorted(vb.items(), key=lambda x: -x[1]):
            if cnt > 0:
                print(f"    {reason:<20} {cnt:>4} trades blocked")

        # Exit breakdown for CDP-Anchor
        print(f"\n  Exit reasons — '{cdp_anchor['label']}':")
        for reason, cnt in sorted(cdp_anchor["exit_reasons"].items(), key=lambda x: -x[1]):
            pnls = [t["pnl"] for t in cdp_anchor["trades"] if t["reason"] == reason]
            avg = sum(pnls) / len(pnls) if pnls else 0
            wr = sum(1 for p in pnls if p > 0) / max(len(pnls), 1)
            print(f"    {reason:<20} {cnt:>4}  WR={wr:.0%}  avg=${avg:>7.1f}")

        # Monte Carlo for variants 1–4
        print(f"\n  Monte Carlo P(pass Topstep 14-day combine) — 5,000 paths:")
        print(f"  {'Config':<38} {'P(pass)':>9} {'p95_DD':>9} {'Med.Days':>10}")
        print(f"  {'-'*70}")
        for r in results:
            mc = monte_carlo_pass_prob(
                r["trades"],
                n_paths=5000,
                days_per_path=14,
                trades_per_day=max(r["trades_per_day"], 0.3),
            )
            print(
                f"  {r['label']:<38} {mc['p_pass']:>8.1%}  "
                f"${mc['p95_dd']:>8,.0f}  {mc['median_days']:>9}d"
            )

    # ── Save results ──────────────────────────────────────────────────────────
    print(f"\n  Saving results...")
    all_results: dict[str, list] = {"in_sample": [], "oos": []}

    for dataset_label, data_path, key in [
        ("in_sample", IS_DATA, "in_sample"),
        ("oos", OOS_DATA, "oos"),
    ]:
        if not data_path.exists():
            continue
        bars = _load_rth(data_path)
        for label, lo, cr, mos, sht, mscdp, lq3, mtpd in configs:
            r = run_backtest(
                bars, long_only=lo, cdp_required=cr, min_other_score=mos,
                allow_cdp_shorts=sht, min_short_cdp_ratio=mscdp,
                use_legacy_q3=lq3, max_trades_per_day=mtpd, label=label,
            )
            mc = monte_carlo_pass_prob(
                r["trades"], n_paths=5000, days_per_path=14,
                trades_per_day=max(r["trades_per_day"], 0.3),
            )
            all_results[key].append({k: v for k, v in r.items() if k != "trades"} | {"monte_carlo": mc})

    out_path = ROOT / "rule_based_v1" / "diagnostics" / "cd_orb_v2_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Results saved → {out_path.relative_to(ROOT)}")
    print()
