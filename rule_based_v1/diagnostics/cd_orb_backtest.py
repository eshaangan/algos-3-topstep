"""Cumulative Delta ORB (CD-ORB) Diagnostic Backtest.

Tests the novel CumulativeDeltaFilter on top of the baseline ORB strategy,
sweeping quality thresholds Q≥1..4 against the confirmed-edge baseline.

Novel signal:
    CDP proxy = volume × (2 × (close−low) / (high−low+ε) − 1)
    Combined with OR positional bias, volume surge, and breakout momentum
    into a 4-point quality score. Only trades with score ≥ threshold proceed.

Academic basis: Glosten-Milgrom (1985), Chordia & Subrahmanyam (2004),
                Lopez de Prado AFML (2018, Ch. 19)

Datasets:
  Primary sweep   : data/processed/mnq_bars_5min.h5   (in-sample, baseline=131 trades)
  OOS validation  : data/processed/jan_feb_2026_oos_test.h5  (Jan-Feb 2026, 32 trades)

Run:
    cd rule_based_v1
    python diagnostics/cd_orb_backtest.py

Output: diagnostics/cd_orb_results.json
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

# ── Execution config (mirrors live 2-contract config) ────────────────────────
N_CONTRACTS   = 2
POINT_VALUE   = 2.0          # $2.00 / point for MNQ
TICK_SIZE     = 0.25
COMMISSION    = 0.62         # per side per contract
SLIPPAGE_TICKS = 1
PT_MULT       = 3.0
SL_MULT       = 1.5
ATR_PERIOD    = 14
TIME_STOP_BARS = 24          # 2 hours on 5-min bars
MAX_DAILY_LOSS = -950.0      # Topstep $1,000 − $50 buffer
DRAWDOWN_BUFFER = 1950.0     # Topstep $2,000 − $50 buffer
MAX_TRADES_DAY  = 1
STARTING_EQUITY = 50_000.0


# ── ORB parameters (best config from regime_sweep) ───────────────────────────
ORB_PARAMS = dict(
    or_end_time="10:04",
    min_or_bars=7,
    min_range_atr=0.3,
    entry_cutoff_time="12:00",
    atr_period=ATR_PERIOD,
    long_only=True,
)


@dataclass
class _Pos:
    entry_price: float
    entry_bar_idx: int
    stop_loss: float
    profit_target: float
    time_stop_bar: int
    n_contracts: int


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
    min_quality: int | None = None,
    label: str = "baseline",
) -> dict:
    """Run a single backtest variant.

    Parameters
    ----------
    bars : pd.DataFrame
        RTH 5-min bars, timezone-aware (US/Eastern).
    min_quality : int or None
        CD-ORB quality threshold. None = baseline (no filter).
    label : str
        Human-readable label for this variant.

    Returns
    -------
    dict with keys: label, n, wr, pnl, sharpe, dd, pf, exit_reasons, trades
    """
    orb = OpeningRangeBreakoutRule(**ORB_PARAMS)
    if min_quality is not None:
        cd_filter = CumulativeDeltaFilter(
            or_end_time=ORB_PARAMS["or_end_time"],
            min_quality_score=min_quality,
        )
        agg = SignalAggregator(
            primary_rule=orb,
            filter_rules=[cd_filter],
            confirmation_rules=[],
            min_confirmations=0,
        )
    else:
        agg = SignalAggregator(
            primary_rule=orb,
            filter_rules=[],
            confirmation_rules=[],
            min_confirmations=0,
        )

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

    for i in range(min_bars, len(bars)):
        bar = bars.iloc[i]
        ts_et = bars.index[i].tz_convert("US/Eastern")
        bdate = ts_et.date()
        bh, bm = ts_et.hour, ts_et.minute

        # Day rollover
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
                pnl = _calc_pnl(pos.entry_price, exit_p, pos.n_contracts)
                trades.append(
                    {
                        "date": str(bdate),
                        "entry": pos.entry_price,
                        "exit": exit_p,
                        "pnl": round(pnl, 2),
                        "reason": reason,
                    }
                )
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
            and trades_today < MAX_TRADES_DAY
            and daily_loss > MAX_DAILY_LOSS
            and (equity - peak) > -DRAWDOWN_BUFFER
        )

        if can_enter:
            lookback = bars.iloc[max(0, i - min_bars + 1) : i + 1]
            dec = agg.evaluate(lookback)
            if dec.should_trade:
                ep = _slip(bar["close"], dec.direction, True)
                sl = ep - SL_MULT * atr_now * dec.direction
                pt = ep + PT_MULT * atr_now * dec.direction
                pos = _Pos(ep, i, sl, pt, i + TIME_STOP_BARS, N_CONTRACTS)
                trades_today += 1

    # Flush last day
    if cur_date and cur_date not in daily_pnl:
        daily_pnl[cur_date] = daily_loss

    # ── Metrics ───────────────────────────────────────────────────────────────
    wins = [t for t in trades if t["pnl"] > 0]
    total_pnl = sum(t["pnl"] for t in trades)
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))

    daily_series = pd.Series(
        {k: v for k, v in daily_pnl.items() if v != 0}
    )
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
        "exit_reasons": exit_counts,
        "trades_per_day": round(len(trades) / max(len(daily_pnl), 1), 3),
        "mll_ok": max_dd > -2000,
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
    """Monte Carlo simulation of Topstep combine pass probability.

    Samples from the observed trade P&L distribution, simulating 14 trading
    days with Topstep's daily loss and trailing drawdown limits applied.

    Returns
    -------
    dict: p_pass, p95_dd, median_days, n_paths
    """
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
            day_trades = rng.choice(pnls, size=n_per_day, replace=True)
            for trade_pnl in day_trades:
                daily_total += trade_pnl
                balance += trade_pnl
                peak = max(peak, balance)

                # Trailing drawdown check (per trade)
                if (balance - peak) <= trailing_dd_limit:
                    halted = True
                    break

                # Check daily loss mid-day
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

    p95_dd = float(np.percentile(drawdowns, 5))  # 5th pctile = worst 5%
    median_days = int(np.median(days_to_pass)) if days_to_pass else days_per_path

    return {
        "p_pass": round(passes / n_paths, 4),
        "p95_dd": round(p95_dd, 2),
        "median_days": median_days,
        "n_paths": n_paths,
    }


def _load_rth(path: Path, key: str = "bars_5min") -> pd.DataFrame:
    bars = pd.read_hdf(str(path), key=key)
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("US/Eastern")
    # RTH only: 09:30–15:59
    et = bars.index.tz_convert("US/Eastern")
    rth_mask = (
        (et.hour > 9) | ((et.hour == 9) & (et.minute >= 30))
    ) & (et.hour < 16)
    return bars[rth_mask].copy()


def _print_header(title: str) -> None:
    print(f"\n{'='*82}")
    print(f"  {title}")
    print(f"{'='*82}")
    print(
        f"  {'Config':<35} {'N':>5}  {'WR':>6}  {'PnL':>9}  "
        f"{'Sharpe':>7}  {'Max DD':>9}  {'T/Day':>6}  MLL"
    )
    print(f"  {'-'*78}")


def _print_row(r: dict) -> None:
    mll = "✅" if r["mll_ok"] else "❌"
    print(
        f"  {r['label']:<35} {r['n']:>5}  {r['wr']:>5.1%}  "
        f"${r['pnl']:>8,.0f}  {r['sharpe']:>7.2f}  "
        f"${r['dd']:>8,.0f}  {r['trades_per_day']:>6.2f}  {mll}"
    )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    IS_DATA  = ROOT / "data" / "processed" / "mnq_bars_5min.h5"
    OOS_DATA = ROOT / "data" / "processed" / "jan_feb_2026_oos_test.h5"

    # ── In-sample sweep ───────────────────────────────────────────────────────
    print("\nLoading in-sample data (mnq_bars_5min.h5)...")
    bars_is = _load_rth(IS_DATA)
    print(f"  {len(bars_is):,} bars  |  "
          f"{bars_is.index[0].date()} → {bars_is.index[-1].date()}")

    configs = [
        ("Baseline (no filter)",  None),
        ("CD-ORB Q≥1",            1),
        ("CD-ORB Q≥2",            2),
        ("CD-ORB Q≥3 (★ rec.)",   3),
        ("CD-ORB Q≥4",            4),
    ]

    _print_header("CD-ORB IN-SAMPLE SWEEP — MNQ  |  2 contracts  |  PT=3.0x SL=1.5x  |  LONG-only")
    is_results = []
    for label, q in configs:
        r = run_backtest(bars_is, min_quality=q, label=label)
        is_results.append(r)
        _print_row(r)

    # Exit reason breakdown for Q≥3
    q3_is = next(r for r in is_results if "Q≥3" in r["label"])
    print(f"\n  Exit reasons (Q≥3 in-sample):")
    for reason, cnt in sorted(q3_is["exit_reasons"].items(), key=lambda x: -x[1]):
        pnls = [t["pnl"] for t in q3_is["trades"] if t["reason"] == reason]
        avg  = sum(pnls) / len(pnls) if pnls else 0
        print(f"    {reason:<20} {cnt:>4}  avg ${avg:>7.1f}")

    # ── OOS validation ────────────────────────────────────────────────────────
    if OOS_DATA.exists():
        print(f"\nLoading OOS data (jan_feb_2026_oos_test.h5)...")
        bars_oos = _load_rth(OOS_DATA)
        print(f"  {len(bars_oos):,} bars  |  "
              f"{bars_oos.index[0].date()} → {bars_oos.index[-1].date()}")

        _print_header("CD-ORB OOS VALIDATION — Jan–Feb 2026  |  2 contracts")
        oos_results = []
        for label, q in configs:
            r = run_backtest(bars_oos, min_quality=q, label=label)
            oos_results.append(r)
            _print_row(r)
    else:
        print(f"\n  [OOS data not found at {OOS_DATA}; skipping OOS section]")
        oos_results = []

    # ── Monte Carlo P(pass) for Q≥3 ──────────────────────────────────────────
    print(f"\n{'='*82}")
    print(f"  MONTE CARLO COMBINE SIM — Q≥3  |  5,000 paths  |  14-day window")
    print(f"{'='*82}")

    for label, results, dataset in [
        ("In-sample", is_results, "IS"),
        ("OOS",       oos_results, "OOS"),
    ]:
        q3 = next((r for r in results if "Q≥3" in r["label"]), None)
        baseline = next((r for r in results if "Baseline" in r["label"]), None)
        if q3 is None:
            continue

        for variant_label, variant in [("Baseline", baseline), ("Q≥3", q3)]:
            if variant is None:
                continue
            mc = monte_carlo_pass_prob(
                variant["trades"],
                n_paths=5000,
                days_per_path=14,
                trades_per_day=max(variant["trades_per_day"], 0.5),
            )
            print(
                f"  [{dataset}] {variant_label:<12}  "
                f"P(pass)={mc['p_pass']:.1%}  "
                f"p95_dd=${mc['p95_dd']:,.0f}  "
                f"median_days={mc['median_days']}"
            )

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "in_sample": [
            {k: v for k, v in r.items() if k != "trades"}
            for r in is_results
        ],
        "oos": [
            {k: v for k, v in r.items() if k != "trades"}
            for r in oos_results
        ],
    }
    out_path = ROOT / "rule_based_v1" / "diagnostics" / "cd_orb_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved → {out_path.relative_to(ROOT)}")
    print()
