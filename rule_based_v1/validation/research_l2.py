"""Validate order-book (L2) signals through the pre-registered gate.

Runs the moment we have recorded L2 data (data/l2_raw/, pulled from the remote via
data_collection/pull_l2.sh). With no real data yet it self-tests on synthetic L2
(including a planted-edge case) so the pipeline is proven end-to-end in advance.

Discipline (same as everything else): select on the development slice, test once on
the holdout, count trials honestly in the DSR, model real MNQ costs. Signals are
minute-scale (not tick scalping) to (a) clear the ~$2.24 round-trip cost wall and
(b) stay clear of Lucid's HFT / ≤5s microscalping bans.

    python rule_based_v1/validation/research_l2.py            # real data if present, else synthetic
    python rule_based_v1/validation/research_l2.py --synthetic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rule_based_v1.validation.harness import (  # noqa: E402
    SimParams, simulate, aggregate_stats, monthly_breakdown, deflated_sharpe_ratio,
)
from rule_based_v1.validation.l2_features import (  # noqa: E402
    load_l2_raw, build_bars, synthesize_l2,
)

L2_RAW = ROOT / "data" / "l2_raw"
# MNQ micro: $2/pt, 0.25 tick. Cost ≈ commission both sides + 1 tick slip each way.
MNQ = dict(point_value=2.0, tick_size=0.25, commission_per_side=0.62, slippage_ticks=1)


def _zscore(s: pd.Series, win: int = 60) -> pd.Series:
    return (s - s.rolling(win, min_periods=win // 2).mean()) / s.rolling(win, min_periods=win // 2).std()


# ── candidate signals (each: bars → direction Series in {1,-1,0}) ─────────────

def sig_depth_continuation(bars: pd.DataFrame, z: float) -> pd.Series:
    """Strong resting-depth imbalance ⇒ go with it (book pressure persists)."""
    zi = _zscore(bars["depth_imb"])
    d = pd.Series(0, index=bars.index, dtype=int)
    d[zi >= z] = 1
    d[zi <= -z] = -1
    return d


def sig_micro_tilt(bars: pd.DataFrame, z: float) -> pd.Series:
    """Microprice tilted above/below mid ⇒ short-horizon drift in that direction."""
    zi = _zscore(bars["micro_tilt"])
    d = pd.Series(0, index=bars.index, dtype=int)
    d[zi >= z] = 1
    d[zi <= -z] = -1
    return d


def sig_depth_reversion(bars: pd.DataFrame, z: float) -> pd.Series:
    """Extreme imbalance ⇒ fade (liquidity exhaustion / spoofy pressure reverts)."""
    return -sig_depth_continuation(bars, z)


SIGNALS = {
    "depth_continuation": (sig_depth_continuation, [1.0, 1.5, 2.0]),
    "micro_tilt": (sig_micro_tilt, [1.0, 1.5, 2.0]),
    "depth_reversion": (sig_depth_reversion, [1.5, 2.0]),
}


def _run_family(name, fn, grid, dev, hold, p, n_trials):
    best, best_pnl = None, -1e18
    for z in grid:
        s = fn(dev, z)
        t = simulate(dev[["open", "high", "low", "close"]],
                     pd.DataFrame({"direction": s, "atr": dev["atr"]}), p)
        a = aggregate_stats(t)
        if a["n_trades"] >= 30 and a["total_pnl"] > best_pnl:
            best_pnl, best = a["total_pnl"], z
    if best is None:
        print(f"  {name:20s} no viable DEV config")
        return
    s = fn(hold, best)
    t = simulate(hold[["open", "high", "low", "close"]],
                 pd.DataFrame({"direction": s, "atr": hold["atr"]}), p)
    a = aggregate_stats(t)
    mb = monthly_breakdown(t)
    dsr = deflated_sharpe_ratio(t["pnl"].to_numpy() if len(t) else [], n_trials=n_trials).get("dsr")
    posm = float((mb["pnl"] > 0).mean()) if len(mb) else 0.0
    print(f"  {name:20s} z={best} | DEV ${best_pnl:>7,.0f} | HOLD {a['n_trades']:>4}tr "
          f"WR={a['win_rate']:.0%} ${a['total_pnl']:>7,.0f} DD=${a['max_dd']:>6,.0f} "
          f"posMo={posm:.0%} DSR={dsr:.3f}" if dsr is not None else
          f"  {name:20s} z={best} | HOLD {a['n_trades']}tr ${a['total_pnl']:,.0f} DSR=n/a")


def run(bars: pd.DataFrame, dev_end: str, freq_label: str):
    bars = bars.dropna(subset=["depth_imb", "micro_tilt", "atr"])
    dev = bars[bars.index < pd.Timestamp(dev_end, tz="UTC")]
    hold = bars[bars.index >= pd.Timestamp(dev_end, tz="UTC")]
    print(f"\nL2 signals @ {freq_label}: DEV={len(dev)} bars (<{dev_end}), HOLD={len(hold)} bars")
    if len(dev) < 200 or len(hold) < 100:
        print("  (insufficient bars — need more recorded data before this is meaningful)")
        return
    p = SimParams(pt_atr=1.5, sl_atr=1.0, horizon_bars=10, cooldown_bars=1,
                  max_trades_per_day=8, n_contracts=2, **MNQ)
    n_trials = sum(len(g) for _, g in SIGNALS.values())
    for name, (fn, grid) in SIGNALS.items():
        _run_family(name, fn, grid, dev, hold, p, n_trials)


def _synth_multi(days: int, strength: float, seed0: int, freq: str) -> pd.DataFrame:
    """Build multi-day synthetic L2 bars (one ~9h session per weekday) so the daily
    trade cap behaves like real multi-day data."""
    books, trs, day = [], [], 1
    d = pd.Timestamp("2026-06-01")
    made = 0
    while made < days:
        if d.weekday() < 5:  # weekday session only
            b, t = synthesize_l2(7000, seed=seed0 + made, signal_strength=strength,
                                 start=f"{d.date()} 14:00")
            books.append(b); trs.append(t); made += 1
        d += pd.Timedelta(days=1)
    book = pd.concat(books, ignore_index=True); trades = pd.concat(trs, ignore_index=True)
    book["ts"] = pd.to_datetime(book["recv_ns"], unit="ns", utc=True)
    trades["ts"] = pd.to_datetime(trades["recv_ns"], unit="ns", utc=True)
    return build_bars(book, trades, freq=freq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="force synthetic data")
    ap.add_argument("--freq", default="1min")
    ap.add_argument("--raw-dir", default=str(L2_RAW))
    args = ap.parse_args()

    book = pd.DataFrame() if args.synthetic else load_l2_raw(args.raw_dir, "book")
    if book.empty:
        if not args.synthetic:
            print(f"No recorded L2 in {args.raw_dir} yet — running PIPELINE SELF-TEST on synthetic data.")
            print("Pull real data with data_collection/pull_l2.sh once a few days have accumulated.\n")
        # planted-edge synthetic: the gate SHOULD light up when an edge truly exists
        bars = _synth_multi(20, strength=0.30, seed0=100, freq=args.freq)
        run(bars, dev_end="2026-06-18", freq_label=f"{args.freq} (SYNTHETIC planted edge → gate should pass)")
        # control: no edge → gate should NOT fire
        bars0 = _synth_multi(20, strength=0.0, seed0=500, freq=args.freq)
        run(bars0, dev_end="2026-06-18", freq_label=f"{args.freq} (SYNTHETIC no edge → gate should reject)")
        return

    # Full depth often unavailable on the Lucid feed (NO_BOOK). Fall back to BBO
    # top-of-book (queue imbalance / microprice / spread), which IS entitled.
    n_levels = 5
    src = "ORDER_BOOK (depth)"
    if book.empty or float(book.get("bid_sz_0", pd.Series([0])).max() or 0) == 0:
        bbo = load_l2_bbo(args.raw_dir)
        if not bbo.empty and float(bbo["bid_sz_0"].max() or 0) > 0:
            book, n_levels, src = bbo, 1, "BBO (top-of-book)"
        else:
            print("Neither full depth nor BBO has usable data yet — keep recording.")
            return

    trades = load_l2_raw(args.raw_dir, "trade")
    bars = build_bars(book, trades, freq=args.freq, n_levels=n_levels)
    dates = sorted({d.date() for d in bars.index})
    print(f"Source: {src}. Loaded {len(book):,} snapshots → {len(bars):,} {args.freq} bars over {len(dates)} days")
    # 60/40 dev/holdout split by time
    if len(dates) >= 4:
        split = dates[int(len(dates) * 0.6)]
        run(bars, dev_end=str(split), freq_label=args.freq)
    else:
        print("Need >=4 days of data for a dev/holdout split. Keep recording.")


if __name__ == "__main__":
    main()
