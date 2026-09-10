#!/usr/bin/env python3
"""
Turn MBO microstructure features into a trade series and screen it.

Pipeline: features (ml_intraday_v3.features.mbo_features) -> a simple threshold
strategy on the one feature that showed any predictivity (queue imbalance at
best) -> a trade log -> the edge-attribution screens.

The strategy is deliberately the simplest thing that could work: when queue
imbalance at the touch exceeds +thr go long, below -thr go short, hold H
seconds, exit. Non-overlapping. This is NOT meant to be clever; it is meant to
convert a measured IC into a costed P/L so the screens can rule on it.

Costs are charged honestly and are the whole game at this horizon:
    round trip = cross the spread (enter marketable, exit marketable)
               + 2 x commission
For MNQ: point value $2, 1-tick spread = 0.25pt = $0.50, commission $0.62/side.
A signal must clear ~$1.74/contract round trip before it is an edge.

Usage:
    python -m ml_intraday_v3.diagnostics.mbo_signal_backtest \
        --features data/processed/mbo_vps/features_20260903.parquet
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

PV, TICK, COMM = 2.0, 0.25, 0.62      # MNQ micro


def load_features(paths: list) -> pd.DataFrame:
    frames = []
    for p in paths:
        f = pd.read_parquet(p)
        f["ts"] = pd.to_datetime(f["ts"], utc=True)
        frames.append(f)
    f = pd.concat(frames, ignore_index=True).sort_values("ts").reset_index(drop=True)
    return f


def backtest(f: pd.DataFrame, feature: str, thr: float, hold_s: int) -> pd.DataFrame:
    """
    Non-overlapping threshold strategy. Enter marketable at the touch, exit
    marketable hold_s bars later. Returns a trade log with gross and net P/L.
    """
    f = f.dropna(subset=["mid", "best_bid", "best_ask"]).reset_index(drop=True)
    sig = f[feature].to_numpy()
    bid, ask = f["best_bid"].to_numpy(), f["best_ask"].to_numpy()
    ts = f["ts"].to_numpy()

    trades = []
    i, n = 0, len(f)
    while i < n - hold_s:
        s = sig[i]
        if s > thr or s < -thr:
            side = 1 if s > thr else -1
            j = i + hold_s
            if side == 1:                       # buy at ask, sell at bid
                entry, exit_ = ask[i], bid[j]
            else:                               # sell at bid, buy at ask
                entry, exit_ = bid[i], ask[j]
            gross_pts = (exit_ - entry) * side
            pnl_gross = gross_pts * PV
            pnl_net = pnl_gross - 2 * COMM      # spread already in entry/exit
            trades.append({
                "entry_time": ts[i], "exit_time": ts[j], "direction": side,
                "entry": entry, "exit": exit_, "pnl_gross": pnl_gross,
                "pnl": pnl_net, "signal": s,
            })
            i = j + 1                            # non-overlapping
        else:
            i += 1
    return pd.DataFrame(trades)


def summarize(tr: pd.DataFrame, label: str) -> dict:
    if len(tr) == 0:
        return {"label": label, "n": 0}
    g, npl = tr["pnl_gross"], tr["pnl"]
    return {
        "label": label, "n": int(len(tr)),
        "gross_total": round(g.sum(), 2), "gross_mean": round(g.mean(), 4),
        "net_total": round(npl.sum(), 2), "net_mean": round(npl.mean(), 4),
        "gross_t": round(g.mean() / g.std(ddof=1) * np.sqrt(len(g)), 2) if g.std() > 0 else None,
        "net_t": round(npl.mean() / npl.std(ddof=1) * np.sqrt(len(npl)), 2) if npl.std() > 0 else None,
        "win_rate": round((npl > 0).mean(), 3),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True,
                    help="feature parquet(s); globs allowed")
    ap.add_argument("--feature", default="queue_imbalance")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    paths = []
    for pat in args.features:
        paths.extend(sorted(glob.glob(pat)))
    if not paths:
        print("no feature files matched", file=sys.stderr)
        return 2

    f = load_features(paths)
    print(f"features: {len(f)} bars over {len(paths)} day(s), "
          f"{f.ts.min()} -> {f.ts.max()}\n")

    # Grid over threshold and hold. Gross first: if gross has no edge, cost is moot.
    print(f"{'thr':>5} {'hold':>5} {'n':>7} {'gross_mean':>11} {'gross_t':>8} "
          f"{'net_mean':>10} {'net_t':>7} {'win':>6}")
    best = None
    for thr in (0.2, 0.4, 0.6, 0.8):
        for hold in (5, 15, 30, 60):
            tr = backtest(f, args.feature, thr, hold)
            s = summarize(tr, f"thr{thr}_h{hold}")
            if s["n"] < 30:
                continue
            print(f"{thr:>5} {hold:>5} {s['n']:>7} {s['gross_mean']:>11.4f} "
                  f"{str(s['gross_t']):>8} {s['net_mean']:>10.4f} "
                  f"{str(s['net_t']):>7} {s['win_rate']:>6}")
            # track best by gross_t (is there ANY signal before costs?)
            if s["gross_t"] is not None and (best is None or s["gross_t"] > best[0]):
                best = (s["gross_t"], thr, hold, tr)

    if best is None:
        print("\nno config produced enough trades")
        return 0

    gt, thr, hold, tr = best
    print(f"\nbest gross_t = {gt} at thr={thr}, hold={hold}s, n={len(tr)}")
    print(f"  gross total ${tr.pnl_gross.sum():,.0f}  net total ${tr.pnl.sum():,.0f}  "
          f"(cost drag ${tr.pnl_gross.sum()-tr.pnl.sum():,.0f})")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        tr.to_parquet(args.out)
        print(f"\nwrote best-config trade log -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
