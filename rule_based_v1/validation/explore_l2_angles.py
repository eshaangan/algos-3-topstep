"""Exploratory scan of NEW angles on the recorded BBO+trade data — DEV SLICE ONLY.

Discipline: everything here runs on bars before DEV_END (2026-06-27). The holdout
stays untouched, so any angle that pops can still be honestly promoted through the
pre-registered gate in research_l2.py. This script makes NO go/no-go claims; it
ranks hypotheses by information coefficient (IC = Spearman corr of feature vs
forward return) and simple event studies.

Angles scanned:
  A. IC ladder — every feature x horizon (1..20 bars): where does ANY signal live?
  B. Time-of-day — is predictability concentrated (e.g. first hour of RTH)?
  C. Conditional IC — flow imbalance when the book is THIN vs THICK (impact
     asymmetry, Kyle-lambda flavor), and imbalance when spread is wide vs tight.
  D. Sweep events — big one-sided aggressive volume bursts: continuation or fade?
  E. Liquidity-pull events — sudden drop in resting size: does price follow?

    python rule_based_v1/validation/explore_l2_angles.py --freq 1min
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rule_based_v1.validation.l2_features import build_bars_by_day  # noqa: E402

L2_RAW = ROOT / "data" / "l2_raw"
DEV_END = "2026-06-27"
HORIZONS = [1, 2, 5, 10, 20]
FEATURES = ["l1_imb", "micro_tilt", "spread", "depth_total", "t_ofi_imb", "t_vol"]


def ic(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < 100:
        return np.nan, n
    return float(x[m].corr(y[m], method="spearman")), n


def fwd_ret(bars: pd.DataFrame, h: int) -> pd.Series:
    return bars["close"].shift(-h) - bars["close"]


def scan_ic_ladder(dev: pd.DataFrame) -> None:
    print("\n=== A. IC ladder (Spearman, feature vs forward mid move) ===")
    print(f"{'feature':14s} " + " ".join(f"h={h:<3d}   " for h in HORIZONS))
    for f in FEATURES:
        if f not in dev.columns:
            continue
        row = []
        for h in HORIZONS:
            v, _ = ic(dev[f], fwd_ret(dev, h))
            row.append(f"{v:+.3f}  " if np.isfinite(v) else "  n/a  ")
        print(f"{f:14s} " + " ".join(row))
    # |IC| > ~0.03 with thousands of bars is worth a second look; >0.05 is interesting.


def scan_time_of_day(dev: pd.DataFrame, feat: str = "l1_imb", h: int = 5) -> None:
    print(f"\n=== B. Time-of-day IC ({feat}, h={h}) — ET hours ===")
    et = dev.index.tz_convert("America/New_York")
    y = fwd_ret(dev, h)
    for hr in range(9, 16):
        m = et.hour == hr
        v, n = ic(dev.loc[m, feat], y[m])
        bar = "#" * int(min(abs(v) * 200, 40)) if np.isfinite(v) else ""
        print(f"  {hr:02d}:00  IC={v:+.3f} (n={n:>5})  {bar}" if np.isfinite(v)
              else f"  {hr:02d}:00  n/a (n={n})")


def scan_conditional(dev: pd.DataFrame, h: int = 5) -> None:
    print(f"\n=== C. Conditional IC (h={h}) ===")
    y = fwd_ret(dev, h)
    combos = [
        ("t_ofi_imb | thin book (depth_total<q25)", "t_ofi_imb", dev["depth_total"] < dev["depth_total"].quantile(0.25)),
        ("t_ofi_imb | thick book (depth_total>q75)", "t_ofi_imb", dev["depth_total"] > dev["depth_total"].quantile(0.75)),
        ("l1_imb    | wide spread (>q75)", "l1_imb", dev["spread"] > dev["spread"].quantile(0.75)),
        ("l1_imb    | tight spread (<=q25)", "l1_imb", dev["spread"] <= dev["spread"].quantile(0.25)),
        ("l1_imb    | high vol bar (t_vol>q75)", "l1_imb", dev["t_vol"] > dev["t_vol"].quantile(0.75)),
        ("l1_imb x t_ofi agree (product)", None, None),
    ]
    for label, feat, mask in combos:
        if feat is None:  # interaction: book and flow point the same way
            x = dev["l1_imb"] * dev["t_ofi_imb"]
            sign_agree = np.sign(dev["l1_imb"]) == np.sign(dev["t_ofi_imb"])
            v, n = ic(dev.loc[sign_agree, "l1_imb"], y[sign_agree])
            print(f"  {label:42s} IC={v:+.3f} (n={n})")
            continue
        v, n = ic(dev.loc[mask, feat], y[mask])
        print(f"  {label:42s} IC={v:+.3f} (n={n})" if np.isfinite(v) else f"  {label:42s} n/a (n={n})")


def scan_sweeps(dev: pd.DataFrame, h: int = 5) -> None:
    print(f"\n=== D. Sweep events (one-sided volume burst) fwd move @ h={h} ===")
    vol_hi = dev["t_vol"] > dev["t_vol"].quantile(0.90)
    y = fwd_ret(dev, h)
    for thr in (0.5, 0.7):
        for side, name in ((1, "BUY "), (-1, "SELL")):
            ev = vol_hi & (side * dev["t_ofi_imb"] > thr)
            n = int(ev.sum())
            if n < 30:
                print(f"  {name} sweep imb>{thr}: n={n} (too few)")
                continue
            mv = float((side * y[ev]).mean())  # + = continuation, - = fade
            t = mv / (float((side * y[ev]).std()) / np.sqrt(n) + 1e-12)
            print(f"  {name} sweep imb>{thr}: n={n:>4}  fwd move in sweep dir = {mv:+.2f} pts  (t={t:+.1f})")


def scan_liquidity_pull(dev: pd.DataFrame, h: int = 5) -> None:
    print(f"\n=== E. Liquidity-pull events (depth_total drops >50% bar-over-bar) @ h={h} ===")
    chg = dev["depth_total"].pct_change()
    ev = chg < -0.5
    y = fwd_ret(dev, h)
    n = int(ev.sum())
    if n < 30:
        print(f"  n={n} (too few)")
        return
    # direction proxy: side that got thinner ⇒ sign of imbalance CHANGE
    d_imb = dev["l1_imb"].diff()
    signed = np.sign(d_imb[ev]) * y[ev]
    print(f"  n={n}  |mean fwd move|={float(y[ev].abs().mean()):.2f} vs all-bar {float(y.abs().mean()):.2f} pts")
    print(f"  move in direction of imbalance shift: {float(signed.mean()):+.2f} pts (t={float(signed.mean() / (signed.std() / np.sqrt(n) + 1e-12)):+.1f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", default="1min")
    ap.add_argument("--raw-dir", default=str(L2_RAW))
    args = ap.parse_args()

    bars = build_bars_by_day(args.raw_dir, freq=args.freq, n_levels=1, use_bbo=True)
    bars = bars.dropna(subset=["close"])
    dev = bars[bars.index < pd.Timestamp(DEV_END, tz="UTC")]
    print(f"Bars: {len(bars):,} total @ {args.freq} | DEV slice: {len(dev):,} (< {DEV_END}) — holdout untouched")
    for col in ("t_ofi_imb", "t_vol"):
        if col not in dev.columns:
            print(f"  (no {col} column — trade features missing)")
            return

    scan_ic_ladder(dev)
    scan_time_of_day(dev)
    scan_conditional(dev)
    scan_sweeps(dev)
    scan_liquidity_pull(dev)
    print("\nReminder: exploratory, dev-only. Anything interesting must pass the "
          "pre-registered gate (research_l2.py) on the untouched holdout.")


if __name__ == "__main__":
    main()
