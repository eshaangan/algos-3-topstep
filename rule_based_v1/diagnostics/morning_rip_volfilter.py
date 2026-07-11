"""Vol-gated Morning Rip: does a PRE-OPEN volatility filter separate the +$24/day
open-drive days from the January-style whipsaw bleed?

Leakage safety: filters use ONLY prints before 9:30 ET —
  pm_range  : high-low of 08:25-09:29 ET prints (premarket hour, incl. 8:30 data)
  on_range  : high-low of 00:00-09:29 ET prints (overnight session proxy)
Both are known before the entry stops would be placed.

Protocol: Dec-May tape = dev (thresholds selected here, terciles). The Jun-Jul
recorder tape gets the FROZEN rule only (quasi-holdout — unfiltered variants were
already seen there once; the fully clean test is forward days). Trials this run:
2 filters x 3 terciles x 2 variants = 12 cells, counted.

    python rule_based_v1/diagnostics/morning_rip_volfilter.py            # dev CSVs
    python rule_based_v1/diagnostics/morning_rip_volfilter.py --recorder-dir <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FILES = [ROOT / "data/processed/mnq_trades_dec2025.csv.gz",
         ROOT / "data/processed/mnq_trades_jan_feb9_2026.csv.gz",
         ROOT / "data/processed/mnq_trades_mar_may_2026.csv.gz"]
COMM, STOP_SLIP, PV, NC = 0.62, 0.50, 2.0, 2
VARIANTS = [("8/15/8", 8, 15, 8), ("5/10/5", 5, 10, 5)]
THRESH_OUT = "/tmp/morning_rip_volfilter_thresholds.json"


def score_day(px, off, tp, sl):
    open_px = px[0]
    up, dn = open_px + off, open_px - off
    hu = np.flatnonzero(px >= up); hd = np.flatnonzero(px <= dn)
    if len(hu) == 0 and len(hd) == 0:
        return 0.0
    iu = hu[0] if len(hu) else np.inf
    idn = hd[0] if len(hd) else np.inf
    d, i0, entry = (1, int(iu), up + STOP_SLIP) if iu < idn else (-1, int(idn), dn - STOP_SLIP)
    seg = px[i0:]
    ti_ = np.flatnonzero(d * (seg - (entry + d * tp)) >= 0)
    si_ = np.flatnonzero(d * (seg - (entry - d * sl)) <= 0)
    ti = ti_[0] if len(ti_) else np.inf
    si = si_[0] if len(si_) else np.inf
    if ti == si == np.inf:
        return float(d * (seg[-1] - entry) - STOP_SLIP)
    return float(tp) if ti < si else float(-sl - STOP_SLIP)


def add_prints(store, ts_et, px):
    """Split one day's ET-indexed prints into premarket stats + RTH array."""
    hm = ts_et.dt.hour * 60 + ts_et.dt.minute
    day = ts_et.dt.strftime("%Y-%m-%d")
    for d, g in pd.DataFrame({"hm": hm, "px": px, "day": day}).groupby("day"):
        s = store.setdefault(d, {"pm_lo": np.inf, "pm_hi": -np.inf,
                                 "on_lo": np.inf, "on_hi": -np.inf, "rth": []})
        pm = g[(g.hm >= 8 * 60 + 25) & (g.hm < 9 * 60 + 30)]["px"]
        on = g[g.hm < 9 * 60 + 30]["px"]
        rth = g[(g.hm >= 9 * 60 + 30) & (g.hm < 16 * 60)]["px"]
        if len(pm):
            s["pm_lo"] = min(s["pm_lo"], pm.min()); s["pm_hi"] = max(s["pm_hi"], pm.max())
        if len(on):
            s["on_lo"] = min(s["on_lo"], on.min()); s["on_hi"] = max(s["on_hi"], on.max())
        if len(rth):
            s["rth"].append(rth.to_numpy())


def day_table(store):
    rows = []
    for d, s in sorted(store.items()):
        if not s["rth"]:
            continue
        px = np.concatenate(s["rth"])
        if len(px) < 5000:
            continue
        row = {"day": d,
               "pm_range": s["pm_hi"] - s["pm_lo"] if np.isfinite(s["pm_hi"]) else np.nan,
               "on_range": s["on_hi"] - s["on_lo"] if np.isfinite(s["on_hi"]) else np.nan}
        for name, off, tp, sl in VARIANTS:
            row[name] = score_day(px, off, tp, sl) * PV * NC - 2 * COMM * NC
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recorder-dir", help="consolidated trade_MNQ_*.parquet (frozen-rule eval)")
    args = ap.parse_args()
    store: dict = {}

    if args.recorder_dir:
        for f in sorted(glob.glob(os.path.join(args.recorder_dir, "trade_MNQ_*.parquet"))):
            t = pd.read_parquet(f)
            if len(t) < 1000:
                continue
            ts = pd.to_datetime(t["recv_ns"], unit="ns", utc=True).dt.tz_convert("America/New_York")
            add_prints(store, ts, t["price"].to_numpy(float))
    else:
        for f in FILES:
            print(f"streaming {f.name}...", flush=True)
            for chunk in pd.read_csv(f, compression=None, chunksize=20_000_000,
                                     usecols=["ts_recv", "price"]):
                ts = pd.to_datetime(chunk["ts_recv"], unit="ns", utc=True).dt.tz_convert("America/New_York")
                add_prints(store, ts, chunk["price"].to_numpy(float) * 1e-9)

    df = day_table(store).dropna(subset=["pm_range"])
    print(f"\n{len(df)} scored days ({df.day.iloc[0]}..{df.day.iloc[-1]})")

    if args.recorder_dir:
        thr = json.load(open(THRESH_OUT))
        for name, _, _, _ in VARIANTS:
            for filt in ("pm_range", "on_range"):
                gate = df[filt] >= thr[filt]["t2"]
                on_, off_ = df.loc[gate, name], df.loc[~gate, name]
                print(f"FROZEN {filt}>=p67({thr[filt]['t2']:.0f}pt) {name}: "
                      f"ON n={len(on_)} ${on_.sum():+,.0f} (${on_.mean():+.0f}/d) | "
                      f"OFF n={len(off_)} ${off_.sum():+,.0f}")
        return

    # DEV: tercile analysis per filter per variant
    thr_out = {}
    for filt in ("pm_range", "on_range"):
        t1, t2 = df[filt].quantile([1 / 3, 2 / 3])
        thr_out[filt] = {"t1": float(t1), "t2": float(t2)}
        print(f"\n-- {filt} terciles: <{t1:.0f} / {t1:.0f}-{t2:.0f} / >{t2:.0f} pts --")
        for name, _, _, _ in VARIANTS:
            lo = df[df[filt] < t1][name]
            md = df[(df[filt] >= t1) & (df[filt] < t2)][name]
            hi = df[df[filt] >= t2][name]
            print(f"  {name}: LO n={len(lo)} ${lo.mean():+6.0f}/d | "
                  f"MID n={len(md)} ${md.mean():+6.0f}/d | HI n={len(hi)} ${hi.mean():+6.0f}/d"
                  f"   (corr={df[filt].corr(df[name]):+.2f})")
    json.dump(thr_out, open(THRESH_OUT, "w"))
    df.to_csv("/tmp/morning_rip_volfilter_dev.csv", index=False)
    print(f"\nthresholds frozen -> {THRESH_OUT}; dev table -> /tmp/morning_rip_volfilter_dev.csv")


if __name__ == "__main__":
    main()
