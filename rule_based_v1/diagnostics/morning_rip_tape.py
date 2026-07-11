"""Morning Rip (opening-range straddle) — TAPE-ACCURATE backtest on recorded prints.

Minute bars can't score this strategy: the 9:30 ET bar's median range (55pts) dwarfs
the 8/15/8 geometry, so >60% of outcomes are ambiguous. This walks actual trade
prints, so stop-entry, TP, and SL ordering is exact.

Mechanics per day: opening print at 9:30:00 ET -> buy-stop at open+off and
sell-stop at open-off (OCO). First print through a level fills it (with stop
slippage); then TP (resting limit, no slip) vs SL (stop-market, slip) — whichever
prints first. Costs: commission $0.62/side/contract, stop fills slip 2 ticks.

Runs against consolidated day files (trade_MNQ_YYYYMMDD.parquet).
    python morning_rip_tape.py --data-dir /Users/jg/mnq_l2_data
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

COMM = 0.62          # per side per contract
STOP_SLIP = 0.50     # 2 ticks on stop-triggered fills (entry + SL)
PV = 2.0             # $/pt/contract
OPEN_UTC = "13:30"   # 9:30 ET (EDT)
CUTOFF_UTC = "20:00"


def day_result(t: pd.DataFrame, off: float, tp: float, sl: float):
    ts = pd.to_datetime(t["recv_ns"], unit="ns", utc=True)
    day = ts.iloc[0].strftime("%Y-%m-%d")
    m = (ts.dt.strftime("%H:%M") >= OPEN_UTC) & (ts.dt.strftime("%H:%M") < CUTOFF_UTC)
    px = t.loc[m, "price"].to_numpy(float)
    if len(px) < 100:
        return None
    open_px = px[0]
    up, dn = open_px + off, open_px - off
    # entry: first print at/through either stop level
    hit_up = np.flatnonzero(px >= up)
    hit_dn = np.flatnonzero(px <= dn)
    if len(hit_up) == 0 and len(hit_dn) == 0:
        return {"day": day, "outcome": "no_fill", "pnl_pts": 0.0}
    iu = hit_up[0] if len(hit_up) else np.inf
    idn = hit_dn[0] if len(hit_dn) else np.inf
    if iu < idn:
        d, i0, entry = 1, int(iu), up + STOP_SLIP
    else:
        d, i0, entry = -1, int(idn), dn - STOP_SLIP
    tp_px, sl_px = entry + d * tp, entry - d * sl
    seg = px[i0:]
    tp_i = np.flatnonzero(d * (seg - tp_px) >= 0)
    sl_i = np.flatnonzero(d * (seg - sl_px) <= 0)
    ti = tp_i[0] if len(tp_i) else np.inf
    si = sl_i[0] if len(sl_i) else np.inf
    if ti == si == np.inf:                       # neither -> exit at last RTH print
        pnl = d * (seg[-1] - entry) - STOP_SLIP
        out = "eod"
    elif ti < si:
        pnl = tp                                  # resting limit, no slip
        out = "tp"
    else:
        pnl = -sl - STOP_SLIP                     # stop-market: extra slip
        out = "sl"
    return {"day": day, "outcome": out, "dir": d, "pnl_pts": float(pnl)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--contracts", type=int, default=2)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.data_dir, "trade_MNQ_*.parquet")))
    variants = [("preston 8/15/8", 8, 15, 8), ("scalp 5/10/5", 5, 10, 5),
                ("main 10/10/10", 10, 10, 10), ("12/18/9", 12, 18, 9)]
    print(f"{len(files)} day files | {args.contracts} contracts | tape-accurate")
    for name, off, tp, sl in variants:
        rows = []
        for f in files:
            t = pd.read_parquet(f)
            if len(t) < 1000:
                continue
            r = day_result(t.sort_values("recv_ns"), off, tp, sl)
            if r:
                rows.append(r)
        df = pd.DataFrame(rows)
        tr = df[df.outcome != "no_fill"]
        if not len(tr):
            continue
        gross = tr["pnl_pts"] * PV * args.contracts
        net = gross - 2 * COMM * args.contracts
        wr = (tr.outcome == "tp").mean()
        print(f"\n{name}: days={len(df)} filled={len(tr)} WR(tp)={wr:.0%} "
              f"net=${net.sum():,.0f} (${net.mean():+.0f}/day) "
              f"outcomes={tr.outcome.value_counts().to_dict()}")
        for r in tr.itertuples():
            print(f"    {r.day} {'L' if getattr(r,'dir',0)>0 else 'S'} {r.outcome:4s} {r.pnl_pts:+6.2f}pts")


if __name__ == "__main__":
    main()
