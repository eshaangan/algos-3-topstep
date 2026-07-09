"""Morning Rip tape backtest over the historical Databento print files
(Dec 2025 - May 2026, ~7GB plain CSV: ts_recv[ns], side, price[x1e-9], size).

Streams each file in chunks, keeps only the RTH window, groups prints by day,
and scores each day with the same exact-ordering logic as morning_rip_tape.py.

    python rule_based_v1/diagnostics/morning_rip_tape_hist.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FILES = [
    ROOT / "data/processed/mnq_trades_dec2025.csv.gz",
    ROOT / "data/processed/mnq_trades_jan_feb9_2026.csv.gz",
    ROOT / "data/processed/mnq_trades_mar_may_2026.csv.gz",
]
COMM, STOP_SLIP, PV = 0.62, 0.50, 2.0
OPEN_UTC, CUTOFF_UTC = "13:30", "20:00"   # NOTE: EDT; winter months open 14:30 UTC
VARIANTS = [("preston 8/15/8", 8, 15, 8), ("scalp 5/10/5", 5, 10, 5),
            ("main 10/10/10", 10, 10, 10), ("12/18/9", 12, 18, 9)]


def score_day(px: np.ndarray, off: float, tp: float, sl: float):
    open_px = px[0]
    up, dn = open_px + off, open_px - off
    hu = np.flatnonzero(px >= up)
    hd = np.flatnonzero(px <= dn)
    if len(hu) == 0 and len(hd) == 0:
        return "no_fill", 0.0
    iu = hu[0] if len(hu) else np.inf
    idn = hd[0] if len(hd) else np.inf
    d, i0, entry = (1, int(iu), up + STOP_SLIP) if iu < idn else (-1, int(idn), dn - STOP_SLIP)
    seg = px[i0:]
    ti_ = np.flatnonzero(d * (seg - (entry + d * tp)) >= 0)
    si_ = np.flatnonzero(d * (seg - (entry - d * sl)) <= 0)
    ti = ti_[0] if len(ti_) else np.inf
    si = si_[0] if len(si_) else np.inf
    if ti == si == np.inf:
        return "eod", float(d * (seg[-1] - entry) - STOP_SLIP)
    if ti < si:
        return "tp", float(tp)
    return "sl", float(-sl - STOP_SLIP)


def rth_open_utc(ts: pd.Series) -> pd.Series:
    """9:30 ET expressed in UTC varies with DST — convert properly."""
    et = ts.dt.tz_convert("America/New_York")
    hm = et.dt.hour * 60 + et.dt.minute
    return (hm >= 9 * 60 + 30) & (hm < 16 * 60)


def main() -> None:
    day_px: dict[str, list] = {}
    for f in FILES:
        if not f.exists():
            continue
        print(f"streaming {f.name}...", flush=True)
        for chunk in pd.read_csv(f, compression=None, chunksize=20_000_000,
                                 usecols=["ts_recv", "price"]):
            ts = pd.to_datetime(chunk["ts_recv"], unit="ns", utc=True)
            m = rth_open_utc(ts)
            if not m.any():
                continue
            sub = pd.DataFrame({"ts": ts[m], "px": chunk.loc[m, "price"] * 1e-9})
            for day, g in sub.groupby(sub["ts"].dt.strftime("%Y-%m-%d")):
                day_px.setdefault(day, []).append(g["px"].to_numpy())
    days = sorted(day_px)
    print(f"{len(days)} RTH days ({days[0]}..{days[-1]})", flush=True)

    per_var = {}
    for name, off, tp, sl in VARIANTS:
        rows = []
        for day in days:
            px = np.concatenate(day_px[day])
            if len(px) < 5000:
                continue
            out, pnl = score_day(px, off, tp, sl)
            rows.append({"day": day, "out": out, "pnl": pnl})
        df = pd.DataFrame(rows)
        tr = df[df.out != "no_fill"]
        net = tr["pnl"] * PV * 2 - 2 * COMM * 2         # 2 contracts
        wr = (tr.out == "tp").mean()
        mo = tr.assign(m=tr.day.str[:7], usd=net.values).groupby("m")["usd"].sum()
        pos = (mo > 0).mean()
        print(f"\n{name}: days={len(df)} filled={len(tr)} WR={wr:.0%} "
              f"net=${net.sum():>7,.0f} (${net.mean():+.0f}/day) posMonths={pos:.0%}")
        print("  " + "  ".join(f"{m}:{v:+,.0f}" for m, v in mo.items()))
        per_var[name] = net
    pd.DataFrame(per_var).to_csv("/tmp/morning_rip_hist_daily.csv")
    print("\nsaved daily $ to /tmp/morning_rip_hist_daily.csv")


if __name__ == "__main__":
    main()
