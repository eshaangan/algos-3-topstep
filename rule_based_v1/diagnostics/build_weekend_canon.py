"""Generate the canonical weekend_hold_v1 series (deterministic, committed).

Long MNQ from the first 1-min bar at/after 18:00 ET Sunday to the last bar
<= 16:00 ET Monday. Skips weekends whose Monday ends before 15:00 ET (holiday /
early close). Costs: $0.62/side/contract + 1 tick slip/side, 2 micros.
Timestamps are naive fixed UTC-5 (see research_mgc.load_1m).
    python rule_based_v1/diagnostics/build_weekend_canon.py
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from rule_based_v1.validation.research_mgc import load_1m

PV, NC, COMM, TICK = 2.0, 2, 0.62, 0.25
COST = 2 * COMM * NC + 2 * TICK * PV * NC

def main():
    b = load_1m("data/hist_1m24")
    rows = []
    for d0 in sorted({x for x in b.index.normalize().unique() if x.dayofweek == 6}):
        w = b.loc[d0 + pd.Timedelta(hours=17, minutes=58): d0 + pd.Timedelta(days=1, hours=16)]
        if len(w) < 600:
            continue
        ent = w[(w.index.hour == 18) & (w.index.minute <= 5) & (w.index.normalize() == d0)]
        if not len(ent):
            continue
        mon = w[w.index.normalize() == d0 + pd.Timedelta(days=1)]
        if not len(mon) or mon.index[-1].hour < 15:
            continue
        e = ent["open"].iloc[0]
        seg = w.loc[ent.index[0]:]
        rows.append({"day": (d0 + pd.Timedelta(days=1)).tz_localize(None),
                     "pnl": (mon["close"].iloc[-1] - e) * PV * NC - COST,
                     "mae": (seg["low"].min() - e) * PV * NC})
    d = pd.DataFrame(rows).set_index("day").sort_index()
    d.to_csv("data/processed/weekend_hold_canon.csv")
    u = d["pnl"]; t = u.mean() / (u.std(ddof=1) / np.sqrt(len(u)))
    print(f"n={len(u)} mean=${u.mean():+.2f}/wk t={t:+.2f} "
          f"posYrs={(u.groupby(u.index.year).sum() > 0).mean():.0%} worstMAE=${d['mae'].min():,.0f}")

if __name__ == "__main__":
    main()
