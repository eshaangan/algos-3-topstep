"""weekend_hold_v1 on the REPAIRED tape - winter weekends included for the first time.

The 202 weekends this strategy was validated on were all DST-period, because the
fetcher's len(bars)>100 guard discarded every non-DST Sunday's reopen hour. Those
winter weekends are now fetched and merged, so they are a genuine out-of-sample
season: never seen by any prior test, tuning or selection.
"""
import numpy as np, pandas as pd
ET = "America/New_York"; PV = 2.0; COST = 2.06; STOP_USD = 600.0; MICROS = 2

d = pd.read_parquet("/Users/jg/auction/data/mnq_1m_all.parquet").tz_convert(ET).sort_index()

rows = []
for s in pd.date_range(d.index.min().date(), d.index.max().date(), freq="W-SUN"):
    ent = pd.Timestamp.combine(s.date(), pd.Timestamp("18:00").time()).tz_localize(ET)
    ex = pd.Timestamp.combine(s.date() + pd.Timedelta(days=1), pd.Timestamp("15:59").time()).tz_localize(ET)
    i = d.index.searchsorted(ent, side="left")
    if i >= len(d) or (d.index[i] - ent).total_seconds() / 60 > 2:      # strict reopen
        continue
    j = d.index.searchsorted(ex, side="right") - 1
    if j <= i or (ex - d.index[j]).total_seconds() / 60 > 90:
        continue
    seg = d.iloc[i:j + 1]
    ep = float(seg["open"].iloc[0]); stop = ep - STOP_USD / PV
    hit = np.where(seg["low"].to_numpy() <= stop)[0]
    if len(hit):
        k = hit[0]; xp = min(stop, float(seg["open"].iloc[k])); stopped = True
    else:
        xp = float(seg["close"].iloc[-1]); stopped = False
    rows.append({"sunday": s.date(), "year": s.year,
                 "dst": bool(ent.dst().total_seconds()),
                 "pnl": (xp - ep) * PV * MICROS - COST * MICROS, "stopped": stopped})
r = pd.DataFrame(rows)

def stat(p, label, width=34):
    p = np.asarray(p, float)
    if len(p) < 2:
        print(f"  {label:<{width}} n={len(p):3d}  (too few)"); return
    t = p.mean() / (p.std(ddof=1) / np.sqrt(len(p)))
    print(f"  {label:<{width}} n={len(p):3d}  mean=${p.mean():8.2f}  t={t:6.2f}  "
          f"WR={100*(p>0).mean():5.1f}%  worst=${p.min():9.2f}")

print("=== weekend_hold_v1, 2 micros, $600/micro stop, REPAIRED TAPE ===")
stat(r.pnl, "ALL weekends")
print()
print("--- the split that matters ---")
stat(r.loc[r.dst, "pnl"],  "DST (the validated sample)")
stat(r.loc[~r.dst, "pnl"], "non-DST (NEW, never tested)")
print()
print("--- per year ---")
for y, g in r.groupby("year"):
    stat(g.pnl, str(y))
print()
print("--- the ledger's stress cuts, re-run on the full tape ---")
stat(r.loc[~r.year.isin([2026]), "pnl"], "excl 2026 (partial year)")
stat(r.loc[~r.year.isin([2020, 2026]), "pnl"], "excl 2020 + 2026")
stat(r.loc[r.year.isin([2021, 2022, 2023, 2024]), "pnl"], "2021-2024 only")
print()
print(f"stopped out: {100*r.stopped.mean():.1f}%   "
      f"top-20 share of total P/L: {100*r.pnl.nlargest(20).sum()/r.pnl.sum():.0f}%")
r.to_csv("/Users/jg/auction/runs/weekend_repaired.csv", index=False)
