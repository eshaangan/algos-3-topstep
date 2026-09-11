"""Reconcile the weekend leg: strict entry tolerance vs my 90-minute one."""
import pandas as pd, numpy as np, glob
ET = "America/New_York"; PV = 2.0; COST = 2.06; STOP_USD = 600.0

# 1. Does winter Sunday 18:00-19:00 ET data exist in the RAW files at all?
raw = sorted(glob.glob("/Users/jg/.svc-3hKye0/hist_1m24v/m_*.parquet"))
def raw_day(p):
    d = pd.read_parquet(p)
    ts = pd.DatetimeIndex(pd.to_datetime(d["ts"].values))
    return ts
winter_sun = [p for p in raw if p[-12:-8] in ("2020","2021","2022","2023","2024","2025")
              and p[-8:-6] in ("01","02","12")]
print("checking raw files for winter Sunday evening coverage...")
found = 0; checked = 0
for p in winter_sun[:400]:
    ts = raw_day(p)
    sun = ts[(ts.dayofweek == 6)]
    if len(sun) == 0: continue
    checked += 1
    if ((sun.hour == 17)).any(): found += 1
print(f"  winter files containing a Sunday: {checked}; of those with a 17:0x local bar: {found}")

# 2. The weekend leg under both tolerances
d = pd.read_parquet("/Users/jg/auction/data/mnq_1m_all.parquet").tz_convert(ET).sort_index()
def leg(tol_min):
    out = []
    allsun = pd.date_range(d.index.min().date(), d.index.max().date(), freq="W-SUN")
    for s in allsun:
        ent = pd.Timestamp.combine(s.date(), pd.Timestamp("18:00").time()).tz_localize(ET)
        ex  = pd.Timestamp.combine(s.date()+pd.Timedelta(days=1), pd.Timestamp("15:59").time()).tz_localize(ET)
        i = d.index.searchsorted(ent, side="left")
        if i >= len(d) or (d.index[i]-ent).total_seconds()/60 > tol_min: continue
        j = d.index.searchsorted(ex, side="right")-1
        if j <= i or (ex-d.index[j]).total_seconds()/60 > 90: continue
        seg = d.iloc[i:j+1]
        ep = float(seg["open"].iloc[0]); stop = ep - STOP_USD/PV
        hit = np.where(seg["low"].to_numpy() <= stop)[0]
        if len(hit):
            k = hit[0]; xp = min(stop, float(seg["open"].iloc[k]))
        else:
            xp = float(seg["close"].iloc[-1])
        out.append({"sunday": s.date(), "dst": bool(ent.dst().total_seconds()),
                    "pnl": (xp-ep)*PV*2 - COST*2})
    return pd.DataFrame(out)

print()
for tol, label in ((90, "tol=90min  (my rebuild)"), (2, "tol=2min  (strict reopen)")):
    r = leg(tol)
    p = r.pnl.to_numpy()
    t = p.mean()/(p.std(ddof=1)/np.sqrt(len(p)))
    print(f"{label:28s} n={len(p):3d}  mean=${p.mean():7.2f}  t={t:5.2f}  worst=${p.min():8.2f}")
r = leg(90)
print()
print("SPLIT of the tol=90 run by season:")
for dst, g in r.groupby("dst"):
    p = g.pnl.to_numpy()
    t = p.mean()/(p.std(ddof=1)/np.sqrt(len(p)))
    print(f"  {'summer/DST (real 18:00 entry)' if dst else 'winter (entry an hour late)':32s} "
          f"n={len(p):3d} mean=${p.mean():7.2f} t={t:5.2f}")
print()
print("ledger records: n=202, $126.60/micro stopped => $253.20 at 2 micros, worst -$603/micro")
