"""Rebuild the live books two legs from raw MNQ, with the punitive stop fill.

weekend_hold_v1 : long Sun 18:00 ET -> Mon 15:59 ET
fomc_drift_v1   : long 18:00 ET day-before-FOMC -> 13:55 ET announcement day
Stop is $600/micro (300 MNQ pts). Gap-through fills at min(stop, bar OPEN), so
Sunday/Monday gaps are never assumed tradeable.
"""
import pandas as pd, numpy as np, json
PV=2.0; COST=2.06; STOP_USD=600.0

d=pd.read_parquet("/Users/jg/auction/data/mnq_1m_all.parquet")
ET="America/New_York"

def first_bar_after(ts, within=90):
    i=d.index.searchsorted(ts,side="left")
    if i>=len(d): return None
    if (d.index[i]-ts)>pd.Timedelta(minutes=within): return None
    return i

def last_bar_before(ts, within=90):
    i=d.index.searchsorted(ts,side="right")-1
    if i<0: return None
    if (ts-d.index[i])>pd.Timedelta(minutes=within): return None
    return i

def run(entry_ts, exit_ts, micros):
    """Long `micros` MNQ. Returns dict or None."""
    i=first_bar_after(entry_ts); j=last_bar_before(exit_ts)
    if i is None or j is None or j<=i: return None
    seg=d.iloc[i:j+1]
    ep=float(seg["open"].iloc[0])
    stop_px=ep-STOP_USD/PV
    hit=seg.index[seg["low"]<=stop_px]
    if len(hit):
        k=seg.index.get_loc(hit[0])
        fill=min(stop_px, float(seg["open"].iloc[k]))   # punitive: gap fills at the open
        xp, stopped, xt = fill, True, seg.index[k]
    else:
        xp, stopped, xt = float(seg["close"].iloc[-1]), False, seg.index[-1]
    mae=(float(seg["low"].min())-ep)*PV*micros
    return {"entry":seg.index[0].isoformat(),"exit":xt.isoformat(),
            "exit_day":xt.date().isoformat(),
            "pnl":round((xp-ep)*PV*micros - COST*micros,2),
            "stopped":bool(stopped),"mae":round(mae,2)}

# ---- weekend: every Sunday 18:00 ET in the tape
days=pd.Series(sorted({x.date() for x in d.index}))
sundays=[x for x in days if x.weekday()==6]
wk=[]
for s in sundays:
    ent=pd.Timestamp.combine(s,pd.Timestamp("18:00").time()).tz_localize(ET)
    ex =pd.Timestamp.combine(s+pd.Timedelta(days=1),pd.Timestamp("15:59").time()).tz_localize(ET)
    r=run(ent,ex,2)
    if r: r["leg"]="weekend"; wk.append(r)

# ---- fomc: 18:00 ET the prior session -> 13:55 ET announcement day
fo=[]
cal=pd.read_csv("/Users/jg/auction/data/fomc_announcements.csv")
dates=[pd.Timestamp(x).date() for x in cal["announcement_date"]]
dayset=set(days)
for a in dates:
    prior=a-pd.Timedelta(days=1)
    while prior not in dayset and (a-prior).days<5: prior-=pd.Timedelta(days=1)
    if prior not in dayset: continue
    ent=pd.Timestamp.combine(prior,pd.Timestamp("18:00").time()).tz_localize(ET)
    ex =pd.Timestamp.combine(a,pd.Timestamp("13:55").time()).tz_localize(ET)
    r=run(ent,ex,2)
    if r: r["leg"]="fomc"; fo.append(r)

def rep(name,rows):
    p=np.array([r["pnl"] for r in rows])
    t=p.mean()/(p.std(ddof=1)/np.sqrt(len(p)))
    print(f"{name}: n={len(p)} mean=${p.mean():.2f} t={t:.2f} worst=${p.min():.2f} "
          f"best=${p.max():.2f} stopped={np.mean([r[chr(34)+chr(34)] if False else r['stopped'] for r in rows]):.1%} WR={(p>0).mean():.1%}")
rep("weekend(2mic,$600 stop)",wk); rep("fomc(2mic,$600 stop)",fo)
json.dump({"weekend":wk,"fomc":fo},open("/Users/jg/auction/runs/legs_2mic_600.json","w"))
print("wrote runs/legs_2mic_600.json")
