"""Rebuild both legs on the REPAIRED tape and re-run the pass-probability MC."""
import numpy as np, pandas as pd, json
ET="America/New_York"; PV=2.0; COST=2.06; STOP_USD=600.0
d=pd.read_parquet("/Users/jg/auction/data/mnq_1m_all.parquet").tz_convert(ET).sort_index()

def trade(ent, ex, micros, entry_tol):
    i=d.index.searchsorted(ent,side="left")
    if i>=len(d) or (d.index[i]-ent).total_seconds()/60>entry_tol: return None
    j=d.index.searchsorted(ex,side="right")-1
    if j<=i or (ex-d.index[j]).total_seconds()/60>90: return None
    seg=d.iloc[i:j+1]; ep=float(seg["open"].iloc[0]); stop=ep-STOP_USD/PV
    hit=np.where(seg["low"].to_numpy()<=stop)[0]
    xp = min(stop,float(seg["open"].iloc[hit[0]])) if len(hit) else float(seg["close"].iloc[-1])
    return {"day":seg.index[0].date(), "pnl":(xp-ep)*PV*micros-COST*micros}

wk=[]
for s in pd.date_range(d.index.min().date(), d.index.max().date(), freq="W-SUN"):
    t=trade(pd.Timestamp.combine(s.date(),pd.Timestamp("18:00").time()).tz_localize(ET),
            pd.Timestamp.combine(s.date()+pd.Timedelta(days=1),pd.Timestamp("15:59").time()).tz_localize(ET),
            2, 2)
    if t: wk.append(t)
cal=pd.read_csv("/Users/jg/auction/data/fomc_announcements.csv")
dayset={x.date() for x in d.index}
fo=[]
for a in [pd.Timestamp(x).date() for x in cal["announcement_date"]]:
    p=a-pd.Timedelta(days=1)
    while p not in dayset and (a-p).days<5: p-=pd.Timedelta(days=1)
    if p not in dayset: continue
    t=trade(pd.Timestamp.combine(p,pd.Timestamp("18:00").time()).tz_localize(ET),
            pd.Timestamp.combine(a,pd.Timestamp("13:55").time()).tz_localize(ET), 2, 90)
    if t: fo.append(t)

w=np.array([x["pnl"] for x in wk]); f=np.array([x["pnl"] for x in fo])
print(f"weekend n={len(w)} mean=${w.mean():.2f} t={w.mean()/(w.std(ddof=1)/np.sqrt(len(w))):.2f} worst=${w.min():.2f}")
print(f"fomc    n={len(f)} mean=${f.mean():.2f} t={f.mean()/(f.std(ddof=1)/np.sqrt(len(f))):.2f} worst=${f.min():.2f}")
worst=sorted(wk,key=lambda x:x["pnl"])[:3]
print("worst 3 weekends:", [(str(x['day']), round(x['pnl'])) for x in worst])

TARGET=6000.; MLL=3000.; rng=np.random.default_rng(7)
def mc(weeks,n=40000):
    p_f=8/52.; passed=busted=0; wtp=[]
    for i in range(n):
        eq=0.;peak=0.;floor=-MLL;dead=False
        for k in range(weeks):
            tr=[w[rng.integers(len(w))]]
            if rng.random()<p_f: tr.append(f[rng.integers(len(f))])
            for t in tr:
                eq+=t
                if eq<=floor: busted+=1; dead=True; break
            if dead: break
            peak=max(peak,eq); floor=max(floor,min(peak-MLL,0.))
            if eq>=TARGET: passed+=1; wtp.append(k+1); break
    return passed/n, busted/n, (float(np.median(wtp)) if wtp else None)
print()
print("P(pass), repaired tape, weekend x2 + fomc x2, $600 stop:")
for wks in (16,32,52):
    p,b,m=mc(wks); print(f"  {wks:>2} weeks: p_pass={p:.3f}  bust={b:.3f}  median={m}")
print()
print("for comparison the frozen spec claims 50.4% @16wk and 73.4% @32wk")
