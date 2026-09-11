import pandas as pd, glob, time
t0=time.time()
f=sorted(glob.glob("/Users/jg/.svc-3hKye0/hist_1m24v/m_*.parquet"))
print(len(f),"files",flush=True)
d=pd.concat([pd.read_parquet(x) for x in f],ignore_index=True)
print("cols",list(d.columns),d.shape,flush=True)
ts=pd.DatetimeIndex(pd.to_datetime(d["ts"].values))
g=pd.Series(ts).diff().dt.total_seconds()/60
gi=pd.Series(pd.Series(ts)[(g>50)&(g<80)].values)
print("RAW halt-return hr winter:",gi[gi.dt.month.isin([1,2,12])].dt.hour.value_counts().head(2).to_dict(),flush=True)
print("RAW halt-return hr summer:",gi[gi.dt.month.isin([6,7,8])].dt.hour.value_counts().head(2).to_dict(),flush=True)
ts=ts.tz_localize("America/Chicago",ambiguous="NaT",nonexistent="NaT").tz_convert("America/New_York")
d=d.drop(columns=["ts"]).set_index(ts).sort_index()
d=d[~d.index.isna()]; d=d[~d.index.duplicated(keep="last")]; d.index.name="et"
g=d.index.to_series().diff().dt.total_seconds()/60
gi=pd.Series(g[(g>50)&(g<80)].index)
print("ET halt-return winter:",gi[gi.dt.month.isin([1,2,12])].dt.hour.value_counts().head(2).to_dict(),flush=True)
print("ET halt-return summer:",gi[gi.dt.month.isin([6,7,8])].dt.hour.value_counts().head(2).to_dict(),flush=True)
# RTH-open volume spike guard (ledger requirement): must be 09:30-09:33 ET both seasons
if "volume" in d.columns:
    for nm,mo in (("winter",[1,2,12]),("summer",[6,7,8])):
        s=d[d.index.month.isin(mo)]
        v=s.groupby([s.index.hour,s.index.minute])["volume"].sum()
        print(nm,"peak-volume minute ET:",v.idxmax(),flush=True)
d.to_parquet("/Users/jg/auction/data/mnq_1m_all.parquet")
print("WROTE",d.shape,d.index.min(),d.index.max(),f"{time.time()-t0:.1f}s",flush=True)
