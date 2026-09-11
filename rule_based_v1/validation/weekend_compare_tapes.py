"""Are the two MNQ tapes the same tape?"""
import pandas as pd, numpy as np
ET = "America/New_York"

def norm(p, label):
    d = pd.read_parquet(p)
    print(f"\n--- {label}: {p.split('/')[-1]} ---")
    print("  cols", list(d.columns)[:8], " rows", f"{len(d):,}")
    if d.index.tz is None:
        if "ts" in d.columns:
            idx = pd.DatetimeIndex(pd.to_datetime(d["ts"].values))
            print("  index naive, has ts column")
        else:
            idx = pd.DatetimeIndex(d.index)
            print("  index naive")
        # probe: CME halt returns 17:01 local if Chicago-stamped
        g = pd.Series(idx).diff().dt.total_seconds()/60
        at = pd.Series(pd.Series(idx)[(g > 50) & (g < 80)].values)
        print("  RAW halt-return hr winter", at[at.dt.month.isin([1,2,12])].dt.hour.value_counts().head(1).to_dict(),
              "summer", at[at.dt.month.isin([6,7,8])].dt.hour.value_counts().head(1).to_dict())
        idx = idx.tz_localize("America/Chicago", ambiguous="NaT", nonexistent="NaT").tz_convert(ET)
        d = d.set_index(idx)
    else:
        d = d.tz_convert(ET)
        print("  index already tz-aware:", d.index.tz)
    d = d[~d.index.isna()].sort_index()
    d = d[~d.index.duplicated(keep="last")]
    g = d.index.to_series().diff().dt.total_seconds()/60
    at = pd.Series(g[(g > 50) & (g < 80)].index)
    print("  ET halt-return hr winter", at[at.dt.month.isin([1,2,12])].dt.hour.value_counts().head(1).to_dict(),
          "summer", at[at.dt.month.isin([6,7,8])].dt.hour.value_counts().head(1).to_dict())
    print("  span", d.index.min(), "->", d.index.max())
    return d

a = norm("/Users/jg/auction/data/mnq_1m_all.parquet", "MINE (rebuilt from hist_1m24v)")
b = norm("/Users/jg/auction/data/mnq_1m_all_LEDGER.parquet", "LEDGER (data/processed)")

print("\n=== SUNDAY 18:00-18:05 ET BARS PER YEAR (the weekend entry) ===")
def sun(d):
    s = d[(d.index.dayofweek == 6) & (d.index.hour == 18) & (d.index.minute <= 5)]
    return s.groupby(s.index.year).size()
sa, sb = sun(a), sun(b)
print(pd.DataFrame({"mine": sa, "ledger": sb}).fillna(0).astype(int).to_string())
print("\ntotal distinct Sundays with an 18:0x bar:  mine",
      len({d.date() for d in a[(a.index.dayofweek==6)&(a.index.hour==18)&(a.index.minute<=5)].index}),
      " ledger",
      len({d.date() for d in b[(b.index.dayofweek==6)&(b.index.hour==18)&(b.index.minute<=5)].index}))
