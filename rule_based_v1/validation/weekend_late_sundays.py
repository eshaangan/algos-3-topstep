import pandas as pd, numpy as np
ET = "America/New_York"
d = pd.read_parquet("/Users/jg/auction/data/mnq_1m_all.parquet").tz_convert(ET).sort_index()
allsun = pd.date_range(d.index.min().date(), d.index.max().date(), freq="W-SUN")
rows = []
for s in allsun:
    t = pd.Timestamp.combine(s.date(), pd.Timestamp("18:00").time()).tz_localize(ET)
    i = d.index.searchsorted(t, side="left")
    if i >= len(d): continue
    first = d.index[i]
    rows.append({"sunday": s.date(), "delay_min": (first - t).total_seconds()/60.0,
                 "first_bar_et": first, "month": s.month, "year": s.year,
                 "dst": bool(t.dst().total_seconds())})
g = pd.DataFrame(rows)
late = g[g.delay_min > 2]
print(f"late Sundays: {len(late)}")
print("\ndelay distribution (minutes):")
print(late.delay_min.value_counts().head(12).to_string())
print("\nlate Sundays by DST flag:")
print(g.assign(late=g.delay_min > 2).groupby(["dst","late"]).size().to_string())
print("\nfirst-bar clock time on late Sundays (ET):")
print(late.first_bar_et.dt.strftime("%H:%M").value_counts().head(8).to_string())
print("\nlate Sundays by year:")
print(late.groupby("year").size().to_string())
print("\nsample of late Sundays:")
print(late[["sunday","delay_min","first_bar_et","dst"]].head(10).to_string(index=False))
