"""How many Sundays actually have a Globex reopen bar, and what does my
weekend leg do on the ones that do not?"""
import pandas as pd, numpy as np
ET = "America/New_York"
d = pd.read_parquet("/Users/jg/auction/data/mnq_1m_all.parquet").tz_convert(ET).sort_index()

days = sorted({x.date() for x in d.index})
sundays = [x for x in days if x.weekday() == 6]
print(f"calendar Sundays in tape span: {len(pd.date_range(d.index.min().date(), d.index.max().date(), freq='W-SUN'))}")
print(f"Sundays that appear as a date in the tape at all: {len(sundays)}")

# For every Sunday in the span, how far after 18:00 ET is the first available bar?
allsun = pd.date_range(d.index.min().date(), d.index.max().date(), freq="W-SUN")
gaps = []
for s in allsun:
    t = pd.Timestamp.combine(s.date(), pd.Timestamp("18:00").time()).tz_localize(ET)
    i = d.index.searchsorted(t, side="left")
    if i >= len(d):
        gaps.append((s.date(), np.nan)); continue
    gaps.append((s.date(), (d.index[i] - t).total_seconds() / 60.0))
g = pd.DataFrame(gaps, columns=["sunday", "mins_to_first_bar"])
print("\ndelay from Sun 18:00 ET to the first available bar:")
for lo, hi, lbl in [(-1, 2, "<=2 min  (real reopen)"), (2, 15, "2-15 min"),
                    (15, 90, "15-90 min  (ADMITTED by within=90)"),
                    (90, 1e9, ">90 min  (rejected)")]:
    n = ((g.mins_to_first_bar > lo) & (g.mins_to_first_bar <= hi)).sum()
    print(f"  {lbl:38s} {n:4d}")
print(f"  no bar at all                          {g.mins_to_first_bar.isna().sum():4d}")

print("\nSundays with a bar within 2 min of 18:00, by year:")
ok = g[g.mins_to_first_bar <= 2]
print(ok.groupby(pd.to_datetime(ok.sunday).dt.year).size().to_string())
print(f"\nTOTAL Sundays with a genuine 18:00 reopen: {len(ok)}")
print("ledger's recorded weekend n = 202")
