import pandas as pd, numpy as np
ET="America/New_York"
d=pd.read_parquet("/Users/jg/auction/data/mnq_1m_all.parquet").tz_convert(ET).sort_index()
sun=d[d.index.dayofweek==6]
sun=sun.assign(dst=[bool(x.dst().total_seconds()) for x in sun.index], hr=sun.index.hour)
print("Sunday bars per session-hour, ET, by season (mean bars per Sunday):")
n_dst=len({x.date() for x in sun.index[sun.dst]}); n_std=len({x.date() for x in sun.index[~sun.dst]})
print(f"  DST Sundays={n_dst}  non-DST Sundays={n_std}")
print(f"{'hour ET':>8} {'DST':>10} {'non-DST':>10}")
for h in (17,18,19,20,21,22,23):
    a=(sun.dst&(sun.hr==h)).sum()/max(n_dst,1)
    b=((~sun.dst)&(sun.hr==h)).sum()/max(n_std,1)
    print(f"{h:>8} {a:>10.1f} {b:>10.1f}")
print()
print("=> a HOLE shows as ~0 bars in that hour; a SHIFT shows the bars moved to the next hour")
tot_dst=sun.dst.sum()/max(n_dst,1); tot_std=(~sun.dst).sum()/max(n_std,1)
print(f"total Sunday bars per session: DST {tot_dst:.0f}   non-DST {tot_std:.0f}   diff {tot_dst-tot_std:.0f}")
