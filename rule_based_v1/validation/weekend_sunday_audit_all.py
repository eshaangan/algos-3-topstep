"""Does the len(bars)>100 guard damage the other instrument tapes too?"""
import pandas as pd, os, zoneinfo
ET = zoneinfo.ZoneInfo("America/New_York")
sun = pd.date_range("2020-01-01", "2026-07-09", freq="W-SUN")
hdr = ("dataset", "DST Sun present", "non-DST present", "non-DST missing")
print("%-20s %16s %16s %16s" % hdr)
for name in ("hist_1m24v", "hist_1m24_zn", "hist_1m24_mgc", "hist_1m24_mcl",
             "hist_1m24_m6e", "hist_1m24_mes", "hist_1m24v_sungap"):
    D = "/Users/jg/.svc-3hKye0/" + name
    if not os.path.isdir(D):
        continue
    hd = hs = ms = 0
    for s in sun:
        dst = bool(pd.Timestamp(s.date(), tz=ET).dst().total_seconds())
        ex = os.path.exists(os.path.join(D, "m_%s.parquet" % s.strftime("%Y%m%d")))
        if dst:
            hd += ex
        else:
            hs += ex
            ms += (not ex)
    print("%-20s %16d %16d %16d" % (name, hd, hs, ms))
