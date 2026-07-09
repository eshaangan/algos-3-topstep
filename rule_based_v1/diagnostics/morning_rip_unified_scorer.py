"""Score 8/15/8 Morning Rip per day from: (a) hist_1s Rithmic bars, (b) friend's
NQ tick csvs for our gap windows. Emits /tmp/rip_1s_daily.csv + /tmp/rip_gap_daily.csv."""
import glob, gzip, os
from pathlib import Path
import numpy as np, pandas as pd

OFF, TP, SL = 8.0, 15.0, 8.0
STOP_SLIP, COMM, PV, NC = 0.50, 0.62, 2.0, 2
GAPS = [("2026-02-10", "2026-02-28"), ("2026-05-14", "2026-06-11")]

def score_prints(px):
    open_px = px[0]; up, dn = open_px + OFF, open_px - OFF
    hu = np.flatnonzero(px >= up); hd = np.flatnonzero(px <= dn)
    if not len(hu) and not len(hd): return None
    iu = hu[0] if len(hu) else np.inf; idn = hd[0] if len(hd) else np.inf
    d, i0, entry = (1, int(iu), up + STOP_SLIP) if iu < idn else (-1, int(idn), dn - STOP_SLIP)
    seg = px[i0:]
    ti_ = np.flatnonzero(d * (seg - (entry + d * TP)) >= 0)
    si_ = np.flatnonzero(d * (seg - (entry - d * SL)) <= 0)
    ti = ti_[0] if len(ti_) else np.inf; si = si_[0] if len(si_) else np.inf
    if ti == si == np.inf: return float(d * (seg[-1] - entry) - STOP_SLIP)
    return float(TP) if ti < si else float(-SL - STOP_SLIP)

def score_1s(o, h, l, c):
    open_px = o[0]; up, dn = open_px + OFF, open_px - OFF
    ent = None
    for i in range(len(o)):
        hit_u, hit_d = h[i] >= up, l[i] <= dn
        if hit_u and hit_d:  # both stops in one second: conservative = adverse fill,
            return float(-SL - STOP_SLIP)   # entered then immediately stopped
        if hit_u: ent = (1, i, up + STOP_SLIP); break
        if hit_d: ent = (-1, i, dn - STOP_SLIP); break
    if ent is None: return None
    d, i0, entry = ent
    tp_px, sl_px = entry + d * TP, entry - d * SL
    for i in range(i0, len(o)):
        hit_sl = (l[i] <= sl_px) if d > 0 else (h[i] >= sl_px)
        hit_tp = (h[i] >= tp_px) if d > 0 else (l[i] <= tp_px)
        if hit_sl: return float(-SL - STOP_SLIP)   # both-hit -> stop (conservative)
        if hit_tp: return float(TP)
    return float(d * (c[-1] - entry) - STOP_SLIP)

def dollars(pts): return pts * PV * NC - 2 * COMM * NC

rows = []
for f in sorted(glob.glob(str(Path.home() / ".svc-3hKye0/hist_1s/bars1s_*.parquet"))):
    day = f.split("_")[-1][:8]
    b = pd.read_parquet(f)
    ts = pd.to_datetime(b["ts"], utc=True, format="mixed", errors="coerce")
    et = ts.dt.tz_convert("America/New_York")
    hm = et.dt.hour * 60 + et.dt.minute
    m = (hm >= 570) & (hm < 960)
    b = b[m]
    if len(b) < 3000: continue
    pts = score_1s(b["open"].to_numpy(), b["high"].to_numpy(), b["low"].to_numpy(), b["close"].to_numpy())
    if pts is not None:
        rows.append({"day": f"{day[:4]}-{day[4:6]}-{day[6:]}", "pnl": dollars(pts), "src": "1s"})
pd.DataFrame(rows).to_csv("/tmp/rip_1s_daily.csv", index=False)
print(f"1s days scored: {len(rows)}")

rows = []
for f in sorted(glob.glob(str(Path.home() / "trading-bot/ticks_ytd/NQ_*.csv.gz")) +
                sorted(glob.glob(str(Path.home() / "trading-bot/ticks/NQ_*.csv.gz")))):
    day = os.path.basename(f)[3:13]
    if not any(a <= day <= b_ for a, b_ in GAPS): continue
    with gzip.open(f, "rt") as fh:
        t = pd.read_csv(fh)
    ts = pd.to_datetime(t["ts"], unit="s", utc=True).dt.tz_convert("America/New_York")
    hm = ts.dt.hour * 60 + ts.dt.minute
    px = t.loc[(hm >= 570) & (hm < 960), "price"].to_numpy(float)
    if len(px) < 5000: continue
    pts = score_prints(px)
    if pts is not None:
        rows.append({"day": day, "pnl": dollars(pts), "src": "nqticks"})
pd.DataFrame(rows).to_csv("/tmp/rip_gap_daily.csv", index=False)
print(f"gap days scored: {len(rows)}")
