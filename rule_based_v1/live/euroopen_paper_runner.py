"""Paper euro_open_v1: long MNQ 2:00->5:00 ET when prior RTH day down & high-range."""
import glob, json, os, time
from datetime import datetime, timezone
import pandas as pd
NC, PV, COMM, TICK = 2, 2.0, 0.62, 0.25
ET = "America/New_York"
def log(m): print(f"[{datetime.now(timezone.utc):%m-%d %H:%M:%S}Z] {m}", flush=True)
def quote(raw):
    fs = sorted(glob.glob(os.path.join(raw, "stream_MNQ_*.csv")))
    if not fs: return None, 1e9
    with open(fs[-1], "rb") as f:
        try: f.seek(-300, 2)
        except OSError: pass
        for line in reversed(f.read().decode(errors="ignore").strip().splitlines()):
            p = line.split(",")
            if len(p) == 5:
                try: return (float(p[1])+float(p[3]))/2, time.time()-float(p[0])/1e9
                except ValueError: continue
    return None, 1e9
def prior_day_state(raw):
    """From yesterday's stream file: RTH ret + range vs nothing fancy (approx via stream mids)."""
    fs = sorted(glob.glob(os.path.join(raw, "stream_MNQ_*.csv")))
    if len(fs) < 2: return None
    rows = []
    for f in fs[-3:]:
        day = f.split("_")[-1][:8]
        try:
            df = pd.read_csv(f, header=None, names=["ns","bp","bs","ap","asz"], usecols=[0,1,3])
        except Exception: continue
        ts = pd.to_datetime(df["ns"], unit="ns", utc=True).dt.tz_convert(ET)
        mid = (df["bp"]+df["ap"])/2
        m = (ts.dt.hour*60+ts.dt.minute >= 570) & (ts.dt.hour*60+ts.dt.minute < 960)
        if m.sum() < 1000: continue
        v = mid[m]
        rows.append({"day": day, "ret": v.iloc[-1]-v.iloc[0], "rng": v.max()-v.min()})
    if not rows: return None
    d = pd.DataFrame(rows).drop_duplicates("day").sort_values("day")
    last = d.iloc[-1]
    med = d["rng"].median()
    return {"down": last["ret"] < 0, "hirange": last["rng"] > med, "day": last["day"]}
def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--raw-dir", required=True); ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    sp = os.path.join(a.out_dir, "eo_state.json"); tp = os.path.join(a.out_dir, "eo_trades.jsonl")
    st = json.load(open(sp)) if os.path.exists(sp) else {"open": None, "done": []}
    log("euro-open paper runner up (2:00->5:00 ET, conditional)")
    while True:
        now = pd.Timestamp.now(tz=ET); key = str(now.date())
        ent = now.normalize() + pd.Timedelta(hours=2)
        ext = now.normalize() + pd.Timedelta(hours=5)
        if now.weekday() < 5 and key not in st["done"] and st["open"] is None and ent <= now <= ent + pd.Timedelta(minutes=10):
            cond = prior_day_state(a.raw_dir)
            if cond and cond["down"] and cond["hirange"]:
                mid, age = quote(a.raw_dir)
                if mid and age < 300:
                    st["open"] = {"key": key, "entry": mid}
                    open(tp, "a").write(json.dumps({"ev": "entry", "key": key, "px": mid, "cond": cond}) + "\n")
                    log(f"PAPER ENTRY long 2 MNQ @ {mid:.2f} (euro-open, prior down+range)")
            elif cond:
                if key not in st["done"]:
                    st["done"].append(key)   # condition not met: no trade today
        if st["open"] and st["open"]["key"] == key and now >= ext:
            mid, age = quote(a.raw_dir)
            if mid:
                pnl = (mid - st["open"]["entry"])*PV*NC - 2*COMM*NC - 2*TICK*PV*NC
                open(tp, "a").write(json.dumps({"ev": "exit", "key": key, "px": mid, "pnl": round(pnl, 2)}) + "\n")
                log(f"PAPER EXIT @ {mid:.2f} pnl ${pnl:+.2f}")
                st["done"].append(key); st["open"] = None
        json.dump(st, open(sp, "w")); time.sleep(120)
if __name__ == "__main__": main()
