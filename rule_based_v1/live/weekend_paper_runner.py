"""Paper-trade weekend_hold_v1: long MNQ Sun 18:00 -> Mon 16:00 ET. Stream quotes only."""
import glob, json, os, time
from datetime import datetime, timezone
import pandas as pd
NC, PV, COMM, TICK = 2, 2.0, 0.62, 0.25
ET = "America/New_York"
def log(m): print(f"[{datetime.now(timezone.utc):%m-%d %H:%M:%S}Z] {m}", flush=True)
def quote(raw):
    # recorder stream is ONE-SIDED BBO per line (bid OR ask; other NaN). Forward-fill
    # most-recent finite bid AND ask over a 64KB tail. (fixed 2026-07-13: NaN-at-reopen)
    import math
    fs = sorted(glob.glob(os.path.join(raw, "stream_MNQ_*.csv")))
    if not fs: return None, 1e9
    with open(fs[-1], "rb") as f:
        try: f.seek(-65536, 2)
        except OSError: f.seek(0)
        lines = f.read().decode(errors="ignore").strip().splitlines()
    bid = ask = newest = None
    for line in reversed(lines):
        p = line.split(",")
        if len(p) != 5: continue
        try: ns, b, a = float(p[0]), float(p[1]), float(p[3])
        except ValueError: continue
        if newest is None: newest = ns
        if bid is None and math.isfinite(b): bid = b
        if ask is None and math.isfinite(a): ask = a
        if bid is not None and ask is not None: break
    if bid is None or ask is None or newest is None: return None, 1e9
    mid = (bid + ask) / 2
    if mid <= 0: return None, 1e9
    return mid, time.time() - newest / 1e9
def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--raw-dir", required=True); ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    sp = os.path.join(a.out_dir, "wk_state.json"); tp = os.path.join(a.out_dir, "wk_trades.jsonl")
    st = json.load(open(sp)) if os.path.exists(sp) else {"open": None, "done": []}
    log("weekend paper runner up (Sun 18:00 -> Mon 16:00)")
    while True:
        now = pd.Timestamp.now(tz=ET)
        if now.weekday() == 6:                              # Sunday: enter 18:00-18:10
            key = str((now + pd.Timedelta(days=1)).date())  # keyed by Monday
            ent = now.normalize() + pd.Timedelta(hours=18)
            if key not in st["done"] and st["open"] is None and ent <= now <= ent + pd.Timedelta(minutes=10):
                mid, age = quote(a.raw_dir)
                if mid and age < 300:
                    st["open"] = {"key": key, "entry": mid}
                    open(tp, "a").write(json.dumps({"ev": "entry", "key": key, "px": mid}) + "\n")
                    log(f"PAPER ENTRY long 2 MNQ @ {mid:.2f} (weekend hold)")
        if now.weekday() == 0 and st["open"] and st["open"]["key"] == str(now.date()):
            ext = now.normalize() + pd.Timedelta(hours=15, minutes=59)
            if now >= ext:
                mid, age = quote(a.raw_dir)
                if mid:
                    pnl = (mid - st["open"]["entry"])*PV*NC - 2*COMM*NC - 2*TICK*PV*NC
                    open(tp, "a").write(json.dumps({"ev": "exit", "key": st["open"]["key"], "px": mid, "pnl": round(pnl, 2)}) + "\n")
                    log(f"PAPER EXIT @ {mid:.2f} pnl ${pnl:+.2f}")
                    st["done"].append(st["open"]["key"]); st["open"] = None
        json.dump(st, open(sp, "w")); time.sleep(60)
if __name__ == "__main__": main()
