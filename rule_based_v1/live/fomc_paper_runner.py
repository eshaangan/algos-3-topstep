"""Paper-trade fomc_drift_v1 (registered + passed, commit 288dbec). NO real orders.

Spec (FROZEN): long MNQ at 14:00 ET the day before each FOMC decision; exit
13:55 ET decision day (always flat before the 14:00 announcement). 2 micros.
Quotes come from the recorder's live BBO stream file — no Rithmic connection.

Calendar verified vs federalreserve.gov 2026-07-09. Extend WINDOWS when the
Fed publishes 2027 dates.

    python fomc_paper_runner.py --raw-dir <l2_raw> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from datetime import datetime, timezone

import pandas as pd

# (entry day 14:00 ET, decision day 13:55 ET) — official Fed calendar
EVENTS = [("2026-07-28", "2026-07-29"),
          ("2026-09-15", "2026-09-16"),
          ("2026-10-27", "2026-10-28"),
          ("2026-12-08", "2026-12-09")]
NC, PV, COMM, TICK = 2, 2.0, 0.62, 0.25
ET = "America/New_York"


def log(msg):
    print(f"[{datetime.now(timezone.utc):%m-%d %H:%M:%S}Z] {msg}", flush=True)


def last_quote(raw_dir: str):
    """Mid + staleness from the newest stream csv line."""
    day = time.strftime("%Y%m%d", time.gmtime())
    files = sorted(glob.glob(os.path.join(raw_dir, f"stream_MNQ_*.csv")))
    if not files:
        return None, 1e9
    import math
    with open(files[-1], "rb") as f:
        try:
            f.seek(-65536, 2)
        except OSError:
            f.seek(0)
        lines = f.read().decode(errors="ignore").strip().splitlines()
    bid = ask = newest = None                       # one-sided BBO -> forward-fill both sides
    for line in reversed(lines):
        p = line.split(",")
        if len(p) != 5:
            continue
        try:
            ns, b, a = float(p[0]), float(p[1]), float(p[3])
        except ValueError:
            continue
        if newest is None:
            newest = ns
        if bid is None and math.isfinite(b):
            bid = b
        if ask is None and math.isfinite(a):
            ask = a
        if bid is not None and ask is not None:
            break
    if bid is None or ask is None or newest is None:
        return None, 1e9
    mid = (bid + ask) / 2
    if mid <= 0:
        return None, 1e9
    return mid, time.time() - newest / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    state_p = os.path.join(args.out_dir, "fomc_state.json")
    trades_p = os.path.join(args.out_dir, "fomc_trades.jsonl")
    st = json.load(open(state_p)) if os.path.exists(state_p) else {"open": None, "done": []}
    log(f"fomc paper runner up — {len(EVENTS)} scheduled events")

    while True:
        now = pd.Timestamp.now(tz=ET)
        for ent_d, dec_d in EVENTS:
            key = dec_d
            if key in st["done"]:
                continue
            ent_t = pd.Timestamp(f"{ent_d} 14:00", tz=ET)
            exit_t = pd.Timestamp(f"{dec_d} 13:55", tz=ET)
            # entry window: 14:00-14:10 on entry day
            if st["open"] is None and ent_t <= now <= ent_t + pd.Timedelta(minutes=10):
                mid, age = last_quote(args.raw_dir)
                if mid is None or age > 180:
                    log(f"WARN no fresh quote (age={age:.0f}s) — retrying")
                else:
                    st["open"] = {"event": key, "entry_px": mid, "entry_t": str(now)}
                    rec = {**st["open"], "ev": "entry", "quote_age_s": round(age, 1)}
                    open(trades_p, "a").write(json.dumps(rec) + "\n")
                    log(f"PAPER ENTRY long 2 MNQ @ {mid:.2f} (FOMC {key})")
            # exit window
            if st["open"] and st["open"]["event"] == key and now >= exit_t:
                mid, age = last_quote(args.raw_dir)
                if mid is None or age > 300:
                    log(f"WARN stale exit quote (age={age:.0f}s) — using anyway")
                    mid = mid or st["open"]["entry_px"]
                pnl = (mid - st["open"]["entry_px"]) * PV * NC - 2 * COMM * NC - 2 * TICK * PV * NC
                rec = {"ev": "exit", "event": key, "exit_px": mid,
                       "entry_px": st["open"]["entry_px"], "pnl": round(pnl, 2)}
                open(trades_p, "a").write(json.dumps(rec) + "\n")
                log(f"PAPER EXIT @ {mid:.2f} -> pnl ${pnl:+.2f} (FOMC {key})")
                st["done"].append(key)
                st["open"] = None
        json.dump(st, open(state_p, "w"))
        time.sleep(60)


if __name__ == "__main__":
    main()
