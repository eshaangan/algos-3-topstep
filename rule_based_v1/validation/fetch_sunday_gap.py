"""Re-fetch the winter-Sunday hour that fetch_mnq_vol.py silently discarded.

THE BUG BEING REPAIRED
----------------------
`fetch_mnq_vol.py` requests one UTC day at a time and keeps the result only if
`len(bars) > 100`. A Sunday UTC day contains only the Globex reopen:

    summer  Sun 00:00-24:00 UTC = Sat 20:00 -> Sun 20:00 EDT  -> 18:00-20:00 ET traded = ~121 bars  KEPT
    winter  Sun 00:00-24:00 UTC = Sat 19:00 -> Sun 19:00 EST  -> 18:00-19:00 ET traded = ~61 bars   DISCARDED

So every non-DST Sunday lost its 18:00-19:00 ET hour, which is exactly the
weekend_hold_v1 entry. 112 of 126 non-DST Sundays are missing outright.

This fetcher takes the same one-request-per-UTC-day path but keeps anything with
real bars, and writes to a STAGING directory so the existing 1,906-file set
cannot be corrupted. Merge only after validation.

Requests are serialised with a pause between them: Rithmic allows one historical
request per user at a time, and the account has hit ForcedLogout from concurrent
sessions before.
"""
import argparse
import asyncio
import os
import sys
import zoneinfo
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from async_rithmic import RithmicClient, SysInfraType, TimeBarType

ET = zoneinfo.ZoneInfo("America/New_York")
# instrument -> (raw dir, Rithmic symbol, exchange)
INSTRUMENTS = {
    "mnq": ("hist_1m24v", "MNQ", "CME"),
    "zn": ("hist_1m24_zn", "ZN", "CBOT"),
    "mgc": ("hist_1m24_mgc", "MGC", "COMEX"),
    "mcl": ("hist_1m24_mcl", "MCL", "NYMEX"),
    "m6e": ("hist_1m24_m6e", "M6E", "CME"),
    "mes": ("hist_1m24_mes", "MES", "CME"),
}
MIN_BARS = 30          # a winter Sunday legitimately has ~61; the old guard was 100


def missing_sundays(src: Path, start: str, end: str) -> list[datetime]:
    """UTC Sunday days whose file is absent from the existing set."""
    out = []
    for s in pd.date_range(start, end, freq="W-SUN"):
        f = src / f"m_{s:%Y%m%d}.parquet"
        if not f.exists():
            out.append(datetime(s.year, s.month, s.day, tzinfo=timezone.utc))
    return out


async def run(days, pause: float, dry: bool, stage: Path, symbol: str, exch: str) -> None:
    stage.mkdir(exist_ok=True)
    todo = [d for d in days if not (stage / f"m_{d:%Y%m%d}.parquet").exists()]
    print(f"[{symbol}] {len(days)} missing Sundays, {len(todo)} still to fetch", flush=True)
    if dry:
        print("DRY RUN - no connection made. First 5:",
              [f"{d:%Y-%m-%d}" for d in todo[:5]], flush=True)
        return
    if not todo:
        print("nothing to do", flush=True)
        return

    c = RithmicClient(
        user=os.environ["RITHMIC_USERNAME"],
        password=os.environ["RITHMIC_PASSWORD"],
        system_name=os.environ.get("RITHMIC_SYSTEM_NAME", "LucidTrading"),
        app_name=os.environ.get("RITHMIC_APP_NAME", "x"),
        app_version="1.0.0",
        url=os.environ.get("RITHMIC_GATEWAY_URI", "wss://rprotocol.rithmic.com:443"),
    )
    await c.connect(plants=[SysInfraType.HISTORY_PLANT])
    print("history plant connected", flush=True)
    ok = thin = err = 0
    try:
        for i, d in enumerate(todo, 1):
            try:
                bars = await c.get_historical_time_bars(
                    symbol, exch, d, d + timedelta(days=1), TimeBarType.MINUTE_BAR, 1)
                if len(bars) >= MIN_BARS:
                    pd.DataFrame([{
                        "ts": b["bar_end_datetime"], "open": b["open_price"],
                        "high": b["high_price"], "low": b["low_price"],
                        "close": b["close_price"], "vol": int(b.get("volume") or 0),
                    } for b in bars]).to_parquet(stage / f"m_{d:%Y%m%d}.parquet")
                    ok += 1
                else:
                    thin += 1
                    print(f"  {d:%Y-%m-%d} only {len(bars)} bars - not written", flush=True)
            except Exception as e:                      # noqa: BLE001
                err += 1
                print(f"  {d:%Y-%m-%d} ERR {str(e)[:60]}", flush=True)
            if i % 20 == 0:
                print(f"  {i}/{len(todo)}  ok={ok} thin={thin} err={err}", flush=True)
            await asyncio.sleep(pause)
    finally:
        await c.disconnect()
    print(f"DONE ok={ok} thin={thin} err={err}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-07-09")
    ap.add_argument("--pause", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="fetch at most N days")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--instrument", default="mnq", choices=sorted(INSTRUMENTS))
    a = ap.parse_args()
    raw, symbol, exch = INSTRUMENTS[a.instrument]
    src = Path.home() / ".svc-3hKye0" / raw
    stage = Path.home() / ".svc-3hKye0" / (raw + "_sungap")
    days = missing_sundays(src, a.start, a.end)
    if a.limit:
        days = days[:a.limit]
    asyncio.run(run(days, a.pause, a.dry, stage, symbol, exch))


if __name__ == "__main__":
    main()
