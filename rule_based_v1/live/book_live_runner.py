"""LIVE runner for the two-leg book: weekend_hold_v1 + fomc_drift_v1 (legal window).

THE FROZEN SPEC — this is the exact configuration the Monte Carlo scored
-----------------------------------------------------------------------
From `runs/disaster_stop_lucid_32.json`, best cell by P(pass) on LucidFlex 100k:

    stop        $600 per micro   (= 300.0 MNQ points; MNQ is $2/point)
    weekend     2 micros         Sun 18:00 ET -> Mon 15:59 ET
    fomc        2 micros         18:00 ET day-before-decision -> 13:55 ET decision day
    expected    $288.75/week
    worst trade -$1,205 (40% of the $3,000 MLL, inside the 60% tail budget)
    P(pass)     73.4% over 32 weeks, median 16 weeks

**Deviating from any of these numbers invalidates the 73.4%.** In particular the
stop is not optional garnish: it is what makes 2 micros tail-legal at all. The
unstopped worst weekend is -$1,117 per micro, i.e. -$2,234 at this size, which is
74% of the MLL on a single trade.

WHY ONE PROCESS AND NOT TWO RUNNERS
-----------------------------------
`cancel_all_orders` and `exit_position` are ACCOUNT-WIDE. Two independent runners
sharing one Rithmic account would cancel each other's protective stops -- the
weekend leg's stop would silently vanish the moment the FOMC leg closed a trade.
So both legs live in one process, one state file, and **only one position is ever
open at a time**: a leg that wants to enter while another is open logs and skips.

The legs do not overlap by design (weekend exits Monday 15:59; the earliest FOMC
entry is Monday 18:00), so the serialisation should never actually bind. It is
there because "should never" is not an execution guarantee.

SAFETY, inherited from the shakedown-hardened weekend runner
------------------------------------------------------------
* **Verified resting stop.** After the entry fills, a STOP_MARKET is submitted and
  then CONFIRMED to be resting by listing orders. If it is not there, the position
  is flattened immediately. This exists because this project has already observed
  a bracket template being silently ignored by the platform -- a stop you believe
  in but which is not resting is worse than no stop at all.
* **KILL file.** Touch `<out-dir>/KILL` and the runner flattens and exits.
* **Conservative startup recovery.** Any open position found at startup is
  flattened rather than adopted; the runner cannot know what protection it has.
* **Offline front-month roll.** No TICKER_PLANT: Rithmic permits one concurrent
  ticker session per login and `record_l2.py` holds it. Subscribing here forced
  the two processes into a reconnect loop that corrupted the quote stream.
* Fills are booked from notifications via `create_task`, never awaited inside the
  callback (that deadlocked an earlier runner).

Quotes come from `record_l2.py`'s stream file via RAW_DIR. No quote at the entry
instant means NO TRADE -- the runner logs and skips rather than entering blind.

    RAW_DIR=data/l2_raw python3 -m rule_based_v1.live.book_live_runner \\
        --out-dir logs/book_live --live
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

# ---- FROZEN CONFIG (see module docstring; do not tune) ----------------------
STOP_USD_PER_MICRO = 600.0
WEEKEND_QTY = 2
FOMC_QTY = 2
MAX_CONCURRENT_QTY = 2          # hard cap: one leg at a time, at its own size

SYMBOL, EXCHANGE = "MNQ", "CME"
ET = "America/New_York"
PV, COMM, TICK = 2.0, 0.62, 0.25
STOP_PTS = STOP_USD_PER_MICRO / PV      # 300.0 points

# FOMC decision dates, verified against federalreserve.gov/monetarypolicy/
# fomccalendars.htm on 2026-09-09. Decision day = the SECOND day of each meeting.
# Entry is 18:00 ET on the day BEFORE, which is after Lucid's 16:45 flatten and
# after the 18:00 reopen, so the hold is legal end to end.
FOMC_DECISION_DATES = [
    "2026-09-16", "2026-10-28", "2026-12-09",
    "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09",
    "2027-07-28", "2027-09-15", "2027-10-27", "2027-12-08",
]

ENTRY_WINDOW_MIN = 15           # how long after 18:00 an entry may still be taken
QUOTE_MAX_AGE_S = 120           # a staler quote than this is treated as no quote


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%m-%d %H:%M:%S}Z] {msg}", flush=True)


# ===========================================================================
# Leg schedule logic -- pure functions, unit-tested without a broker
# ===========================================================================

@dataclass(frozen=True)
class LegSignal:
    leg: str
    qty: int
    key: str


def weekend_signal(now: pd.Timestamp) -> LegSignal | None:
    """Sunday 18:00-18:15 ET. Keyed by the Monday it exits on."""
    if now.weekday() != 6:
        return None
    ent = now.normalize() + pd.Timedelta(hours=18)
    if ent <= now <= ent + pd.Timedelta(minutes=ENTRY_WINDOW_MIN):
        return LegSignal("weekend", WEEKEND_QTY,
                         f"wk-{(now + pd.Timedelta(days=1)).date()}")
    return None


def fomc_signal(now: pd.Timestamp, decisions: list[str]) -> LegSignal | None:
    """18:00-18:15 ET on the calendar day before a decision date."""
    nxt = str((now + pd.Timedelta(days=1)).date())
    if nxt not in decisions:
        return None
    ent = now.normalize() + pd.Timedelta(hours=18)
    if ent <= now <= ent + pd.Timedelta(minutes=ENTRY_WINDOW_MIN):
        return LegSignal("fomc", FOMC_QTY, f"fomc-{nxt}")
    return None


def due_exit(now: pd.Timestamp, open_trade: dict | None) -> str | None:
    """Is the open position at or past its scheduled exit? Returns a reason."""
    if not open_trade:
        return None
    leg, key = open_trade["leg"], open_trade["key"]
    if leg == "weekend":
        # exit Monday 15:59 ET, the validated close; Lucid force-flattens at 16:45
        if now.weekday() == 0 and key == f"wk-{now.date()}":
            if now >= now.normalize() + pd.Timedelta(hours=15, minutes=59):
                return "scheduled_exit_mon_1559"
        # safety net: if we ever wake up past Monday still holding, get flat
        if now.weekday() not in (6, 0):
            return "stale_weekend_position"
    elif leg == "fomc":
        if key == f"fomc-{now.date()}":
            # 13:55 ET, always flat before the 14:00 announcement
            if now >= now.normalize() + pd.Timedelta(hours=13, minutes=55):
                return "scheduled_exit_fomc_1355"
        elif key < f"fomc-{now.date()}":
            return "stale_fomc_position"
    return None


def stop_price(fill: float) -> float:
    """Protective stop, rounded to a tradeable tick."""
    return round((fill - STOP_PTS) / TICK) * TICK


def front_month(today: pd.Timestamp | None = None) -> str:
    """Front-month MNQ from the CME calendar. No network, no ticker plant.

    Quarterly H/M/U/Z expiring the third Friday, volume rolling ~8 days before.
    FADE_CONTRACT overrides for testing.
    """
    override = os.environ.get("FADE_CONTRACT")
    if override:
        return override
    d = (today or pd.Timestamp.now(tz=ET)).normalize()
    codes = {3: "H", 6: "M", 9: "U", 12: "Z"}

    def third_friday(y: int, m: int) -> pd.Timestamp:
        first = pd.Timestamp(year=y, month=m, day=1)
        return first + pd.Timedelta(days=(4 - first.weekday()) % 7 + 14)

    y, m = d.year, d.month
    naive = d.tz_localize(None) if d.tz is not None else d
    while True:
        qm = m + (-m % 3 if m % 3 else 0)
        if qm > 12:
            y, m = y + 1, 1
            continue
        if naive < third_friday(y, qm) - pd.Timedelta(days=8):
            return f"{SYMBOL}{codes[qm]}{y % 10}"
        m = qm + 1
        if m > 12:
            y, m = y + 1, 1


# ===========================================================================
# Runner
# ===========================================================================

class BookLive:
    def __init__(self, out_dir: str, live: bool,
                 decisions: list[str] | None = None) -> None:
        self.out, self.live = out_dir, live
        self.decisions = decisions or FOMC_DECISION_DATES
        os.makedirs(out_dir, exist_ok=True)
        self.state_p = os.path.join(out_dir, "state.json")
        self.ev_p = os.path.join(out_dir, "events.jsonl")
        self.kill_p = os.path.join(out_dir, "KILL")
        self.st = (json.load(open(self.state_p)) if os.path.exists(self.state_p)
                   else {"open": None, "done": []})
        self.client = self.acct = self.contract = None

    # ---- plumbing ---------------------------------------------------------
    def jlog(self, **rec) -> None:
        rec["t"] = str(pd.Timestamp.now(tz="UTC"))
        with open(self.ev_p, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    def save(self) -> None:
        json.dump(self.st, open(self.state_p, "w"), default=str)

    def quote(self) -> float | None:
        """Freshest mid from the recorder stream, or None if missing/stale.

        The stream is one-sided per line (bid OR ask, the other NaN), so both
        sides are forward-filled from the tail independently.
        """
        import math
        raw = os.environ.get("RAW_DIR", "")
        files = sorted(glob.glob(os.path.join(raw, "stream_MNQ_*.csv")))
        if not files:
            return None
        with open(files[-1], "rb") as fh:
            try:
                fh.seek(-65536, 2)
            except OSError:
                fh.seek(0)
            lines = fh.read().decode(errors="ignore").strip().splitlines()
        bid = ask = newest = None
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
            return None
        if time.time() - newest / 1e9 > QUOTE_MAX_AGE_S:
            return None
        mid = (bid + ask) / 2
        return mid if mid > 0 else None

    # ---- broker -----------------------------------------------------------
    async def connect(self) -> None:
        from async_rithmic import ReconnectionSettings, RithmicClient, SysInfraType
        self.client = RithmicClient(
            user=os.environ["RITHMIC_USERNAME"],
            password=os.environ["RITHMIC_PASSWORD"],
            system_name=os.environ.get("RITHMIC_SYSTEM_NAME", "LucidTrading"),
            app_name=os.environ.get("RITHMIC_APP_NAME", "x") + ":book",
            app_version="1.0.0",
            url=os.environ.get("RITHMIC_GATEWAY_URI",
                               "wss://rprotocol.rithmic.com:443"),
            reconnection_settings=ReconnectionSettings(
                max_retries=None, backoff_type="exponential",
                interval=2, max_delay=60))
        # NO TICKER_PLANT -- record_l2.py holds the single permitted ticker session.
        await self.client.connect(plants=[SysInfraType.ORDER_PLANT,
                                          SysInfraType.PNL_PLANT])
        self.acct = (await self.client.plants["order"].list_accounts())[0].account_id
        self.contract = front_month()
        self.client.on_exchange_order_notification += self._note
        log(f"connected acct={self.acct} contract={self.contract} "
            f"mode={'LIVE' if self.live else 'DRY'}")
        self.jlog(ev="connected", live=self.live, acct=self.acct,
                  contract=self.contract)

    async def _note(self, n) -> None:
        from async_rithmic import ExchangeOrderNotificationType as NT
        if getattr(n, "notify_type", None) != NT.FILL:
            return
        px, side = getattr(n, "fill_price", None), getattr(n, "transaction_type", None)
        self.jlog(ev="fill", px=px, side=side, tag=getattr(n, "user_tag", ""))
        log(f"FILL side={side} px={px}")
        ot = self.st.get("open")
        if not (ot and px):
            return
        if side == 1 and ot.get("fill") is None:              # entry BUY
            ot["fill"] = float(px)
            self.save()
        elif side == 2:                                        # any SELL = flat
            qty = ot["qty"]
            pnl = (float(px) - (ot.get("fill") or ot["ref"])) * PV * qty - 2 * COMM * qty
            self.jlog(ev="closed", leg=ot["leg"], exit_px=px, qty=qty,
                      pnl=round(pnl, 2))
            log(f"CLOSED {ot['leg']} @ {px} qty={qty} pnl=${pnl:+.2f}")
            self.st["done"].append(ot["key"])
            self.st["open"] = None
            self.save()
            asyncio.create_task(self._cancel_all())

    async def _cancel_all(self) -> None:
        try:
            await self.client.plants["order"].cancel_all_orders(account_id=self.acct)
        except Exception as exc:                              # noqa: BLE001
            log(f"cancel err {exc}")

    async def enter(self, sig: LegSignal, ref: float) -> None:
        from async_rithmic import OrderType, TransactionType
        oid = f"{sig.leg[:2]}-{uuid.uuid4().hex[:8]}"
        self.st["open"] = {"leg": sig.leg, "key": sig.key, "qty": sig.qty,
                           "oid": oid, "ref": ref, "fill": None,
                           "verified": False, "stp_sent": False,
                           "verify_after": time.time() + 5}
        self.save()
        if self.live:
            await self.client.plants["order"].submit_order(
                order_id=oid, symbol=self.contract, exchange=EXCHANGE,
                qty=sig.qty, transaction_type=TransactionType.BUY,
                order_type=OrderType.MARKET, account_id=self.acct)
            log(f"LIVE ENTRY {sig.leg} long {sig.qty} {self.contract} MKT oid={oid}")
        else:
            log(f"DRY would ENTER {sig.leg} long {sig.qty} ref~{ref}")
        self.jlog(ev="entry_order", leg=sig.leg, oid=oid, qty=sig.qty,
                  ref=ref, live=self.live)

    async def protect(self, ot: dict) -> None:
        from async_rithmic import OrderType, TransactionType
        px = stop_price(ot["fill"])
        if not self.live:
            ot["verified"] = True
            log(f"DRY would place stop @ {px} (${STOP_USD_PER_MICRO:.0f}/micro)")
            return
        try:
            await self.client.plants["order"].submit_order(
                order_id=f"{ot['oid']}-stp", symbol=self.contract,
                exchange=EXCHANGE, qty=ot["qty"],
                transaction_type=TransactionType.SELL,
                order_type=OrderType.STOP_MARKET, trigger_price=px,
                account_id=self.acct)
            ot["verify_after"] = time.time() + 4
            self.jlog(ev="stop_placed", px=px, qty=ot["qty"])
            log(f"STOP placed @ {px} qty={ot['qty']} — verifying")
        except Exception as exc:                              # noqa: BLE001
            log(f"CRITICAL protect fail {exc} — flatten")
            await self.flatten("protect_fail")

    async def verify(self, ot: dict) -> None:
        """Confirm the stop is actually resting. If not, get flat."""
        try:
            orders = await self.client.plants["order"].list_orders(
                account_id=self.acct)
        except Exception:                                     # noqa: BLE001
            ot["verify_after"] = time.time() + 6
            return
        if f"{ot['oid']}-stp" in {getattr(o, "user_tag", "") for o in orders}:
            ot["verified"], ot["verify_after"] = True, None
            log("STOP verified resting")
            self.jlog(ev="protection_verified", leg=ot["leg"])
        else:
            log("CRITICAL: stop NOT resting — flatten")
            await self.flatten("protection_not_resting")

    async def flatten(self, why: str) -> None:
        ot = self.st.get("open")
        self.jlog(ev="flatten", why=why, leg=(ot or {}).get("leg"), live=self.live)
        if self.live:
            await self._cancel_all()
            try:
                await self.client.plants["order"].exit_position(
                    account_id=self.acct, symbol=self.contract, exchange=EXCHANGE)
            except Exception as exc:                          # noqa: BLE001
                log(f"exit err {exc}")
            log(f"FLATTEN ({why})")
        else:
            if ot:
                self.st["done"].append(ot["key"])
            self.st["open"] = None
            self.save()
            log(f"DRY would FLATTEN ({why})")

    # ---- main loop --------------------------------------------------------
    async def run(self) -> None:
        await self.connect()
        if self.st.get("open"):
            # We cannot know what protection an inherited position has.
            log("startup with open state — flattening (conservative)")
            await self.flatten("startup_recovery")

        while True:
            now = pd.Timestamp.now(tz=ET)

            if os.path.exists(self.kill_p):
                await self.flatten("kill")
                self.save()
                log("KILL file present — exiting")
                return

            ot = self.st.get("open")

            # 1. protection state machine, before anything else
            if ot and ot.get("fill") and not ot.get("verified") and self.live \
                    and ot.get("verify_after") and time.time() >= ot["verify_after"]:
                if not ot.get("stp_sent"):
                    await self.protect(ot)
                    ot["stp_sent"] = True
                else:
                    await self.verify(ot)
                self.save()

            # 2. scheduled exits
            why = due_exit(now, ot)
            if why:
                await self.flatten(why)
                ot = None

            # 3. entries -- at most one position at a time
            if ot is None:
                for sig in (weekend_signal(now), fomc_signal(now, self.decisions)):
                    if sig is None or sig.key in self.st["done"]:
                        continue
                    if sig.qty > MAX_CONCURRENT_QTY:
                        log(f"REFUSING {sig.leg}: qty {sig.qty} over cap")
                        break
                    self.contract = front_month()      # may have rolled since start
                    ref = self.quote()
                    if ref is None:
                        log(f"NO FRESH QUOTE at {sig.leg} entry — is record_l2.py "
                            f"running and RAW_DIR set? SKIPPING (no blind entry)")
                        self.jlog(ev="entry_skipped", leg=sig.leg, why="no_quote")
                        break
                    await self.enter(sig, ref)
                    break

            self.save()
            await asyncio.sleep(20)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--live", action="store_true",
                    help="place REAL orders; without this the runner is dry-run")
    a = ap.parse_args()
    if a.live:
        log(f"*** LIVE *** weekend x{WEEKEND_QTY}, fomc x{FOMC_QTY}, "
            f"stop ${STOP_USD_PER_MICRO:.0f}/micro ({STOP_PTS:.0f} pts)")
    asyncio.run(BookLive(a.out_dir, a.live).run())


if __name__ == "__main__":
    main()
