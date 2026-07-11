"""LIVE weekend_hold_v1 — 1 micro, DRY-RUN unless --live. Sun 18:00 entry -> Mon 16:00 exit.
Catastrophic stop only (-450 pts, beyond p95 MAE): account protection, not strategy.
Reuses the shakedown-hardened patterns: verified resting stop, fills booked via
notifications (create_task, never await in callbacks), KILL file, supervisor-ready."""
from __future__ import annotations
import asyncio, glob, json, os, time, uuid
from datetime import datetime, timezone
import pandas as pd

QTY = 1                      # HARD CAP: ladder rung one
CAT_STOP_PTS = 450.0
SYMBOL, EXCHANGE = "MNQ", "CME"
ET = "America/New_York"
PV, COMM, TICK = 2.0, 0.62, 0.25

def log(m): print(f"[{datetime.now(timezone.utc):%m-%d %H:%M:%S}Z] {m}", flush=True)

class WkLive:
    def __init__(self, out_dir, live):
        self.out, self.live = out_dir, live
        os.makedirs(out_dir, exist_ok=True)
        self.state_p = os.path.join(out_dir, "state.json")
        self.ev_p = os.path.join(out_dir, "events.jsonl")
        self.kill_p = os.path.join(out_dir, "KILL")
        self.st = json.load(open(self.state_p)) if os.path.exists(self.state_p) else {"open": None, "done": []}
        self.client = None; self.acct = None; self.contract = None

    def jlog(self, **rec):
        rec["t"] = str(pd.Timestamp.now(tz="UTC"))
        open(self.ev_p, "a").write(json.dumps(rec, default=str) + "\n")

    def save(self):
        json.dump(self.st, open(self.state_p, "w"), default=str)

    async def connect(self):
        from async_rithmic import RithmicClient, ReconnectionSettings, SysInfraType
        self.client = RithmicClient(
            user=os.environ["RITHMIC_USERNAME"], password=os.environ["RITHMIC_PASSWORD"],
            system_name=os.environ.get("RITHMIC_SYSTEM_NAME", "LucidTrading"),
            app_name=os.environ.get("RITHMIC_APP_NAME", "x") + ":wk", app_version="1.0.0",
            url=os.environ.get("RITHMIC_GATEWAY_URI", "wss://rprotocol.rithmic.com:443"),
            reconnection_settings=ReconnectionSettings(max_retries=None,
                backoff_type="exponential", interval=2, max_delay=60))
        await self.client.connect(plants=[SysInfraType.ORDER_PLANT, SysInfraType.PNL_PLANT])
        self.acct = (await self.client.plants["order"].list_accounts())[0].account_id
        self.contract = os.environ.get("FADE_CONTRACT", "MNQU6")
        self.client.on_exchange_order_notification += self._note
        log(f"connected acct={self.acct} contract={self.contract} mode={'LIVE' if self.live else 'DRY'}")
        self.jlog(ev="connected", live=self.live)

    async def _note(self, n):
        from async_rithmic import ExchangeOrderNotificationType as NT
        if getattr(n, "notify_type", None) != NT.FILL: return
        px = getattr(n, "fill_price", None)
        side = getattr(n, "transaction_type", None)
        self.jlog(ev="fill", px=px, side=side, tag=getattr(n, "user_tag", ""))
        log(f"FILL side={side} px={px}")
        ot = self.st.get("open")
        if ot and px and side == 1 and ot.get("fill") is None:      # entry BUY fill
            ot["fill"] = float(px); self.save()
        elif ot and px and side == 2:                                # any SELL = we're flat
            pnl = (float(px) - (ot.get("fill") or ot["ref"]))*PV*QTY - 2*COMM*QTY
            self.jlog(ev="closed", exit_px=px, pnl=round(pnl, 2))
            log(f"CLOSED @ {px} pnl=${pnl:+.2f}")
            self.st["done"].append(ot["key"]); self.st["open"] = None; self.save()
            asyncio.create_task(self._cancel_all())

    async def _cancel_all(self):
        try: await self.client.plants["order"].cancel_all_orders(account_id=self.acct)
        except Exception as e: log(f"cancel err {e}")

    async def enter(self, ref):
        from async_rithmic import OrderType, TransactionType
        oid = f"wk-{uuid.uuid4().hex[:8]}"
        self.st["open"] = {"key": self.key(), "oid": oid, "ref": ref, "fill": None,
                           "verified": False, "verify_after": time.time() + 5}
        self.save()
        if self.live:
            await self.client.plants["order"].submit_order(
                order_id=oid, symbol=self.contract, exchange=EXCHANGE, qty=QTY,
                transaction_type=TransactionType.BUY, order_type=OrderType.MARKET,
                account_id=self.acct)
            log(f"LIVE ENTRY long {QTY} {self.contract} MKT oid={oid}")
        else:
            log(f"DRY would ENTER long {QTY} ref~{ref}")
        self.jlog(ev="entry_order", oid=oid, ref=ref, live=self.live)

    async def protect(self, ot):
        from async_rithmic import OrderType, TransactionType
        stop_px = round((ot["fill"] - CAT_STOP_PTS) / TICK) * TICK
        if self.live:
            try:
                await self.client.plants["order"].submit_order(
                    order_id=f"{ot['oid']}-stp", symbol=self.contract, exchange=EXCHANGE,
                    qty=QTY, transaction_type=TransactionType.SELL,
                    order_type=OrderType.STOP_MARKET, trigger_price=stop_px,
                    account_id=self.acct)
                ot["verify_after"] = time.time() + 4
                self.jlog(ev="cat_stop_placed", px=stop_px)
                log(f"CAT-STOP placed @ {stop_px} — verifying")
            except Exception as e:
                log(f"CRITICAL protect fail {e} — flatten")
                await self.flatten("protect_fail")
        else:
            ot["verified"] = True
            log(f"DRY would place cat-stop @ {stop_px}")

    async def verify(self, ot):
        try:
            orders = await self.client.plants["order"].list_orders(account_id=self.acct)
        except Exception:
            ot["verify_after"] = time.time() + 6; return
        tags = {getattr(o, "user_tag", "") for o in orders}
        if f"{ot['oid']}-stp" in tags:
            ot["verified"] = True; ot["verify_after"] = None
            log("CAT-STOP verified resting ✓"); self.jlog(ev="protection_verified")
        else:
            log("CRITICAL: stop NOT resting — flatten + disable")
            await self.flatten("protection_not_resting")

    async def flatten(self, why):
        self.jlog(ev="flatten", why=why, live=self.live)
        if self.live:
            await self._cancel_all()
            try:
                await self.client.plants["order"].exit_position(
                    account_id=self.acct, symbol=self.contract, exchange=EXCHANGE)
            except Exception as e: log(f"exit err {e}")
            log(f"FLATTEN ({why})")
        else:
            ot = self.st.get("open")
            if ot: self.st["done"].append(ot["key"])
            self.st["open"] = None; self.save()
            log(f"DRY would FLATTEN ({why})")

    def key(self):
        now = pd.Timestamp.now(tz=ET)
        return str((now + pd.Timedelta(days=1)).date()) if now.weekday() == 6 else str(now.date())

    def quote(self):
        fs = sorted(glob.glob(os.path.join(os.environ.get("RAW_DIR", ""), "stream_MNQ_*.csv")))
        if not fs: return None
        with open(fs[-1], "rb") as f:
            try: f.seek(-300, 2)
            except OSError: pass
            for line in reversed(f.read().decode(errors="ignore").strip().splitlines()):
                p = line.split(",")
                if len(p) == 5:
                    try: return (float(p[1]) + float(p[3]))/2
                    except ValueError: continue
        return None

    async def run(self):
        await self.connect()
        if self.st.get("open") and self.live:
            log("startup with open state — flattening (conservative)")
            await self.flatten("startup_recovery")
        while True:
            now = pd.Timestamp.now(tz=ET)
            if os.path.exists(self.kill_p):
                await self.flatten("kill"); self.save(); return
            ot = self.st.get("open")
            if ot and ot.get("fill") and not ot.get("verified") and self.live and \
               ot.get("verify_after") and time.time() >= ot["verify_after"]:
                if not ot.get("stp_sent"):
                    await self.protect(ot)
                    ot["stp_sent"] = True
                else:
                    await self.verify(ot)
                self.save()
            # ENTRY: Sunday 18:00-18:15
            if now.weekday() == 6 and ot is None and self.key() not in self.st["done"]:
                ent = now.normalize() + pd.Timedelta(hours=18)
                if ent <= now <= ent + pd.Timedelta(minutes=15):
                    ref = self.quote()
                    if ref: await self.enter(ref)
            # EXIT: Monday 16:00
            if now.weekday() == 0 and ot and ot["key"] == str(now.date()) and \
               now >= now.normalize() + pd.Timedelta(hours=16):
                await self.flatten("scheduled_exit_mon16")
            self.save()
            await asyncio.sleep(20)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    if a.live: log("*** LIVE: 1 micro weekend hold, cat-stop -450pts ***")
    asyncio.run(WkLive(a.out_dir, a.live).run())

if __name__ == "__main__": main()
