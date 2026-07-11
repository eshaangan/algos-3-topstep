"""UNIFIED live daemon for the validated book — one Rithmic order-plant session.

Strategies (all pre-registered):
  WK   weekend_hold_v1 : long Sun 18:00 -> Mon 16:00 ET          (ladder rung sizing)
  EO   euro_open_v1    : long 2:00 -> 5:00 ET weeknights IF prior RTH day down & high-range
  FOMC fomc_drift_v1   : long 14:00 day-before -> 13:55 decision day (verified Fed calendar)

Guardrails: hard size ladder (banked-PnL cushion), catastrophic verified stop on
every position, fill-timeout reconciliation, flatten verification sweep, holiday
guard, KILL file, watchdog, atomic state, ntfy phone alerts. DRY unless --live.
"""
from __future__ import annotations
import asyncio, glob, json, os, time, urllib.request, uuid
from datetime import datetime, timezone
import pandas as pd

ET = "America/New_York"
SYMBOL, EXCHANGE = "MNQ", "CME"
PV, COMM, TICK = 2.0, 0.62, 0.25
CAT_STOP_PTS = 450.0
FOMC_ENTRY = ["2026-07-28", "2026-09-15", "2026-10-27", "2026-12-08"]   # day before decision
FOMC_EXIT  = {"2026-07-28": "2026-07-29", "2026-09-15": "2026-09-16",
              "2026-10-27": "2026-10-28", "2026-12-08": "2026-12-09"}
NO_ENTRY_SUNDAYS = {"2026-09-06"}          # Labor Day Monday
HOLIDAYS = {"2026-09-07", "2026-11-26", "2026-12-25"}
NTFY = os.environ.get("NTFY_TOPIC", "")

def log(m): print(f"[{datetime.now(timezone.utc):%m-%d %H:%M:%S}Z] {m}", flush=True)
def push(m):
    if not NTFY: return
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"https://ntfy.sh/{NTFY}", data=m.encode(), method="POST"), timeout=5)
    except Exception: pass

def ladder_size(kind, banked):
    """EOD-trailing floor rises with banked PnL until it locks at breakeven
    (+$3k banked). Cushion is ~$3k until lock, then grows. Sizes chosen so the
    HISTORICAL WORST event MAE at that size stays under the cushion at that rung."""
    if kind == "WK":                       # worst MAE ~ -$2,092/micro
        return 2 if banked >= 4500 else 1
    # EO/FOMC: worst MAE ~ -$416/micro
    if banked >= 6000: return 4
    if banked >= 3000: return 3
    return 2

class Daemon:
    def __init__(self, out, live, now_fn=None, quote_fn=None):
        self.out, self.live = out, live
        os.makedirs(out, exist_ok=True)
        self.state_p = os.path.join(out, "state.json")
        self.ev_p = os.path.join(out, "events.jsonl")
        self.kill_p = os.path.join(out, "KILL")
        self.st = json.load(open(self.state_p)) if os.path.exists(self.state_p) else \
            {"open": None, "done": [], "banked": 0.0}
        self.client = None; self.acct = None; self.contract = None
        self.now = now_fn or (lambda: pd.Timestamp.now(tz=ET))
        self.quote = quote_fn or self._stream_quote
        self.beat = time.time()

    def jlog(self, **r):
        r["t"] = str(pd.Timestamp.now(tz="UTC"))
        open(self.ev_p, "a").write(json.dumps(r, default=str) + "\n")
    def save(self):
        tmp = self.state_p + ".tmp"
        json.dump(self.st, open(tmp, "w"), default=str)
        os.replace(tmp, self.state_p)

    def _stream_quote(self):
        fs = sorted(glob.glob(os.path.join(os.environ.get("RAW_DIR", ""), "stream_MNQ_*.csv")))
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

    async def connect(self):
        from async_rithmic import RithmicClient, ReconnectionSettings, SysInfraType
        self.client = RithmicClient(
            user=os.environ["RITHMIC_USERNAME"], password=os.environ["RITHMIC_PASSWORD"],
            system_name=os.environ.get("RITHMIC_SYSTEM_NAME", "LucidTrading"),
            app_name=os.environ.get("RITHMIC_APP_NAME", "x") + ":ladder", app_version="1.0.0",
            url=os.environ.get("RITHMIC_GATEWAY_URI", "wss://rprotocol.rithmic.com:443"),
            reconnection_settings=ReconnectionSettings(max_retries=None,
                backoff_type="exponential", interval=3, max_delay=120))
        await self.client.connect(plants=[SysInfraType.ORDER_PLANT, SysInfraType.PNL_PLANT])
        accounts = await self.client.plants["order"].list_accounts()
        want = os.environ.get("LUCID_ACCOUNT_ID")
        match = [a for a in accounts if not want or str(a.account_id) == want]
        if not match:
            raise RuntimeError(f"account {want} not found in {[a.account_id for a in accounts]}")
        self.acct = match[0].account_id
        self.contract = os.environ.get("FADE_CONTRACT", "MNQU6")
        self.pos_qty = None           # live position per PNL plant
        self.client.on_exchange_order_notification += self._note
        self.client.on_instrument_pnl_update += self._pnl_note
        log(f"connected acct={self.acct} {self.contract} mode={'LIVE' if self.live else 'DRY'}")
        push(f"ladder daemon up ({'LIVE' if self.live else 'DRY'})")

    async def _note(self, n):
        from async_rithmic import ExchangeOrderNotificationType as NT
        if getattr(n, "notify_type", None) != NT.FILL: return
        px, side = getattr(n, "fill_price", None), getattr(n, "transaction_type", None)
        sym = getattr(n, "symbol", ""); tag = str(getattr(n, "user_tag", "") or "")
        acct = getattr(n, "account_id", None)
        qty = int(getattr(n, "fill_size", 0) or 0)
        self.jlog(ev="fill", px=px, side=side, tag=tag, sym=sym, qty=qty, acct=acct)
        ot = self.st.get("open")
        if not (ot and px): return
        # STRICT attribution: our contract, our account (if reported), our order tags only
        if sym != self.contract: return
        if acct and self.acct and str(acct) != str(self.acct): return
        ours = tag.startswith(ot["oid"]) or tag == ""      # exit_position fills carry no tag
        if not ours: return
        if side == 1 and tag == ot["oid"]:
            filled = ot.get("filled_qty", 0) + (qty or ot["qty"])
            ot["filled_qty"] = filled
            ot["fill"] = float(px) if ot.get("fill") is None else ot["fill"]
            self.save()
            log(f"ENTRY FILL {filled}/{ot['qty']} @ {px}")
            if filled >= ot["qty"]: push(f"{ot['kind']} entry filled @ {px}")
        elif side == 2:
            qty = ot["qty"]
            pnl = (float(px) - (ot.get("fill") or ot["ref"]))*PV*qty - 2*COMM*qty
            self.st["banked"] = float(self.st["banked"]) + pnl
            self.jlog(ev="closed", kind=ot["kind"], exit_px=px, pnl=round(pnl, 2),
                      banked=round(self.st["banked"], 2))
            log(f"CLOSED {ot['kind']} @ {px} pnl=${pnl:+.2f} banked=${self.st['banked']:+.2f}")
            push(f"{ot['kind']} closed {pnl:+.2f} (banked {self.st['banked']:+.0f})")
            self.st["done"].append(ot["key"]); self.st["open"] = None; self.save()
            asyncio.create_task(self._cancel_all())

    async def _pnl_note(self, n):
        try:
            if getattr(n, "symbol", "") == self.contract:
                q = getattr(n, "open_position_quantity", None)
                if q is None: q = getattr(n, "fill_buy_qty", 0) - getattr(n, "fill_sell_qty", 0)
                self.pos_qty = int(q)
        except Exception: pass

    async def _cancel_all(self):
        try: await self.client.plants["order"].cancel_all_orders(account_id=self.acct)
        except Exception as e: log(f"cancel err {e}")

    async def _working_orders(self):
        try:
            orders = await self.client.plants["order"].list_orders(account_id=self.acct)
            return [getattr(o, "user_tag", "") for o in orders]
        except Exception:
            return None

    async def enter(self, kind, key):
        from async_rithmic import OrderType, TransactionType
        qty = ladder_size(kind, self.st["banked"])
        worst = 2100 if kind == "WK" else 420              # historical worst MAE $/micro
        cushion = 3000 + max(0, float(self.st["banked"]) - 3000)   # floor-lock model
        if self.live and qty * worst > 0.75 * cushion:
            log(f"{kind}: cushion gate blocked entry (need {qty*worst} vs {0.75*cushion:.0f})")
            self.jlog(ev="cushion_block", kind=kind, qty=qty)
            self.st["done"].append(key); self.save(); return
        ref, age = self.quote()
        if ref is None or age > 300:
            log(f"{kind}: no fresh quote (age {age:.0f}s) — retry next loop"); return
        oid = f"{kind.lower()}-{uuid.uuid4().hex[:6]}"
        self.st["open"] = {"kind": kind, "key": key, "oid": oid, "qty": qty, "ref": ref,
                           "fill": None, "verified": False, "stp_sent": False,
                           "entered_at": time.time(), "verify_after": time.time() + 5}
        self.save()
        if self.live:
            try:
                await self.client.plants["order"].submit_order(
                    order_id=oid, symbol=self.contract, exchange=EXCHANGE, qty=qty,
                    transaction_type=TransactionType.BUY, order_type=OrderType.MARKET,
                    account_id=self.acct)
            except Exception as e:
                log(f"{kind} submit FAILED {e} — clearing for retry"); self.jlog(ev="submit_fail", err=str(e))
                self.st["open"] = None; self.save(); return
            log(f"LIVE ENTRY {kind} {qty} {self.contract} @~{ref}")
        else:
            self.st["open"]["fill"] = ref     # dry: simulate instant fill for branch coverage
            log(f"DRY ENTER {kind} {qty} @~{ref}")
        self.jlog(ev="entry", kind=kind, qty=qty, ref=ref, live=self.live)

    async def protect(self, ot):
        from async_rithmic import OrderType, TransactionType
        stop_px = round((ot["fill"] - CAT_STOP_PTS)/TICK)*TICK
        if self.live:
            try:
                await self.client.plants["order"].submit_order(
                    order_id=f"{ot['oid']}-stp", symbol=self.contract, exchange=EXCHANGE,
                    qty=ot["qty"], transaction_type=TransactionType.SELL,
                    order_type=OrderType.STOP_MARKET, trigger_price=stop_px,
                    account_id=self.acct)
                ot["verify_after"] = time.time() + 4
                log(f"cat-stop placed @ {stop_px}")
            except Exception as e:
                log(f"CRITICAL protect fail {e}"); push(f"CRITICAL {ot['kind']} protect fail")
                await self.flatten("protect_fail")
        else:
            ot["verified"] = True; log(f"DRY cat-stop @ {stop_px}")

    async def verify(self, ot):
        tags = await self._working_orders()
        if tags is None:
            ot["verify_after"] = time.time() + 6; return
        if f"{ot['oid']}-stp" in tags:
            ot["verified"] = True; ot["verify_after"] = None
            log("protection verified ✓"); push(f"{ot['kind']} protected ✓")
            self.jlog(ev="protection_verified")
        else:
            log("CRITICAL stop not resting — flatten"); push(f"CRITICAL {ot['kind']} stop missing — flattening")
            await self.flatten("protection_not_resting")

    async def flatten(self, why):
        self.jlog(ev="flatten", why=why, live=self.live)
        ot = self.st.get("open")
        if self.live:
            confirmed = False
            for attempt in range(4):                       # verification sweep
                await self._cancel_all()
                try:
                    await self.client.plants["order"].exit_position(
                        account_id=self.acct, symbol=self.contract, exchange=EXCHANGE)
                except Exception as e: log(f"exit err {e}")
                await asyncio.sleep(4)
                tags = await self._working_orders()
                orders_clear = tags is not None and not any(t for t in tags)
                pos_clear = (self.pos_qty == 0) if self.pos_qty is not None else None
                if orders_clear and pos_clear in (True, None):
                    confirmed = pos_clear is True or attempt >= 1
                    if confirmed: break
                log(f"flatten sweep {attempt+1}: orders_clear={orders_clear} pos={self.pos_qty}")
            if not confirmed:
                log("CRITICAL: flatten UNCONFIRMED — manual check required")
                push("CRITICAL: flatten unconfirmed — CHECK ACCOUNT NOW")
                self.jlog(ev="flatten_unconfirmed")
            log(f"FLATTEN ({why}) confirmed={confirmed}")
        else:
            if ot:
                px, _ = self.quote()
                pnl = ((px or ot["ref"]) - (ot.get("fill") or ot["ref"]))*PV*ot["qty"] - 2*COMM*ot["qty"]
                self.st["banked"] = float(self.st["banked"]) + pnl
                self.jlog(ev="closed_dry", kind=ot["kind"], pnl=round(pnl, 2))
                log(f"DRY FLATTEN {ot['kind']} pnl≈${pnl:+.2f} ({why})")
        if ot: self.st["done"].append(ot["key"])
        self.st["open"] = None; self.save()

    def schedule(self, now):
        """Return (kind, key, is_exit) for anything due at `now`."""
        d, wd, hm = str(now.date()), now.weekday(), now.hour*60 + now.minute
        ot = self.st.get("open")
        # exits first
        if ot:
            k = ot["kind"]
            if k == "WK" and wd == 0 and ot["key"] == d and hm >= 960: return ("WK", ot["key"], True)
            if k == "EO" and ot["key"] == f"E{d}" and hm >= 300: return ("EO", ot["key"], True)
            if k == "FOMC" and FOMC_EXIT.get(ot["key"]) == d and hm >= 835: return ("FOMC", ot["key"], True)
            return None
        # entries (one position at a time, by design)
        if wd == 6 and d not in NO_ENTRY_SUNDAYS:
            key = str((now + pd.Timedelta(days=1)).date())
            if key not in self.st["done"] and 1080 <= hm <= 1095: return ("WK", key, False)
        if d in FOMC_ENTRY and f"F{d}" not in self.st["done"] and 840 <= hm <= 850:
            return ("FOMC", d, False)
        if wd in (0,1,2,3,4) and d not in HOLIDAYS and f"E{d}" not in self.st["done"] and 120 <= hm <= 130:
            if self._eo_condition(): return ("EO", f"E{d}", False)
            self.st["done"].append(f"E{d}"); self.save()
        return None

    def _eo_condition(self):
        try:
            fs = sorted(glob.glob(os.path.join(os.environ.get("RAW_DIR", ""), "stream_MNQ_*.csv")))
            rows = []
            for f in fs[-4:]:
                df = pd.read_csv(f, header=None, names=["ns","bp","bs","ap","asz"], usecols=[0,1,3])
                ts = pd.to_datetime(df["ns"], unit="ns", utc=True).dt.tz_convert(ET)
                mid = (df["bp"]+df["ap"])/2
                m = (ts.dt.hour*60+ts.dt.minute >= 570) & (ts.dt.hour*60+ts.dt.minute < 960)
                if m.sum() < 500: continue
                v = mid[m]
                rows.append({"day": f.split("_")[-1][:8], "ret": v.iloc[-1]-v.iloc[0], "rng": v.max()-v.min()})
            if not rows: return False
            dd = pd.DataFrame(rows).drop_duplicates("day").sort_values("day")
            last = dd.iloc[-1]
            return bool(last["ret"] < 0 and last["rng"] > dd["rng"].median())
        except Exception as e:
            log(f"eo condition err {e}"); return False

    async def run(self):
        await self.connect()
        if self.st.get("open") and self.live:
            log("startup with open state — recovery flatten")
            await self.flatten("startup_recovery")
        while True:
            self.beat = time.time()
            now = self.now()
            if os.path.exists(self.kill_p):
                await self.flatten("kill"); push("KILL — daemon down"); return
            ot = self.st.get("open")
            # fill-timeout reconciliation
            if ot and self.live and ot.get("fill") is None and time.time() - ot["entered_at"] > 90:
                log("fill timeout — reconciling via flatten"); push("fill timeout — reconciling")
                await self.flatten("fill_timeout")
            # protection lifecycle
            ot = self.st.get("open")
            if ot and ot.get("fill") and not ot.get("verified") and self.live and \
               ot.get("verify_after") and time.time() >= ot["verify_after"]:
                if not ot.get("stp_sent"):
                    await self.protect(ot); ot["stp_sent"] = True
                else:
                    await self.verify(ot)
                self.save()
            due = self.schedule(now)
            if due:
                kind, key, is_exit = due
                if is_exit: await self.flatten(f"{kind}_scheduled_exit")
                else: await self.enter(kind, key)
            await asyncio.sleep(15)

def selftest():
    """Mock-clock walk through every branch in DRY mode."""
    import tempfile
    td = tempfile.mkdtemp()
    seq = [
        pd.Timestamp("2026-07-12 18:02", tz=ET),   # Sunday entry
        pd.Timestamp("2026-07-12 18:03", tz=ET),
        pd.Timestamp("2026-07-13 16:01", tz=ET),   # Monday exit
        pd.Timestamp("2026-07-14 02:03", tz=ET),   # Tue euro-open (condition mocked True)
        pd.Timestamp("2026-07-14 05:01", tz=ET),   # euro-open exit
        pd.Timestamp("2026-07-28 14:05", tz=ET),   # FOMC entry
        pd.Timestamp("2026-07-29 13:56", tz=ET),   # FOMC exit
        pd.Timestamp("2026-09-06 18:05", tz=ET),   # holiday Sunday: must NOT enter
    ]
    i = [0]
    def now(): return seq[min(i[0], len(seq)-1)]
    d = Daemon(td, live=False, now_fn=now, quote_fn=lambda: (23000.0, 1.0))
    d._eo_condition = lambda: True
    async def drive():
        d.acct = "TEST"; d.contract = "MNQU6"
        for step in range(len(seq)):
            i[0] = step
            nowv = d.now()
            due = d.schedule(nowv)
            if due:
                kind, key, is_exit = due
                if is_exit: await d.flatten(f"{kind}_exit")
                else: await d.enter(kind, key)
        assert not d.st["open"], "position left open at end"
        assert any(k.startswith("2026-07-13") for k in d.st["done"]), "weekend trade missing"
        assert any(k.startswith("E2026-07-14") for k in d.st["done"]), "euro-open missing"
        assert "2026-07-28" in d.st["done"], "FOMC missing"
        assert not any(k == "2026-09-07" for k in d.st["done"] if not k.startswith("E")), "holiday entry fired!"
        print("SELFTEST PASS — all schedule branches exercised, holiday guard held,"
              f" banked=${d.st['banked']:+.2f}")
    asyncio.run(drive())

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: selftest(); return
    if a.live: log(f"*** LIVE: ladder daemon (WK/EO/FOMC), cat-stop {CAT_STOP_PTS}pts ***")
    asyncio.run(Daemon(a.out_dir, a.live).run())

if __name__ == "__main__": main()
