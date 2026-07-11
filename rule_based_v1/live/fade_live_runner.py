"""LIVE 1-micro execution shakedown for imb_fade_midday. DRY-RUN BY DEFAULT.

Purpose: prove the plumbing (routing, fills, brackets, flatten) and measure REAL
slippage — NOT to validate the edge (the July gate does that). Signal logic is
imported from paper_fade_runner so live and paper can never drift apart.

HARD GUARDRAILS (constants, not flags — changing them requires a code change):
  QTY = 1 micro, always.
  DAILY_LOSS_STOP = -$100 realized -> flatten + disable for the day.
  No entries after 14:44 ET; hard flatten 14:59 ET; never holds overnight.
  Kill switch: `touch <out-dir>/KILL` -> cancel all, flatten, exit.
  Exchange-side OCO bracket (stop + target) placed WITH the entry, so a dead
  runner never leaves a naked position.
  Real orders ONLY with --live; otherwise every order action is logged as DRY.

    python fade_live_runner.py --raw-dir <l2_raw> --out-dir <dir>          # dry-run
    python fade_live_runner.py --raw-dir <l2_raw> --out-dir <dir> --live   # real orders
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_fade_runner import (  # noqa: E402  (frozen signal — single source of truth)
    IncrementalBars, start_watchdog, signal_at,
    PT_ATR, SL_ATR, HORIZON_BARS, COOLDOWN_BARS, MAX_TRADES_DAY, TICK, ET,
)

# ── hard guardrails ───────────────────────────────────────────────────────────
QTY = 1                       # micro contracts — HARD CAP
DAILY_LOSS_STOP = -100.0      # realized $ -> stop trading for the day
LAST_ENTRY_ET = (14, 44)      # no new entries after this
FLATTEN_ET = (14, 59)         # hard flatten
SYMBOL, EXCHANGE = "MNQ", "CME"
POLL_SECS = 2   # stream csv makes each poll ~free; 2s keeps decision lag ~bar-close+2s


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}Z] {msg}", flush=True)


def jlog(path: str, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps({"t": str(pd.Timestamp.now(tz='UTC')), **rec}, default=str) + "\n")


class LiveFade:
    def __init__(self, raw_dir: str, out_dir: str, live: bool):
        self.raw_dir, self.out_dir, self.live = raw_dir, out_dir, live
        os.makedirs(out_dir, exist_ok=True)
        self.events_p = os.path.join(out_dir, "live_events.jsonl")
        self.trades_p = os.path.join(out_dir, "live_trades.jsonl")
        self.state_p = os.path.join(out_dir, "live_state.json")
        self.kill_p = os.path.join(out_dir, "KILL")
        self._closing: dict | None = None   # trade being flattened, awaiting exit fill
        self._last_rms = 0.0
        self.client = None
        self.account_id = None
        self.contract = None
        self.st = self._load_state()

    # ── state ──────────────────────────────────────────────────────────────
    def _load_state(self) -> dict:
        if os.path.exists(self.state_p):
            with open(self.state_p) as f:
                return json.load(f)
        return {"day": None, "trades_today": 0, "realized_pnl": 0.0,
                "open_trade": None, "processed": [], "disabled": False,
                "cooldown_until": None}

    def _save(self) -> None:
        tmp = self.state_p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.st, f, indent=1, default=str)
        os.replace(tmp, self.state_p)

    # ── rithmic ────────────────────────────────────────────────────────────
    async def connect(self) -> None:
        from async_rithmic import RithmicClient, ReconnectionSettings, SysInfraType
        self.client = RithmicClient(
            user=os.environ["RITHMIC_USERNAME"], password=os.environ["RITHMIC_PASSWORD"],
            system_name=os.environ.get("RITHMIC_SYSTEM_NAME", "LucidTrading"),
            app_name=(os.environ.get("RITHMIC_APP_NAME_ORDERS")
                      or os.environ.get("RITHMIC_APP_NAME", "XXXX:fade_live")),
            app_version="1.0.0",
            url=os.environ.get("RITHMIC_GATEWAY_URI", "wss://rprotocol.rithmic.com:443"),
            reconnection_settings=ReconnectionSettings(
                max_retries=None, backoff_type="exponential", interval=2, max_delay=30),
        )
        # ORDER + PNL plants only; TICKER belongs to the recorder process.
        await self.client.connect(plants=[SysInfraType.ORDER_PLANT, SysInfraType.PNL_PLANT])
        accounts = await self.client.plants["order"].list_accounts()
        if not accounts:
            raise RuntimeError("no accounts on order plant")
        self.account_id = accounts[0].account_id
        self.contract = await self._front_month()
        self.client.on_exchange_order_notification += self._on_exch_note
        self.client.on_rithmic_order_notification += self._on_rith_note
        self.client.on_account_pnl_update += self._on_pnl
        self.client.on_instrument_pnl_update += self._on_pnl
        log(f"connected: account={self.account_id} contract={self.contract} "
            f"mode={'LIVE' if self.live else 'DRY-RUN'}")
        jlog(self.events_p, {"ev": "connected", "account": str(self.account_id),
                             "contract": self.contract, "live": self.live})

    async def _front_month(self) -> str:
        # explicit override wins (set FADE_CONTRACT=MNQU6 etc. — the recorder's
        # startup log line "Recording <contract> @ CME" shows the current one)
        if os.environ.get("FADE_CONTRACT"):
            return os.environ["FADE_CONTRACT"]
        try:
            return await self.client.get_front_month_contract(SYMBOL, EXCHANGE)
        except Exception:
            # ticker plant not connected — resolve via reference data (best effort)
            ref = await self.client.plants["order"].get_reference_data(SYMBOL, EXCHANGE)
            return getattr(ref, "symbol", SYMBOL)

    async def _on_pnl(self, note) -> None:
        """PNL plant push — authoritative realized PnL for the daily loss stop.
        Field names differ across payload types; log everything in the shakedown,
        bind whichever realized field is present."""
        payload = {f: getattr(note, f) for f in
                   ("day_pl", "day_pnl", "realized_pnl", "closed_position_pnl",
                    "account_id", "symbol") if hasattr(note, f)}
        jlog(self.events_p, {"ev": "pnl_update", **payload})
        for f in ("day_pl", "day_pnl", "realized_pnl", "closed_position_pnl"):
            v = payload.get(f)
            if v is not None:
                try:
                    self.st["realized_pnl"] = float(v)
                except (TypeError, ValueError):
                    pass
                break

    async def _on_exch_note(self, note) -> None:
        from async_rithmic import ExchangeOrderNotificationType as NT
        ntype = getattr(note, "notify_type", None)
        rec = {"ev": "exchange_note", "type": str(ntype),
               "sym": getattr(note, "symbol", "?"),
               "side": getattr(note, "transaction_type", None),
               "fill_px": getattr(note, "fill_price", None),
               "fill_sz": getattr(note, "fill_size", None),
               "order_id": getattr(note, "user_tag", "") or getattr(note, "basket_id", "")}
        jlog(self.events_p, rec)
        log(f"EXCH {rec['type']} {rec['sym']} side={rec['side']} px={rec['fill_px']}")

        # Bind FILLs to trade state. BUY=1/SELL=2; entry dir: +1 -> BUY, -1 -> SELL.
        if ntype != NT.FILL or rec["sym"] != self.contract:
            return
        px = rec["fill_px"]
        if px is None:
            return
        fill_dir = 1 if rec["side"] == 1 else -1

        # exit fill for a trade we are actively flattening (time exit / kill / eod)
        if self._closing is not None and fill_dir == -self._closing["dir"]:
            self._book_close(self._closing, float(px), self._closing.get("close_reason", "flatten"))
            self._closing = None
            return

        ot = self.st["open_trade"]
        if not ot:
            return
        if fill_dir == ot["dir"] and ot.get("entry_fill_px") is None:
            ot["entry_fill_px"] = float(px)          # real entry -> true slippage
            slip = ot["dir"] * (float(px) - ot["ref_px"])
            jlog(self.events_p, {"ev": "entry_fill", "oid": ot["oid"], "px": px,
                                 "slip_pts_vs_ref": round(slip, 2)})
            log(f"ENTRY FILLED @ {px} (slip vs ref {slip:+.2f} pts)")
            # NEVER await plant requests inside this callback: it runs in the
            # plant's listener task, so awaiting a request response here
            # deadlocks the whole order pipe (2026-07-08: every request timed
            # out, flatten included, position sat naked until process restart).
            asyncio.create_task(self._place_protection(ot))
        elif fill_dir == -ot["dir"]:                  # stop or target leg filled
            which = "stop" if str(rec["order_id"]).endswith("-stp") else \
                    "target" if str(rec["order_id"]).endswith("-tgt") else "exit"
            self._book_close(ot, float(px), which)
            self.st["open_trade"] = None
            asyncio.create_task(self._cancel_sibling())

    async def _cancel_sibling(self) -> None:
        try:
            await self.client.plants["order"].cancel_all_orders(account_id=self.account_id)
        except Exception as e:
            log(f"sibling cancel error: {e}")

    def _book_close(self, trade: dict, exit_px: float, reason: str) -> None:
        entry = trade.get("entry_fill_px") or trade["ref_px"]
        pnl = trade["dir"] * (exit_px - entry) * 2.0 * QTY - 2 * 0.62 * QTY
        self.st["realized_pnl"] += pnl
        self.st["cooldown_until"] = str(pd.Timestamp.now(tz="UTC").floor("1min")
                                        + pd.Timedelta(minutes=1 + COOLDOWN_BARS))
        rec = {"oid": trade["oid"], "dir": trade["dir"], "entry_px": entry,
               "exit_px": exit_px, "reason": reason, "pnl": round(pnl, 2),
               "day_pnl": round(self.st["realized_pnl"], 2)}
        jlog(self.trades_p, rec)
        jlog(self.events_p, {"ev": "trade_closed", **rec})
        log(f"CLOSED ({reason}) @ {exit_px} pnl=${pnl:+.2f} day=${self.st['realized_pnl']:+.2f}")
        self._save()

    async def _on_rith_note(self, note) -> None:
        jlog(self.events_p, {"ev": "rithmic_note",
                             "type": str(getattr(note, "notify_type", "?")),
                             "status": getattr(note, "status", "")})

    # ── order actions (every path honors dry-run) ──────────────────────────
    @staticmethod
    def _tick_round(px: float) -> float:
        return round(px / TICK) * TICK

    async def enter(self, direction: int, atr: float, ref_px: float, z: float) -> None:
        """Market entry ONLY. Protective stop+target are placed as EXPLICIT separate
        orders on the entry fill (see _place_protection) and then VERIFIED resting —
        the native bracket template was silently ignored on Lucid (2026-07-08:
        trade ran 53pts past its stop level with nothing at the exchange)."""
        from async_rithmic import OrderType, TransactionType
        oid = f"fade-{uuid.uuid4().hex[:8]}"
        side = TransactionType.BUY if direction > 0 else TransactionType.SELL
        rec = {"ev": "entry_order", "oid": oid, "dir": direction, "qty": QTY,
               "ref_px": ref_px, "z": z, "atr": atr, "live": self.live}
        if self.live:
            await self.client.plants["order"].submit_order(
                order_id=oid, symbol=self.contract, exchange=EXCHANGE, qty=QTY,
                transaction_type=side, order_type=OrderType.MARKET,
                account_id=self.account_id)
            log(f"LIVE ENTRY {'LONG' if direction > 0 else 'SHORT'} {QTY} {self.contract} MKT oid={oid}")
        else:
            log(f"DRY would ENTER {'LONG' if direction > 0 else 'SHORT'} {QTY} {self.contract} "
                f"MKT ref~{ref_px:.2f}")
        jlog(self.events_p, rec)
        self.st["open_trade"] = {"oid": oid, "dir": direction, "entry_bar": None,
                                 "ref_px": ref_px, "atr": atr, "bars_held": 0,
                                 "entry_fill_px": None, "stop_px": None, "tgt_px": None,
                                 "protection_verified": False, "verify_after": None}

    async def _place_protection(self, ot: dict) -> None:
        """Explicit STOP_MARKET + LIMIT exits, opposite side, right after entry fill.
        We manage the OCO ourselves: any leg fill -> cancel_all (this account trades
        nothing else). Placement failure -> immediate flatten + disable."""
        from async_rithmic import OrderType, TransactionType
        d, entry = ot["dir"], ot["entry_fill_px"]
        exit_side = TransactionType.SELL if d > 0 else TransactionType.BUY
        ot["stop_px"] = self._tick_round(entry - d * SL_ATR * ot["atr"])
        ot["tgt_px"] = self._tick_round(entry + d * PT_ATR * ot["atr"])
        op = self.client.plants["order"]
        try:
            await op.submit_order(order_id=f"{ot['oid']}-stp", symbol=self.contract,
                                  exchange=EXCHANGE, qty=QTY, transaction_type=exit_side,
                                  order_type=OrderType.STOP_MARKET,
                                  trigger_price=ot["stop_px"], account_id=self.account_id)
            await op.submit_order(order_id=f"{ot['oid']}-tgt", symbol=self.contract,
                                  exchange=EXCHANGE, qty=QTY, transaction_type=exit_side,
                                  order_type=OrderType.LIMIT,
                                  price=ot["tgt_px"], account_id=self.account_id)
            ot["verify_after"] = time.time() + 3
            log(f"PROTECTION placed: stop {ot['stop_px']} / target {ot['tgt_px']} — verifying")
            jlog(self.events_p, {"ev": "protection_placed", "oid": ot["oid"],
                                 "stop_px": ot["stop_px"], "tgt_px": ot["tgt_px"]})
        except Exception as e:
            log(f"CRITICAL: protection placement failed ({e}) — flattening")
            jlog(self.events_p, {"ev": "protection_failed", "err": str(e)})
            await self.flatten("protection_placement_failed")
            self.st["disabled"] = True

    async def _verify_protection(self, ot: dict) -> None:
        """Confirm both protective orders are actually WORKING at the exchange.
        Not resting -> we are naked -> flatten + disable. No more silent trust."""
        try:
            orders = await self.client.plants["order"].list_orders(account_id=self.account_id)
        except Exception as e:
            log(f"verify: list_orders failed ({e}); retrying next cycle")
            ot["verify_after"] = time.time() + 5
            return
        tags = {getattr(o, "user_tag", "") for o in orders}
        have = {f"{ot['oid']}-stp" in tags, f"{ot['oid']}-tgt" in tags}
        jlog(self.events_p, {"ev": "protection_verify", "tags_seen": sorted(t for t in tags if t),
                             "ok": have == {True}})
        if have == {True}:
            ot["protection_verified"] = True
            ot["verify_after"] = None
            log("PROTECTION verified resting at exchange ✓")
        else:
            log("CRITICAL: protection NOT resting — flattening + disabling for the day")
            await self.flatten("protection_not_resting")
            self.st["disabled"] = True

    async def flatten(self, reason: str) -> None:
        jlog(self.events_p, {"ev": "flatten", "reason": reason, "live": self.live})
        if self.st["open_trade"]:
            # keep the trade for the fill handler — clearing it before the exit
            # fill arrives is exactly the race that blinded the daily stop on
            # 2026-07-08 (realized_pnl stayed 0 across two closed trades)
            self._closing = {**self.st["open_trade"], "close_reason": reason}
        if self.live:
            op = self.client.plants["order"]
            try:
                await op.cancel_all_orders(account_id=self.account_id)
            except Exception as e:
                log(f"cancel_all error: {e}")
            try:
                await op.exit_position(account_id=self.account_id,
                                       symbol=self.contract, exchange=EXCHANGE)
            except Exception as e:
                log(f"exit_position error: {e}")
            log(f"FLATTENED ({reason})")
        else:
            log(f"DRY would FLATTEN ({reason})")
        self.st["open_trade"] = None

    async def _reconcile(self) -> None:
        """Periodic RMS snapshot for the audit log (throttled — it was spamming
        the events file every poll cycle)."""
        if not self.live or time.time() - self._last_rms < 300:
            return
        self._last_rms = time.time()
        try:
            rms = await self.client.plants["order"].get_account_rms()
            jlog(self.events_p, {"ev": "rms", "raw": str(rms)[:400]})
        except Exception:
            pass

    # ── main loop ──────────────────────────────────────────────────────────
    async def run(self) -> None:
        await self.connect()
        if self.st["open_trade"] and self.live:
            # restarted mid-trade (crash/watchdog): flatten rather than trust
            # reconstructed state — conservative by design
            log("startup with open trade in state — flattening")
            await self.flatten("startup_recovery")
        inc = IncrementalBars(self.raw_dir)
        beat = [time.time()]
        start_watchdog(lambda: beat[0], limit_secs=900)  # > longest sleep (300s)
        while True:
            beat[0] = time.time()
            now_et = pd.Timestamp.now(tz=ET)
            day = f"{now_et:%Y%m%d}"
            if self.st["day"] != day:
                if self.st["open_trade"]:
                    # never silently forget a position across a day boundary
                    await self.flatten("day_rollover_safety")
                self.st.update(day=day, trades_today=0, realized_pnl=0.0,
                               processed=[], disabled=False, cooldown_until=None,
                               open_trade=None)

            if os.path.exists(self.kill_p):
                log("KILL switch found")
                await self.flatten("kill_switch")
                self._save()
                return

            in_session = 11 <= now_et.hour < 16
            if not in_session:
                self._save()
                await asyncio.sleep(300)
                continue

            # hard flatten window
            past_flatten = (now_et.hour, now_et.minute) >= FLATTEN_ET
            if past_flatten and self.st["open_trade"]:
                await self.flatten("eod_1459")
                self.st["disabled"] = True

            try:
                closed = inc.update(day)              # completed bars only
            except Exception as e:
                log(f"data error: {e}")
                closed = pd.DataFrame()

            if len(closed) >= 1:
                key = str(closed.index[-1])

                # verify protection is resting once placed (self.live only)
                ot0 = self.st["open_trade"]
                if (self.live and ot0 and not ot0.get("protection_verified")
                        and ot0.get("verify_after") and time.time() >= ot0["verify_after"]):
                    await self._verify_protection(ot0)

                # manage open trade: count held bars -> time exit
                if self.st["open_trade"]:
                    ot = self.st["open_trade"]
                    if ot["entry_bar"] is None:
                        ot["entry_bar"] = key
                    else:
                        held = (closed.index[-1] - pd.Timestamp(ot["entry_bar"]))
                        ot["bars_held"] = int(held.total_seconds() // 60)
                        if ot["bars_held"] >= HORIZON_BARS:
                            await self.flatten("time_exit")
                            self.st["cooldown_until"] = str(
                                closed.index[-1] + pd.Timedelta(minutes=COOLDOWN_BARS))

                # entries
                sig_bar = closed.iloc[-1]
                can_enter = (
                    self.st["open_trade"] is None and not self.st["disabled"]
                    and key not in self.st["processed"]
                    and self.st["trades_today"] < MAX_TRADES_DAY
                    and self.st["realized_pnl"] > DAILY_LOSS_STOP
                    and (now_et.hour, now_et.minute) <= LAST_ENTRY_ET
                    and (self.st["cooldown_until"] is None or
                         sig_bar.name >= pd.Timestamp(self.st["cooldown_until"]))
                )
                if can_enter:
                    d = signal_at(sig_bar)
                    if d != 0:
                        await self.enter(d, float(sig_bar["atr"]),
                                         float(sig_bar["close"]), float(sig_bar["z"]))
                        self.st["trades_today"] += 1
                self.st["processed"] = (self.st["processed"] + [key])[-500:]

            await self._reconcile()
            if self.st["realized_pnl"] <= DAILY_LOSS_STOP and self.st["open_trade"]:
                await self.flatten("daily_loss_stop")
                self.st["disabled"] = True
            self._save()
            await asyncio.sleep(POLL_SECS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--live", action="store_true",
                    help="place REAL orders (default: dry-run logging only)")
    args = ap.parse_args()
    if args.live:
        log("*** LIVE MODE: real orders, 1 micro, daily stop -$100 ***")
    runner = LiveFade(args.raw_dir, args.out_dir, live=args.live)
    try:
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
