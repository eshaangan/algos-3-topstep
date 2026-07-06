"""Standalone Rithmic L2 (order book) + trade recorder.

WHY THIS EXISTS
---------------
Price-only strategies on liquid index futures are arbitraged away (confirmed on
16 years of ES). The one place a real intraday edge can still live is the order
book — and we have ZERO L2 history because the live runner only ever subscribed
to LAST_TRADE. Rithmic streams full depth live but does not give deep historical
L2, so the only way to get a validatable L2 dataset is to start recording it now.
Every day not recording is data we can never get back.

This is a SEPARATE process from the trading runner — it places no orders and
shares no state, so it cannot affect live trading. Run it alongside the trader.

WHAT IT CAPTURES (per front-month MNQ, CME)
  - Order book: up to 10 levels of {bid,ask} × {price,size,orders} on every update,
    with exchange ssboe/usecs timestamps and update_type.
  - Trades: price, size, aggressor side, exchange timestamp.
Both are written to chunked parquet under data/l2_raw/ (rotated by UTC date), so a
crash loses at most one buffer flush.

USAGE
    python data_collection/record_l2.py                  # MNQ, default out dir
    python data_collection/record_l2.py --symbol MES
    OUT_DIR=/path python data_collection/record_l2.py --flush-secs 30

ENV (same as the trader): RITHMIC_USERNAME, RITHMIC_PASSWORD, RITHMIC_SYSTEM_NAME,
RITHMIC_APP_NAME, RITHMIC_GATEWAY_URI.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from async_rithmic import RithmicClient, DataType, ReconnectionSettings

_DEFAULT_SYSTEM = "LucidTrading"
_DEFAULT_APP = "XXXX:mnq_l2_recorder"
_DEFAULT_GATEWAY = "rprotocol.rithmic.com:443"
_EXCHANGE = "CME"
_N_LEVELS = 10

ROOT = Path(__file__).resolve().parents[1]


class Recorder:
    def __init__(self, symbol: str, out_dir: Path, flush_secs: float):
        self.symbol = symbol
        self.out_dir = out_dir
        self.flush_secs = flush_secs
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._book_buf: list[dict] = []
        self._trade_buf: list[dict] = []
        self._bbo_buf: list[dict] = []
        self._last_flush = time.monotonic()
        self._last_data = time.monotonic()   # staleness watchdog
        self._stale_exit = False
        self._book_count = 0
        self._trade_count = 0
        self._bbo_count = 0
        self._running = True
        self._client: RithmicClient | None = None
        self._contract: str | None = None

    # ── callbacks ────────────────────────────────────────────────────────────

    async def _on_order_book(self, msg) -> None:
        try:
            row = {
                "recv_ns": time.time_ns(),
                "ssboe": int(getattr(msg, "ssboe", 0) or 0),
                "usecs": int(getattr(msg, "usecs", 0) or 0),
                "update_type": int(getattr(msg, "update_type", 0) or 0),
            }
            bp, bs, bo = list(msg.bid_price), list(msg.bid_size), list(msg.bid_orders)
            ap, as_, ao = list(msg.ask_price), list(msg.ask_size), list(msg.ask_orders)
            for i in range(_N_LEVELS):
                row[f"bid_px_{i}"] = bp[i] if i < len(bp) else float("nan")
                row[f"bid_sz_{i}"] = bs[i] if i < len(bs) else 0
                row[f"bid_ord_{i}"] = bo[i] if i < len(bo) else 0
                row[f"ask_px_{i}"] = ap[i] if i < len(ap) else float("nan")
                row[f"ask_sz_{i}"] = as_[i] if i < len(as_) else 0
                row[f"ask_ord_{i}"] = ao[i] if i < len(ao) else 0
            self._book_buf.append(row)
            self._book_count += 1
            self._last_data = time.monotonic()
        except Exception as exc:  # never let a bad message kill the recorder
            print(f"[book parse err] {exc}", flush=True)

    async def _on_tick(self, data: dict) -> None:
        """Both LAST_TRADE and BBO arrive here; split by data_type."""
        try:
            dt = data.get("data_type")
            if dt == DataType.BBO:
                self._bbo_buf.append({
                    "recv_ns": time.time_ns(),
                    "ssboe": int(data.get("ssboe") or 0),
                    "usecs": int(data.get("usecs") or 0),
                    "bid_px": float(data.get("bid_price") or "nan"),
                    "bid_sz": int(data.get("bid_size") or 0),
                    "bid_ord": int(data.get("bid_orders") or 0),
                    "ask_px": float(data.get("ask_price") or "nan"),
                    "ask_sz": int(data.get("ask_size") or 0),
                    "ask_ord": int(data.get("ask_orders") or 0),
                })
                self._bbo_count += 1
                self._last_data = time.monotonic()
                return
            # LAST_TRADE
            price = float(data.get("trade_price") or data.get("last_trade") or 0)
            size = int(data.get("trade_size") or data.get("trade_volume") or 0)
            if price <= 0 or size <= 0:
                return
            self._trade_buf.append({
                "recv_ns": time.time_ns(),
                "ssboe": int(data.get("ssboe") or 0),
                "usecs": int(data.get("usecs") or 0),
                "price": price,
                "size": size,
                "aggressor": int(data.get("aggressor") or data.get("transaction_type") or 0),
            })
            self._trade_count += 1
            self._last_data = time.monotonic()
        except Exception as exc:
            print(f"[tick parse err] {exc}", flush=True)

    # ── disk ─────────────────────────────────────────────────────────────────

    def _flush(self) -> None:
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        stamp = int(time.time() * 1000)
        for buf, kind in ((self._book_buf, "book"), (self._trade_buf, "trade"),
                          (self._bbo_buf, "bbo")):
            if not buf:
                continue
            df = pd.DataFrame(buf)
            path = self.out_dir / f"{kind}_{self.symbol}_{date}_{stamp}.parquet"
            try:
                df.to_parquet(path, compression="zstd", index=False)
            except Exception:
                df.to_parquet(path, index=False)  # fallback if zstd unavailable
            buf.clear()
        self._last_flush = time.monotonic()
        print(f"[{datetime.now(timezone.utc):%H:%M:%S}Z] flushed "
              f"book={self._book_count} bbo={self._bbo_count} trade={self._trade_count} (cumulative)",
              flush=True)

    def _maybe_flush(self) -> None:
        if (time.monotonic() - self._last_flush) >= self.flush_secs or \
           (len(self._book_buf) + len(self._trade_buf)) >= 100_000:
            self._flush()

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def run(self) -> None:
        user = os.environ["RITHMIC_USERNAME"]
        pw = os.environ["RITHMIC_PASSWORD"]
        system = os.getenv("RITHMIC_SYSTEM_NAME", _DEFAULT_SYSTEM)
        app = os.getenv("RITHMIC_APP_NAME", _DEFAULT_APP)
        gateway = os.getenv("RITHMIC_GATEWAY_URI", _DEFAULT_GATEWAY)

        self._client = RithmicClient(
            user=user, password=pw, system_name=system,
            app_name=app, app_version="1.0.0", url=gateway,
            reconnection_settings=ReconnectionSettings(max_retries=None, backoff_type="exponential",
                                                       interval=2, max_delay=30),
        )
        # TICKER_PLANT only: the recorder needs nothing else, and holding an
        # ORDER_PLANT login here would collide with fade_live_runner's session
        # (Rithmic allows one login per plant per user; two would ping-pong).
        from async_rithmic import SysInfraType
        await self._client.connect(plants=[SysInfraType.TICKER_PLANT])
        self._contract = await self._client.get_front_month_contract(self.symbol, _EXCHANGE)
        print(f"Connected. Recording {self._contract} @ {_EXCHANGE} → {self.out_dir}", flush=True)

        self._client.on_order_book += self._on_order_book
        self._client.on_tick += self._on_tick
        await self._client.subscribe_to_market_data(
            self._contract, _EXCHANGE,
            DataType.ORDER_BOOK.value | DataType.BBO.value | DataType.LAST_TRADE.value,
        )
        print("Subscribed to ORDER_BOOK + BBO + LAST_TRADE. Recording…", flush=True)

        stale_secs = 600.0   # exit (→ watchdog restart) if no data this long; survives CME break
        self._last_data = time.monotonic()
        while self._running:
            await asyncio.sleep(1.0)
            self._maybe_flush()
            if (time.monotonic() - self._last_data) > stale_secs:
                print(f"[STALE] no market data for {stale_secs:.0f}s — exiting for watchdog restart",
                      flush=True)
                self._stale_exit = True
                break

        self._flush()  # final flush on shutdown
        try:
            await self._client.disconnect()
        except Exception:
            pass
        print("Recorder stopped.", flush=True)
        if self._stale_exit:
            import sys as _sys
            _sys.exit(1)   # non-zero → .l2wd watchdog restarts a fresh connection

    def stop(self, *_):
        print("Shutdown signal received…", flush=True)
        self._running = False


def main() -> None:
    ap = argparse.ArgumentParser(description="Rithmic L2 order book + trade recorder")
    ap.add_argument("--symbol", default="MNQ")
    ap.add_argument("--out-dir", default=os.getenv("OUT_DIR", str(ROOT / "data" / "l2_raw")))
    ap.add_argument("--flush-secs", type=float, default=60.0)
    args = ap.parse_args()

    rec = Recorder(args.symbol, Path(args.out_dir), args.flush_secs)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, rec.stop)
        except NotImplementedError:
            signal.signal(sig, rec.stop)
    loop.run_until_complete(rec.run())


if __name__ == "__main__":
    main()
