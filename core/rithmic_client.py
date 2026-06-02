"""
Synchronous Rithmic client for Lucid Trading — drop-in replacement for
core/projectx_client.py AND TopstepXRestDataFetcher.

Wraps async_rithmic.RithmicClient behind a synchronous interface by running
the async event loop in a background daemon thread.  All blocking sync calls
dispatch coroutines via asyncio.run_coroutine_threadsafe() so the existing
synchronous MLStrategyRunner code needs no changes.

Environment variables (all required unless noted):
    RITHMIC_USERNAME       Rithmic login username
    RITHMIC_PASSWORD       Rithmic login password
    RITHMIC_SYSTEM_NAME    e.g. "Lucid Trading"  (default: Lucid Trading)
    RITHMIC_APP_NAME       e.g. "XXXX:mnq_ml_trader"  (default: XXXX:mnq_ml_trader)
    RITHMIC_GATEWAY_URI    e.g. "rituz00100.rithmic.com:443"
    LUCID_ACCOUNT_ID       Lucid account ID (optional, auto-discovered if omitted)
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from zoneinfo import ZoneInfo

_TZ_ET = ZoneInfo("America/New_York")
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from async_rithmic import RithmicClient as _AsyncRithmicClient
from async_rithmic.enums import TimeBarType, OrderType, TransactionType, OrderPlacement, DataType
from core.microstructure import compute_bar_features, compute_vpin

# Re-export shared dataclasses so that `from core.rithmic_client import ...` works
# for any code that previously imported these from core.projectx_client.
from core.projectx_client import (  # noqa: F401
    AccountState,
    BracketInstruction,
    HistoryBar,
    OrderSnapshot,
    OrderState,
    PositionState,
)

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM   = "LucidTrading"
_DEFAULT_APP_NAME = "XXXX:mnq_ml_trader"
_DEFAULT_GATEWAY  = "rprotocol.rithmic.com:443"
_SYMBOL           = "MNQ"
_EXCHANGE         = "CME"
_BAR_SIZE_MIN     = 5

# Path to precomputed microstructure parquet for historical backfill
_MICRO_PARQUET = Path(__file__).parents[1] / "data" / "processed" / "mnq_microstructure_5min.parquet"


class RithmicClientError(RuntimeError):
    pass


class TickAccumulator:
    """
    Accumulates live trade ticks within the current 5-min bar and computes
    microstructure features (OFI, Kyle's lambda, VPIN, sub-bar slices, etc.)
    when the bar closes.

    Thread-safe: add_tick() is called from the async event-loop thread;
    all other methods are called from that same thread via _on_time_bar.
    """

    def __init__(self, bar_size_minutes: int = 5) -> None:
        self._bar_size_minutes = bar_size_minutes
        self._ticks: list[dict] = []            # current bar's ticks
        self._current_bar_ts: Optional[pd.Timestamp] = None
        # 1-min OFI accumulator for cross-timeframe features
        self._min_ofi: dict[pd.Timestamp, float] = defaultdict(float)

    def add_tick(self, trade_price: float, trade_size: int,
                 aggressor: int, ssboe: int, usecs: int) -> None:
        """Add one live trade tick.  Called from the async event-loop thread."""
        ts_ns = ssboe * 1_000_000_000 + usecs * 1_000
        # Rithmic aggressor: 1=BUY(B), 2=SELL(A), 0/other=neutral(N)
        side = "B" if aggressor == 1 else ("A" if aggressor == 2 else "N")
        dt   = datetime.fromtimestamp(ssboe, tz=timezone.utc)

        bar_ts = pd.Timestamp(dt).tz_convert("UTC").floor(f"{self._bar_size_minutes}min")
        min_ts = pd.Timestamp(dt).tz_convert("UTC").floor("1min")

        if self._current_bar_ts is None:
            self._current_bar_ts = bar_ts

        # Ticks for the current bar only; late-arriving ticks for a closed bar are dropped
        if bar_ts == self._current_bar_ts:
            self._ticks.append({
                "ts_recv": ts_ns,
                "price_f": float(trade_price),
                "size":    float(trade_size),
                "side":    side,
            })
            signed = trade_size if side == "B" else (-trade_size if side == "A" else 0)
            self._min_ofi[min_ts] += signed

    def finalize_bar(self, bar_ts: pd.Timestamp) -> dict:
        """
        Compute and return microstructure features for the bar that just closed.
        Called from _on_time_bar once the bar event arrives.
        """
        features: dict = {}
        if self._ticks:
            grp = pd.DataFrame(self._ticks)
            features = compute_bar_features(grp)

        # Cross-timeframe OFI (1-min sub-bars)
        features["ofi_1m_first"] = float(self._min_ofi.get(bar_ts, 0.0))
        features["ofi_1m_last"]  = float(self._min_ofi.get(
            bar_ts + pd.Timedelta(minutes=self._bar_size_minutes - 1), 0.0))

        return features

    def advance(self, new_bar_ts: pd.Timestamp) -> None:
        """Reset for the next bar.  Prune old 1-min OFI entries."""
        self._ticks = []
        self._current_bar_ts = new_bar_ts
        cutoff = new_bar_ts - pd.Timedelta(minutes=30)
        stale  = [k for k in self._min_ofi if k < cutoff]
        for k in stale:
            del self._min_ofi[k]

    def tick_count(self) -> int:
        return len(self._ticks)


class RithmicClient:
    """
    Synchronous wrapper around async_rithmic.RithmicClient.

    Implements the same public interface as ProjectXClient (order placement,
    position/order queries) plus the data-fetcher interface used by
    TopstepXRestDataFetcher (fetch_latest_bar / update_buffer / get_latest_bar /
    get_buffer), so a single instance can serve as both self.client and
    self.data_fetcher in MLStrategyRunner.
    """

    def __init__(
        self,
        account_id: Optional[str] = None,
        contract_id: Optional[str] = None,   # accepted but unused — Rithmic resolves automatically
        bar_size_minutes: int = _BAR_SIZE_MIN,
        lookback_bars: int = 500,
    ) -> None:
        self._account_id_override = account_id or os.getenv("LUCID_ACCOUNT_ID")
        self._bar_size_minutes    = bar_size_minutes
        self._lookback_bars       = lookback_bars

        # Bar buffer — written from event-loop thread, read from main thread
        self._bar_buffer: deque[pd.Series] = deque(maxlen=lookback_bars)
        self._bar_lock   = threading.Lock()
        self._last_delivered_bar_time: Optional[pd.Timestamp] = None

        # Timestamps used by subscription watchdog (kept separate intentionally):
        #   _last_bar_received_at  — only updated when a real bar arrives
        #   _last_resubscribe_at   — updated after each repair attempt (cooldown guard)
        self._last_bar_received_at: float = time.monotonic()
        self._last_resubscribe_at: float = 0.0

        # Live tick accumulator for intrabar microstructure features
        self._tick_acc = TickAccumulator(bar_size_minutes=bar_size_minutes)

        # Internal async client and resolved contract
        self._arith: Optional[_AsyncRithmicClient] = None
        self._contract: Optional[str]              = None
        self._account_id: Optional[str]            = None

        # Start background event loop
        self._loop   = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="rithmic-loop")
        self._thread.start()

        # Block until fully connected and buffered
        logger.info("Connecting to Rithmic (%s)…", os.getenv("RITHMIC_SYSTEM_NAME", _DEFAULT_SYSTEM))
        self._run_async(self._connect_async(), timeout=120)
        logger.info("Rithmic client ready. Contract=%s, Account=%s", self._contract, self._account_id)

    # ------------------------------------------------------------------ loop

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_async(self, coro, timeout: float = 30) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    # ------------------------------------------------------------------ connection

    async def _connect_async(self) -> None:
        user     = os.environ["RITHMIC_USERNAME"]
        password = os.environ["RITHMIC_PASSWORD"]
        system   = os.getenv("RITHMIC_SYSTEM_NAME", _DEFAULT_SYSTEM)
        app_name = os.getenv("RITHMIC_APP_NAME",   _DEFAULT_APP_NAME)
        gateway  = os.getenv("RITHMIC_GATEWAY_URI", _DEFAULT_GATEWAY)

        self._arith = _AsyncRithmicClient(
            user=user,
            password=password,
            system_name=system,
            app_name=app_name,
            app_version="1.0.0",
            url=gateway,
            manual_or_auto=OrderPlacement.AUTO,
        )
        await self._arith.connect()

        # Resolve account ID
        accounts = self._arith.accounts or []
        if self._account_id_override:
            matched = [a for a in accounts if a.account_id == self._account_id_override]
            self._account_id = matched[0].account_id if matched else (accounts[0].account_id if accounts else self._account_id_override)
        else:
            self._account_id = accounts[0].account_id if accounts else None
        logger.info("Account: %s (from %d accounts)", self._account_id, len(accounts))

        # Resolve front-month contract
        self._contract = await self._arith.get_front_month_contract(_SYMBOL, _EXCHANGE)
        logger.info("Front-month contract: %s", self._contract)

        # Subscribe to PnL updates for position tracking
        await self._arith.subscribe_to_pnl_updates()

        # Backfill bar history from local parquet (has full microstructure features)
        await self._backfill_bars()

        # Subscribe to live tick stream for intrabar microstructure computation
        await self._arith.subscribe_to_market_data(
            self._contract, _EXCHANGE, DataType.LAST_TRADE
        )
        self._arith.on_tick += self._on_tick

        # Subscribe to live 5-min bars (triggers feature finalization at bar close)
        await self._arith.subscribe_to_time_bar_data(
            self._contract, _EXCHANGE, TimeBarType.MINUTE_BAR, self._bar_size_minutes
        )
        self._arith.on_time_bar += self._on_time_bar

        # Start async watchdog that re-subscribes if Rithmic silently drops time bars
        self._loop.create_task(self._time_bar_subscription_watchdog())

    async def _time_bar_subscription_watchdog(self) -> None:
        """
        Runs forever inside the async event loop. If no on_time_bar callback fires
        for >10 minutes during active ET hours (4 AM - 5 PM ET, CME break excluded),
        re-sends subscribe_to_time_bar_data() on the live connection without disconnecting.

        After MAX_CONSECUTIVE_FAILURES failed repair attempts, escalates to a full
        disconnect so the outer watchdog-bash restarts the process cleanly.

        Uses two separate timestamps:
          _last_bar_received_at   — updated only when a real bar arrives (truthful)
          _last_resubscribe_at    — updated after each repair attempt (cooldown guard)
        """
        STALE_SECONDS            = 10 * 60
        RESUBSCRIBE_COOLDOWN     = 12 * 60
        SUBSCRIBE_TIMEOUT        = 30
        MAX_CONSECUTIVE_FAILURES = 3
        CHECK_INTERVAL           = 60
        consecutive_failures     = 0

        while True:
            await asyncio.sleep(CHECK_INTERVAL)
            try:
                now_et = datetime.now(tz=_TZ_ET)
                h_et = now_et.hour
                # CME MNQ active: 4 AM – 5 PM ET (excludes 5–6 PM daily maintenance)
                in_active_hours = 4 <= h_et < 17
                stale      = (time.monotonic() - self._last_bar_received_at) > STALE_SECONDS
                cooldown_ok = (time.monotonic() - self._last_resubscribe_at) > RESUBSCRIBE_COOLDOWN

                if not (in_active_hours and stale and cooldown_ok and self._contract and self._arith):
                    continue

                logger.warning(
                    "Time-bar watchdog: no bar for >%ds at %s ET — re-subscribing (attempt %d/%d)",
                    STALE_SECONDS, now_et.strftime("%H:%M"), consecutive_failures + 1, MAX_CONSECUTIVE_FAILURES,
                )
                self._last_resubscribe_at = time.monotonic()
                await asyncio.wait_for(
                    self._arith.subscribe_to_time_bar_data(
                        self._contract, _EXCHANGE,
                        TimeBarType.MINUTE_BAR, self._bar_size_minutes,
                    ),
                    timeout=SUBSCRIBE_TIMEOUT,
                )
                logger.info("Time-bar re-subscription sent successfully")
                consecutive_failures = 0

            except asyncio.TimeoutError:
                consecutive_failures += 1
                logger.warning(
                    "Time-bar watchdog: re-subscribe timed out (%d/%d consecutive failures)",
                    consecutive_failures, MAX_CONSECUTIVE_FAILURES,
                )
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.error(
                        "Time-bar watchdog: %d consecutive failures — exiting so bash watchdog restarts process",
                        consecutive_failures,
                    )
                    try:
                        await self._arith.disconnect()
                    except Exception:
                        pass
                    os._exit(1)
            except Exception as exc:
                consecutive_failures += 1
                logger.warning("Time-bar watchdog error (%d/%d): %s", consecutive_failures, MAX_CONSECUTIVE_FAILURES, exc)
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.error(
                        "Time-bar watchdog: %d consecutive failures — exiting so bash watchdog restarts process",
                        consecutive_failures,
                    )
                    try:
                        await self._arith.disconnect()
                    except Exception:
                        pass
                    os._exit(1)

    async def _backfill_bars(self) -> None:
        """Load historical bars from the precomputed microstructure parquet."""
        if _MICRO_PARQUET.exists():
            try:
                micro = pd.read_parquet(_MICRO_PARQUET)
                micro.index = pd.to_datetime(micro.index, utc=True)
                # Take the last lookback_bars rows
                tail = micro.tail(self._lookback_bars)
                for ts, row in tail.iterrows():
                    series = row.copy()
                    series.name = ts
                    with self._bar_lock:
                        self._bar_buffer.append(series)
                logger.info("Backfilled %d bars from parquet (%s → %s)",
                            len(tail), tail.index[0].date(), tail.index[-1].date())
                return
            except Exception as exc:
                logger.warning("Parquet backfill failed: %s — falling back to HISTORY_PLANT", exc)

        # Fallback: pull OHLCV from Rithmic (no microstructure features)
        end_time   = datetime.now(tz=timezone.utc)
        start_time = end_time - timedelta(minutes=self._bar_size_minutes * (self._lookback_bars + 20))
        try:
            bars = await self._arith.get_historical_time_bars(
                self._contract, _EXCHANGE,
                start_time=start_time, end_time=end_time,
                bar_type=TimeBarType.MINUTE_BAR,
                bar_type_periods=self._bar_size_minutes,
                idle_timeout=30.0,
            )
            for bar_data in (bars or []):
                self._ingest_bar(bar_data, micro_features={})
            logger.info("Backfilled %d OHLCV bars from Rithmic (no microstructure)", len(bars or []))
        except Exception as exc:
            logger.warning("HISTORY_PLANT backfill failed: %s", exc)

    async def _on_tick(self, data: dict) -> None:
        """Route live trade tick to the tick accumulator."""
        try:
            price     = float(data.get("trade_price") or data.get("last_trade") or 0)
            size      = int(data.get("trade_size")    or data.get("trade_volume") or 0)
            aggressor = int(data.get("aggressor")     or data.get("transaction_type") or 0)
            ssboe     = int(data.get("ssboe") or 0)
            usecs     = int(data.get("usecs") or 0)
            if price > 0 and size > 0 and ssboe > 0:
                self._tick_acc.add_tick(price, size, aggressor, ssboe, usecs)
        except Exception as exc:
            logger.debug("_on_tick parse error: %s | data=%s", exc, data)

    async def _on_time_bar(self, data: dict) -> None:
        """Bar close event: finalize microstructure from accumulated ticks then ingest."""
        self._last_bar_received_at = time.monotonic()
        # Prefer marker (Unix epoch seconds → unambiguous UTC).
        # bar_end_datetime from async_rithmic is a naive local-time datetime on some
        # systems; replace(tzinfo=utc) would mislabel CDT 10:05 as "UTC 10:05", shifting
        # every bar 5 hours into the past.  marker has no such ambiguity.
        marker = data.get("marker")
        if marker is not None:
            end_dt: datetime = datetime.fromtimestamp(int(marker), tz=timezone.utc)
        else:
            raw = data.get("bar_end_datetime")
            if raw is None:
                return
            if isinstance(raw, datetime):
                # astimezone() treats naive datetimes as local time → correct UTC
                end_dt = raw.astimezone(timezone.utc)
            else:
                return

        bar_start = pd.Timestamp(end_dt).tz_convert("UTC") - pd.Timedelta(minutes=self._bar_size_minutes)

        # Compute microstructure features from ticks accumulated in this bar
        micro = self._tick_acc.finalize_bar(bar_start)
        if self._tick_acc.tick_count() > 0:
            logger.debug("Bar %s: %d ticks → %d micro features",
                         bar_start, self._tick_acc.tick_count(), len(micro))

        # Advance accumulator for the next bar
        next_bar_ts = bar_start + pd.Timedelta(minutes=self._bar_size_minutes)
        self._tick_acc.advance(next_bar_ts)

        self._ingest_bar(data, micro_features=micro)

    def _ingest_bar(self, data: dict, micro_features: Optional[dict] = None) -> None:
        """
        Convert async_rithmic bar dict to pd.Series (with microstructure) and append to buffer.

        micro_features: dict from TickAccumulator.finalize_bar() — merged into the row.
                        Pass {} for historical OHLCV-only bars (missing features → NaN).
        """
        try:
            # Prefer marker (Unix epoch → unambiguous UTC); see _on_time_bar for rationale.
            marker = data.get("marker")
            if marker is not None:
                end_dt: datetime = datetime.fromtimestamp(int(marker), tz=timezone.utc)
            else:
                raw = data.get("bar_end_datetime")
                if raw is None:
                    return
                if isinstance(raw, datetime):
                    end_dt = raw.astimezone(timezone.utc)
                else:
                    return

            bar_start = pd.Timestamp(end_dt).tz_convert("UTC") - pd.Timedelta(minutes=self._bar_size_minutes)

            ohlcv = {
                "open":   float(data.get("open_price",  data.get("open",  0))),
                "high":   float(data.get("high_price",  data.get("high",  0))),
                "low":    float(data.get("low_price",   data.get("low",   0))),
                "close":  float(data.get("close_price", data.get("close", 0))),
                "volume": float(data.get("volume", 0)),
            }

            # Merge OHLCV + microstructure into one Series
            combined = {**ohlcv, **(micro_features or {})}
            row = pd.Series(combined, name=bar_start)

            with self._bar_lock:
                if self._bar_buffer and self._bar_buffer[-1].name >= bar_start:
                    return
                self._bar_buffer.append(row)
                self._update_rolling_micro_features()

        except Exception as exc:
            logger.warning("Failed to ingest bar: %s | data=%s", exc, data)

    def _update_rolling_micro_features(self) -> None:
        """
        Update VPIN and ofi_15m on the last bar in the buffer.
        Must be called with _bar_lock held.
        """
        if len(self._bar_buffer) < 5:
            return
        try:
            buf = list(self._bar_buffer)
            buy_s  = pd.Series([b.get("buy_vol",  np.nan) for b in buf])
            sell_s = pd.Series([b.get("sell_vol", np.nan) for b in buf])
            ofi_s  = pd.Series([b.get("ofi",      np.nan) for b in buf])

            vpin_d = compute_vpin(buy_s, sell_s, windows=(5, 20))
            ofi_15 = ofi_s.shift(1).rolling(3, min_periods=3).sum()

            last = self._bar_buffer[-1].copy()
            last["vpin_5"]   = float(vpin_d["vpin_5"].iloc[-1])  if not vpin_d["vpin_5"].isna().iloc[-1]  else np.nan
            last["vpin_20"]  = float(vpin_d["vpin_20"].iloc[-1]) if not vpin_d["vpin_20"].isna().iloc[-1] else np.nan
            last["ofi_15m"]  = float(ofi_15.iloc[-1])            if not pd.isna(ofi_15.iloc[-1])          else np.nan
            self._bar_buffer[-1] = last
        except Exception as exc:
            logger.debug("Rolling micro feature update failed: %s", exc)

    # ------------------------------------------------------------------ data-fetcher interface

    def initialize_buffer(self) -> None:
        """No-op: buffer is populated during __init__ via _backfill_bars."""

    def fetch_latest_bar(self) -> Optional[pd.Series]:
        """
        Return the most recent bar if it is newer than the last one delivered,
        otherwise return None.  Mirrors TopstepXRestDataFetcher.fetch_latest_bar().
        """
        with self._bar_lock:
            if not self._bar_buffer:
                return None
            latest = self._bar_buffer[-1]

        if self._last_delivered_bar_time is None or latest.name > self._last_delivered_bar_time:
            return latest
        return None

    def update_buffer(self, bar: pd.Series) -> None:
        """
        Mark `bar` as delivered.  The bar is already in the internal buffer
        (added via the on_time_bar callback); we only update the delivery pointer.
        """
        self._last_delivered_bar_time = bar.name

    def get_latest_bar(self) -> Optional[pd.Series]:
        with self._bar_lock:
            return self._bar_buffer[-1] if self._bar_buffer else None

    def get_buffer(self) -> pd.DataFrame:
        with self._bar_lock:
            if not self._bar_buffer:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            return pd.DataFrame(list(self._bar_buffer))

    # ------------------------------------------------------------------ ProjectXClient interface: orders

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        client_order_id: Optional[str] = None,
        stop_loss_bracket: Optional[BracketInstruction] = None,  # kept for API compat, ignored
        take_profit_bracket: Optional[BracketInstruction] = None,  # kept for API compat, ignored
        account_id: Optional[int] = None,
        contract_id: Optional[str] = None,   # unused for Rithmic
        linked_order_id: Optional[int] = None,
    ) -> OrderState:
        # Bracket orders (template 330) are silently rejected on paper/sim accounts.
        # Submit a plain MARKET entry only; caller is responsible for placing the
        # stop via place_stop_order() after confirming the fill.
        order_id = client_order_id or str(uuid.uuid4())[:16]
        tx_type  = TransactionType.BUY if side.upper() == "BUY" else TransactionType.SELL

        logger.info("place_order: %s %d×%s MARKET  id=%s", side.upper(), quantity, self._contract, order_id)

        responses = self._run_async(
            self._arith.submit_order(
                order_id=order_id,
                symbol=self._contract,
                exchange=_EXCHANGE,
                qty=quantity,
                transaction_type=tx_type,
                order_type=OrderType.MARKET,
                account_id=self._account_id,
            ),
            timeout=15,
        )

        basket_id = None
        if responses:
            first = responses[0]
            basket_id = getattr(first, "basket_id", None) or getattr(first, "order_id", None)

        return OrderState(
            order_id=str(basket_id or order_id),
            symbol=symbol,
            side=side.upper(),
            quantity=quantity,
            status="ACCEPTED",
            avg_fill_price=None,
        )

    def place_stop_order(
        self,
        stop_price: float,
        quantity: int,
        side: str = "SELL",
        client_order_id: Optional[str] = None,
    ) -> str:
        """Place a STOP_MARKET order to protect an open position.  Returns order_id."""
        order_id = client_order_id or str(uuid.uuid4())[:16]
        tx_type  = TransactionType.BUY if side.upper() == "BUY" else TransactionType.SELL
        logger.info(
            "place_stop_order: %s %d×%s STOP_MARKET @ %.2f  id=%s",
            side.upper(), quantity, self._contract, stop_price, order_id,
        )
        responses = self._run_async(
            self._arith.submit_order(
                order_id=order_id,
                symbol=self._contract,
                exchange=_EXCHANGE,
                qty=quantity,
                transaction_type=tx_type,
                order_type=OrderType.STOP_MARKET,
                account_id=self._account_id,
                trigger_price=stop_price,  # STOP_MARKET requires trigger_price, not price
            ),
            timeout=15,
        )
        basket_id = None
        if responses:
            first = responses[0]
            basket_id = getattr(first, "basket_id", None) or getattr(first, "order_id", None)
        resolved = str(basket_id or order_id)
        logger.info("Stop order accepted: id=%s", resolved)
        return resolved

    def cancel_order(self, order_id: str, account_id: Optional[int] = None) -> None:
        self._run_async(
            self._arith.cancel_order(
                basket_id=order_id,
                account_id=self._account_id,
            ),
            timeout=10,
        )

    def search_open_orders(self, account_id: Optional[int] = None) -> List[OrderSnapshot]:
        """Return open orders for the current account and symbol."""
        try:
            orders = self._run_async(
                self._arith.list_orders(account_id=self._account_id),
                timeout=15,
            )
            result = []
            for o in (orders or []):
                snap = self._order_to_snapshot(o)
                if snap is not None:
                    result.append(snap)
            return result
        except Exception as exc:
            logger.warning("search_open_orders failed: %s", exc)
            return []

    def search_orders(
        self,
        start_timestamp: datetime,
        end_timestamp: Optional[datetime] = None,
        account_id: Optional[int] = None,
    ) -> List[OrderSnapshot]:
        """Return filled orders since start_timestamp using fill history."""
        try:
            if start_timestamp.tzinfo is None:
                start_timestamp = start_timestamp.replace(tzinfo=timezone.utc)
            end_ts = end_timestamp or datetime.now(tz=timezone.utc)
            if end_ts.tzinfo is None:
                end_ts = end_ts.replace(tzinfo=timezone.utc)

            fills = self._run_async(
                self._arith.get_fill_history(
                    start_time=start_timestamp,
                    end_time=end_ts,
                    account_id=self._account_id,
                ),
                timeout=15,
            )
            result = []
            for f in (fills or []):
                snap = self._fill_to_snapshot(f)
                if snap is not None:
                    result.append(snap)
            return result
        except Exception as exc:
            logger.warning("search_orders failed: %s", exc)
            return []

    # ------------------------------------------------------------------ ProjectXClient interface: positions

    def search_open_positions(self) -> List[Dict[str, Any]]:
        """Return list of open positions (non-empty = we are in a trade)."""
        try:
            positions = self._run_async(
                self._arith.list_positions(account_id=self._account_id),
                timeout=15,
            )
            result = []
            for p in (positions or []):
                # Filter for our symbol; skip flat (qty == 0) positions.
                # InstrumentPnLPositionUpdate proto fields (in priority order):
                #   net_quantity          — signed net position (positive=long, negative=short)
                #   buy_qty / sell_qty    — cumulative filled quantities
                #   open_position_quantity — open contracts (long perspective)
                # open_long_quantity / quantity do NOT exist in this proto and return None.
                sym = getattr(p, "symbol", "") or ""
                if _SYMBOL not in sym.upper():
                    continue
                def _int(attr: str) -> int:
                    v = getattr(p, attr, None)
                    try:
                        return int(v) if v is not None else 0
                    except (TypeError, ValueError):
                        return 0
                net_qty = _int("net_quantity")
                if net_qty == 0:
                    # fallback: buy - sell
                    net_qty = _int("buy_qty") - _int("sell_qty")
                if net_qty == 0:
                    net_qty = _int("open_position_quantity")
                qty = abs(net_qty)
                if qty > 0:
                    result.append({
                        "contract_id": sym,
                        "size": qty,
                        "average_price": float(getattr(p, "avg_open_fill_price", 0) or 0),
                    })
            return result
        except Exception as exc:
            logger.warning("search_open_positions failed: %s", exc)
            return []

    # ------------------------------------------------------------------ ProjectXClient interface: account

    def get_account_state(self) -> AccountState:
        try:
            summaries = self._run_async(
                self._arith.list_account_summary(account_id=self._account_id),
                timeout=15,
            )
            s = summaries[0] if summaries else None
            if s is None:
                raise RithmicClientError("No account summary returned")

            balance      = float(getattr(s, "net_liq_liquidating_value",  getattr(s, "balance",       0)) or 0)
            open_pnl     = float(getattr(s, "open_position_pnl",          getattr(s, "open_pnl",      0)) or 0)
            realized_pnl = float(getattr(s, "realized_pnl",               0) or 0)
            daily_pnl    = float(getattr(s, "daily_pnl",                  realized_pnl + open_pnl) or 0)

            return AccountState(
                account_id=self._account_id or "",
                equity=balance + open_pnl,
                balance=balance,
                open_pnl=open_pnl,
                realized_pnl=realized_pnl,
                daily_pnl=daily_pnl,
            )
        except Exception as exc:
            logger.warning("get_account_state failed: %s", exc)
            return AccountState(
                account_id=self._account_id or "",
                equity=0.0, balance=0.0, open_pnl=0.0,
                realized_pnl=0.0, daily_pnl=0.0,
            )

    # ------------------------------------------------------------------ bars (ProjectXClient compat)

    def retrieve_bars(
        self,
        *,
        contract_id: Optional[str] = None,
        live: bool = True,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        unit: int = 2,
        unit_number: int = 5,
        limit: int = 500,
        include_partial_bar: bool = False,
    ) -> List[HistoryBar]:
        """
        Fetch historical bars.  Satisfies the ProjectXClient.retrieve_bars() signature
        used by TopstepXRestDataFetcher; the data-fetcher methods above are preferred
        for the live runner.
        """
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        end_ts = end_time or datetime.now(tz=timezone.utc)
        if end_ts.tzinfo is None:
            end_ts = end_ts.replace(tzinfo=timezone.utc)

        try:
            bars = self._run_async(
                self._arith.get_historical_time_bars(
                    self._contract, _EXCHANGE,
                    start_time=start_time,
                    end_time=end_ts,
                    bar_type=TimeBarType.MINUTE_BAR,
                    bar_type_periods=unit_number,
                    idle_timeout=30.0,
                ),
                timeout=60,
            )
            result = []
            for b in (bars or []):
                end_dt = b.get("bar_end_datetime") or datetime.fromtimestamp(int(b.get("marker", 0)), tz=timezone.utc)
                if isinstance(end_dt, datetime) and end_dt.tzinfo is None:
                    end_dt = end_dt.replace(tzinfo=timezone.utc)
                _ts = pd.Timestamp(end_dt)
                bar_start = (_ts.tz_localize("UTC") if _ts.tzinfo is None else _ts.tz_convert("UTC")) - pd.Timedelta(minutes=unit_number)
                result.append(HistoryBar(
                    timestamp=bar_start.to_pydatetime(),
                    open=float(b.get("open_price",  b.get("open",  0))),
                    high=float(b.get("high_price",  b.get("high",  0))),
                    low=float(b.get("low_price",    b.get("low",   0))),
                    close=float(b.get("close_price", b.get("close", 0))),
                    volume=float(b.get("volume", 0)),
                ))
            return sorted(result, key=lambda x: x.timestamp)
        except Exception as exc:
            logger.error("retrieve_bars failed: %s", exc)
            return []

    # ------------------------------------------------------------------ helpers

    def _order_to_snapshot(self, o: Any) -> Optional[OrderSnapshot]:
        """Convert a Rithmic order notification to an OrderSnapshot."""
        try:
            basket_id = getattr(o, "basket_id", None) or ""
            tx = getattr(o, "transaction_type", "") or ""
            # transaction_type is 'B' (buy) or 'S' (sell) as a string in Rithmic
            # Map to ProjectX convention: 0=BUY, 1=SELL
            side = 1 if str(tx).upper() in ("S", "SELL", "2") else 0

            filled_price_raw = getattr(o, "avg_fill_price", None) or getattr(o, "fill_price", None)
            filled_price = float(filled_price_raw) if filled_price_raw else None

            filled_qty_raw = getattr(o, "filled_quantity", None) or getattr(o, "fill_size", None) or 0
            filled_qty = int(filled_qty_raw) if filled_qty_raw else 0

            ts_raw = getattr(o, "ssboe", None) or getattr(o, "update_timestamp", None)
            if ts_raw and isinstance(ts_raw, (int, float)):
                ts = datetime.fromtimestamp(int(ts_raw), tz=timezone.utc)
            elif isinstance(ts_raw, datetime):
                ts = ts_raw if ts_raw.tzinfo else ts_raw.replace(tzinfo=timezone.utc)
            else:
                ts = datetime.now(tz=timezone.utc)

            return OrderSnapshot(
                order_id=0,
                account_id=0,
                contract_id=getattr(o, "symbol", _SYMBOL),
                symbol_id=None,
                status=0,
                order_type=0,
                side=side,
                size=int(getattr(o, "quantity", 0) or 0),
                filled_volume=filled_qty,
                filled_price=filled_price,
                limit_price=None,
                stop_price=None,
                creation_timestamp=ts,
                update_timestamp=ts,
                custom_tag=basket_id,
            )
        except Exception as exc:
            logger.warning("_order_to_snapshot failed: %s | obj=%s", exc, o)
            return None

    def _fill_to_snapshot(self, f: Any) -> Optional[OrderSnapshot]:
        """Convert a Rithmic fill history entry to an OrderSnapshot."""
        try:
            tx = getattr(f, "transaction_type", "") or getattr(f, "buy_sell_type", "") or ""
            side = 1 if str(tx).upper() in ("S", "SELL", "2") else 0

            fill_price_raw = getattr(f, "fill_price", None) or getattr(f, "avg_fill_price", None)
            fill_price = float(fill_price_raw) if fill_price_raw else None

            fill_qty_raw = getattr(f, "fill_size", None) or getattr(f, "quantity", None) or 0
            fill_qty = int(fill_qty_raw) if fill_qty_raw else 0

            ssboe = getattr(f, "ssboe", None)
            usecs = getattr(f, "usecs", 0) or 0
            if ssboe:
                ts = datetime.fromtimestamp(int(ssboe) + int(usecs) / 1_000_000, tz=timezone.utc)
            else:
                ts = datetime.now(tz=timezone.utc)

            basket_id = getattr(f, "basket_id", None) or getattr(f, "order_id", None) or ""

            return OrderSnapshot(
                order_id=0,
                account_id=0,
                contract_id=getattr(f, "symbol", _SYMBOL),
                symbol_id=None,
                status=0,
                order_type=0,
                side=side,
                size=fill_qty,
                filled_volume=fill_qty,
                filled_price=fill_price,
                limit_price=None,
                stop_price=None,
                creation_timestamp=ts,
                update_timestamp=ts,
                custom_tag=str(basket_id),
            )
        except Exception as exc:
            logger.warning("_fill_to_snapshot failed: %s | obj=%s", exc, f)
            return None

    # ------------------------------------------------------------------ lifecycle

    def disconnect(self) -> None:
        """Gracefully disconnect from Rithmic."""
        try:
            self._run_async(self._arith.disconnect(), timeout=10)
        except Exception as exc:
            logger.warning("disconnect error: %s", exc)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
