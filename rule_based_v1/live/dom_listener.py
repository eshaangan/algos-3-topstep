"""DOM Listener — Level 2 Order Flow Imbalance via ProjectX GatewayDepth.

Subscribes to the SignalR 'GatewayDepth' WebSocket event from ProjectX API.
Maintains a real-time order book and exposes an OFI metric.

GatewayDepth event fields (DomType enum):
  type=1 (Ask): offer at price level
  type=2 (Bid): bid at price level
  type=3 (BestAsk): top-of-book ask
  type=4 (BestBid): top-of-book bid

OFI metric: (bid_vol_topN - ask_vol_topN) / (bid_vol_topN + ask_vol_topN)
  +1.0 = full buy pressure, -1.0 = full sell pressure

Usage:
    listener = DOMListener(contract_id=..., hub_url=..., token=..., top_levels=5)
    listener.start()
    ofi = listener.get_current_ofi()
    listener.stop()

Run standalone to test connection:
    python rule_based_v1/live/dom_listener.py --contract-id CON.F.US.MNQ.M26
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict

logger = logging.getLogger(__name__)

DOM_TYPE_ASK      = 1
DOM_TYPE_BID      = 2
DOM_TYPE_BEST_ASK = 3
DOM_TYPE_BEST_BID = 4


class DOMListener:
    """Real-time Level 2 order book listener for a single futures contract."""

    def __init__(
        self,
        contract_id: str,
        hub_url: str,
        token: str,
        top_levels: int = 5,
        reconnect_delay: int = 5,
    ):
        self.contract_id    = contract_id
        self.hub_url        = hub_url
        self.token          = token
        self.top_levels     = top_levels
        self.reconnect_delay = reconnect_delay

        # Order book: {price: volume} per side
        self._bids: dict[float, int] = defaultdict(int)
        self._asks: dict[float, int] = defaultdict(int)
        self._lock = threading.Lock()
        self._ofi: float = 0.0
        self._connected: bool = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the WebSocket listener in a background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="DOMListener")
        self._thread.start()
        logger.info(f"DOMListener started for {self.contract_id}")

    def stop(self) -> None:
        """Signal the listener to stop and wait for thread exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        logger.info("DOMListener stopped")

    def get_current_ofi(self) -> float:
        """Return the current order flow imbalance metric (-1 to +1).

        Returns 0.0 when disconnected (neutral — does not block trades).
        """
        return self._ofi

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # WebSocket loop (background thread)
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Persistent reconnect loop."""
        while not self._stop_event.is_set():
            try:
                self._connect()
            except Exception as e:
                logger.warning(f"DOMListener connection error: {e}")
            finally:
                self._connected = False
                self._ofi = 0.0

            if not self._stop_event.is_set():
                logger.info(f"DOMListener reconnecting in {self.reconnect_delay}s...")
                time.sleep(self.reconnect_delay)

    def _connect(self) -> None:
        """Open SignalR connection and subscribe to GatewayDepth."""
        try:
            from signalrcore.hub_connection_builder import HubConnectionBuilder  # type: ignore
        except ImportError:
            raise ImportError(
                "signalrcore is required for DOM streaming. "
                "Install with: pip install signalrcore"
            )

        connection = (
            HubConnectionBuilder()
            .with_url(
                self.hub_url,
                options={"headers": {"Authorization": f"Bearer {self.token}"}},
            )
            .build()
        )

        connection.on("GatewayDepth", self._on_depth_event)
        connection.on_open(self._on_open)
        connection.on_close(self._on_close)
        connection.on_error(self._on_error)

        connection.start()

        # Subscribe to market depth for our contract
        connection.send("SubscribeContractMarketDepth", [self.contract_id])
        logger.info(f"Subscribed to GatewayDepth for {self.contract_id}")
        self._connected = True

        # Block until stop requested
        while not self._stop_event.is_set():
            time.sleep(1)

        connection.stop()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_open(self) -> None:
        logger.info("DOMListener WebSocket connected")
        self._connected = True

    def _on_close(self) -> None:
        logger.warning("DOMListener WebSocket disconnected")
        self._connected = False
        self._ofi = 0.0

    def _on_error(self, error) -> None:
        logger.error(f"DOMListener WebSocket error: {error}")

    def _on_depth_event(self, data: list) -> None:
        """Handle GatewayDepth message and update order book + OFI."""
        if not data:
            return

        for event in (data if isinstance(data, list) else [data]):
            try:
                dom_type = int(event.get("type", 0))
                price    = float(event.get("price", 0.0))
                volume   = int(event.get("volume", 0) or event.get("currentVolume", 0))

                if price <= 0:
                    continue

                with self._lock:
                    if dom_type in (DOM_TYPE_BID, DOM_TYPE_BEST_BID):
                        if volume == 0:
                            self._bids.pop(price, None)
                        else:
                            self._bids[price] = volume
                    elif dom_type in (DOM_TYPE_ASK, DOM_TYPE_BEST_ASK):
                        if volume == 0:
                            self._asks.pop(price, None)
                        else:
                            self._asks[price] = volume

                    self._ofi = self._compute_ofi()
            except Exception as e:
                logger.debug(f"DOMListener parse error: {e}")

    # ------------------------------------------------------------------
    # OFI computation
    # ------------------------------------------------------------------

    def _compute_ofi(self) -> float:
        """Compute OFI from top N bid/ask levels (must hold self._lock)."""
        if not self._bids or not self._asks:
            return 0.0

        top_bids = sorted(self._bids.keys(), reverse=True)[: self.top_levels]
        top_asks = sorted(self._asks.keys())[: self.top_levels]

        bid_vol = sum(self._bids[p] for p in top_bids)
        ask_vol = sum(self._asks[p] for p in top_asks)

        total = bid_vol + ask_vol
        if total == 0:
            return 0.0

        return (bid_vol - ask_vol) / total


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(ROOT))

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(description="Test DOMListener connection")
    parser.add_argument("--contract-id", default=os.getenv("TOPSTEPX_CONTRACT_ID"))
    parser.add_argument("--duration", type=int, default=60, help="Seconds to listen")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Pull hub URL and token from ProjectXClient
    sys.path.insert(0, str(ROOT / "core"))
    try:
        from core.projectx_client import ProjectXClient  # type: ignore
        client = ProjectXClient()
        token   = client.token
        hub_url = client.base_url.replace("/api", "/realtime")  # typical SignalR endpoint
    except Exception as e:
        logger.error(f"Failed to get client token: {e}")
        sys.exit(1)

    contract_id = args.contract_id
    if not contract_id:
        logger.error("--contract-id required (or set TOPSTEPX_CONTRACT_ID env var)")
        sys.exit(1)

    listener = DOMListener(
        contract_id=contract_id,
        hub_url=hub_url,
        token=token,
        top_levels=5,
    )
    listener.start()

    try:
        for i in range(args.duration):
            time.sleep(1)
            ofi = listener.get_current_ofi()
            connected = listener.is_connected
            print(f"[{i+1:>3}s] connected={connected}  OFI={ofi:+.3f}")
    finally:
        listener.stop()
