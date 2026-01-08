"""
Real-time data fetcher for TopstepX live trading using SignalR WebSockets.

Fetches live tick data from TopstepX Market Hub and aggregates into OHLCV bars.
"""

import logging
import os
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from collections import deque
import json

import pandas as pd
import pytz

# SignalR for Python (requires signalrcore)
from signalrcore.hub_connection_builder import HubConnectionBuilder
from signalrcore.protocol.messagepack_protocol import MessagePackHubProtocol

logger = logging.getLogger(__name__)


class TopstepXDataFetcher:
    """
    Fetches real-time market data from TopstepX SignalR Market Hub.

    Aggregates tick/quote data into OHLCV bars for trading.
    """

    def __init__(
        self,
        contract_id: str,
        bar_size_minutes: int = 5,
        lookback_bars: int = 100,
        session_token: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the TopstepX data fetcher.

        Args:
            contract_id: TopstepX contract ID (e.g., "CON.F.US.EP.H26")
            bar_size_minutes: Bar size in minutes (default: 5)
            lookback_bars: Number of historical bars to maintain
            session_token: TopstepX JWT session token
            base_url: TopstepX base URL for WebSocket connection
        """
        self.contract_id = contract_id
        self.bar_size_minutes = bar_size_minutes
        self.lookback_bars = lookback_bars

        # Get credentials from environment
        self.session_token = session_token or os.getenv("TOPSTEPX_SESSION_TOKEN")
        self.base_url = base_url or os.getenv("TOPSTEPX_PROJECTX_BASE_URL", "https://api.topstepx.com")

        if not self.session_token:
            raise ValueError("TOPSTEPX_SESSION_TOKEN not found in environment")

        # Extract base domain for WebSocket URL
        # Convert REST API URL to WebSocket RTC URL following TopstepX pattern:
        # - Production: https://api.topstepx.com -> wss://gateway-rtc.s2f.projectx.com
        # - Demo: https://gateway-api-demo.s2f.projectx.com -> wss://gateway-rtc-demo.s2f.projectx.com

        # NOTE: TopstepX appears to use gateway-rtc-demo for all environments (production and demo)
        # The production RTC endpoint (gateway-rtc.s2f.projectx.com) does not exist
        # SignalR with skipNegotiation uses https:// and converts to wss:// internally
        if "api.topstepx.com" in self.base_url:
            # Production REST API -> use demo RTC (only one that exists)
            ws_base = "https://gateway-rtc-demo.s2f.projectx.com"
        elif "gateway-api-demo.s2f.projectx.com" in self.base_url:
            # Demo environment
            ws_base = "https://gateway-rtc-demo.s2f.projectx.com"
        elif "gateway-api.s2f.projectx.com" in self.base_url:
            # Production environment (alternative) -> use demo RTC
            ws_base = "https://gateway-rtc-demo.s2f.projectx.com"
        else:
            # Fallback: use demo RTC endpoint (only one that resolves)
            ws_base = "https://gateway-rtc-demo.s2f.projectx.com"

        # Include access token in URL query string as shown in TopstepX docs
        # This is required even with skip_negotiation
        self.market_hub_url = f"{ws_base}/hubs/market?access_token={self.session_token}"

        # Rolling buffer of completed bars
        self.bars_buffer = pd.DataFrame()

        # Current bar being built
        self.current_bar: Optional[Dict[str, Any]] = None
        self.current_bar_start: Optional[datetime] = None

        # Latest quote data
        self.latest_quote: Optional[Dict[str, Any]] = None

        # SignalR connection
        self.hub_connection = None
        self.is_connected = False

        # Chicago timezone (market hours)
        self.chicago_tz = pytz.timezone('America/Chicago')

        logger.info(
            f"TopstepXDataFetcher initialized: contract={contract_id}, "
            f"bar_size={bar_size_minutes}m, lookback={lookback_bars}"
        )

    def _setup_signalr_connection(self):
        """Setup SignalR connection to TopstepX Market Hub."""
        logger.info(f"Setting up SignalR connection to {self.market_hub_url}")

        # CRITICAL: skipNegotiation=True is required for TopstepX WebSocket connection
        # The access token is passed in the URL query string, not via options
        self.hub_connection = HubConnectionBuilder() \
            .with_url(
                self.market_hub_url,
                options={
                    "skip_negotiation": True,  # Skip HTTP negotiation, go straight to WebSocket
                    "access_token_factory": lambda: self.session_token,
                    "headers": {
                        "Authorization": f"Bearer {self.session_token}"
                    }
                }
            ) \
            .configure_logging(logging.INFO) \
            .with_automatic_reconnect({
                "type": "interval",
                "keep_alive_interval": 10,
                "intervals": [0, 2, 5, 10, 30]
            }) \
            .build()

        # Register event handlers
        self.hub_connection.on_open(self._on_connected)
        self.hub_connection.on_close(self._on_disconnected)
        self.hub_connection.on_error(self._on_error)

        # Register market data handlers
        self.hub_connection.on("GatewayQuote", self._on_quote)
        self.hub_connection.on("GatewayTrade", self._on_trade)

        logger.info("SignalR connection configured")

    def _on_connected(self):
        """Called when SignalR connection is established."""
        logger.info("✓ Connected to TopstepX Market Hub")
        self.is_connected = True

        # Subscribe to contract quotes and trades
        try:
            logger.info(f"Subscribing to contract quotes: {self.contract_id}")
            self.hub_connection.send("SubscribeContractQuotes", [self.contract_id])

            logger.info(f"Subscribing to contract trades: {self.contract_id}")
            self.hub_connection.send("SubscribeContractTrades", [self.contract_id])

            logger.info("✓ Subscribed to market data")
        except Exception as e:
            logger.error(f"Failed to subscribe to market data: {e}")

    def _on_disconnected(self):
        """Called when SignalR connection is closed."""
        logger.warning("✗ Disconnected from TopstepX Market Hub")
        self.is_connected = False

    def _on_error(self, error):
        """Called when SignalR connection error occurs."""
        logger.error(f"SignalR error: {error}")

    def _on_quote(self, contract_id: str, quote_data: Dict[str, Any]):
        """
        Handle incoming GatewayQuote message.

        Quote format:
        {
            "symbol": "F.US.EP",
            "symbolName": "/ES",
            "lastPrice": 2100.25,
            "bestBid": 2100.00,
            "bestAsk": 2100.50,
            "change": 25.50,
            "changePercent": 0.14,
            "open": 2090.00,
            "high": 2110.00,
            "low": 2080.00,
            "volume": 12000,
            "lastUpdated": "2024-07-21T13:45:00Z",
            "timestamp": "2024-07-21T13:45:00Z"
        }
        """
        if contract_id != self.contract_id:
            return

        logger.debug(f"Received quote: {quote_data}")
        self.latest_quote = quote_data

        # Extract price for bar building
        price = quote_data.get('lastPrice')
        timestamp_str = quote_data.get('timestamp')

        if price is None or timestamp_str is None:
            logger.warning("Quote missing price or timestamp")
            return

        # Parse timestamp
        timestamp = pd.to_datetime(timestamp_str).tz_convert(self.chicago_tz)

        # Update current bar
        self._update_current_bar(price, timestamp)

    def _on_trade(self, contract_id: str, trade_data: Dict[str, Any]):
        """
        Handle incoming GatewayTrade message.

        Trade format:
        {
            "symbolId": "F.US.EP",
            "price": 2100.25,
            "timestamp": "2024-07-21T13:45:00Z",
            "type": 0,  # 0=Buy, 1=Sell
            "volume": 2
        }
        """
        if contract_id != self.contract_id:
            return

        logger.debug(f"Received trade: {trade_data}")

        # Extract trade price and volume
        price = trade_data.get('price')
        volume = trade_data.get('volume', 0)
        timestamp_str = trade_data.get('timestamp')

        if price is None or timestamp_str is None:
            return

        # Parse timestamp
        timestamp = pd.to_datetime(timestamp_str).tz_convert(self.chicago_tz)

        # Update current bar with trade data
        self._update_current_bar(price, timestamp, volume)

    def _update_current_bar(self, price: float, timestamp: datetime, volume: int = 0):
        """Update the current OHLCV bar being built."""
        # Round timestamp down to bar start time
        bar_start = self._round_to_bar_start(timestamp)

        # If we're starting a new bar
        if self.current_bar_start is None or bar_start > self.current_bar_start:
            # Save completed bar
            if self.current_bar is not None:
                self._finalize_current_bar()

            # Start new bar
            self.current_bar_start = bar_start
            self.current_bar = {
                'timestamp': bar_start,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume
            }
            logger.debug(f"Started new bar at {bar_start}")

        # Update current bar
        else:
            self.current_bar['high'] = max(self.current_bar['high'], price)
            self.current_bar['low'] = min(self.current_bar['low'], price)
            self.current_bar['close'] = price
            self.current_bar['volume'] += volume

    def _round_to_bar_start(self, timestamp: datetime) -> datetime:
        """Round timestamp down to the start of the bar interval."""
        minutes = (timestamp.minute // self.bar_size_minutes) * self.bar_size_minutes
        return timestamp.replace(minute=minutes, second=0, microsecond=0)

    def _finalize_current_bar(self):
        """Add completed current bar to the buffer."""
        if self.current_bar is None:
            return

        # Convert to Series
        bar = pd.Series({
            'open': self.current_bar['open'],
            'high': self.current_bar['high'],
            'low': self.current_bar['low'],
            'close': self.current_bar['close'],
            'volume': self.current_bar['volume']
        }, name=self.current_bar['timestamp'])

        # Validate bar
        if not self._validate_ohlc(bar):
            logger.error(f"Invalid OHLC bar: {bar}")
            return

        # Add to buffer
        if self.bars_buffer.empty:
            self.bars_buffer = pd.DataFrame([bar])
        else:
            self.bars_buffer.loc[bar.name] = bar

            # Keep only last N bars
            if len(self.bars_buffer) > self.lookback_bars:
                self.bars_buffer = self.bars_buffer.iloc[-self.lookback_bars:]

        # Ensure sorted
        self.bars_buffer = self.bars_buffer.sort_index()

        logger.info(
            f"✓ Bar completed: {bar.name.strftime('%H:%M')} | "
            f"O:{bar['open']:.2f} H:{bar['high']:.2f} L:{bar['low']:.2f} C:{bar['close']:.2f} V:{int(bar['volume'])}"
        )

    def _validate_ohlc(self, bar: pd.Series) -> bool:
        """Validate OHLC relationships."""
        try:
            h, l, o, c = bar['high'], bar['low'], bar['open'], bar['close']
            return (l <= o <= h) and (l <= c <= h) and (l <= h)
        except:
            return False

    def connect(self):
        """Connect to TopstepX Market Hub."""
        try:
            self._setup_signalr_connection()
            logger.info("Starting SignalR connection...")
            self.hub_connection.start()

            # Wait for connection to establish
            import time
            timeout = 10
            start = time.time()
            while not self.is_connected and (time.time() - start) < timeout:
                time.sleep(0.1)

            if not self.is_connected:
                raise TimeoutError("Failed to connect to TopstepX Market Hub within timeout")

            logger.info("✓ Connected and subscribed to market data")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to market data: {e}")
            raise

    def disconnect(self):
        """Disconnect from TopstepX Market Hub."""
        if self.hub_connection:
            try:
                # Unsubscribe from contract data
                self.hub_connection.send("UnsubscribeContractQuotes", [self.contract_id])
                self.hub_connection.send("UnsubscribeContractTrades", [self.contract_id])

                # Stop connection
                self.hub_connection.stop()
                logger.info("✓ Disconnected from TopstepX Market Hub")
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")

    def get_latest_bar(self) -> Optional[pd.Series]:
        """
        Get the most recent completed bar.

        Returns:
            Latest bar as a Series, or None if no bars available
        """
        # Check if we have a completed bar that's new
        if self.bars_buffer.empty:
            return None

        latest_bar = self.bars_buffer.iloc[-1]
        return latest_bar

    def get_buffer(self) -> pd.DataFrame:
        """
        Get the current rolling buffer of bars.

        Returns:
            DataFrame with historical bars
        """
        return self.bars_buffer.copy()

    def check_connection(self) -> bool:
        """
        Check if connection to market data is healthy.

        Returns:
            True if connected, False otherwise
        """
        return self.is_connected and self.hub_connection is not None
