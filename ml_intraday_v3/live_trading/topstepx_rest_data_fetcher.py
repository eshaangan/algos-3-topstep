"""
Real-time data fetcher for TopstepX using REST API polling.

This is the simpler, proven approach that polls the REST API for completed bars
rather than using WebSocket streaming.
"""

import importlib.util
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import pytz

# Import ProjectX client with a robust fallback for test environments
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from core.projectx_client import ProjectXClient
except ModuleNotFoundError:
    core_path = project_root / "core" / "projectx_client.py"
    spec = importlib.util.spec_from_file_location("projectx_client", core_path)
    if spec is None or spec.loader is None:
        raise
    module = importlib.util.module_from_spec(spec)
    sys.modules["projectx_client"] = module
    spec.loader.exec_module(module)
    ProjectXClient = module.ProjectXClient

logger = logging.getLogger(__name__)


class TopstepXRestDataFetcher:
    """
    Fetches real-time market data from TopstepX using REST API polling.

    Polls the REST API periodically for completed OHLCV bars.
    """

    def __init__(
        self,
        contract_id: str,
        bar_size_minutes: int = 5,
        lookback_bars: int = 100,
    ):
        """
        Initialize the TopstepX REST data fetcher.

        Args:
            contract_id: TopstepX contract ID (e.g., "CON.F.US.EP.H26")
            bar_size_minutes: Bar size in minutes (default: 5)
            lookback_bars: Number of historical bars to maintain
        """
        self.contract_id = contract_id
        self.bar_size_minutes = bar_size_minutes
        self.lookback_bars = lookback_bars

        # Initialize ProjectX client
        self.client = ProjectXClient()

        # Rolling buffer of completed bars
        self.bars_buffer = pd.DataFrame()

        # Last bar timestamp we've seen
        self.last_bar_timestamp: Optional[pd.Timestamp] = None

        # Chicago timezone (market hours)
        self.chicago_tz = pytz.timezone('America/Chicago')

        logger.info(
            f"TopstepXRestDataFetcher initialized: contract={contract_id}, "
            f"bar_size={bar_size_minutes}m, lookback={lookback_bars}"
        )

    def connect(self):
        """Connect to TopstepX API (just verify credentials)."""
        try:
            # Test connection by getting account state
            account = self.client.get_account_state()
            logger.info(f"✓ Connected to TopstepX API: Account {account.account_id}, ${account.equity:,.2f}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to TopstepX API: {e}")
            raise

    def disconnect(self):
        """Disconnect from TopstepX API."""
        logger.info("✓ Disconnected from TopstepX API")

    def initialize_buffer(self):
        """Initialize the rolling buffer with historical bars."""
        logger.info("Initializing buffer with historical bars...")

        # Fetch historical bars
        # Fetch more than needed to account for gaps
        end_time = datetime.now(timezone.utc)
        # Calculate start time based on bar size and lookback
        minutes_back = self.lookback_bars * self.bar_size_minutes * 2  # 2x buffer
        start_time = end_time - timedelta(minutes=minutes_back)

        try:
            bars = self.client.retrieve_bars(
                contract_id=self.contract_id,
                start_time=start_time,
                end_time=end_time,
                unit=2,  # 2 = minutes
                unit_number=self.bar_size_minutes,
                limit=self.lookback_bars * 2,
                include_partial_bar=False,
                live=False,  # Use historical data mode
            )

            if not bars:
                logger.warning("No historical bars returned")
                return

            # Convert to DataFrame
            rows = []
            for bar in bars:
                rows.append({
                    'timestamp': pd.to_datetime(bar.timestamp, utc=True).tz_convert(self.chicago_tz),
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                })

            df = pd.DataFrame(rows)
            df = df.set_index('timestamp')
            df = df.sort_index()

            # Keep only last N bars
            if len(df) > self.lookback_bars:
                df = df.iloc[-self.lookback_bars:]

            self.bars_buffer = df

            if not df.empty:
                self.last_bar_timestamp = df.index[-1]
                logger.info(
                    f"✓ Buffer initialized with {len(df)} bars: "
                    f"{df.index[0].strftime('%Y-%m-%d %H:%M')} to {df.index[-1].strftime('%Y-%m-%d %H:%M')}"
                )
            else:
                logger.warning("Buffer is empty after initialization")

        except Exception as e:
            logger.error(f"Error initializing buffer: {e}")
            raise

    def fetch_latest_bar(self) -> Optional[pd.Series]:
        """
        Fetch the most recent completed bar from TopstepX API.

        Returns:
            Latest bar as a Series, or None if no new bar available
        """
        try:
            # Fetch recent bars with a generous lookback window
            # 1 hour lookback to be safe
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=self.bar_size_minutes * 12)  # 60 mins for 5m bars

            bars = self.client.retrieve_bars(
                contract_id=self.contract_id,
                start_time=start_time,
                end_time=end_time,
                unit=2,  # 2 = minutes
                unit_number=self.bar_size_minutes,
                limit=20, # Increased limit
                include_partial_bar=True,  # Include partial to force latest data
                live=False,  # Use historical data mode (sim)
            )

            if not bars:
                return None

            # Find the latest COMPLETE bar
            # A bar is complete if its end time (start + duration) is in the past
            latest_bar = None
            bar_duration = timedelta(minutes=self.bar_size_minutes)
            
            # Iterate backwards
            for bar in reversed(bars):
                bar_ts = pd.to_datetime(bar.timestamp, utc=True)
                # Check if bar is complete
                # Allow a small buffer (e.g. 5 seconds) for clock skew
                if bar_ts + bar_duration <= end_time + timedelta(seconds=5):
                    latest_bar = bar
                    break
            
            if latest_bar is None:
                return None

            bar_timestamp = pd.to_datetime(latest_bar.timestamp, utc=True).tz_convert(self.chicago_tz)

            # Check if this is a new bar
            if self.last_bar_timestamp is not None and bar_timestamp <= self.last_bar_timestamp:
                return None

            # Create Series for the bar
            bar_series = pd.Series({
                'open': latest_bar.open,
                'high': latest_bar.high,
                'low': latest_bar.low,
                'close': latest_bar.close,
                'volume': latest_bar.volume
            }, name=bar_timestamp)

            return bar_series

        except Exception as e:
            logger.error(f"Error fetching latest bar: {e}")
            return None

    def update_buffer(self, new_bar: pd.Series):
        """
        Add new bar to the rolling buffer.

        Args:
            new_bar: New bar to add
        """
        # Add to buffer
        if self.bars_buffer.empty:
            self.bars_buffer = pd.DataFrame([new_bar])
        else:
            self.bars_buffer.loc[new_bar.name] = new_bar

            # Keep only last N bars
            if len(self.bars_buffer) > self.lookback_bars:
                self.bars_buffer = self.bars_buffer.iloc[-self.lookback_bars:]

        # Ensure sorted
        self.bars_buffer = self.bars_buffer.sort_index()

        # Update last timestamp
        self.last_bar_timestamp = new_bar.name

        logger.debug(
            f"Buffer updated: {len(self.bars_buffer)} bars, "
            f"latest: {new_bar.name.strftime('%H:%M')}"
        )

    def get_buffer(self) -> pd.DataFrame:
        """
        Get the current rolling buffer of bars.

        Returns:
            DataFrame with historical bars
        """
        return self.bars_buffer.copy()

    def get_latest_bar(self) -> Optional[pd.Series]:
        """
        Get the most recent bar from the buffer.

        Returns:
            Latest bar as a Series, or None if buffer is empty
        """
        if self.bars_buffer.empty:
            return None
        return self.bars_buffer.iloc[-1]

    def check_connection(self) -> bool:
        """
        Check if connection to TopstepX API is healthy.

        Returns:
            True if connected, False otherwise
        """
        try:
            account = self.client.get_account_state()
            return True
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False
