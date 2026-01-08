"""
Real-time data fetcher for live trading.

Fetches live bars from Databento and maintains a rolling buffer
for feature calculation.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import databento as db
import pandas as pd

logger = logging.getLogger(__name__)


class LiveDataFetcher:
    """
    Fetches real-time OHLCV bars from Databento.

    Maintains a rolling buffer of historical bars needed for
    feature calculation (ATR, EMA, etc.).
    """

    def __init__(
        self,
        symbol: str,
        bar_size: str = "1m",
        lookback_bars: int = 100,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the data fetcher.

        Args:
            symbol: Symbol to fetch (e.g., "MES")
            bar_size: Bar size (e.g., "1m", "5m")
            lookback_bars: Number of historical bars to maintain for features
            api_key: Databento API key (if None, reads from env)
        """
        self.symbol = symbol
        self.bar_size = bar_size
        self.lookback_bars = lookback_bars

        # Get API key
        self.api_key = api_key or os.getenv("DATABENTO_API_KEY")
        if not self.api_key:
            raise ValueError("DATABENTO_API_KEY not found in environment")

        # Initialize client
        self.client = db.Historical(key=self.api_key)

        # Rolling buffer of bars
        self.bars_buffer = pd.DataFrame()

        logger.info(
            f"LiveDataFetcher initialized: symbol={symbol}, "
            f"bar_size={bar_size}, lookback={lookback_bars}"
        )

    def fetch_historical(self, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch historical bars to initialize the buffer.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today

        Returns:
            DataFrame with OHLCV bars
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Fetching historical bars: {start_date} to {end_date}")

        try:
            # Fetch data from Databento
            data = self.client.timeseries.get_range(
                dataset="GLBX.MDP3",
                symbols=[self.symbol],
                schema="ohlcv-1m",
                start=start_date,
                end=end_date,
                stype_in="continuous",
            )

            # Convert to DataFrame
            df = data.to_df()

            # Rename columns to match our convention
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
            })

            # Set timestamp as index
            df.index = pd.to_datetime(df.index)
            df.index.name = 'timestamp'

            # Select only OHLCV columns
            df = df[['open', 'high', 'low', 'close', 'volume']]

            logger.info(f"Fetched {len(df)} historical bars")

            return df

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise

    def initialize_buffer(self, start_date: Optional[str] = None):
        """
        Initialize the rolling buffer with historical bars.

        Args:
            start_date: Start date for historical fetch
                       If None, fetches last N bars based on lookback_bars
        """
        if start_date is None:
            # Fetch last lookback_bars + buffer
            # Assume trading days, so fetch ~2x bars to ensure enough data
            days_back = int(self.lookback_bars * 2 / 390)  # ~390 1m bars per day
            start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        logger.info(f"Initializing buffer from {start_date}")

        # Fetch historical data
        df = self.fetch_historical(start_date)

        # Keep only the last lookback_bars
        if len(df) > self.lookback_bars:
            df = df.iloc[-self.lookback_bars:]

        self.bars_buffer = df

        logger.info(f"Buffer initialized with {len(self.bars_buffer)} bars")
        logger.info(f"Buffer range: {self.bars_buffer.index[0]} to {self.bars_buffer.index[-1]}")

    def fetch_latest_bar(self, max_retries: int = 3) -> Optional[pd.Series]:
        """
        Fetch the most recent completed bar with retry logic and validation.

        Args:
            max_retries: Maximum number of retry attempts

        Returns:
            Latest bar as a Series, or None if no new bar
        """
        import time

        for attempt in range(max_retries):
            try:
                # Fetch last 5 minutes (to ensure we get the most recent completed bar)
                end_time = datetime.now()
                start_time = end_time - timedelta(minutes=5)

                data = self.client.timeseries.get_range(
                    dataset="GLBX.MDP3",
                    symbols=[self.symbol],
                    schema="ohlcv-1m",
                    start=start_time.strftime("%Y-%m-%d"),
                    end=end_time.strftime("%Y-%m-%d"),
                    stype_in="continuous",
                )

                df = data.to_df()

                if df.empty:
                    logger.warning("No bars fetched")
                    return None

                # Rename columns
                df = df.rename(columns={
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume',
                })

                df.index = pd.to_datetime(df.index)
                df.index.name = 'timestamp'

                # Get the most recent bar
                latest_bar = df.iloc[-1][['open', 'high', 'low', 'close', 'volume']]

                # Validate bar completeness
                if not self._validate_bar_complete(latest_bar.name):
                    logger.warning(f"Bar at {latest_bar.name} may be incomplete (too recent)")
                    return None

                # Check for duplicates
                if self._check_for_duplicates(latest_bar):
                    logger.debug(f"Duplicate bar at {latest_bar.name}, skipping")
                    return None

                # Check if this bar is older than buffer
                if not self.bars_buffer.empty:
                    if latest_bar.name <= self.bars_buffer.index[-1]:
                        logger.debug("No new bar available")
                        return None

                # Check for gaps
                gap_msg = self._check_for_gaps(latest_bar)
                if gap_msg:
                    logger.warning(gap_msg)

                # Validate OHLC
                if not self._validate_ohlc(latest_bar):
                    logger.error(f"Invalid OHLC data: {latest_bar}")
                    return None

                return latest_bar

            except Exception as e:
                logger.error(f"Error fetching bar (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.critical("Max retries exceeded for bar fetch")
                    return None

    def update_buffer(self, new_bar: pd.Series):
        """
        Add new bar to the rolling buffer efficiently with validation.

        Args:
            new_bar: New bar to add
        """
        # Validate before adding
        if not self._validate_ohlc(new_bar):
            logger.error("Skipping invalid bar")
            return

        # Create new dataframe if needed
        if self.bars_buffer.empty:
            self.bars_buffer = pd.DataFrame([new_bar])
        else:
            # More efficient: use loc instead of concat
            self.bars_buffer.loc[new_bar.name] = new_bar

            # Keep only last N bars (sorted by index)
            if len(self.bars_buffer) > self.lookback_bars:
                self.bars_buffer = self.bars_buffer.iloc[-self.lookback_bars:]

        # Ensure sorted
        self.bars_buffer = self.bars_buffer.sort_index()

        logger.debug(
            f"Buffer updated: {len(self.bars_buffer)} bars, "
            f"latest timestamp: {self.bars_buffer.index[-1]}"
        )

    def get_buffer(self) -> pd.DataFrame:
        """
        Get the current rolling buffer of bars.

        Returns:
            DataFrame with historical bars
        """
        return self.bars_buffer.copy()

    def check_connection(self) -> bool:
        """
        Test connection to Databento API.

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            # Try to fetch a small amount of recent data
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)

            data = self.client.timeseries.get_range(
                dataset="GLBX.MDP3",
                symbols=[self.symbol],
                schema="ohlcv-1m",
                start=start_time.strftime("%Y-%m-%d"),
                end=end_time.strftime("%Y-%m-%d"),
                stype_in="continuous",
            )

            logger.info("Data connection healthy")
            return True

        except Exception as e:
            logger.error(f"Data connection failed: {e}")
            return False

    def _validate_bar_complete(self, bar_timestamp: pd.Timestamp) -> bool:
        """
        Check if bar is likely complete (not currently forming).

        Args:
            bar_timestamp: Timestamp of the bar

        Returns:
            True if bar appears complete, False otherwise
        """
        now = pd.Timestamp.now(tz='America/Chicago')
        # Bar should be at least 1 minute old
        age_seconds = (now - bar_timestamp).total_seconds()
        return age_seconds >= 60

    def _check_for_duplicates(self, new_bar: pd.Series) -> bool:
        """
        Check if bar already exists in buffer.

        Args:
            new_bar: Bar to check

        Returns:
            True if duplicate, False otherwise
        """
        if self.bars_buffer.empty:
            return False
        return new_bar.name in self.bars_buffer.index

    def _check_for_gaps(self, new_bar: pd.Series) -> Optional[str]:
        """
        Check for time gaps in bar sequence.

        Args:
            new_bar: New bar to check

        Returns:
            Warning message if gap detected, None otherwise
        """
        if self.bars_buffer.empty:
            return None

        last_bar_time = self.bars_buffer.index[-1]
        expected_gap = pd.Timedelta(minutes=1)
        actual_gap = new_bar.name - last_bar_time

        if actual_gap > expected_gap * 1.5:  # Allow 50% tolerance
            return f"Gap detected: {actual_gap.total_seconds()/60:.1f} minutes"
        return None

    def _validate_ohlc(self, bar: pd.Series) -> bool:
        """
        Validate OHLC relationships.

        Args:
            bar: Bar to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            h, l, o, c = bar['high'], bar['low'], bar['open'], bar['close']
            # Low <= Open, Close <= High
            # Low <= High
            return (l <= o <= h) and (l <= c <= h) and (l <= h)
        except:
            return False
