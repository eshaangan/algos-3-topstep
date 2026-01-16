"""
Real-time event detection for live trading.

Implements CUSUM filtering to match training event generation,
ensuring predictions only happen on significant price moves.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LiveEventDetector:
    """
    Detects trading events in real-time using CUSUM filter.

    Matches the training event generation policy to ensure
    live predictions are on the same distribution as training.
    """

    def __init__(
        self,
        atr_period: int = 14,
        cusum_threshold_atr_mult: float = 0.8,
    ):
        """
        Initialize event detector.

        Args:
            atr_period: ATR lookback period (matches training)
            cusum_threshold_atr_mult: CUSUM threshold as multiple of ATR (matches labeling.yaml)
        """
        self.atr_period = atr_period
        self.cusum_threshold_atr_mult = cusum_threshold_atr_mult

        # CUSUM state (symmetric filter)
        self.s_pos = 0.0  # Positive cumulative sum
        self.s_neg = 0.0  # Negative cumulative sum
        self.last_price: Optional[float] = None

        logger.info(
            f"LiveEventDetector initialized: atr_period={atr_period}, "
            f"cusum_mult={cusum_threshold_atr_mult}"
        )

    def _compute_atr(self, bars_df: pd.DataFrame) -> float:
        """
        Compute ATR from recent bars.

        Args:
            bars_df: DataFrame with OHLC data (sorted by time)

        Returns:
            Current ATR value (or NaN if insufficient data)
        """
        if len(bars_df) < self.atr_period + 1:
            return np.nan

        df = bars_df.tail(self.atr_period + 1).copy()
        prev_close = df["close"].shift(1)

        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=self.atr_period, adjust=False).mean().iloc[-1]

        return float(atr)

    def is_event(self, bars_df: pd.DataFrame, current_bar_close: float) -> tuple[bool, dict]:
        """
        Check if current bar is a CUSUM event.

        Args:
            bars_df: Recent bars for ATR calculation (sorted by time)
            current_bar_close: Close price of the current bar

        Returns:
            (is_event, info_dict) where info_dict contains diagnostic info
        """
        # Compute current ATR
        atr = self._compute_atr(bars_df)

        info = {
            "atr": atr,
            "threshold": np.nan,
            "s_pos": self.s_pos,
            "s_neg": self.s_neg,
            "price_diff": np.nan,
            "event_type": None,
        }

        if not np.isfinite(atr) or atr <= 0.0:
            logger.debug("ATR not available or invalid, skipping event check")
            self.last_price = current_bar_close
            return False, info

        threshold = atr * self.cusum_threshold_atr_mult
        info["threshold"] = threshold

        # Need at least one previous price to compute diff
        if self.last_price is None:
            logger.debug("No previous price, initializing CUSUM")
            self.last_price = current_bar_close
            return False, info

        # Compute price change
        price_diff = current_bar_close - self.last_price
        info["price_diff"] = price_diff

        # Update CUSUM accumulators
        self.s_pos = max(0.0, self.s_pos + price_diff)
        self.s_neg = min(0.0, self.s_neg + price_diff)

        info["s_pos"] = self.s_pos
        info["s_neg"] = self.s_neg

        is_event = False

        # Check if positive CUSUM exceeds threshold (upward move)
        if self.s_pos > threshold:
            logger.info(
                f"CUSUM EVENT (UP): s_pos={self.s_pos:.2f} > threshold={threshold:.2f}, "
                f"price_diff={price_diff:.2f}, close={current_bar_close:.2f}"
            )
            self.s_pos = 0.0  # Reset
            is_event = True
            info["event_type"] = "up"

        # Check if negative CUSUM exceeds threshold (downward move)
        elif self.s_neg < -threshold:
            logger.info(
                f"CUSUM EVENT (DOWN): s_neg={self.s_neg:.2f} < -threshold={-threshold:.2f}, "
                f"price_diff={price_diff:.2f}, close={current_bar_close:.2f}"
            )
            self.s_neg = 0.0  # Reset
            is_event = True
            info["event_type"] = "down"

        else:
            logger.debug(
                f"No event: s_pos={self.s_pos:.2f}, s_neg={self.s_neg:.2f}, "
                f"threshold={threshold:.2f}, price_diff={price_diff:.2f}"
            )

        # Update last price
        self.last_price = current_bar_close

        return is_event, info

    def reset(self):
        """Reset CUSUM state (e.g., at session start)."""
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.last_price = None
        logger.info("CUSUM state reset")
