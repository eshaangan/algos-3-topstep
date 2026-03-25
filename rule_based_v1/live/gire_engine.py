"""GIRE live signal engine — Gap Inventory Repair Engine.

Processes 5-min RTH bars and emits one signal per day when gap fade conditions
are met (up-gap SHORT, down-gap LONG).

Validated best config (from backtest research):
    g_star=0.2, clv_thresh=0.8, use_opening_failure=True, time_stop_hour=12

Usage:
    engine = GIREEngine(g_star=0.2, clv_thresh=0.8)
    signal = engine.update(bar_time, bar, bars_df)   # bar has open/high/low/close keys
    engine.reset_day()                                # explicit daily reset (optional)
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GIREEngine:
    """Gap Inventory Repair Engine for 5-min live trading.

    Emits at most one signal per session day. The signal fires when:
      - A qualifying gap is detected (G_t vs g_star threshold)
      - Yesterday's CLV supports the gap direction
      - The opening failure condition is satisfied (if use_opening_failure=True)
      - Price breaks through the bar1 range boundary at or after 09:40

    Parameters
    ----------
    g_star : float
        Minimum |G_t| = |O_t - C_y| / ATR_d required to qualify a gap.
    clv_thresh : float
        CLV_y threshold: SHORT requires CLV_y >= clv_thresh; LONG requires
        CLV_y <= (1 - clv_thresh).
    use_opening_failure : bool
        When True, require opening failure pattern on bar1/bar2 before signalling.
    time_stop_hour : int
        Hour (ET) at which pending untripped signals are discarded.
    n_contracts : int
        Number of contracts included in emitted signal dicts.
    """

    def __init__(
        self,
        g_star: float = 0.2,
        clv_thresh: float = 0.8,
        use_opening_failure: bool = True,
        time_stop_hour: int = 12,
        n_contracts: int = 1,
    ) -> None:
        self.g_star = g_star
        self.clv_thresh = clv_thresh
        self.use_opening_failure = use_opening_failure
        self.time_stop_hour = time_stop_hour
        self.n_contracts = n_contracts

        # Per-day state — reset on new calendar date
        self._current_date: date | None = None
        self._bar1: Any | None = None          # 09:30 bar
        self._bar2: Any | None = None          # 09:35 bar
        self._signal_dir: int | None = None    # 1=LONG, -1=SHORT, None=no signal
        self._signal_data: dict | None = None  # precomputed entry/exit prices
        self._entry_triggered: bool = False

        # Cached per-day values
        self._atr_d: float = float("nan")
        self._c_y: float = float("nan")
        self._clv_y: float = float("nan")
        self._bar1_low: float = float("nan")
        self._bar1_high: float = float("nan")
        self._o_t: float = float("nan")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset_day(self) -> None:
        """Explicit daily reset — clears all per-day state."""
        self._bar1 = None
        self._bar2 = None
        self._signal_dir = None
        self._signal_data = None
        self._entry_triggered = False
        self._atr_d = float("nan")
        self._c_y = float("nan")
        self._clv_y = float("nan")
        self._bar1_low = float("nan")
        self._bar1_high = float("nan")
        self._o_t = float("nan")
        logger.info("GIREEngine: day reset")

    def update(
        self,
        bar_time: Any,
        bar: Any,
        bars_df: pd.DataFrame,
    ) -> dict[str, Any] | None:
        """Process one 5-min bar.

        Parameters
        ----------
        bar_time :
            Timestamp of the bar (pandas Timestamp or datetime).
        bar :
            Object/dict-like with keys or attributes: open, high, low, close.
        bars_df :
            Rolling bar buffer (tz-aware US/Eastern DatetimeIndex, 5-min RTH).

        Returns
        -------
        dict or None
            Signal dict when a valid gap-fade entry is triggered, otherwise None.
        """
        # ---- Auto-reset on new calendar day ----
        try:
            bar_date = bar_time.date() if hasattr(bar_time, "date") else bar_time
        except Exception:
            bar_date = None

        if bar_date is not None and bar_date != self._current_date:
            if self._current_date is not None:
                logger.info(f"GIREEngine: auto-reset for new day {bar_date}")
            self.reset_day()
            self._current_date = bar_date

        hour = bar_time.hour
        minute = bar_time.minute

        # ---- Capture bar1 at 09:30 ----
        if hour == 9 and minute == 30:
            self._bar1 = bar
            self._bar1_low = float(_get(bar, "low"))
            self._bar1_high = float(_get(bar, "high"))
            self._o_t = float(_get(bar, "open"))
            return None

        # ---- Capture bar2 at 09:35 and compute signal ----
        if hour == 9 and minute == 35:
            if self._bar1 is None:
                return None
            self._bar2 = bar
            self._compute_signal(bars_df)
            return None

        # ---- Entry evaluation at 09:40+ ----
        if hour < 9 or (hour == 9 and minute < 40):
            return None

        # Time stop: discard pending signal after time_stop_hour
        if hour >= self.time_stop_hour:
            if self._signal_dir is not None and not self._entry_triggered:
                logger.info(
                    f"GIREEngine: signal discarded — past time stop {self.time_stop_hour:02d}:00"
                )
                self._signal_dir = None
                self._signal_data = None
            return None

        # No pending signal or already triggered
        if self._signal_dir is None or self._entry_triggered:
            return None

        bar_low = float(_get(bar, "low"))
        bar_high = float(_get(bar, "high"))

        if self._signal_dir == -1:
            # SHORT: entry when price breaks below bar1_low
            if bar_low < self._bar1_low:
                entry_price = self._bar1_low - 0.25
                signal = self._build_signal(entry_price)
                self._entry_triggered = True
                logger.info(
                    f"GIREEngine ENTRY triggered: SHORT @ {entry_price:.2f} "
                    f"(bar1_low={self._bar1_low:.2f}, bar_low={bar_low:.2f})"
                )
                return signal

        elif self._signal_dir == 1:
            # LONG: entry when price breaks above bar1_high
            if bar_high > self._bar1_high:
                entry_price = self._bar1_high + 0.25
                signal = self._build_signal(entry_price)
                self._entry_triggered = True
                logger.info(
                    f"GIREEngine ENTRY triggered: LONG @ {entry_price:.2f} "
                    f"(bar1_high={self._bar1_high:.2f}, bar_high={bar_high:.2f})"
                )
                return signal

        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_signal(self, bars_df: pd.DataFrame) -> None:
        """Evaluate gap and opening-failure conditions after bar2 is captured.

        Sets self._signal_dir and self._signal_data if conditions are met.
        Called exactly once per day, at 09:35.
        """
        if self._bar1 is None or self._bar2 is None:
            return

        today_date = self._current_date
        atr_d = _compute_daily_atr(bars_df, today_date)
        if np.isnan(atr_d) or atr_d <= 0:
            logger.info("GIREEngine: insufficient ATR history — no signal today")
            return

        self._atr_d = atr_d

        # Yesterday's last RTH bar for C_y, H_y, L_y
        prev_data = _get_prev_day_data(bars_df, today_date)
        if prev_data is None:
            logger.info("GIREEngine: no prior-day data — no signal today")
            return

        c_y, h_y, l_y = prev_data
        self._c_y = c_y

        # Gap metric
        o_t = self._o_t
        g_t = (o_t - c_y) / atr_d

        # CLV of yesterday
        clv_y = (c_y - l_y) / (h_y - l_y + 1e-9)

        # Opening failure flags
        bar1 = self._bar1
        bar2 = self._bar2
        b1_o = float(_get(bar1, "open"))
        b1_h = float(_get(bar1, "high"))
        b1_l = float(_get(bar1, "low"))
        b1_c = float(_get(bar1, "close"))
        b2_h = float(_get(bar2, "high"))
        b2_l = float(_get(bar2, "low"))

        f_short = (b1_c < b1_o + 0.3 * (b1_h - b1_l)) or (b2_l < b1_l)
        f_long  = (b1_c > b1_l + 0.7 * (b1_h - b1_l)) or (b2_h > b1_h)

        opening_high_taken = b2_h > b1_h
        opening_low_taken  = b2_l < b1_l

        logger.info(
            f"GIREEngine signal eval: G_t={g_t:.3f}, CLV_y={clv_y:.3f}, ATR_d={atr_d:.4f}, "
            f"g_star={self.g_star}, clv_thresh={self.clv_thresh}, "
            f"f_short={f_short}, f_long={f_long}, "
            f"opening_high_taken={opening_high_taken}, opening_low_taken={opening_low_taken}"
        )

        # Cache CLV_y for use in _build_signal
        self._clv_y = clv_y

        # SHORT: up-gap fade
        if (
            g_t > self.g_star
            and clv_y > self.clv_thresh
            and (not self.use_opening_failure or f_short)
            and not opening_high_taken
        ):
            self._signal_dir = -1
            logger.info(
                f"GIREEngine: SHORT signal queued "
                f"(G_t={g_t:.3f}, CLV_y={clv_y:.3f}, ATR_d={atr_d:.4f})"
            )
            return

        # LONG: down-gap fade
        if (
            g_t < -self.g_star
            and clv_y < (1.0 - self.clv_thresh)
            and (not self.use_opening_failure or f_long)
            and not opening_low_taken
        ):
            self._signal_dir = 1
            logger.info(
                f"GIREEngine: LONG signal queued "
                f"(G_t={g_t:.3f}, CLV_y={clv_y:.3f}, ATR_d={atr_d:.4f})"
            )
            return

        logger.info("GIREEngine: conditions not met — no signal today")

    def _build_signal(self, entry_price: float) -> dict[str, Any]:
        """Build and return the signal dict for the given entry price."""
        direction = self._signal_dir
        atr_d = self._atr_d
        c_y = self._c_y
        b1_h = self._bar1_high
        b1_l = self._bar1_low
        o_t = self._o_t
        g_t = (o_t - c_y) / (atr_d if atr_d > 0 else 1e-9)

        if direction == -1:
            # SHORT
            stop_ref = max(b1_h, o_t)
            sl_dist = min(stop_ref + 0.5 * atr_d * 0.25 - entry_price, 2.0 * atr_d)
            sl_dist = max(sl_dist, 0.25)  # floor at 1 tick
            stop_price = entry_price + sl_dist + 0.25
            profit_target = c_y - 0.25
        else:
            # LONG
            stop_ref = min(b1_l, o_t)
            sl_dist = min(entry_price - (stop_ref - 0.5 * atr_d * 0.25), 2.0 * atr_d)
            sl_dist = max(sl_dist, 0.25)  # floor at 1 tick
            stop_price = entry_price - sl_dist - 0.25
            profit_target = c_y + 0.25

        return {
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss": stop_price,
            "profit_target": profit_target,
            "n_contracts": self.n_contracts,
            "source": "GIRE",
            "G_t": g_t,
            "CLV_y": self._clv_y,
            "ATR_d": atr_d,
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _get(obj: Any, key: str) -> float:
    """Read a field from a dict-like or attribute-style bar object."""
    try:
        return float(obj[key])
    except (KeyError, TypeError):
        return float(getattr(obj, key))


def _compute_daily_atr(bars_df: pd.DataFrame, today_date: date) -> float:
    """Compute rolling 20-day mean of daily RTH range ending before today_date.

    Parameters
    ----------
    bars_df :
        5-min RTH bar buffer with tz-aware DatetimeIndex (US/Eastern).
    today_date :
        Calendar date of the current session; only prior-day bars are used.

    Returns
    -------
    float
        Rolling 20-day ATR, or NaN when fewer than 5 prior days are available.
    """
    if bars_df is None or len(bars_df) < 2:
        return float("nan")

    prior = bars_df[bars_df.index.map(lambda t: t.date()) < today_date]
    if len(prior) == 0:
        return float("nan")

    prior_dates = prior.index.map(lambda t: t.date())
    daily_range = (
        prior.groupby(prior_dates)
        .apply(lambda g: g["high"].max() - g["low"].min())
    )

    if len(daily_range) < 5:
        return float("nan")

    rolling_atr = daily_range.rolling(20, min_periods=5).mean()
    return float(rolling_atr.iloc[-1])


def _get_prev_day_data(
    bars_df: pd.DataFrame, today_date: date
) -> tuple[float, float, float] | None:
    """Return (close, high, low) of the last prior RTH session.

    Parameters
    ----------
    bars_df :
        5-min RTH bar buffer.
    today_date :
        Current session date; looks backward from this date.

    Returns
    -------
    tuple (c_y, h_y, l_y) or None
    """
    if bars_df is None or len(bars_df) == 0:
        return None

    prior = bars_df[bars_df.index.map(lambda t: t.date()) < today_date]
    if len(prior) == 0:
        return None

    prev_day = prior.index[-1].date()
    day_bars = prior[prior.index.map(lambda t: t.date()) == prev_day]
    if len(day_bars) == 0:
        return None

    c_y = float(day_bars["close"].iloc[-1])
    h_y = float(day_bars["high"].max())
    l_y = float(day_bars["low"].min())
    return c_y, h_y, l_y
