"""MSITE live signal engine — 5-min calibration.

Wraps the DCClock / StateMachine machinery from msite_backtest.py into a
stateful per-session engine suitable for the live runner.

Usage:
    engine = MSITEEngine(n_contracts=12, pt_mult=2.0, sl_frac=0.55)
    engine.reset_day(open_price)
    signal = engine.update(bar_time, bar)   # bar has open/high/low/close keys
"""

from __future__ import annotations

import logging
import sys
from collections import deque
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import shared components from msite_backtest (same repo, diagnostics dir)
# ---------------------------------------------------------------------------
_DIAG_DIR = str(Path(__file__).resolve().parent.parent / "diagnostics")
if _DIAG_DIR not in sys.path:
    sys.path.insert(0, _DIAG_DIR)

from msite_backtest import (  # noqa: E402
    DCClock,
    StateMachine,
    yang_zhang_vol,
    compute_signature_scores,
    realized_semivariance_imbalance,
    roughness_proxy,
)

# ---------------------------------------------------------------------------
# 5-min calibration constants
# ---------------------------------------------------------------------------
YZ_WINDOW = 60
SIG_WINDOW = 10
SIG_WINDOW_RELEASE = 3
RSV_WINDOW = 10
ROUGH_WINDOW = 16
WARMUP_BARS = 70

DC_THETA_SCALE = 2.7
DC_THETA_FACTORS = [0.35 * 2.7, 0.80 * 2.7, 1.60 * 2.7]
DC_FLIP_LOOKBACK = 8
COIL_DC2_MIN_BARS = 5
COIL_DC3_MIN_BARS = 14
COIL_DC1_FLIPS_MIN = 2
COIL_M_MAX = 0.50
COIL_B_MAX = 0.60
RELEASE_M_MIN = 0.20
RELEASE_B_MIN = 0.0
RELEASE_ROUGH_MIN = -0.50

TICK_SIZE = 0.25


class MSITEEngine:
    """Multi-Scale Intrinsic-Time Engine for 5-min live trading.

    Processes one bar at a time and emits a signal dict when a RELEASE state
    is detected and all gate conditions are met.

    Parameters
    ----------
    n_contracts : int
        Number of contracts to include in emitted signal dicts.
    pt_mult : float
        Profit-target multiplier applied to the SL distance.
    sl_frac : float
        Fraction of sigma_abs used for SL calculation (floor: 8 ticks).
    afternoon_cutoff : tuple[int, int]
        (hour, minute) in ET after which pending signals are discarded.
    """

    def __init__(
        self,
        n_contracts: int = 12,
        pt_mult: float = 2.0,
        sl_frac: float = 0.55,
        afternoon_cutoff: tuple[int, int] = (14, 0),
    ) -> None:
        self.n_contracts = n_contracts
        self.pt_mult = pt_mult
        self.sl_frac = sl_frac
        self.afternoon_cutoff = afternoon_cutoff

        # Rolling OHLC deques (maxlen=75 gives headroom above WARMUP_BARS=70)
        self._opens: deque[float] = deque(maxlen=75)
        self._highs: deque[float] = deque(maxlen=75)
        self._lows: deque[float] = deque(maxlen=75)
        self._closes: deque[float] = deque(maxlen=75)

        # DC clocks — one per theta factor
        self.dc1 = DCClock(DC_THETA_FACTORS[0])
        self.dc2 = DCClock(DC_THETA_FACTORS[1])
        self.dc3 = DCClock(DC_THETA_FACTORS[2])

        # State machine
        self.sm = StateMachine()

        # Pending signal (queued on RELEASE, emitted next bar)
        self._pending_signal: dict[str, Any] | None = None

        # Track current trading date to auto-reset on day change
        self._current_date: date | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset_day(self, open_price: float) -> None:
        """Reset DC clocks and state machine for a new session.

        Parameters
        ----------
        open_price : float
            Reference price used to initialise DC clock extremes.
            Pass 0.0 when the actual open is not yet known.
        """
        ref = open_price if open_price > 0.0 else 1.0
        self.dc1.reset(ref)
        self.dc2.reset(ref)
        self.dc3.reset(ref)
        self.sm.reset_day()
        self._pending_signal = None
        logger.info(f"MSITEEngine: day reset (open_price={open_price})")

    def update(self, bar_time: Any, bar: Any) -> dict[str, Any] | None:
        """Process one 5-min bar.

        Parameters
        ----------
        bar_time :
            Timestamp of the bar (pandas Timestamp or datetime).
        bar :
            Object/dict-like with attributes or keys: open, high, low, close.

        Returns
        -------
        dict or None
            Signal dict when a valid RELEASE entry is ready, otherwise None.
        """
        # ---- Helper to read bar fields from dict or object ----
        def _get(obj: Any, key: str) -> float:
            try:
                return float(obj[key])
            except (KeyError, TypeError):
                return float(getattr(obj, key))

        # ---- Auto-reset on new calendar day ----
        try:
            bar_date = bar_time.date() if hasattr(bar_time, "date") else bar_time.to_pydatetime().date()
        except Exception:
            bar_date = None

        if bar_date is not None and bar_date != self._current_date:
            if self._current_date is not None:
                logger.info(f"MSITEEngine: auto-reset for new day {bar_date}")
            # Reset DC clocks and state machine for the new day.
            # Use the current bar's open price so DC extremes start at
            # a realistic level (matches msite_backtest.py behaviour).
            # NOTE: rolling deques are NOT cleared — the YZ/signature windows
            # intentionally span day boundaries, exactly as in msite_backtest.py.
            self.reset_day(_get(bar, "open"))
            self._current_date = bar_date

        # ---- Append bar to rolling buffers ----
        self._opens.append(_get(bar, "open"))
        self._highs.append(_get(bar, "high"))
        self._lows.append(_get(bar, "low"))
        self._closes.append(_get(bar, "close"))

        closes_arr = np.array(self._closes, dtype=float)
        opens_arr = np.array(self._opens, dtype=float)
        highs_arr = np.array(self._highs, dtype=float)
        lows_arr = np.array(self._lows, dtype=float)
        n_bars = len(closes_arr)

        # ---- DC clocks updated on every bar (before warmup guard) ----
        # Use full YZ vol if available, otherwise fall back to price * 0.1%.
        if n_bars >= YZ_WINDOW:
            yz_quick = yang_zhang_vol(opens_arr, highs_arr, lows_arr, closes_arr, n=YZ_WINDOW)
            sigma_quick = float(yz_quick[-1])
            if np.isnan(sigma_quick) or sigma_quick <= 0.0:
                sigma_quick = closes_arr[-1] * 0.001
        else:
            sigma_quick = closes_arr[-1] * 0.001
        self.dc1.update(closes_arr[-1], sigma_quick)
        self.dc2.update(closes_arr[-1], sigma_quick)
        self.dc3.update(closes_arr[-1], sigma_quick)

        # ---- Warmup guard (full signal computation requires WARMUP_BARS) ----
        if n_bars < WARMUP_BARS:
            return None

        opens = opens_arr
        highs = highs_arr
        lows = lows_arr
        closes = closes_arr

        # ---- Component 1: Yang-Zhang volatility (full window) ----
        yz_arr = yang_zhang_vol(opens, highs, lows, closes, n=YZ_WINDOW)
        sigma_abs = float(yz_arr[-1])
        if np.isnan(sigma_abs) or sigma_abs <= 0.0:
            sigma_abs = closes[-1] * 0.001

        # ---- Component 3: Path signature scores ----
        sig_closes = closes[-SIG_WINDOW:]
        sig_highs = highs[-SIG_WINDOW:]
        sig_lows = lows[-SIG_WINDOW:]
        M, K, _dir_sign = compute_signature_scores(sig_closes, sig_highs, sig_lows, sigma_abs)

        sig_closes_short = closes[-SIG_WINDOW_RELEASE:]
        sig_highs_short = highs[-SIG_WINDOW_RELEASE:]
        sig_lows_short = lows[-SIG_WINDOW_RELEASE:]
        M_short, _K_short, _dir_short = compute_signature_scores(
            sig_closes_short, sig_highs_short, sig_lows_short, sigma_abs
        )

        # ---- Component 4: Realized semivariance imbalance ----
        B = realized_semivariance_imbalance(closes, window=RSV_WINDOW)

        # ---- Component 5: Roughness proxy ----
        roughness = roughness_proxy(yz_arr, window=ROUGH_WINDOW)

        # ---- State machine update ----
        new_state, sig_dir = self.sm.update(
            self.dc1, self.dc2, self.dc3,
            M, M_short, K, B, roughness, sigma_abs,
        )

        # ---- Check pending signal from previous bar ----
        emitted = None
        if self._pending_signal is not None:
            pending = self._pending_signal
            self._pending_signal = None

            # Afternoon cutoff check
            blocked = False
            try:
                cut_h, cut_m = self.afternoon_cutoff
                bar_dt = bar_time.to_pydatetime() if hasattr(bar_time, "to_pydatetime") else bar_time
                if (bar_dt.hour, bar_dt.minute) >= (cut_h, cut_m):
                    logger.info(
                        f"MSITEEngine: pending signal discarded — past afternoon cutoff "
                        f"{cut_h:02d}:{cut_m:02d}"
                    )
                    blocked = True
            except Exception:
                pass

            if not blocked:
                direction = pending["direction"]
                entry_price = closes[-1] + direction * TICK_SIZE
                sl_dist = max(8 * TICK_SIZE, self.sl_frac * 0.80 * sigma_abs)
                stop_loss = entry_price - direction * sl_dist
                profit_target = entry_price + direction * self.pt_mult * sl_dist

                emitted = {
                    "direction": direction,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "profit_target": profit_target,
                    "n_contracts": self.n_contracts,
                    "source": "MSITE",
                    "sigma_abs": sigma_abs,
                }
                logger.info(
                    f"MSITEEngine SIGNAL: dir={direction:+d}  entry={entry_price:.2f}  "
                    f"SL={stop_loss:.2f}  PT={profit_target:.2f}  "
                    f"sigma={sigma_abs:.2f}  n={self.n_contracts}"
                )

        # ---- Queue new RELEASE signal (long-only: sig_dir == 1) ----
        if new_state == StateMachine.RELEASE and sig_dir == 1:
            self._pending_signal = {"direction": sig_dir}
            logger.info(
                f"MSITEEngine: RELEASE detected (long), queuing signal for next bar "
                f"(M_short={M_short:.3f}, B={B:.3f}, roughness={roughness:.3f})"
            )

        return emitted
