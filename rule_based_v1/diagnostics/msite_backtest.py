"""MSITE Backtest — Multi-Scale Intrinsic-Time Engine diagnostic.

Components:
  1. Yang-Zhang range-based volatility (n=30)
  2. Three Directional-Change clocks (theta_factors = 0.35, 0.80, 1.60)
  3. Path Signature scores (depth-2, no external libs)
  4. Realized Semivariance Imbalance (window=20)
  5. Roughness proxy (lag-1 autocorr of Δlog(vol), window=40)
  6. State machine: NEUTRAL → COIL → RELEASE / EXHAUSTION

Instrument: MNQ 5-min bars, 2026 YTD
  2 contracts, point_value=2.0, tick_size=0.25
  commission=0.62/side, slippage=1 tick

Run (from project root):
    python rule_based_v1/diagnostics/msite_backtest.py
    python rule_based_v1/diagnostics/msite_backtest.py --release-only
    python rule_based_v1/diagnostics/msite_backtest.py --exhaustion-only
    python rule_based_v1/diagnostics/msite_backtest.py --save
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
RBV1 = ROOT / "rule_based_v1"
for _p in [str(ROOT), str(RBV1)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = ROOT / "data" / "processed" / "mnq_2026ytd_5min.h5"
RESULTS_PATH = ROOT / "rule_based_v1" / "diagnostics" / "msite_results.json"

POINT_VALUE = 2.0
TICK_SIZE = 0.25
N_CONTRACTS = 2
COMMISSION_PER_SIDE = 0.62
SLIPPAGE_TICKS = 1

STARTING_EQUITY = 50_000.0
MAX_DAILY_LOSS = -950.0
DRAWDOWN_BUFFER = 1_950.0

# ---------------------------------------------------------------------------
# Timeframe-aware calibration
# ---------------------------------------------------------------------------
# σ_abs from YZ scales as sqrt(tf_minutes) for a diffusion process.
# So DC_THETA_SCALE must scale as 1/sqrt(tf_minutes) to keep thresholds
# in the same ~40-50pt "meaningful intraday move" range across timeframes.
# All bar-count thresholds scale as 1/tf_minutes (same real-time duration).

_TF_CONFIGS = {
    1: dict(
        yz_window=200, sig_window=20, sig_window_release=5,
        rsv_window=20, rough_window=40, warmup_bars=220,
        dc_theta_scale=6.0, dc_flip_lookback=30,
        coil_dc2_min_bars=15, coil_dc3_min_bars=60, coil_dc1_flips_min=2,
    ),
    5: dict(
        yz_window=60,  sig_window=10, sig_window_release=3,
        rsv_window=10, rough_window=16, warmup_bars=70,
        dc_theta_scale=2.7, dc_flip_lookback=8,
        coil_dc2_min_bars=5, coil_dc3_min_bars=14, coil_dc1_flips_min=2,
    ),
    15: dict(
        yz_window=25,  sig_window=5,  sig_window_release=2,
        rsv_window=6,  rough_window=8,  warmup_bars=30,
        dc_theta_scale=1.55, dc_flip_lookback=4,
        coil_dc2_min_bars=3, coil_dc3_min_bars=6, coil_dc1_flips_min=2,
    ),
}


def _apply_tf_config(tf_minutes: int) -> None:
    """Set module-level constants for the given bar timeframe."""
    global YZ_WINDOW, SIG_WINDOW, SIG_WINDOW_RELEASE, RSV_WINDOW, ROUGH_WINDOW
    global WARMUP_BARS, DC_THETA_SCALE, DC_THETA_FACTORS, DC_FLIP_LOOKBACK
    global COIL_DC2_MIN_BARS, COIL_DC3_MIN_BARS, COIL_DC1_FLIPS_MIN

    if tf_minutes not in _TF_CONFIGS:
        raise ValueError(f"Unsupported timeframe: {tf_minutes}min. Choose from {list(_TF_CONFIGS)}")
    cfg = _TF_CONFIGS[tf_minutes]

    YZ_WINDOW             = cfg["yz_window"]
    SIG_WINDOW            = cfg["sig_window"]
    SIG_WINDOW_RELEASE    = cfg["sig_window_release"]
    RSV_WINDOW            = cfg["rsv_window"]
    ROUGH_WINDOW          = cfg["rough_window"]
    WARMUP_BARS           = cfg["warmup_bars"]
    DC_THETA_SCALE        = cfg["dc_theta_scale"]
    DC_THETA_FACTORS      = [0.35 * DC_THETA_SCALE,
                              0.80 * DC_THETA_SCALE,
                              1.60 * DC_THETA_SCALE]
    DC_FLIP_LOOKBACK      = cfg["dc_flip_lookback"]
    COIL_DC2_MIN_BARS     = cfg["coil_dc2_min_bars"]
    COIL_DC3_MIN_BARS     = cfg["coil_dc3_min_bars"]
    COIL_DC1_FLIPS_MIN    = cfg["coil_dc1_flips_min"]


# Default: 1-min constants (overridden by --tf flag at runtime)
YZ_WINDOW = 200
SIG_WINDOW = 20
SIG_WINDOW_RELEASE = 5
RSV_WINDOW = 20
ROUGH_WINDOW = 40
WARMUP_BARS = 220
DC_THETA_SCALE = 6.0
DC_THETA_FACTORS = [0.35 * DC_THETA_SCALE,
                    0.80 * DC_THETA_SCALE,
                    1.60 * DC_THETA_SCALE]
DC_FLIP_LOOKBACK = 30
COIL_DC2_MIN_BARS = 15
COIL_DC3_MIN_BARS = 60
COIL_DC1_FLIPS_MIN = 2

COIL_M_MAX = 0.50
COIL_B_MAX = 0.60

RELEASE_M_MIN = 0.20
RELEASE_B_MIN = 0.0
RELEASE_ROUGH_MIN = -0.50

EXHAUSTION_OVERSHOOT_2 = 2.0
EXHAUSTION_OVERSHOOT_3 = 1.5
EXHAUSTION_M_MAX = 0.50
EXHAUSTION_K_THRESH = -0.15
EXHAUSTION_B_THRESH = -0.10

# Exit — K trailing
K_TRAIL_THRESH = -0.30

# Baseline reference (for comparison row in output)
BASELINE = {"name": "Baseline ORB+PrevVWAP", "n": 16, "wr": 0.75,
            "avg_pnl": 277.0, "sharpe": 8.29, "maxdd": -796.0, "exp": 277.0}


# ===========================================================================
# Component 1: Yang-Zhang Volatility
# ===========================================================================

def yang_zhang_vol(opens: np.ndarray, highs: np.ndarray,
                   lows: np.ndarray, closes: np.ndarray,
                   n: int = YZ_WINDOW) -> np.ndarray:
    """Return Yang-Zhang rolling volatility series (same length as input).

    Returns sigma in *price units* (sigma * close).
    First n-1 values are NaN.
    """
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    size = len(closes)
    sigma_abs = np.full(size, np.nan)

    # Pre-compute per-bar terms
    overnight = np.full(size, np.nan)
    oc = np.full(size, np.nan)
    rs = np.full(size, np.nan)

    for i in range(1, size):
        overnight[i] = np.log(opens[i] / closes[i - 1])
        oc[i] = np.log(closes[i] / opens[i])
        log_hc = np.log(highs[i] / closes[i])
        log_ho = np.log(highs[i] / opens[i])
        log_lc = np.log(lows[i] / closes[i])
        log_lo = np.log(lows[i] / opens[i])
        rs[i] = log_hc * log_ho + log_lc * log_lo

    for i in range(n, size):
        window_on = overnight[i - n + 1: i + 1]
        window_oc = oc[i - n + 1: i + 1]
        window_rs = rs[i - n + 1: i + 1]

        var_on = np.nanvar(window_on, ddof=1)
        var_oc = np.nanvar(window_oc, ddof=1)
        mean_rs = np.nanmean(window_rs)

        sigma_log = np.sqrt(max(0.0, var_on + k * var_oc + (1 - k) * mean_rs))
        sigma_abs[i] = sigma_log * closes[i]

    return sigma_abs


# ===========================================================================
# Component 2: DC Clock
# ===========================================================================

class DCClock:
    """Single Directional-Change clock.

    Tracks price extremes and fires a DC event when the price reverses
    more than theta (= theta_factor * sigma_abs) from the extreme.
    """

    def __init__(self, theta_factor: float) -> None:
        self.theta_factor = theta_factor
        self.direction: int = 1      # +1 = uptrend, -1 = downtrend
        self.extreme: float = 0.0
        self.dc_price: float = 0.0   # price at which last DC fired
        self.bars_since_dc: int = 0
        self.dc_fired: bool = False  # True only on the bar where DC fired

    def reset(self, price: float) -> None:
        """Reset at start of new day."""
        self.extreme = price
        self.dc_price = price
        self.bars_since_dc = 0
        self.dc_fired = False

    def update(self, price: float, sigma_abs: float) -> bool:
        """Update clock with new price. Returns True if DC fired this bar."""
        theta = self.theta_factor * sigma_abs

        self.dc_fired = False

        if self.direction == 1:
            if price > self.extreme:
                self.extreme = price
            elif price < self.extreme - theta:
                # DC fired — direction reversal to downtrend
                self.direction = -1
                self.dc_price = price
                self.extreme = price
                self.bars_since_dc = 0
                self.dc_fired = True
                return True
        else:  # direction == -1
            if price < self.extreme:
                self.extreme = price
            elif price > self.extreme + theta:
                # DC fired — direction reversal to uptrend
                self.direction = 1
                self.dc_price = price
                self.extreme = price
                self.bars_since_dc = 0
                self.dc_fired = True
                return True

        self.bars_since_dc += 1
        return False

    def overshoot(self, sigma_abs: float) -> float:
        """Overshoot ratio: how far price has moved past last DC point."""
        theta = self.theta_factor * sigma_abs + 1e-10
        if self.direction == 1:
            # In uptrend: extreme went above dc_price
            return max(0.0, (self.extreme - self.dc_price) / theta)
        else:
            # In downtrend: extreme went below dc_price
            return max(0.0, (self.dc_price - self.extreme) / theta)


# ===========================================================================
# Component 3: Path Signature Scores
# ===========================================================================

def compute_signature_scores(
    window_closes: np.ndarray,
    window_highs: np.ndarray,
    window_lows: np.ndarray,
    sigma_abs: float,
) -> tuple[float, float, float]:
    """Compute monotonicity M, curvature K, and net direction sign.

    Depth-2 path signature — no external libraries required.

    Returns:
        M: monotonicity in [0, 1]
        K: curvature in [-1, 1]
        direction_sign: np.sign(net), +1 or -1 (0 if flat)
    """
    denom = max(sigma_abs, 1e-10)
    c = (window_closes - window_closes[0]) / denom
    mid = ((window_highs + window_lows) / 2 - window_closes[0]) / denom

    diffs = np.diff(c)
    path_length = np.sum(np.abs(diffs)) + 1e-10
    net = c[-1]
    M = abs(net) / path_length  # monotonicity [0, 1]

    # Lévy area (depth-2 signature cross-term)
    dc = np.diff(c)
    dm = np.diff(mid)
    levy = 0.5 * np.sum(c[:-1] * dm - mid[:-1] * dc)
    K = np.tanh(levy / (abs(net) + 1e-10))  # curvature [-1, 1]

    return M, K, float(np.sign(net))


# ===========================================================================
# Component 4: Realized Semivariance Imbalance
# ===========================================================================

def realized_semivariance_imbalance(closes: np.ndarray,
                                    window: int = RSV_WINDOW) -> float:
    """Semivariance imbalance B in [-1, 1].

    Uses the last (window+1) closes to compute window returns.
    """
    slice_ = closes[-(window + 1):]
    rets = np.diff(slice_) / slice_[:-1]
    rs_plus = float(np.sum(rets[rets > 0] ** 2))
    rs_minus = float(np.sum(rets[rets <= 0] ** 2))
    return (rs_plus - rs_minus) / (rs_plus + rs_minus + 1e-12)


# ===========================================================================
# Component 5: Roughness Proxy
# ===========================================================================

def roughness_proxy(yz_vol_series: np.ndarray,
                    window: int = ROUGH_WINDOW) -> float:
    """Lag-1 autocorrelation of Δlog(vol) over last `window` bars."""
    v = yz_vol_series[-window:]
    v = v[~np.isnan(v)]
    if len(v) < 4:
        return 0.0
    dlv = np.diff(np.log(np.maximum(v, 1e-10)))
    if len(dlv) < 3:
        return 0.0
    cc = np.corrcoef(dlv[:-1], dlv[1:])
    return float(cc[0, 1]) if cc.shape == (2, 2) else 0.0


# ===========================================================================
# State Machine
# ===========================================================================

class StateMachine:
    """NEUTRAL / COIL / RELEASE / EXHAUSTION state logic."""

    NEUTRAL = "NEUTRAL"
    COIL = "COIL"
    RELEASE = "RELEASE"
    EXHAUSTION = "EXHAUSTION"

    def __init__(self) -> None:
        self.state = self.NEUTRAL
        self.prev_state = self.NEUTRAL
        # Track dc1 flips in last DC_FLIP_LOOKBACK bars
        self.dc1_flip_window: deque[int] = deque(maxlen=DC_FLIP_LOOKBACK)

    def reset_day(self) -> None:
        """Called at start of each new trading day."""
        self.state = self.NEUTRAL
        self.prev_state = self.NEUTRAL
        self.dc1_flip_window.clear()

    def update(
        self,
        dc1: DCClock,
        dc2: DCClock,
        dc3: DCClock,
        M: float,
        M_short: float,
        K: float,
        B: float,
        roughness: float,
        sigma_abs: float,
    ) -> tuple[str, int]:
        """Advance state machine by one bar.

        Returns (new_state, signal_direction):
          signal_direction = dc2.direction for RELEASE/EXHAUSTION,
          0 for NEUTRAL/COIL.
        M      : 20-bar signature monotonicity (used for COIL characterization)
        M_short: 5-bar signature monotonicity (used for RELEASE — captures breakout)
        """
        self.prev_state = self.state

        # Track dc1 flip for COIL detection
        self.dc1_flip_window.append(1 if dc1.dc_fired else 0)
        dc1_flips = sum(self.dc1_flip_window)

        # ---- RELEASE check (before COIL so COIL→RELEASE transition works) ----
        release_ok = (
            self.prev_state == self.COIL
            and dc2.dc_fired
            and (dc3.direction == dc2.direction or dc3.bars_since_dc > 20)
            and M_short >= RELEASE_M_MIN
            and B * dc2.direction > RELEASE_B_MIN
            and roughness > RELEASE_ROUGH_MIN
        )
        if release_ok:
            self.state = self.RELEASE
            return self.RELEASE, dc2.direction

        # ---- EXHAUSTION check ----
        exhaustion_ok = (
            dc2.direction == dc3.direction
            and (dc2.overshoot(sigma_abs) > EXHAUSTION_OVERSHOOT_2
                 or dc3.overshoot(sigma_abs) > EXHAUSTION_OVERSHOOT_3)
            and M < EXHAUSTION_M_MAX
            and K * dc2.direction < EXHAUSTION_K_THRESH
            and B * dc2.direction < EXHAUSTION_B_THRESH
        )
        if exhaustion_ok:
            self.state = self.EXHAUSTION
            return self.EXHAUSTION, -dc2.direction

        # ---- COIL check ----
        coil_ok = (
            dc2.bars_since_dc >= COIL_DC2_MIN_BARS
            and dc3.bars_since_dc >= COIL_DC3_MIN_BARS
            and dc1_flips >= COIL_DC1_FLIPS_MIN
            and M < COIL_M_MAX
            and abs(B) < COIL_B_MAX
        )
        if coil_ok:
            self.state = self.COIL
            return self.COIL, 0

        # Default: NEUTRAL
        self.state = self.NEUTRAL
        return self.NEUTRAL, 0


# ===========================================================================
# Trade Record
# ===========================================================================

class Trade:
    """Lightweight trade container."""

    __slots__ = (
        "entry_idx", "entry_price", "sl", "pt", "time_stop_bars",
        "direction", "signal_type", "M_at_entry", "B_at_entry",
        "sigma_abs", "date", "bars_in_trade"
    )

    def __init__(self, entry_idx: int, entry_price: float, sl: float,
                 pt: float, time_stop_bars: int, direction: int,
                 signal_type: str, M: float, B: float,
                 sigma_abs: float, date) -> None:
        self.entry_idx = entry_idx
        self.entry_price = entry_price
        self.sl = sl
        self.pt = pt
        self.time_stop_bars = time_stop_bars
        self.direction = direction
        self.signal_type = signal_type
        self.M_at_entry = M
        self.B_at_entry = B
        self.sigma_abs = sigma_abs
        self.date = date
        self.bars_in_trade = 0


def calc_pnl(entry: float, exit_price: float, direction: int) -> float:
    gross = (exit_price - entry) * direction * N_CONTRACTS * POINT_VALUE
    cost = 2 * COMMISSION_PER_SIDE * N_CONTRACTS
    return gross - cost


# ===========================================================================
# Backtest Engine
# ===========================================================================

def compute_prevvwap_days(bars: pd.DataFrame) -> set[str]:
    """Return set of date strings where yesterday's close > yesterday's VWAP.

    Computes intraday VWAP from 1-min RTH bars for each session.
    A day is in the returned set if the prior session's close was above VWAP.
    """
    bars = bars.copy()
    bars["date"] = bars.index.date
    bars["typical"] = (bars["high"] + bars["low"] + bars["close"]) / 3.0

    # Daily VWAP and last close per day
    grp = bars.groupby("date")
    daily_vwap = grp.apply(
        lambda g: (g["typical"] * g["volume"]).sum() / g["volume"].sum()
        if g["volume"].sum() > 0 else g["typical"].mean()
    )
    daily_close = grp["close"].last()

    bullish_dates: set[str] = set()
    sorted_dates = sorted(daily_vwap.index)
    for i in range(1, len(sorted_dates)):
        prev_d = sorted_dates[i - 1]
        today_d = sorted_dates[i]
        if daily_close[prev_d] > daily_vwap[prev_d]:
            bullish_dates.add(str(today_d))

    return bullish_dates


def run_backtest(
    bars: pd.DataFrame,
    allow_release: bool = True,
    allow_exhaustion: bool = True,
    long_only: bool = False,
    min_coil_bars: int = 1,
    dc3_aligned: bool = False,
    entry_cutoff: tuple[int, int] | None = None,
    prevvwap_dates: set[str] | None = None,
    pt_mult: float = 2.0,
    sl_frac: float = 0.55,
    time_stop_bars: int = 12,
    afternoon_cutoff: tuple[int, int] | None = None,
) -> tuple[list[dict], dict]:
    """Main MSITE backtest loop.

    Parameters
    ----------
    bars : DataFrame with columns open/high/low/close/volume, RTH only,
           DatetimeIndex (tz-aware US/Eastern).
    allow_release : include RELEASE signals.
    allow_exhaustion : include EXHAUSTION signals.
    long_only : only take LONG (direction=+1) RELEASE signals.
    min_coil_bars : require at least this many consecutive COIL bars before RELEASE.
    dc3_aligned : only enter RELEASE when dc3.direction == sig_dir.
    entry_cutoff : (hour, minute) ET — no new entries at or after this time.
    prevvwap_dates : set of date strings where prevVWAP was bullish; if provided,
                     only take RELEASE signals on those days.
    pt_mult : PT = pt_mult * sl_dist  (default 2.0).
    sl_frac : SL distance = max(8 ticks, sl_frac * theta2_abs)  (default 0.55).
    time_stop_bars : exit after this many bars if neither PT nor SL hit (default 12).
    afternoon_cutoff : (hour, minute) ET — skip RELEASE signals AT OR AFTER this time.
                       Different from entry_cutoff: entry_cutoff blocks the signal queue,
                       afternoon_cutoff filters the signal itself (e.g. (14, 0) = no new
                       RELEASE entries at or after 14:00 ET).

    Returns
    -------
    trades : list of trade dicts
    stats  : component statistics dict
    """
    opens = bars["open"].values
    highs = bars["high"].values
    lows = bars["low"].values
    closes = bars["close"].values
    idx = bars.index

    n = len(closes)
    slippage_val = SLIPPAGE_TICKS * TICK_SIZE

    # Pre-compute Yang-Zhang vol for entire series
    yz_vol = yang_zhang_vol(opens, highs, lows, closes, n=YZ_WINDOW)

    # Instantiate DC clocks
    dc1 = DCClock(DC_THETA_FACTORS[0])
    dc2 = DCClock(DC_THETA_FACTORS[1])
    dc3 = DCClock(DC_THETA_FACTORS[2])

    sm = StateMachine()

    trades: list[dict] = []
    active_trade: Trade | None = None

    # Risk management state
    equity = STARTING_EQUITY
    peak_equity = STARTING_EQUITY
    daily_pnl: dict[str, float] = {}

    # Component statistics counters
    state_counts = {k: 0 for k in ("NEUTRAL", "COIL", "RELEASE", "EXHAUSTION")}
    dc_daily_flips = {0: [], 1: [], 2: []}   # per-day flip counts
    release_count = 0
    exhaustion_count = 0

    # Per-day tracking
    current_date = None
    dc_flips_today = [0, 0, 0]
    pending_signal: dict | None = None  # RELEASE signal fires NEXT bar
    consecutive_coil_bars = 0            # tracks how many bars in a row we've been COIL

    for i in range(n):
        ts = idx[i]
        bar_date = ts.date()

        # ---- Day reset ----
        if bar_date != current_date:
            if current_date is not None:
                dc_daily_flips[0].append(dc_flips_today[0])
                dc_daily_flips[1].append(dc_flips_today[1])
                dc_daily_flips[2].append(dc_flips_today[2])

            current_date = bar_date
            dc_flips_today = [0, 0, 0]
            dc1.reset(closes[i])
            dc2.reset(closes[i])
            dc3.reset(closes[i])
            sm.reset_day()
            pending_signal = None
            consecutive_coil_bars = 0

            # Force-close any open trade that survived overnight
            if active_trade is not None:
                exit_p = closes[i]
                pnl = calc_pnl(active_trade.entry_price, exit_p,
                               active_trade.direction)
                _record_trade(trades, active_trade, exit_p, pnl,
                              "session_close", equity, daily_pnl)
                equity += pnl
                peak_equity = max(peak_equity, equity)
                active_trade = None

        # Skip warmup
        if i < WARMUP_BARS:
            # Still update DC clocks during warmup
            sa = yz_vol[i] if not np.isnan(yz_vol[i]) else closes[i] * 0.001
            dc1.update(closes[i], sa)
            dc2.update(closes[i], sa)
            dc3.update(closes[i], sa)
            continue

        sigma_abs = yz_vol[i]
        if np.isnan(sigma_abs) or sigma_abs <= 0:
            sigma_abs = closes[i] * 0.001

        # ---- Update DC clocks ----
        fired1 = dc1.update(closes[i], sigma_abs)
        fired2 = dc2.update(closes[i], sigma_abs)
        fired3 = dc3.update(closes[i], sigma_abs)

        if fired1:
            dc_flips_today[0] += 1
        if fired2:
            dc_flips_today[1] += 1
        if fired3:
            dc_flips_today[2] += 1

        # ---- Signature scores ----
        start_sig = max(0, i - SIG_WINDOW + 1)
        M, K, _ = compute_signature_scores(
            closes[start_sig: i + 1],
            highs[start_sig: i + 1],
            lows[start_sig: i + 1],
            sigma_abs,
        )
        # Short-window M for RELEASE check (captures breakout momentum, not COIL choppiness)
        start_sig_s = max(0, i - SIG_WINDOW_RELEASE + 1)
        M_short, _, _ = compute_signature_scores(
            closes[start_sig_s: i + 1],
            highs[start_sig_s: i + 1],
            lows[start_sig_s: i + 1],
            sigma_abs,
        )

        # ---- Semivariance imbalance ----
        start_rsv = max(0, i - RSV_WINDOW)
        B = realized_semivariance_imbalance(closes[start_rsv: i + 1])

        # ---- Roughness ----
        start_rough = max(0, i - ROUGH_WINDOW)
        roughness = roughness_proxy(yz_vol[start_rough: i + 1])

        # ---- State machine ----
        new_state, sig_dir = sm.update(
            dc1, dc2, dc3, M, M_short, K, B, roughness, sigma_abs
        )
        state_counts[new_state] += 1

        # Capture streak before resetting (used by min_coil_bars filter at RELEASE)
        coil_streak = consecutive_coil_bars
        if new_state == StateMachine.COIL:
            consecutive_coil_bars += 1
        else:
            consecutive_coil_bars = 0

        # ---- Session close forced exit (15:55) ----
        session_end = (ts.hour == 15 and ts.minute >= 55)

        # ---- Active trade exit check ----
        if active_trade is not None:
            active_trade.bars_in_trade += 1
            direction = active_trade.direction
            exit_p = None
            exit_reason = None

            if session_end:
                exit_p = closes[i]
                exit_reason = "session_close"
            elif direction == 1:
                if highs[i] >= active_trade.pt:
                    exit_p = active_trade.pt
                    exit_reason = "profit_target"
                elif lows[i] <= active_trade.sl:
                    exit_p = active_trade.sl
                    exit_reason = "stop_loss"
            else:  # direction == -1
                if lows[i] <= active_trade.pt:
                    exit_p = active_trade.pt
                    exit_reason = "profit_target"
                elif highs[i] >= active_trade.sl:
                    exit_p = active_trade.sl
                    exit_reason = "stop_loss"

            if exit_p is None and active_trade.bars_in_trade >= active_trade.time_stop_bars:
                exit_p = closes[i]
                exit_reason = "time_stop"

            if exit_p is None and K * direction < K_TRAIL_THRESH:
                exit_p = closes[i]
                exit_reason = "k_flip"

            if exit_p is not None:
                pnl = calc_pnl(active_trade.entry_price, exit_p, direction)
                date_str = str(active_trade.date)
                daily_pnl[date_str] = daily_pnl.get(date_str, 0.0) + pnl
                _record_trade(trades, active_trade, exit_p, pnl,
                              exit_reason, equity, daily_pnl)
                equity += pnl
                peak_equity = max(peak_equity, equity)
                active_trade = None

        # ---- Entry from pending RELEASE signal (previous bar fired) ----
        if (active_trade is None
                and pending_signal is not None
                and not session_end):

            sig = pending_signal
            pending_signal = None
            date_str = str(bar_date)

            # Daily loss check
            day_loss = daily_pnl.get(date_str, 0.0)
            drawdown = equity - peak_equity

            if (day_loss > MAX_DAILY_LOSS
                    and -drawdown < DRAWDOWN_BUFFER
                    and allow_release):
                direction = sig["direction"]
                entry_p = closes[i] + direction * slippage_val
                theta2_abs = 0.80 * sig["sigma_abs"]
                sl_dist = max(8 * TICK_SIZE, sl_frac * theta2_abs)
                sl = entry_p - direction * sl_dist
                pt = entry_p + direction * pt_mult * sl_dist

                active_trade = Trade(
                    entry_idx=i,
                    entry_price=entry_p,
                    sl=sl,
                    pt=pt,
                    time_stop_bars=time_stop_bars,
                    direction=direction,
                    signal_type="RELEASE",
                    M=sig["M"],
                    B=sig["B"],
                    sigma_abs=sig["sigma_abs"],
                    date=bar_date,
                )

        # ---- Generate new signals (if no open trade) ----
        if active_trade is None and not session_end and i + 1 < n:
            date_str = str(bar_date)
            day_loss = daily_pnl.get(date_str, 0.0)
            drawdown = equity - peak_equity

            risk_ok = (day_loss > MAX_DAILY_LOSS
                       and -drawdown < DRAWDOWN_BUFFER)

            if new_state == StateMachine.RELEASE and allow_release and risk_ok:
                # --- Optional filters ---
                # 1. Long-only
                if long_only and sig_dir != 1:
                    pass
                # 2. Minimum consecutive COIL bars before this RELEASE
                elif min_coil_bars > 1 and coil_streak < min_coil_bars:
                    pass
                # 3. DC3 alignment
                elif dc3_aligned and dc3.direction != sig_dir:
                    pass
                # 4. Entry time cutoff (no entries before this time)
                elif (entry_cutoff is not None
                      and (ts.hour, ts.minute) >= entry_cutoff):
                    pass
                # 4b. Afternoon cutoff (no entries at or after this time)
                elif (afternoon_cutoff is not None
                      and (ts.hour, ts.minute) >= afternoon_cutoff):
                    pass
                # 5. PrevVWAP filter
                elif (prevvwap_dates is not None
                      and str(bar_date) not in prevvwap_dates):
                    pass
                else:
                    release_count += 1
                    # Queue for next-bar entry
                    pending_signal = {
                        "direction": sig_dir,
                        "sigma_abs": sigma_abs,
                        "M": M,
                        "B": B,
                    }

            elif new_state == StateMachine.EXHAUSTION and allow_exhaustion and risk_ok:
                # Same-bar entry — check inside prior 3-bar range
                if i >= 3:
                    prior_high = np.max(highs[i - 3: i])
                    prior_low = np.min(lows[i - 3: i])
                    inside_range = prior_low <= closes[i] <= prior_high
                else:
                    inside_range = True

                if inside_range:
                    exhaustion_count += 1
                    fade_dir = sig_dir  # already -dc2.direction from state machine
                    entry_p = closes[i] + fade_dir * slippage_val
                    theta2_abs = 0.80 * sigma_abs
                    sl_dist = max(6 * TICK_SIZE, 0.40 * theta2_abs)
                    sl = entry_p - fade_dir * sl_dist
                    pt = dc2.dc_price  # fade back to DC fire point

                    active_trade = Trade(
                        entry_idx=i,
                        entry_price=entry_p,
                        sl=sl,
                        pt=pt,
                        time_stop_bars=10,
                        direction=fade_dir,
                        signal_type="EXHAUSTION",
                        M=M,
                        B=B,
                        sigma_abs=sigma_abs,
                        date=bar_date,
                    )

    # Finalize any open trade at end of data
    if active_trade is not None and n > 0:
        exit_p = closes[-1]
        pnl = calc_pnl(active_trade.entry_price, exit_p, active_trade.direction)
        _record_trade(trades, active_trade, exit_p, pnl, "end_of_data",
                      equity, daily_pnl)
        equity += pnl
        peak_equity = max(peak_equity, equity)

    # Aggregate daily flips
    avg_flips = [
        np.mean(dc_daily_flips[k]) if dc_daily_flips[k] else 0.0
        for k in range(3)
    ]
    total_bars = sum(state_counts.values())

    component_stats = {
        "dc_avg_daily_flips": avg_flips,
        "state_distribution": {
            k: (v / max(total_bars, 1)) for k, v in state_counts.items()
        },
        "release_signals": release_count,
        "exhaustion_signals": exhaustion_count,
        "total_bars": total_bars,
        "total_days": len(dc_daily_flips[0]),
    }

    return trades, component_stats


def _record_trade(
    trades: list,
    t: Trade,
    exit_p: float,
    pnl: float,
    reason: str,
    equity: float,
    daily_pnl: dict,
) -> None:
    trades.append({
        "date": str(t.date),
        "signal_type": t.signal_type,
        "direction": t.direction,
        "entry": round(t.entry_price, 4),
        "exit": round(exit_p, 4),
        "sl": round(t.sl, 4),
        "pt": round(t.pt, 4),
        "pnl": round(pnl, 2),
        "reason": reason,
        "M_at_entry": round(t.M_at_entry, 4),
        "B_at_entry": round(t.B_at_entry, 4),
        "bars_in_trade": t.bars_in_trade,
    })


# ===========================================================================
# Analytics
# ===========================================================================

def compute_metrics(trades: list[dict]) -> dict:
    """Compute win-rate, Sharpe, max drawdown, expectancy."""
    if not trades:
        return {"n": 0, "wr": 0.0, "avg_pnl": 0.0, "sharpe": 0.0,
                "maxdd": 0.0, "exp": 0.0, "total_pnl": 0.0}

    pnls = np.array([t["pnl"] for t in trades])
    n = len(pnls)
    wins = np.sum(pnls > 0)
    wr = wins / n
    avg_pnl = float(np.mean(pnls))
    total_pnl = float(np.sum(pnls))

    # Sharpe (daily-level)
    dates = sorted(set(t["date"] for t in trades))
    daily = {d: 0.0 for d in dates}
    for t in trades:
        daily[t["date"]] += t["pnl"]
    daily_series = np.array(list(daily.values()))
    if len(daily_series) > 1 and np.std(daily_series) > 0:
        sharpe = float(np.mean(daily_series) / np.std(daily_series)
                       * np.sqrt(252))
    else:
        sharpe = 0.0

    # Max drawdown (cumulative trade PnL)
    cum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum)
    dd_series = cum - running_max
    maxdd = float(np.min(dd_series))

    return {
        "n": n,
        "wr": wr,
        "avg_pnl": avg_pnl,
        "sharpe": sharpe,
        "maxdd": maxdd,
        "exp": avg_pnl,
        "total_pnl": total_pnl,
    }


def exit_distribution(trades: list[dict]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for t in trades:
        dist[t["reason"]] = dist.get(t["reason"], 0) + 1
    return dist


def monthly_breakdown(trades: list[dict]) -> list[dict]:
    from collections import defaultdict
    months: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        month = t["date"][:7]
        months[month].append(t["pnl"])

    rows = []
    cumul = 0.0
    for month in sorted(months):
        pnls = months[month]
        n = len(pnls)
        w = sum(1 for p in pnls if p > 0)
        l = n - w
        pnl = sum(pnls)
        cumul += pnl
        rows.append({
            "month": month,
            "n": n,
            "w": w,
            "l": l,
            "wr": w / n if n else 0.0,
            "pnl": pnl,
            "cumul": cumul,
        })
    return rows


# ===========================================================================
# Output / Printing
# ===========================================================================

def print_component_stats(stats: dict) -> None:
    flips = stats["dc_avg_daily_flips"]
    sd = stats["state_distribution"]
    total_days = stats.get("total_days", 1)
    release_total = stats["release_signals"]
    exhaust_total = stats["exhaustion_signals"]
    release_per_day = release_total / max(total_days, 1)

    print("\n=== MSITE COMPONENT STATISTICS ===")
    print(f"  DC Clock avg daily flips:  "
          f"θ1={flips[0]:.1f}  θ2={flips[1]:.1f}  θ3={flips[2]:.1f}")
    print(f"  State distribution:  "
          f"COIL={sd.get('COIL', 0):.1%}  "
          f"RELEASE={sd.get('RELEASE', 0):.1%}  "
          f"EXHAUST={sd.get('EXHAUSTION', 0):.1%}  "
          f"NEUTRAL={sd.get('NEUTRAL', 0):.1%}")
    print(f"  RELEASE signals: {release_total}  "
          f"(N fired per day avg: {release_per_day:.1f})")
    print(f"  EXHAUSTION signals: {exhaust_total}")


def fmt_row(name: str, m: dict) -> str:
    n = m["n"]
    if n == 0:
        return f"  {name:<28} {'0':>4}   {'N/A':>6}   {'N/A':>6}   {'N/A':>5}  {'N/A':>6}   {'N/A':>6}"
    return (
        f"  {name:<28} {n:>4}  {m['wr']:>6.1%}  "
        f"${m['avg_pnl']:>+7.0f}  {m['sharpe']:>5.2f}  "
        f"${m['maxdd']:>+7.0f}  ${m['exp']:>+7.0f}"
    )


def print_results(rel_trades: list, exh_trades: list, all_trades: list,
                  stats: dict) -> None:
    m_rel = compute_metrics(rel_trades)
    m_exh = compute_metrics(exh_trades)
    m_all = compute_metrics(all_trades)

    print("\n=== MSITE BACKTEST RESULTS ===")
    header = (f"  {'Config':<28} {'N':>4}  {'WR':>6}  "
              f"{'Avg$':>8}  {'Sharpe':>5}  {'MaxDD':>8}  {'Exp$/t':>8}")
    print(header)
    print("  " + "-" * 72)
    print(fmt_row("RELEASE continuation", m_rel))
    print(fmt_row("EXHAUSTION fade", m_exh))
    print(fmt_row("Combined MSITE", m_all))
    # Baseline reference row
    b = BASELINE
    base_row = (
        f"  {b['name']:<28} {b['n']:>4}  {b['wr']:>6.1%}  "
        f"${b['avg_pnl']:>+7.0f}  {b['sharpe']:>5.2f}  "
        f"${b['maxdd']:>+7.0f}  ${b['exp']:>+7.0f}"
    )
    print(base_row)

    # Exit distribution for RELEASE
    if rel_trades:
        dist = exit_distribution(rel_trades)
        total_rel = len(rel_trades)
        parts = []
        for reason in ("profit_target", "time_stop", "stop_loss", "k_flip",
                       "session_close", "end_of_data"):
            cnt = dist.get(reason, 0)
            if cnt:
                parts.append(f"{reason}: {cnt} ({cnt/total_rel:.0%})")
        print("\n=== EXIT DISTRIBUTION ===")
        print("  RELEASE: " + " | ".join(parts))

    # Monthly breakdown
    if all_trades:
        rows = monthly_breakdown(all_trades)
        print("\n=== MONTHLY BREAKDOWN ===")
        print(f"  {'Month':<10} {'N':>4}  {'W':>3}  {'L':>3}  "
              f"{'WR':>6}  {'PnL':>9}  {'Cumul':>9}")
        print("  " + "-" * 52)
        for r in rows:
            print(
                f"  {r['month']:<10} {r['n']:>4}  {r['w']:>3}  {r['l']:>3}  "
                f"{r['wr']:>6.1%}  ${r['pnl']:>+8.0f}  ${r['cumul']:>+8.0f}"
            )


def save_results(rel_trades: list, exh_trades: list, all_trades: list,
                 stats: dict, path: Path) -> None:
    m_rel = compute_metrics(rel_trades)
    m_exh = compute_metrics(exh_trades)
    m_all = compute_metrics(all_trades)

    output = {
        "component_stats": stats,
        "results": {
            "release": m_rel,
            "exhaustion": m_exh,
            "combined": m_all,
            "baseline": BASELINE,
        },
        "exit_distribution": {
            "release": exit_distribution(rel_trades),
            "exhaustion": exit_distribution(exh_trades),
        },
        "monthly_breakdown": monthly_breakdown(all_trades),
        "trades": all_trades,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {path}")


# ===========================================================================
# Data Loading
# ===========================================================================

def resample_bars(bars: pd.DataFrame, tf_minutes: int) -> pd.DataFrame:
    """Resample OHLCV bars to a higher timeframe.

    Input bars must already be RTH-only with a tz-aware DatetimeIndex.
    """
    rule = f"{tf_minutes}min"
    resampled = bars.resample(rule, closed="left", label="left").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna(subset=["open"])
    # Re-apply RTH filter in case resampling introduced off-hours bars
    ts = resampled.index
    rth_mask = (
        ((ts.hour == 9) & (ts.minute >= 30))
        | (ts.hour > 9)
    ) & (ts.hour < 16)
    return resampled[rth_mask].copy()


def load_data(path: str | None = None, key: str | None = None,
              resample_tf: int | None = None) -> pd.DataFrame:
    """Load MNQ bars, apply RTH filter, and optionally resample."""
    data_path = Path(path) if path else DATA_PATH
    hdf_key = key if key else ("bars_1min" if "1min" in str(data_path) else "bars_5min")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    bars = pd.read_hdf(data_path, key=hdf_key)

    # RTH filter: (hour > 9) OR (hour == 9 AND minute >= 30), AND hour < 16
    ts = bars.index
    rth_mask = (
        ((ts.hour == 9) & (ts.minute >= 30))
        | (ts.hour > 9)
    ) & (ts.hour < 16)
    bars = bars[rth_mask].copy()

    if resample_tf and resample_tf > 1:
        orig_len = len(bars)
        bars = resample_bars(bars, resample_tf)
        print(f"  Loaded {orig_len} RTH bars → resampled to {resample_tf}-min: "
              f"{len(bars)} bars "
              f"({bars.index[0].date()} to {bars.index[-1].date()})")
    else:
        print(f"  Loaded {len(bars)} RTH bars "
              f"({bars.index[0].date()} to {bars.index[-1].date()})")
    return bars


# ===========================================================================
# Optimization Sweep
# ===========================================================================

def run_optimization_sweep(bars: pd.DataFrame) -> None:
    """Comprehensive optimization sweep — RELEASE-only, LongOnly baseline.

    Groups:
      A. PT multiplier sweep
      B. SL fraction sweep
      C. Time stop sweep
      D. Afternoon cutoff sweep
      E. Best combos from A-D
    """
    print("\n=== MSITE OPTIMIZATION SWEEP ===")
    total_days = len(set(str(d.date()) for d in bars.index))
    print(f"  Dataset: {bars.index[0].date()} → {bars.index[-1].date()} "
          f"({total_days} trading days)\n")

    prevvwap = compute_prevvwap_days(bars)

    header = (f"  {'Config':<40} {'N':>4}  {'WR':>6}  "
              f"{'Avg$':>7}  {'Sharpe':>6}  {'MaxDD':>8}  {'PnL':>8}")
    sep = "  " + "-" * 86

    BASE = {"long_only": True, "pt_mult": 2.0, "sl_frac": 0.55, "time_stop_bars": 12}

    def _run(label, overrides):
        kw = {**BASE, **overrides}
        trades, _ = run_backtest(bars, allow_release=True, allow_exhaustion=False, **kw)
        rel = [t for t in trades if t["signal_type"] == "RELEASE"]
        m = compute_metrics(rel)
        flag = " ◀" if m["n"] >= 10 and m["sharpe"] > 0 else ""
        if m["n"] == 0:
            row = f"  {label:<40} {'0':>4}   {'—':>6}   {'—':>6}   {'—':>5}   {'—':>7}   {'—':>7}"
        else:
            row = (f"  {label:<40} {m['n']:>4}  {m['wr']:>6.1%}  "
                   f"${m['avg_pnl']:>+6.0f}  {m['sharpe']:>6.2f}  "
                   f"${m['maxdd']:>+7.0f}  ${m['total_pnl']:>+7.0f}{flag}")
        print(row)
        return m

    results = {}

    # ── Group A: PT multiplier ────────────────────────────────────────────────
    print("  GROUP A — PT multiplier  (SL=0.55x, time_stop=12)\n" + header + "\n" + sep)
    for pt in [1.3, 1.5, 2.0, 2.5, 3.0]:
        m = _run(f"LongOnly PT={pt}x", {"pt_mult": pt})
        results[f"PT={pt}x"] = ({"pt_mult": pt}, m)

    # ── Group B: SL fraction ──────────────────────────────────────────────────
    print("\n  GROUP B — SL fraction  (PT=2.0x, time_stop=12)\n" + header + "\n" + sep)
    for sl in [0.35, 0.45, 0.55, 0.65, 0.80]:
        m = _run(f"LongOnly SL={sl}x", {"sl_frac": sl})
        results[f"SL={sl}x"] = ({"sl_frac": sl}, m)

    # ── Group C: Time stop ────────────────────────────────────────────────────
    print("\n  GROUP C — Time stop bars  (PT=2.0x, SL=0.55x)\n" + header + "\n" + sep)
    for ts in [6, 8, 12, 16, 24]:
        m = _run(f"LongOnly time_stop={ts}b ({ts*5}min)", {"time_stop_bars": ts})
        results[f"ts={ts}"] = ({"time_stop_bars": ts}, m)

    # ── Group D: Afternoon cutoff ─────────────────────────────────────────────
    print("\n  GROUP D — Afternoon cutoff  (PT=2.0x, SL=0.55x)\n" + header + "\n" + sep)
    m = _run("LongOnly no cutoff (baseline)", {})
    results["no_cutoff"] = ({}, m)
    for h, label in [(13, "13:00"), (14, "14:00"), (15, "15:00")]:
        m = _run(f"LongOnly no entries ≥{label}", {"afternoon_cutoff": (h, 0)})
        results[f"aft_{h}"] = ({"afternoon_cutoff": (h, 0)}, m)

    # ── Group E: Best combos ──────────────────────────────────────────────────
    # Pick the best single param from each group and combine them
    viable = {k: (ov, m) for k, (ov, m) in results.items()
              if m["n"] >= 10 and m["sharpe"] > 0}
    if viable:
        best_pt_k  = max((k for k in viable if k.startswith("PT=")),
                         key=lambda k: viable[k][1]["sharpe"], default=None)
        best_sl_k  = max((k for k in viable if k.startswith("SL=")),
                         key=lambda k: viable[k][1]["sharpe"], default=None)
        best_ts_k  = max((k for k in viable if k.startswith("ts=")),
                         key=lambda k: viable[k][1]["sharpe"], default=None)
        best_aft_k = max((k for k in viable if k.startswith("aft_")),
                         key=lambda k: viable[k][1]["sharpe"], default=None)

        best_pt_ov  = viable[best_pt_k][0]  if best_pt_k  else {}
        best_sl_ov  = viable[best_sl_k][0]  if best_sl_k  else {}
        best_ts_ov  = viable[best_ts_k][0]  if best_ts_k  else {}
        best_aft_ov = viable[best_aft_k][0] if best_aft_k else {}

        print("\n  GROUP E — Best combos\n" + header + "\n" + sep)
        _run("Best PT + Best SL", {**best_pt_ov, **best_sl_ov})
        _run("Best PT + Best TS", {**best_pt_ov, **best_ts_ov})
        _run("Best PT + Best Aft", {**best_pt_ov, **best_aft_ov})
        _run("Best SL + Best TS", {**best_sl_ov, **best_ts_ov})
        _run("Best PT + SL + TS", {**best_pt_ov, **best_sl_ov, **best_ts_ov})
        _run("Best PT + SL + TS + Aft",
             {**best_pt_ov, **best_sl_ov, **best_ts_ov, **best_aft_ov})
        _run("Best all + PrevVWAP",
             {**best_pt_ov, **best_sl_ov, **best_ts_ov, **best_aft_ov,
              "prevvwap_dates": prevvwap})

    # ── Summary ───────────────────────────────────────────────────────────────
    print(sep)
    all_viable = [(k, ov, m) for k, (ov, m) in results.items()
                  if m["n"] >= 10 and m["sharpe"] > 0]
    if all_viable:
        best = max(all_viable, key=lambda x: x[2]["sharpe"])
        print(f"\n  Best single-param (N≥10): {best[0]}")
        print(f"    N={best[2]['n']}  WR={best[2]['wr']:.1%}  "
              f"Sharpe={best[2]['sharpe']:.2f}  MaxDD=${best[2]['maxdd']:+.0f}  "
              f"PnL=${best[2]['total_pnl']:+.0f}")
    print(f"\n  Reference — ORB+PrevVWAP: N=16  WR=75.0%  Sharpe=8.29  MaxDD=-$796")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MSITE — Multi-Scale Intrinsic-Time Engine backtest"
    )
    parser.add_argument("--release-only", action="store_true",
                        help="Only run RELEASE continuation signals")
    parser.add_argument("--exhaustion-only", action="store_true",
                        help="Only run EXHAUSTION fade signals")
    parser.add_argument("--save", action="store_true",
                        help="Save results to msite_results.json")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Override data file path")
    parser.add_argument("--key", type=str, default=None,
                        help="HDF5 key to read (default: bars_5min or bars_1min)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run optimization sweep across filter combinations")
    parser.add_argument("--tf", type=int, default=1, choices=[1, 5, 15],
                        help="Bar timeframe in minutes (1/5/15). Adjusts all calibration constants.")
    parser.add_argument("--resample", type=int, default=None,
                        help="Resample input data to this timeframe (e.g. --resample 5). "
                             "Requires 1-min input data.")
    args = parser.parse_args()

    # Apply timeframe calibration (uses --resample if set, else --tf)
    effective_tf = args.resample if args.resample else args.tf
    if effective_tf != 1:
        _apply_tf_config(effective_tf)

    print("=== MSITE Backtest ===")
    print(f"  Timeframe: {effective_tf}-min  (calibration applied)")
    bars = load_data(path=args.data_path, key=args.key, resample_tf=args.resample)

    if args.sweep:
        run_optimization_sweep(bars)
        return

    allow_release = not args.exhaustion_only
    allow_exhaustion = not args.release_only

    print(f"  Mode: {'RELEASE only' if args.release_only else 'EXHAUSTION only' if args.exhaustion_only else 'Combined'}")

    trades, stats = run_backtest(
        bars,
        allow_release=allow_release,
        allow_exhaustion=allow_exhaustion,
    )

    rel_trades = [t for t in trades if t["signal_type"] == "RELEASE"]
    exh_trades = [t for t in trades if t["signal_type"] == "EXHAUSTION"]

    print_component_stats(stats)
    print_results(rel_trades, exh_trades, trades, stats)

    if args.save:
        save_results(rel_trades, exh_trades, trades, stats, RESULTS_PATH)


if __name__ == "__main__":
    main()
