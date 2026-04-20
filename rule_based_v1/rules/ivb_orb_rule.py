"""IVB-style ORB Rule — primary signal for LiveRunner.

Implements the IVB/ORB concept with OHLCV data:
  1. Opening range (OR) from the first or_minutes of RTH.
  2. Fixed-range volume profile (FRVP) across that range.
  3. Two entry modes:
       breakout — close beyond OR high/low with body+volume aggression.
       reload   — pullback to VAH/VAL after a confirmed breakout, same aggression filter.
  4. Stop: min(VAL, or_low) - tick for LONG; max(VAH, or_high) + tick for SHORT.
  5. Target: entry + target_range_mult * or_range (direction-adjusted).

Returns stop_price and target_price in signal metadata so the runner
can use them directly instead of ATR-based derivation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import time

from rules.base import BaseRule, RuleSignal


class IVBORBRule(BaseRule):
    """IVB-style Opening Range Breakout primary signal.

    Parameters
    ----------
    or_minutes : int
        Length of the opening range window in minutes. Default 40.
    entry_cutoff_time : str
        Last allowed entry bar (HH:MM ET). Default "14:00".
    min_volume_ratio : float
        Bar volume must be >= this multiple of OR average volume. Default 1.1.
    reload_tolerance_ticks : int
        How many ticks beyond VAH/VAL still counts as a reload touch. Default 4.
    target_range_mult : float
        Target = entry ± target_range_mult * or_range. Default 1.0.
    tick_size : float
        Instrument tick size for stop placement. Default 0.25 (MES).
    mode : str
        "breakout", "reload", or "both". Default "both".
    """

    def __init__(
        self,
        or_minutes: int = 40,
        entry_cutoff_time: str = "14:00",
        min_volume_ratio: float = 1.1,
        reload_tolerance_ticks: int = 4,
        target_range_mult: float = 1.0,
        tick_size: float = 0.25,
        mode: str = "both",
        min_body_ratio: float = 0.35,
    ):
        super().__init__(name="ivb_orb", role="primary")
        self.or_minutes = or_minutes
        self.min_volume_ratio = min_volume_ratio
        self.reload_tolerance_ticks = reload_tolerance_ticks
        self.target_range_mult = target_range_mult
        self.tick_size = tick_size
        self.mode = mode
        self.min_body_ratio = min_body_ratio  # mutable — runner updates per-session based on VIX

        h_cut, m_cut = map(int, entry_cutoff_time.split(":"))
        self._entry_cutoff = time(h_cut, m_cut)
        self._session_start = time(9, 30)
        self._session_end = time(16, 0)

    def required_bars(self) -> int:
        return 90

    def evaluate(self, bars: pd.DataFrame) -> RuleSignal:
        if len(bars) < self.required_bars():
            return RuleSignal.no_signal(self.name, "Insufficient bars")

        idx = bars.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        current_ts = idx[-1].tz_convert("US/Eastern")
        current_time = current_ts.time()
        today = current_ts.date()

        if current_time > self._entry_cutoff:
            return RuleSignal.no_signal(self.name, f"Past entry cutoff {current_time}")
        if current_time < self._session_start:
            return RuleSignal.no_signal(self.name, "Pre-session")

        # --- Build today's bar subsets ---
        idx_et = idx.tz_convert("US/Eastern")
        today_mask = (idx_et.date == today) & (idx_et.time >= self._session_start) & (idx_et.time < self._session_end)
        today_bars = bars.loc[today_mask]
        if today_bars.empty:
            return RuleSignal.no_signal(self.name, "No today bars")

        session_start_ts = today_bars.index[0]
        or_end_ts = session_start_ts + pd.Timedelta(minutes=self.or_minutes)

        # Opening range bars
        or_mask_t = (today_bars.index >= session_start_ts) & (today_bars.index < or_end_ts)
        or_bars = today_bars.loc[or_mask_t]
        if len(or_bars) < 4:
            return RuleSignal.no_signal(self.name, f"Insufficient OR bars: {len(or_bars)}")

        # Current bar must be AFTER OR window
        if current_ts <= or_end_ts:
            return RuleSignal.no_signal(self.name, "Still in OR window")

        or_high = float(or_bars["high"].max())
        or_low = float(or_bars["low"].min())
        or_range = or_high - or_low
        if or_range <= self.tick_size * 4:
            return RuleSignal.no_signal(self.name, f"OR too narrow: {or_range:.2f}")

        vp = self._volume_profile(or_bars)
        avg_volume = float(or_bars["volume"].mean())
        reload_tol = self.reload_tolerance_ticks * self.tick_size

        # Post-OR bars for this session (excluding current bar for history check)
        post_or_mask = (today_bars.index >= or_end_ts)
        post_or_bars = today_bars.loc[post_or_mask]

        current_bar = today_bars.iloc[-1]
        prior_post_or = post_or_bars.iloc[:-1] if len(post_or_bars) > 1 else pd.DataFrame()

        close = float(current_bar["close"])
        high = float(current_bar["high"])
        low = float(current_bar["low"])

        # Did a prior post-OR bar break the range?
        broke_up = len(prior_post_or) > 0 and (prior_post_or["close"] > or_high).any()
        broke_down = len(prior_post_or) > 0 and (prior_post_or["close"] < or_low).any()

        entry_direction = 0
        entry_mode = ""

        # --- Breakout entries ---
        if self.mode in {"breakout", "both"}:
            if close > or_high and self._aggression(current_bar, avg_volume, 1):
                entry_direction = 1
                entry_mode = "breakout"
            elif close < or_low and self._aggression(current_bar, avg_volume, -1):
                entry_direction = -1
                entry_mode = "breakout"

        # --- Reload entries ---
        if entry_direction == 0 and self.mode in {"reload", "both"}:
            if (
                broke_up
                and low <= vp["vah"] + reload_tol
                and close > vp["vah"]
                and self._aggression(current_bar, avg_volume, 1, ratio_mult=0.9)
            ):
                entry_direction = 1
                entry_mode = "reload"
            elif (
                broke_down
                and high >= vp["val"] - reload_tol
                and close < vp["val"]
                and self._aggression(current_bar, avg_volume, -1, ratio_mult=0.9)
            ):
                entry_direction = -1
                entry_mode = "reload"

        if entry_direction == 0:
            return RuleSignal.no_signal(
                self.name,
                f"No signal — close={close:.2f} OR=[{or_low:.2f},{or_high:.2f}] "
                f"broke_up={broke_up} broke_dn={broke_down}",
            )

        # --- Compute stop and target ---
        entry_est = close + entry_direction * self.tick_size  # slippage estimate

        if entry_direction == 1:
            stop_price = min(vp["val"], or_low) - self.tick_size
            target_price = entry_est + self.target_range_mult * or_range
            risk = entry_est - stop_price
        else:
            stop_price = max(vp["vah"], or_high) + self.tick_size
            target_price = entry_est - self.target_range_mult * or_range
            risk = stop_price - entry_est

        if risk <= self.tick_size:
            return RuleSignal.no_signal(self.name, f"Stop too close: risk={risk:.2f}")

        return RuleSignal(
            direction=entry_direction,
            strength=0.7,
            rule_name=self.name,
            reason=(
                f"IVB ORB {'LONG' if entry_direction == 1 else 'SHORT'} [{entry_mode}]: "
                f"close={close:.2f} OR=[{or_low:.2f},{or_high:.2f}] "
                f"VAH={vp['vah']:.2f} VAL={vp['val']:.2f} "
                f"stop={stop_price:.2f} target={target_price:.2f}"
            ),
            metadata={
                "stop_price": stop_price,
                "target_price": target_price,
                "or_high": or_high,
                "or_low": or_low,
                "or_range": or_range,
                "vah": vp["vah"],
                "val": vp["val"],
                "poc": vp["poc"],
                "mode": entry_mode,
                "avg_or_volume": avg_volume,
            },
        )

    def _aggression(
        self, bar: pd.Series, avg_volume: float, direction: int, ratio_mult: float = 1.0
    ) -> bool:
        if avg_volume <= 0:
            return False
        body = float(bar["close"] - bar["open"]) * direction
        rng = max(float(bar["high"] - bar["low"]), 1e-9)
        body_ratio = body / rng
        return (
            body_ratio >= self.min_body_ratio
            and float(bar["volume"]) >= avg_volume * self.min_volume_ratio * ratio_mult
        )

    @staticmethod
    def _volume_profile(
        bars: pd.DataFrame,
        bins: int = 24,
        value_area_pct: float = 0.70,
    ) -> dict[str, float]:
        lo = float(bars["low"].min())
        hi = float(bars["high"].max())
        if hi <= lo:
            return {"poc": hi, "vah": hi, "val": lo}

        step = max(1e-6, (hi - lo) / bins)
        edges = np.arange(lo, hi + step, step)
        if len(edges) < 2:
            edges = np.array([lo, lo + 1e-6])
        hist = np.zeros(len(edges) - 1)

        for _, bar in bars.iterrows():
            bar_lo = float(bar["low"])
            bar_hi = float(bar["high"])
            vol = max(float(bar["volume"]), 0.0)
            touched = np.where((edges[:-1] <= bar_hi) & (edges[1:] >= bar_lo))[0]
            if len(touched):
                hist[touched] += vol / len(touched)

        centers = (edges[:-1] + edges[1:]) / 2
        if hist.sum() <= 0:
            return {"poc": float(centers[len(centers) // 2]), "vah": hi, "val": lo}

        poc_idx = int(hist.argmax())
        selected = {poc_idx}
        total = hist[poc_idx]
        target = hist.sum() * value_area_pct
        left, right = poc_idx - 1, poc_idx + 1
        while total < target and (left >= 0 or right < len(hist)):
            lv = hist[left] if left >= 0 else -1.0
            rv = hist[right] if right < len(hist) else -1.0
            if rv >= lv:
                selected.add(right)
                total += max(rv, 0)
                right += 1
            else:
                selected.add(left)
                total += max(lv, 0)
                left -= 1

        return {
            "poc": float(centers[poc_idx]),
            "vah": float(edges[max(selected) + 1]),
            "val": float(edges[min(selected)]),
        }
