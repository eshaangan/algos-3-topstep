"""Cumulative Delta Filter — FILTER rule for ORB quality gating.

Computes a 4-point order-flow quality score for an ORB breakout using
only OHLCV data. Vetoes entries whose score falls below the threshold.

Academic basis:
  - Glosten & Milgrom (1985): informed vs. uninformed order flow determines
    price discovery; volume distribution within a bar reveals trade direction.
  - Chordia & Subrahmanyam (2004): order imbalance strongly predicts short-term
    returns even after controlling for price momentum.
  - Lopez de Prado, AFML (2018, Ch. 19): tick/volume imbalance bars; the OHLCV
    cumulative delta proxy is the time-bar analogue of tick imbalance.

CDP proxy formula per bar:
    cdp_bar = volume × (2 × (close − low) / (high − low + ε) − 1)

Maps each bar to the interval [−volume, +volume]:
  +volume  →  pure buy bar  (close = high, all aggressive buyers)
  −volume  →  pure sell bar (close = low,  all aggressive sellers)

The normalized cumulative delta over the OR window (cdp_ratio ∈ [−1, +1])
reveals whether smart money was accumulating (bullish) or distributing
(bearish) before the breakout.

Quality score components (0–4):
  1. CDP Direction      : cdp_ratio sign matches breakout direction
  2. OR Positional Bias : time-at-top vs time-at-bottom supports direction
                          (simplified Market Profile TPO concept)
  3. Breakout Volume    : breakout bar volume > breakout_vol_ratio × avg OR vol
  4. Breakout Momentum  : excess size above/below range > min_excess_atr × ATR

v2 additions:
  cdp_required     : if True (default), CDP direction must pass or trade is vetoed
                     immediately — this eliminates the "institutional trap" pattern
                     (high-volume breakout AGAINST order flow).
  allow_cdp_shorts : if True, enables SHORT entries when CDP is strongly bearish.
                     Disabled by default; requires stricter CDP threshold for SHORTs.
"""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from rules.base import BaseRule, RuleSignal
from utils.indicators import atr as _atr


class CumulativeDeltaFilter(BaseRule):
    """Order-flow quality gate for Opening Range Breakout signals.

    Parameters
    ----------
    or_end_time : str
        OR window end time, "HH:MM" Eastern. Must match the primary ORB rule.
        Default "10:04".
    min_quality_score : int
        Minimum total score (0–4) required to pass. Only used when
        cdp_required=False. When cdp_required=True, use min_other_score instead.
        Default 3.
    min_cdp_ratio : float
        Minimum |cdp_ratio| (normalised −1..+1) to award CDP direction point.
        Also the hard-gate threshold when cdp_required=True for LONG entries.
        Default 0.15.
    breakout_vol_ratio : float
        Breakout bar volume must exceed (breakout_vol_ratio × avg OR volume)
        to award the volume surge point. Default 1.25.
    min_breakout_excess_atr : float
        Minimum (breakout_close − range_high) / ATR to award the momentum
        point. Ensures the break is decisive, not a tick-above-range fake.
        Default 0.3.
    atr_period : int
        ATR lookback period. Default 14.
    cdp_required : bool
        If True, CDP direction must pass or the trade is immediately vetoed.
        This eliminates "Type B" traps: high-volume breakouts AGAINST order
        flow. Trades where CDP agrees but other signals fail are still allowed
        if remaining score ≥ min_other_score. Default True.
    min_other_score : int
        When cdp_required=True, minimum score from {Bias, Volume, Momentum}
        (0–3) required after CDP passes. Default 1 (CDP + 1 confirmation).
    allow_cdp_shorts : bool
        If True, allows SHORT breakouts when CDP is strongly bearish. Must be
        explicitly opted-in. Default False.
    min_short_cdp_ratio : float
        Minimum |cdp_ratio| for SHORT entries. Stricter than LONG because
        NQ/MNQ has upward drift bias. Default 0.30 (vs LONG threshold 0.15).
    """

    def __init__(
        self,
        or_end_time: str = "10:04",
        min_quality_score: int = 3,
        min_cdp_ratio: float = 0.15,
        breakout_vol_ratio: float = 1.25,
        min_breakout_excess_atr: float = 0.3,
        atr_period: int = 14,
        # v2 parameters
        cdp_required: bool = True,
        min_other_score: int = 1,
        allow_cdp_shorts: bool = False,
        min_short_cdp_ratio: float = 0.30,
    ):
        super().__init__(name="cumulative_delta_filter", role="filter")
        self.or_end_time = or_end_time
        self.min_quality_score = min_quality_score
        self.min_cdp_ratio = min_cdp_ratio
        self.breakout_vol_ratio = breakout_vol_ratio
        self.min_breakout_excess_atr = min_breakout_excess_atr
        self.atr_period = atr_period
        self.cdp_required = cdp_required
        self.min_other_score = min_other_score
        self.allow_cdp_shorts = allow_cdp_shorts
        self.min_short_cdp_ratio = min_short_cdp_ratio

        h, m = map(int, or_end_time.split(":"))
        self._or_end = time(h, m)

    def required_bars(self) -> int:
        # Match the primary ORB rule's lookback so SignalAggregator uses the same window.
        return max(90, self.atr_period * 2)

    def evaluate(self, bars: pd.DataFrame) -> RuleSignal:  # noqa: C901
        if len(bars) < self.required_bars():
            return RuleSignal.no_signal(self.name, "Insufficient bars")

        # ── Current timestamp in Eastern ─────────────────────────────────────
        current_ts = bars.index[-1]
        if current_ts.tzinfo is not None:
            current_ts_et = current_ts.tz_convert("US/Eastern")
        else:
            current_ts_et = pd.Timestamp(current_ts).tz_localize("US/Eastern")

        current_date = current_ts_et.date()

        # ── Slice opening range bars ──────────────────────────────────────────
        session_open = pd.Timestamp(f"{current_date} 09:30:00", tz="US/Eastern")
        or_close_ts = pd.Timestamp(
            f"{current_date} {self.or_end_time}:00", tz="US/Eastern"
        )

        or_mask = (bars.index >= session_open) & (bars.index <= or_close_ts)
        or_bars = bars.loc[or_mask]

        if len(or_bars) < 3:
            # Too few OR bars — pass through without vetoing; primary will reject.
            return RuleSignal.no_signal(
                self.name, f"Insufficient OR bars: {len(or_bars)} (need ≥3)"
            )

        # ── Cumulative Delta Proxy ────────────────────────────────────────────
        epsilon = 1e-10
        bar_range = (or_bars["high"] - or_bars["low"]).clip(lower=epsilon)
        cdp_per_bar = or_bars["volume"] * (
            2.0 * (or_bars["close"] - or_bars["low"]) / bar_range - 1.0
        )
        cdp_total = float(cdp_per_bar.sum())
        total_or_volume = float(or_bars["volume"].sum())
        cdp_ratio = cdp_total / total_or_volume if total_or_volume > 0 else 0.0

        # ── OR positional bias (simplified TPO) ──────────────────────────────
        range_high = float(or_bars["high"].max())
        range_low = float(or_bars["low"].min())
        or_mid = (range_high + range_low) / 2.0
        upper_bars = int((or_bars["close"] > or_mid).sum())
        lower_bars = int((or_bars["close"] < or_mid).sum())

        # ── Current bar properties ────────────────────────────────────────────
        current_close = float(bars["close"].iloc[-1])
        current_volume = float(bars["volume"].iloc[-1])
        avg_or_volume = float(or_bars["volume"].mean()) if len(or_bars) > 0 else 1.0

        # ── ATR ───────────────────────────────────────────────────────────────
        atr_series = _atr(bars["high"], bars["low"], bars["close"], self.atr_period)
        atr_val = float(atr_series.iloc[-1])
        if np.isnan(atr_val) or atr_val <= 0:
            return RuleSignal.no_signal(self.name, "Invalid ATR")

        # ── Detect breakout direction ─────────────────────────────────────────
        is_long_break = current_close > range_high
        is_short_break = current_close < range_low

        if not is_long_break and not is_short_break:
            return RuleSignal.no_signal(self.name, "No breakout detected by filter")

        direction = 1 if is_long_break else -1

        # ── SHORT gate (opt-in) ───────────────────────────────────────────────
        if direction == -1 and not self.allow_cdp_shorts:
            return RuleSignal(
                direction=0,
                strength=0.0,
                rule_name=self.name,
                reason="SHORT entry blocked: allow_cdp_shorts=False",
                metadata={"vetoed": True, "veto_reason": "shorts_disabled",
                          "cdp_ratio": round(cdp_ratio, 4)},
            )

        # ── 4-point quality score ─────────────────────────────────────────────
        score_details: dict[str, bool] = {}

        if direction == 1:  # LONG breakout
            cdp_ok = cdp_ratio > self.min_cdp_ratio
            bias_ok = upper_bars >= lower_bars
            vol_ok = current_volume > avg_or_volume * self.breakout_vol_ratio
            excess = (current_close - range_high) / atr_val
            momentum_ok = excess > self.min_breakout_excess_atr
        else:  # SHORT breakout — stricter CDP threshold
            cdp_ok = cdp_ratio < -self.min_short_cdp_ratio
            bias_ok = lower_bars >= upper_bars
            vol_ok = current_volume > avg_or_volume * self.breakout_vol_ratio
            excess = (range_low - current_close) / atr_val
            momentum_ok = excess > self.min_breakout_excess_atr

        for label, flag in [
            ("cdp", cdp_ok),
            ("bias", bias_ok),
            ("volume", vol_ok),
            ("momentum", momentum_ok),
        ]:
            score_details[label] = flag

        score = sum(int(v) for v in score_details.values())

        # ── Common metadata ───────────────────────────────────────────────────
        meta = {
            "quality_score": score,
            "cdp_ratio": round(cdp_ratio, 4),
            "upper_bars": upper_bars,
            "lower_bars": lower_bars,
            "avg_or_volume": round(avg_or_volume, 1),
            "breakout_volume": round(current_volume, 1),
            "breakout_excess_atr": round(excess, 3),
            "score_details": score_details,
        }

        # ── CDP-anchored gating (v2) ──────────────────────────────────────────
        if self.cdp_required:
            if not cdp_ok:
                return RuleSignal(
                    direction=0,
                    strength=0.0,
                    rule_name=self.name,
                    reason=(
                        f"CDP required but failed: cdp_ratio={cdp_ratio:.3f} "
                        f"(threshold={'>' if direction==1 else '<'}"
                        f"{self.min_cdp_ratio if direction==1 else -self.min_short_cdp_ratio:.2f})"
                    ),
                    metadata={"vetoed": True, "veto_reason": "cdp_required", **meta},
                )
            other_score = int(bias_ok) + int(vol_ok) + int(momentum_ok)
            if other_score < self.min_other_score:
                return RuleSignal(
                    direction=0,
                    strength=0.0,
                    rule_name=self.name,
                    reason=(
                        f"CDP passed but other signals insufficient: "
                        f"other_score={other_score} < {self.min_other_score}"
                    ),
                    metadata={"vetoed": True, "veto_reason": "other_score_low", **meta},
                )
            return RuleSignal(
                direction=direction,
                strength=min(1.0, (1 + other_score) / 4.0),
                rule_name=self.name,
                reason=(
                    f"CDP-anchored pass: cdp_ratio={cdp_ratio:.3f}, "
                    f"other_score={other_score}/3, {score_details}"
                ),
                metadata={"vetoed": False, **meta},
            )

        # ── Legacy score-threshold gating (v1, cdp_required=False) ───────────
        if score < self.min_quality_score:
            return RuleSignal(
                direction=0,
                strength=0.0,
                rule_name=self.name,
                reason=(
                    f"CD quality score {score}/{self.min_quality_score} — "
                    f"vetoed ({score_details})"
                ),
                metadata={"vetoed": True, "veto_reason": "min_score", **meta},
            )

        return RuleSignal(
            direction=direction,
            strength=score / 4.0,
            rule_name=self.name,
            reason=(
                f"CD quality score {score}/4 passed — "
                f"cdp_ratio={cdp_ratio:.3f}, {score_details}"
            ),
            metadata={"vetoed": False, **meta},
        )
