"""Unit tests for CumulativeDeltaFilter."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cumulative_delta_filter import CumulativeDeltaFilter


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_session_bars(
    n: int = 90,
    base_price: float = 5000.0,
    volume: float = 200.0,
    or_close_as_fraction: float | None = None,
    breakout_close: float | None = None,
    breakout_volume: float | None = None,
    zero_range_bar_idx: int | None = None,
) -> pd.DataFrame:
    """Build n 5-min RTH bars starting at 09:30 ET on 2025-01-02.

    Parameters
    ----------
    n : int
        Total bars (≥ 90 for filter to fire; default 90 ends at 16:55).
    base_price : float
        Price base used for OR bars.
    volume : float
        Default volume for all bars.
    or_close_as_fraction : float or None
        If given, sets close = low + fraction*(high-low) for all 7 OR bars.
        0.0 = pure sell (close=low), 1.0 = pure buy (close=high), 0.5 = neutral.
    breakout_close : float or None
        If given, overrides the close of the last bar (the simulated breakout bar).
    breakout_volume : float or None
        If given, overrides the volume of the last bar.
    zero_range_bar_idx : int or None
        If given, sets high == low == base_price on that bar index (no-range bar).
    """
    start = "2025-01-02 09:30:00"
    times = pd.date_range(start, periods=n, freq="5min", tz="US/Eastern")

    # Flat bars by default
    opens   = np.full(n, base_price)
    highs   = np.full(n, base_price + 10.0)
    lows    = np.full(n, base_price - 10.0)
    closes  = np.full(n, base_price)
    volumes = np.full(n, volume)

    # OR bars are those at 09:30–10:00 (7 bars: indices 0-6 on 5-min)
    if or_close_as_fraction is not None:
        for i in range(7):
            lows[i] = base_price - 10.0
            highs[i] = base_price + 10.0
            closes[i] = lows[i] + or_close_as_fraction * (highs[i] - lows[i])

    # Override breakout bar (last bar)
    if breakout_close is not None:
        closes[-1] = breakout_close
    if breakout_volume is not None:
        volumes[-1] = breakout_volume

    # Zero-range bar (no spread)
    if zero_range_bar_idx is not None:
        highs[zero_range_bar_idx] = base_price
        lows[zero_range_bar_idx] = base_price
        closes[zero_range_bar_idx] = base_price

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=times,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestCumulativeDeltaFilter:

    def test_required_bars_unchanged(self):
        """Filter must not increase the required lookback beyond the ORB primary (90)."""
        f = CumulativeDeltaFilter()
        assert f.required_bars() == 90, (
            "required_bars() should return 90 to match OpeningRangeBreakoutRule"
        )

    def test_pure_buy_bars_cdp_passes_long(self):
        """All OR bars close=high (pure buy) → cdp_ratio=+1.0 → LONG break scores max."""
        # range_high = 5010.0, range_low = 4990.0
        # Last bar breaks out long: close = 5020.0 (excess = 10/ATR)
        bars = _make_session_bars(
            or_close_as_fraction=1.0,   # close = high = 5010 for all OR bars
            breakout_close=5020.0,      # strong LONG break
            breakout_volume=500.0,      # 2.5× avg OR volume of 200 → vol surge ✓
        )
        f = CumulativeDeltaFilter(min_quality_score=3)
        sig = f.evaluate(bars)

        assert sig.metadata.get("vetoed") is False, (
            f"Expected pass but got vetoed. Details: {sig.reason}"
        )
        assert sig.metadata["quality_score"] == 4, (
            f"Expected quality 4 (pure buy OR + strong break) but got {sig.metadata['quality_score']}"
        )
        assert sig.direction == 1

    def test_pure_sell_bars_veto_long_breakout(self):
        """All OR bars close=low (pure sell) → LONG breakout should be vetoed at Q≥3."""
        # cdp_ratio = -1.0 → CDP direction FAILS for LONG
        # all OR bars close=low → upper_bars=0, lower_bars=7 → bias FAILS for LONG
        # Score at most 2 (volume + momentum may pass) → vetoed at Q≥3
        bars = _make_session_bars(
            or_close_as_fraction=0.0,   # close = low = 4990 for all OR bars
            breakout_close=5020.0,
            breakout_volume=500.0,
        )
        f = CumulativeDeltaFilter(min_quality_score=3)
        sig = f.evaluate(bars)

        assert sig.metadata.get("vetoed") is True, (
            f"Expected veto (pure-sell OR + LONG break) but got: {sig.reason}"
        )
        assert sig.metadata["quality_score"] <= 2, (
            f"Expected score ≤ 2 but got {sig.metadata['quality_score']}"
        )

    def test_neutral_cdp_does_not_award_point(self):
        """Near-zero cdp_ratio (<0.15) should NOT award the CDP direction point."""
        # close = mid of each OR bar → cdp_ratio ≈ 0.0
        # Breakout strong LONG but CDP point missing
        bars = _make_session_bars(
            or_close_as_fraction=0.5,   # close = midpoint → cdp = 0.0
            breakout_close=5020.0,
            breakout_volume=500.0,
        )
        f = CumulativeDeltaFilter(min_cdp_ratio=0.15)
        sig = f.evaluate(bars)

        cdp_ratio = sig.metadata.get("cdp_ratio", 0.0)
        assert abs(cdp_ratio) < 0.15, (
            f"Expected near-zero cdp_ratio but got {cdp_ratio}"
        )
        cdp_awarded = sig.metadata.get("score_details", {}).get("cdp", True)
        assert cdp_awarded is False, (
            f"CDP direction point should NOT be awarded for cdp_ratio={cdp_ratio}"
        )

    def test_zero_range_bar_no_crash(self):
        """A zero-range OR bar (high == low) must not raise ZeroDivisionError."""
        bars = _make_session_bars(
            or_close_as_fraction=1.0,
            breakout_close=5020.0,
            zero_range_bar_idx=2,       # bar 2 has high == low
        )
        f = CumulativeDeltaFilter()
        # Must not raise
        try:
            sig = f.evaluate(bars)
        except ZeroDivisionError:
            pytest.fail("ZeroDivisionError raised on zero-range OR bar")
        # cdp_ratio must be finite
        assert np.isfinite(sig.metadata.get("cdp_ratio", float("nan"))), (
            "cdp_ratio should be finite even with zero-range bars"
        )

    def test_or_bars_missing_returns_no_signal_not_veto(self):
        """If bars contain no OR window data (<3 bars), return no_signal, NOT a veto.

        This lets the primary ORB rule reject the trade independently.
        """
        # Build bars starting at 11:00 — no 09:30–10:04 bars present
        start = "2025-01-02 11:00:00"
        times = pd.date_range(start, periods=90, freq="5min", tz="US/Eastern")
        bars = pd.DataFrame(
            {
                "open": np.full(90, 5000.0),
                "high": np.full(90, 5010.0),
                "low": np.full(90, 4990.0),
                "close": np.full(90, 5020.0),
                "volume": np.full(90, 200.0),
            },
            index=times,
        )
        f = CumulativeDeltaFilter()
        sig = f.evaluate(bars)

        assert sig.direction == 0
        assert sig.metadata.get("vetoed") is not True, (
            "Filter must NOT veto when OR bars are unavailable — "
            "let the primary rule decide."
        )

    def test_insufficient_total_bars_returns_no_signal(self):
        """Fewer than required_bars() total bars → no_signal (not a crash)."""
        bars = _make_session_bars(n=10)
        f = CumulativeDeltaFilter()
        sig = f.evaluate(bars)

        assert sig.direction == 0
        assert "Insufficient bars" in sig.reason


class TestCumulativeDeltaFilterV2:
    """Tests for v2 additions: cdp_required anchor and allow_cdp_shorts."""

    def test_cdp_required_vetoes_when_cdp_fails(self):
        """cdp_required=True: veto immediately when CDP direction does not pass."""
        # OR bars close = midpoint → cdp_ratio ≈ 0.0 < 0.15 threshold → CDP fails
        # Even with high volume and strong momentum on breakout bar, must be vetoed
        bars = _make_session_bars(
            or_close_as_fraction=0.5,   # cdp_ratio ≈ 0.0
            breakout_close=5025.0,      # strong break + high excess
            breakout_volume=600.0,      # 3× avg OR volume
        )
        f = CumulativeDeltaFilter(cdp_required=True, min_cdp_ratio=0.15)
        sig = f.evaluate(bars)

        assert sig.metadata.get("vetoed") is True, (
            "cdp_required=True: neutral-CDP breakout must be vetoed"
        )
        assert sig.metadata.get("veto_reason") == "cdp_required", (
            f"Expected veto_reason='cdp_required', got: {sig.metadata.get('veto_reason')}"
        )

    def test_cdp_required_passes_cdp_plus_one_other(self):
        """cdp_required=True + min_other_score=1: CDP passes + 1 other → trade allowed."""
        # Pure buy OR → cdp_ratio = +1.0 (CDP passes)
        # High volume on breakout (volume passes), but momentum and bias may vary
        bars = _make_session_bars(
            or_close_as_fraction=1.0,   # cdp_ratio = +1.0, all bars close=high
            breakout_close=5015.0,      # modest break (excess ≈ 0.25, below 0.3 threshold)
            breakout_volume=500.0,      # volume surge passes (2.5× avg)
        )
        f = CumulativeDeltaFilter(
            cdp_required=True,
            min_other_score=1,
            min_cdp_ratio=0.15,
            min_breakout_excess_atr=0.3,
        )
        sig = f.evaluate(bars)

        # CDP passes (cdp_ratio=1.0 > 0.15) + at least volume passes → should not veto
        # (bias also passes since all OR bars close=high > or_mid)
        assert sig.metadata.get("vetoed") is False, (
            f"CDP + 1 other should pass with min_other_score=1. Got: {sig.reason}"
        )
        assert sig.direction == 1

    def test_cdp_required_vetoes_when_other_score_too_low(self):
        """cdp_required=True + min_other_score=2: CDP passes but other_score=1 → veto."""
        # Pure buy OR → cdp_ratio = +1.0 (CDP passes)
        # Low breakout volume (fails), modest excess (fails), bias only (1 other)
        bars = _make_session_bars(
            or_close_as_fraction=1.0,   # cdp_ratio = +1.0
            breakout_close=5010.5,      # tiny excess, will be well below 0.3×ATR
            breakout_volume=100.0,      # below avg_or_vol=200 → volume fails
        )
        f = CumulativeDeltaFilter(
            cdp_required=True,
            min_other_score=2,          # stricter: need CDP + 2 others
            min_cdp_ratio=0.15,
        )
        sig = f.evaluate(bars)

        assert sig.metadata.get("vetoed") is True, (
            "Should veto when other_score < min_other_score=2"
        )
        assert sig.metadata.get("veto_reason") == "other_score_low"

    def test_short_disabled_by_default(self):
        """allow_cdp_shorts=False (default): SHORT breakout is always vetoed."""
        # Pure sell OR → strongly bearish CDP; price breaks DOWN
        # Without allow_cdp_shorts, this must be blocked
        bars = _make_session_bars(
            or_close_as_fraction=0.0,   # cdp_ratio = -1.0 (strong sell)
            breakout_close=4975.0,      # SHORT break (below range_low=4990)
            breakout_volume=500.0,
        )
        f = CumulativeDeltaFilter(allow_cdp_shorts=False)
        sig = f.evaluate(bars)

        assert sig.metadata.get("vetoed") is True, (
            "SHORT must be vetoed when allow_cdp_shorts=False"
        )
        assert sig.metadata.get("veto_reason") == "shorts_disabled"

    def test_short_enabled_with_strict_threshold(self):
        """allow_cdp_shorts=True + cdp_ratio below strict threshold → SHORT passes."""
        # cdp_ratio = -1.0 (all sell bars) → must be < -0.30 for SHORT to pass
        # Strong SHORT break with volume
        bars = _make_session_bars(
            or_close_as_fraction=0.0,   # cdp_ratio = -1.0
            breakout_close=4975.0,      # SHORT break (25 points below range_low=4990)
            breakout_volume=500.0,      # volume surge
        )
        f = CumulativeDeltaFilter(
            allow_cdp_shorts=True,
            min_short_cdp_ratio=0.30,   # |cdp_ratio| must be > 0.30 for SHORT
            cdp_required=True,
            min_other_score=1,
        )
        sig = f.evaluate(bars)

        # cdp_ratio = -1.0 → passes strict SHORT threshold (< -0.30)
        # bias: all bars close=low < or_mid → lower_bars=7, upper_bars=0 → bias_ok ✓
        # volume: 500 > 200×1.25=250 → volume_ok ✓
        assert sig.metadata.get("vetoed") is False, (
            f"Strongly bearish CDP SHORT should pass. Got: {sig.reason}"
        )
        assert sig.direction == -1

    def test_short_rejected_when_cdp_not_strong_enough(self):
        """allow_cdp_shorts=True but cdp_ratio too weak → SHORT vetoed at CDP gate."""
        # cdp_ratio ≈ -0.10 (slightly bearish but below -0.30 threshold)
        bars = _make_session_bars(
            or_close_as_fraction=0.45,  # close slightly below mid → mild bearish CDP
            breakout_close=4975.0,      # SHORT break
            breakout_volume=500.0,
        )
        f = CumulativeDeltaFilter(
            allow_cdp_shorts=True,
            min_short_cdp_ratio=0.30,
            cdp_required=True,
        )
        sig = f.evaluate(bars)

        assert sig.metadata.get("vetoed") is True
        assert sig.metadata.get("veto_reason") == "cdp_required", (
            f"Weak SHORT CDP should fail cdp_required gate. Got: {sig.metadata}"
        )
