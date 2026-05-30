"""OFI Filter Rule — Order Flow Imbalance gate using Level 2 DOM.

Role: FILTER (veto trades when book imbalance opposes the signal direction).

Requires a DOMListener instance to provide real-time OFI values.
If the listener is disconnected, pass_through=True allows the trade anyway
so a DOM outage doesn't halt trading.

OFI range: -1.0 (full sell pressure) to +1.0 (full buy pressure).
  LONG gate:  require OFI > ofi_long_threshold  (default 0.2)
  SHORT gate: require OFI < -ofi_short_threshold (default 0.2)
"""

from __future__ import annotations

import logging

import pandas as pd

from rules.base import BaseRule, RuleSignal

logger = logging.getLogger(__name__)


class OFIFilterRule(BaseRule):
    """Veto trades when Level 2 book imbalance contradicts direction."""

    def __init__(
        self,
        dom_listener,
        ofi_long_threshold: float = 0.2,
        ofi_short_threshold: float = 0.2,
        pass_through_if_disconnected: bool = True,
    ):
        super().__init__(name="ofi_filter", role="filter")
        self.dom_listener = dom_listener
        self.ofi_long_threshold  = ofi_long_threshold
        self.ofi_short_threshold = ofi_short_threshold
        self.pass_through_if_disconnected = pass_through_if_disconnected

    def required_bars(self) -> int:
        return 1

    def evaluate(self, bars: pd.DataFrame) -> RuleSignal:
        if self.dom_listener is None:
            return RuleSignal(
                direction=0, strength=0.0, rule_name=self.name,
                reason="No DOM listener", metadata={"vetoed": False},
            )

        if not self.dom_listener.is_connected:
            if self.pass_through_if_disconnected:
                logger.debug("OFI filter: DOM disconnected — pass-through")
                return RuleSignal(
                    direction=0, strength=0.0, rule_name=self.name,
                    reason="DOM disconnected (pass-through)",
                    metadata={"vetoed": False},
                )
            else:
                return RuleSignal(
                    direction=0, strength=0.0, rule_name=self.name,
                    reason="DOM disconnected — vetoing trade",
                    metadata={"vetoed": True},
                )

        ofi = self.dom_listener.get_current_ofi()

        # The filter evaluates against the LAST bar's close to infer direction context.
        # Actual direction gating happens via the vetoed flag checked by SignalAggregator
        # AFTER the primary rule has set a direction. We veto based on OFI.

        # Since filters in SignalAggregator don't receive the primary direction,
        # we store OFI and let the runner read it directly for directional gating.
        # For the base FILTER interface: only veto if OFI is strongly opposing.

        # Conservative: only veto when OFI is clearly against both directions
        # (i.e., use in runner to check direction-specific threshold).
        logger.debug(f"OFI filter: OFI={ofi:+.3f}")

        return RuleSignal(
            direction=0, strength=0.0, rule_name=self.name,
            reason=f"OFI={ofi:+.3f}",
            metadata={"vetoed": False, "ofi": ofi},
        )

    def check_direction(self, direction: int) -> tuple[bool, str]:
        """Direction-specific OFI check for use in runner.

        Args:
            direction: +1 for LONG, -1 for SHORT.

        Returns:
            (approved, reason) — approved=True means OFI confirms direction.
        """
        if self.dom_listener is None or not self.dom_listener.is_connected:
            if self.pass_through_if_disconnected:
                return True, "DOM disconnected (pass-through)"
            return False, "DOM disconnected"

        ofi = self.dom_listener.get_current_ofi()

        if direction == 1:
            if ofi >= self.ofi_long_threshold:
                return True, f"OFI={ofi:+.3f} >= threshold {self.ofi_long_threshold:+.2f}"
            return False, f"OFI={ofi:+.3f} < long threshold {self.ofi_long_threshold:+.2f}"

        if direction == -1:
            if ofi <= -self.ofi_short_threshold:
                return True, f"OFI={ofi:+.3f} <= threshold {-self.ofi_short_threshold:+.2f}"
            return False, f"OFI={ofi:+.3f} > short threshold {-self.ofi_short_threshold:+.2f}"

        return True, "No direction"
