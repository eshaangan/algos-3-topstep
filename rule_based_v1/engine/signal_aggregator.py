"""Signal Aggregator - Combines rule signals into trade decisions.

Mode: "primary_plus_filters"
1. Primary rule decides direction.
2. Filter rules can veto (if vetoed=True in metadata).
3. Confirmation rules boost confidence.
4. Trade executes if: primary signals AND no filter vetoes AND >= 1 confirmation.
"""

from dataclasses import dataclass, field
import logging

import pandas as pd

from rules.base import BaseRule, RuleSignal

logger = logging.getLogger(__name__)


@dataclass
class TradeDecision:
    """Final trade decision after aggregating all rule signals."""
    should_trade: bool
    direction: int          # +1 LONG, -1 SHORT, 0 NO_TRADE
    confidence: float       # Combined confidence score
    primary_signal: RuleSignal | None
    filter_signals: list[RuleSignal] = field(default_factory=list)
    confirmation_signals: list[RuleSignal] = field(default_factory=list)
    veto_reasons: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if not self.should_trade:
            if self.veto_reasons:
                reasons = "; ".join(self.veto_reasons)
            elif self.primary_signal and self.primary_signal.reason:
                reasons = f"No primary signal ({self.primary_signal.reason})"
            else:
                reasons = "No primary signal"
            return f"NO TRADE: {reasons}"
        side = "LONG" if self.direction == 1 else "SHORT"
        confirms = [s.rule_name for s in self.confirmation_signals if s.has_signal]
        return f"{side} (conf={self.confidence:.2f}, confirmations={confirms})"


class SignalAggregator:
    def __init__(
        self,
        primary_rule: BaseRule,
        filter_rules: list[BaseRule] | None = None,
        confirmation_rules: list[BaseRule] | None = None,
        min_confirmations: int = 1,
    ):
        if primary_rule.role != "primary":
            raise ValueError(f"Expected primary rule, got {primary_rule.role}")
        self.primary_rule = primary_rule
        self.filter_rules = filter_rules or []
        self.confirmation_rules = confirmation_rules or []
        self.min_confirmations = min_confirmations

        for r in self.filter_rules:
            if r.role != "filter":
                raise ValueError(f"Expected filter rule '{r.name}', got {r.role}")
        for r in self.confirmation_rules:
            if r.role != "confirmation":
                raise ValueError(f"Expected confirmation rule '{r.name}', got {r.role}")

    def required_bars(self) -> int:
        """Maximum required bars across all rules."""
        all_rules = [self.primary_rule] + self.filter_rules + self.confirmation_rules
        return max(r.required_bars() for r in all_rules)

    def evaluate(self, bars: pd.DataFrame) -> TradeDecision:
        """Evaluate all rules and produce a trade decision."""
        # Step 1: Primary rule
        primary_signal = self.primary_rule.evaluate(bars)
        if not primary_signal.has_signal:
            return TradeDecision(
                should_trade=False, direction=0, confidence=0.0,
                primary_signal=primary_signal,
            )

        direction = primary_signal.direction

        # Step 2: Filter rules (any veto blocks the trade)
        filter_signals = []
        veto_reasons = []
        for rule in self.filter_rules:
            signal = rule.evaluate(bars)
            filter_signals.append(signal)
            if signal.metadata.get("vetoed", False):
                veto_reasons.append(f"{rule.name}: {signal.reason}")

        if veto_reasons:
            return TradeDecision(
                should_trade=False, direction=0, confidence=0.0,
                primary_signal=primary_signal,
                filter_signals=filter_signals,
                veto_reasons=veto_reasons,
            )

        # Step 3: Confirmation rules
        confirmation_signals = []
        confirming_count = 0
        for rule in self.confirmation_rules:
            signal = rule.evaluate(bars)
            confirmation_signals.append(signal)
            # Confirmation agrees if its direction matches the primary direction
            if signal.has_signal and signal.direction == direction:
                confirming_count += 1

        if confirming_count < self.min_confirmations:
            return TradeDecision(
                should_trade=False, direction=0, confidence=0.0,
                primary_signal=primary_signal,
                filter_signals=filter_signals,
                confirmation_signals=confirmation_signals,
                veto_reasons=[
                    f"Insufficient confirmations: {confirming_count}/{self.min_confirmations}"
                ],
            )

        # Step 4: Compute combined confidence
        # Base from primary, boosted by confirmations
        confidence = primary_signal.strength
        for signal in confirmation_signals:
            if signal.has_signal and signal.direction == direction:
                confidence = min(1.0, confidence + signal.strength * 0.2)

        return TradeDecision(
            should_trade=True,
            direction=direction,
            confidence=confidence,
            primary_signal=primary_signal,
            filter_signals=filter_signals,
            confirmation_signals=confirmation_signals,
        )
