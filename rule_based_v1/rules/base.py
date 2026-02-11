"""Base classes for rule-based trading signals."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class RuleSignal:
    """Signal produced by a trading rule.

    Attributes:
        direction: +1 for LONG, -1 for SHORT, 0 for NO_SIGNAL.
        strength: Confidence level from 0.0 to 1.0.
        rule_name: Name of the rule that produced this signal.
        reason: Human-readable explanation.
        metadata: Rule-specific data (e.g., indicator values).
    """
    direction: int
    strength: float
    rule_name: str
    reason: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.direction not in (-1, 0, 1):
            raise ValueError(f"direction must be -1, 0, or 1, got {self.direction}")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be in [0, 1], got {self.strength}")

    @property
    def is_long(self) -> bool:
        return self.direction == 1

    @property
    def is_short(self) -> bool:
        return self.direction == -1

    @property
    def has_signal(self) -> bool:
        return self.direction != 0

    @staticmethod
    def no_signal(rule_name: str, reason: str = "No signal") -> "RuleSignal":
        return RuleSignal(direction=0, strength=0.0, rule_name=rule_name, reason=reason)


class BaseRule(ABC):
    """Abstract base class for all trading rules.

    Rules fall into three categories:
    - PRIMARY: Determines trade direction (only one active at a time).
    - FILTER: Can veto a trade (returns direction=0 to pass, or vetoes).
    - CONFIRMATION: Boosts confidence if conditions align.
    """

    def __init__(self, name: str, role: str):
        """
        Args:
            name: Rule identifier.
            role: One of "primary", "filter", "confirmation".
        """
        if role not in ("primary", "filter", "confirmation"):
            raise ValueError(f"role must be primary/filter/confirmation, got {role}")
        self.name = name
        self.role = role

    @abstractmethod
    def evaluate(self, bars: pd.DataFrame) -> RuleSignal:
        """Evaluate the rule on the given bar history.

        Args:
            bars: DataFrame with columns [open, high, low, close, volume]
                  and DatetimeIndex. Most recent bar is last row.

        Returns:
            RuleSignal with the rule's assessment.
        """
        ...

    @abstractmethod
    def required_bars(self) -> int:
        """Minimum number of bars needed for this rule to produce a valid signal."""
        ...
