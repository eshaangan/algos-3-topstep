"""Time of Day Filter - FILTER rule.

Vetoes trades outside allowed session times.
Optional lunch filter vetoes during low-liquidity midday period.
"""

import pandas as pd

from rules.base import BaseRule, RuleSignal


class TimeOfDayRule(BaseRule):
    def __init__(
        self,
        session_start: str = "09:35",
        session_end: str = "15:45",
        lunch_filter_enabled: bool = False,
        lunch_start: str = "12:00",
        lunch_end: str = "13:00",
    ):
        super().__init__(name="time_of_day", role="filter")
        self.session_start = pd.Timestamp(session_start).time()
        self.session_end = pd.Timestamp(session_end).time()
        self.lunch_filter_enabled = lunch_filter_enabled
        self.lunch_start = pd.Timestamp(lunch_start).time()
        self.lunch_end = pd.Timestamp(lunch_end).time()

    def required_bars(self) -> int:
        return 1

    def evaluate(self, bars: pd.DataFrame) -> RuleSignal:
        if len(bars) < 1:
            return RuleSignal.no_signal(self.name, "No bars")

        current_time = bars.index[-1]

        # Convert to Eastern if needed
        if current_time.tzinfo is not None:
            current_time = current_time.tz_convert("US/Eastern")

        t = current_time.time()

        # Check session bounds
        if t < self.session_start or t > self.session_end:
            return RuleSignal(
                direction=0,
                strength=0.0,
                rule_name=self.name,
                reason=f"Outside session: {t} not in [{self.session_start}, {self.session_end}]",
                metadata={"vetoed": True, "current_time": str(t)},
            )

        # Check lunch filter
        if self.lunch_filter_enabled:
            if self.lunch_start <= t <= self.lunch_end:
                return RuleSignal(
                    direction=0,
                    strength=0.0,
                    rule_name=self.name,
                    reason=f"Lunch filter: {t} in [{self.lunch_start}, {self.lunch_end}]",
                    metadata={"vetoed": True, "current_time": str(t)},
                )

        # Pass - no veto
        return RuleSignal(
            direction=0,
            strength=0.0,
            rule_name=self.name,
            reason=f"Session active: {t}",
            metadata={"vetoed": False, "current_time": str(t)},
        )
