"""Frozen LucidFlex rule constants, transcribed from the Lucid help centre.

Sources, all read 2026-09-10 (the help centre 403s without a browser user-agent):
  evaluation  support.lucidtrading.com/en/articles/12945790-lucidflex-evaluation-account
  funded      support.lucidtrading.com/en/articles/12945795-lucidflex-funded-account
  drawdown    support.lucidtrading.com/en/articles/12945815-lucidflex-drawdown
  payouts     support.lucidtrading.com/en/articles/12945796-lucidflex-payouts
  activities  support.lucidtrading.com/en/articles/11404728-other-trading-activities
  inactivity  support.lucidtrading.com/en/articles/11404632-inactivity-policy

Nothing in here is inferred. If a value is not in one of those pages it is not here.
"""
from __future__ import annotations

from dataclasses import dataclass

PROFIT_SPLIT = 0.90          # 90/10 trader/Lucid
MIN_PAYOUT = 500.0
PAYOUT_FRACTION = 0.50       # a request may be 50% of profit, up to the cap
QUALIFYING_DAYS_REQUIRED = 5
MAX_PAYOUTS = 5              # then the trader is moved to a live account
EVAL_MIN_TRADING_DAYS = 2
EVAL_CONSISTENCY = 0.50      # no single day may exceed 50% of total profit
INACTIVITY_DAYS = 30         # no net P&L within 30 calendar days => deleted


@dataclass(frozen=True)
class AccountSpec:
    """One row of the LucidFlex account table."""

    size: int
    profit_target: float
    mll_amount: float          # the Max Loss Limit, e.g. $3,000 on the 100k
    trail_balance: float       # balance above which the MLL stops trailing
    locked_mll: float          # where it stops: initial balance + $100
    max_minis: int
    max_micros: int
    min_daily_profit: float    # a "qualifying day" for payout eligibility
    payout_cap: float

    @property
    def initial_mll(self) -> float:
        return self.size - self.mll_amount


SPECS: dict[int, AccountSpec] = {
    25_000: AccountSpec(25_000, 1_250.0, 1_000.0, 26_100.0, 25_100.0,
                        2, 20, 100.0, 1_000.0),
    50_000: AccountSpec(50_000, 3_000.0, 2_000.0, 52_100.0, 50_100.0,
                        4, 40, 150.0, 2_000.0),
    100_000: AccountSpec(100_000, 6_000.0, 3_000.0, 103_100.0, 100_100.0,
                         6, 60, 200.0, 2_500.0),
    150_000: AccountSpec(150_000, 9_000.0, 4_500.0, 154_600.0, 150_100.0,
                         10, 100, 250.0, 3_000.0),
}


def spec(size: int) -> AccountSpec:
    if size not in SPECS:
        raise KeyError(f"no LucidFlex account of size {size:,}; "
                       f"have {sorted(SPECS)}")
    return SPECS[size]
