"""LucidFlex account state machine.

The strategy this serves is three lines long: at the session open go to the size
cap, hold, flatten before the close. All of the money is lost or kept in the
bookkeeping, and one rule dominates everything:

    "Once you request a payout from LucidFlex, your MLL automatically adjusts to
     the Locked MLL Balance."   -- lucidflex-drawdown

A payout requested while the Max Loss Limit is still trailing throws the floor up
to the locked level immediately. On a 100k that can destroy $2,900 of buffer, which
is more than the entire expected value of the evaluation ticket. `payout_advice()`
exists to stop that from ever happening by accident.

The breach test here is on the CLOSING balance, which is what the drawdown page
describes. Whether Lucid also breaches on live intraday equity is NOT settled by
their documentation; `intraday_status()` reports both readings so a live probe can
tell them apart.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from enum import Enum
from pathlib import Path

from .rules import (EVAL_CONSISTENCY, EVAL_MIN_TRADING_DAYS, INACTIVITY_DAYS,
                    MAX_PAYOUTS, MIN_PAYOUT, PAYOUT_FRACTION, PROFIT_SPLIT,
                    QUALIFYING_DAYS_REQUIRED, AccountSpec, spec)


class Phase(str, Enum):
    EVAL = "eval"
    FUNDED = "funded"
    BREACHED = "breached"
    LIVE = "live"          # five payouts taken, moved to a live account


class RuleViolation(Exception):
    """An action the Lucid rules forbid, raised before it can cost money."""


@dataclass
class SessionResult:
    day: date
    pnl: float
    balance: float
    mll: float
    buffer: float
    phase: Phase
    qualifying_day: bool
    note: str = ""


@dataclass
class PayoutAdvice:
    eligible: bool
    amount: float                 # what could be requested right now
    cash_to_trader: float         # after the 90/10 split
    buffer_now: float
    buffer_after: float
    mll_jump_cost: float          # buffer destroyed purely by the MLL relocking
    recommendation: str           # "request" | "wait" | "ineligible"
    reasons: list[str] = field(default_factory=list)


@dataclass
class LucidAccount:
    """One LucidFlex account, advanced one closing balance at a time."""

    account_id: str
    size: int = 100_000
    phase: Phase = Phase.EVAL
    balance: float = 0.0
    peak_balance: float = 0.0          # highest CLOSING balance ever
    mll_locked: bool = False
    qualifying_days: int = 0           # >= min_daily_profit, this payout cycle
    payouts_taken: int = 0
    cash_received: float = 0.0
    cycle_start_balance: float = 0.0
    trading_days: int = 0
    max_day_profit: float = 0.0        # for the evaluation consistency rule
    payout_pending: bool = False
    last_trade_day: date | None = None
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.balance:
            self.balance = float(self.size)
        if not self.peak_balance:
            self.peak_balance = float(self.size)
        if not self.cycle_start_balance:
            self.cycle_start_balance = float(self.size)

    # ---------------------------------------------------------------- state --
    @property
    def spec(self) -> AccountSpec:
        return spec(self.size)

    @property
    def mll(self) -> float:
        """Max Loss Limit. Trails the highest closing balance, then locks."""
        s = self.spec
        if self.mll_locked:
            return s.locked_mll
        return min(self.peak_balance - s.mll_amount, s.locked_mll)

    @property
    def buffer(self) -> float:
        """Dollars of closing-balance loss available before a breach."""
        return self.balance - self.mll

    @property
    def profit(self) -> float:
        return self.balance - self.size

    @property
    def cycle_profit(self) -> float:
        return self.balance - self.cycle_start_balance

    @property
    def active(self) -> bool:
        return self.phase in (Phase.EVAL, Phase.FUNDED)

    def max_size(self, unit: str = "micros") -> int:
        s = self.spec
        return s.max_micros if unit == "micros" else s.max_minis

    def days_until_deletion(self, today: date) -> int | None:
        """Inactivity policy: an untraded account is deleted after 30 days."""
        if self.last_trade_day is None or not self.active:
            return None
        return INACTIVITY_DAYS - (today - self.last_trade_day).days

    # ------------------------------------------------------------- advance ---
    def close_session(self, day: date, pnl: float) -> SessionResult:
        """Apply one session's realised P&L at the mandatory 16:45 ET flatten."""
        if not self.active:
            raise RuleViolation(f"{self.account_id} is {self.phase.value}; "
                                "it cannot trade")
        if self.payout_pending:
            raise RuleViolation(
                f"{self.account_id} has a payout request pending. Lucid may deny a "
                "request if a trade drops the balance below the required amount. "
                "Settle the payout before trading again.")
        if self.last_trade_day is not None and day <= self.last_trade_day:
            raise RuleViolation(f"session {day} is not after the last recorded "
                                f"session {self.last_trade_day}")

        s = self.spec
        self.balance += pnl
        if pnl:
            self.trading_days += 1
            self.last_trade_day = day
        qualifying = self.phase is Phase.FUNDED and pnl >= s.min_daily_profit
        if qualifying:
            self.qualifying_days += 1
        if pnl > self.max_day_profit:
            self.max_day_profit = pnl

        # the MLL trails the highest CLOSING balance, and locks past the trail
        self.peak_balance = max(self.peak_balance, self.balance)
        if self.peak_balance >= s.trail_balance:
            self.mll_locked = True

        note = ""
        if self.balance <= self.mll:
            self.phase = Phase.BREACHED
            note = f"BREACHED: closing balance {self.balance:,.2f} <= MLL {self.mll:,.2f}"
        elif self.phase is Phase.EVAL and self.profit >= s.profit_target:
            ok_days = self.trading_days >= EVAL_MIN_TRADING_DAYS
            ok_cons = self.max_day_profit <= EVAL_CONSISTENCY * self.profit
            if ok_days and ok_cons:
                # The funded account is a NEW account. It opens at the account
                # size with the MLL back at its initial, still-trailing level --
                # it does NOT inherit the evaluation's ending balance. This is
                # what puts the funded phase below the trail balance, which is
                # the only place the payout trap can bite.
                note = (f"PASSED evaluation at {self.balance:,.2f}; funded account "
                        f"opens fresh at {self.size:,}")
                self.phase = Phase.FUNDED
                self.balance = float(self.size)
                self.peak_balance = float(self.size)
                self.cycle_start_balance = float(self.size)
                self.mll_locked = False
                self.qualifying_days = 0
                self.trading_days = 0
                self.max_day_profit = 0.0
            elif not ok_days:
                note = (f"target reached but only {self.trading_days} trading day(s); "
                        f"{EVAL_MIN_TRADING_DAYS} required")
            else:
                need = self.max_day_profit / EVAL_CONSISTENCY
                note = (f"target reached but consistency fails: best day "
                        f"{self.max_day_profit:,.0f} > 50% of profit "
                        f"{self.profit:,.0f}; need profit >= {need:,.0f}")

        res = SessionResult(day, pnl, self.balance, self.mll, self.buffer,
                            self.phase, qualifying, note)
        self.history.append({**asdict(res), "day": day.isoformat(),
                             "phase": self.phase.value})
        return res

    # -------------------------------------------------------------- payout ---
    def payout_advice(self) -> PayoutAdvice:
        """Whether to request a payout now, and what requesting now would cost.

        The trap: while the MLL is still trailing it sits at
        `peak - mll_amount`, which may be far below the locked level. Requesting
        a payout relocks it upward on the spot. Waiting until the balance clears
        the trail balance makes that jump free, because the MLL has locked anyway.
        """
        s = self.spec
        reasons: list[str] = []
        if self.phase is not Phase.FUNDED:
            return PayoutAdvice(False, 0.0, 0.0, self.buffer, self.buffer, 0.0,
                                "ineligible", [f"account is {self.phase.value}"])

        amount = min(PAYOUT_FRACTION * self.profit, s.payout_cap)
        eligible = True
        if self.qualifying_days < QUALIFYING_DAYS_REQUIRED:
            eligible = False
            reasons.append(
                f"{self.qualifying_days}/{QUALIFYING_DAYS_REQUIRED} qualifying days "
                f"(need a session of >= ${s.min_daily_profit:,.0f})")
        if self.cycle_profit <= 0:
            eligible = False
            reasons.append(f"cycle net profit is {self.cycle_profit:,.2f}; "
                           "must be positive")
        if amount < MIN_PAYOUT:
            eligible = False
            reasons.append(f"requestable ${amount:,.2f} is below the "
                           f"${MIN_PAYOUT:,.0f} minimum; need profit >= "
                           f"${MIN_PAYOUT / PAYOUT_FRACTION:,.0f}")

        jump = 0.0 if self.mll_locked else max(0.0, s.locked_mll - self.mll)
        buffer_after = self.balance - amount - (s.locked_mll if not self.mll_locked
                                                else self.mll)
        if not eligible:
            rec = "ineligible"
        elif jump > 0:
            rec = "wait"
            reasons.append(
                f"requesting now relocks the MLL from {self.mll:,.0f} to "
                f"{s.locked_mll:,.0f} and destroys ${jump:,.0f} of buffer. "
                f"Wait until the balance clears ${s.trail_balance:,.0f}, where the "
                "MLL locks anyway and the request costs nothing.")
        else:
            rec = "request"
            reasons.append("MLL is already locked; requesting costs no buffer")

        return PayoutAdvice(eligible, amount, PROFIT_SPLIT * amount, self.buffer,
                            buffer_after, jump, rec, reasons)

    def request_payout(self, amount: float | None = None,
                       override_wait: bool = False) -> float:
        """Request a payout. Returns cash to the trader after the 90/10 split."""
        adv = self.payout_advice()
        if not adv.eligible:
            raise RuleViolation(f"{self.account_id} cannot request a payout: "
                                + "; ".join(adv.reasons))
        if adv.recommendation == "wait" and not override_wait:
            raise RuleViolation(
                f"{self.account_id}: refusing to request. " + " ".join(adv.reasons)
                + " Pass override_wait=True to do it anyway.")
        amount = adv.amount if amount is None else float(amount)
        if not (MIN_PAYOUT <= amount <= adv.amount):
            raise RuleViolation(f"payout must be between ${MIN_PAYOUT:,.0f} and "
                                f"${adv.amount:,.2f}; got ${amount:,.2f}")

        self.balance -= amount
        self.mll_locked = True                 # the rule that costs people money
        cash = PROFIT_SPLIT * amount
        self.cash_received += cash
        self.payouts_taken += 1
        self.qualifying_days = 0
        self.cycle_start_balance = self.balance
        if self.payouts_taken >= MAX_PAYOUTS:
            self.phase = Phase.LIVE
        return cash

    # ------------------------------------------------------------ intraday ---
    def intraday_status(self, open_equity: float) -> dict:
        """Both readings of the breach rule, for the live probe that settles it.

        Lucid documents the MLL as computed from closing balances but never says
        whether a breach is evaluated on live equity too. Drive an account below
        `mll` intraday and close above it: survival proves the closing-balance
        reading and makes the whole approach viable; an instant breach kills it.
        """
        return {
            "open_equity": open_equity,
            "mll": self.mll,
            "below_mll_now": open_equity <= self.mll,
            "breaches_if_intraday_rule": open_equity <= self.mll,
            "breaches_if_closing_rule": False,   # only the close can breach
            "closing_balance_needed": self.mll + 0.01,
        }

    # ---------------------------------------------------------- persistence --
    def to_dict(self) -> dict:
        d = asdict(self)
        d["phase"] = self.phase.value
        d["last_trade_day"] = (self.last_trade_day.isoformat()
                               if self.last_trade_day else None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LucidAccount":
        d = dict(d)
        d["phase"] = Phase(d["phase"])
        if d.get("last_trade_day"):
            d["last_trade_day"] = date.fromisoformat(d["last_trade_day"])
        return cls(**d)
