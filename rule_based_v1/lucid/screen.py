"""Prop-firm barrier screen.

Everything in the LucidFlex work collapses to one closed form. For an account with
max drawdown D, profit target T and profit split s, bought for fee F:

    P(pass)      = D / (T + D)                 optional stopping, any strategy
    E[extract]   = s * D                       the gifted allowance, never repaid
    EV/attempt   = s * D^2 / (T + D)  -  F

Validation: Lucid 100k gives 0.9 * 3000^2 / 9000 = $900, less $158 = $742. A full
contiguous-window simulation of the same account returned $755. Within 2%.

The closed form is an UPPER BOUND. Three mechanics only ever subtract from it:
  trailing drawdown   the floor ratchets up as you profit  (measured: ~-25% on P(pass))
  intraday breach     no free end-of-day forgiveness       (measured: EV -> ~0 or negative)
  payout friction     minimum qualifying days, caps, buffers
So the screen ranks by the bound, then the flags say how much of it survives.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Calibrated on MNQ, 1,685 contiguous sessions, Lucid 100k, 60 micros.
# The closed form assumes a DRIFTLESS fair game, so the honest comparison is the
# beta-stripped simulation: gross $452 against a $900 bound => 0.50.
# (With 2020-2026 long beta left in, the same cell returned $913, i.e. ~1.01 of the
# bound. Long equity drift historically paid back the entire trailing penalty. That
# is a bet on the tape, not a property of the account, so the screen does not use it.)
TRAILING_HAIRCUT = 0.50
BETA_ADDBACK = 2.02          # measured ratio, beta-on / beta-off. Informational only.
INTRADAY_HAIRCUT = 0.05     # EV collapses to ~zero when live equity is checked


@dataclass
class Account:
    firm: str
    plan: str
    size: int
    target: float
    drawdown: float
    fee: float                 # one-time or first month, after any standard code
    split: float = 0.90
    trailing: bool = True      # False = static/fixed drawdown (strictly better)
    intraday_breach: bool | None = None   # None = UNVERIFIED, the key unknown
    monthly: bool = False      # True = recurring subscription, not one-time
    max_micros: int = 0
    payout_note: str = ""
    source: str = ""
    verified: bool = False     # True only if read on the firm's own site/help centre

    # ------------------------------------------------------------------ math --
    @property
    def p_pass(self) -> float:
        return self.drawdown / (self.target + self.drawdown)

    @property
    def gross(self) -> float:
        """s * D^2 / (T + D): the closed-form upper bound before fees."""
        return self.split * self.drawdown ** 2 / (self.target + self.drawdown)

    @property
    def ev_bound(self) -> float:
        return self.gross - self.fee

    @property
    def ev_expected(self) -> float:
        """The bound with the mechanics applied. This is the number to rank on."""
        g = self.gross
        if self.trailing:
            g *= TRAILING_HAIRCUT
        if self.intraday_breach is True:
            g *= INTRADAY_HAIRCUT
        return g - self.fee

    @property
    def ev_per_dollar(self) -> float:
        """Capital efficiency: expected return per dollar of fee laid out."""
        return self.ev_expected / self.fee if self.fee else float("nan")

    @property
    def flag(self) -> str:
        if not self.verified:
            return "UNVERIFIED"
        if self.intraday_breach is None:
            return "breach rule unknown"
        return "static" if not self.trailing else "trailing"


def rank(accounts: list[Account], by: str = "ev_expected") -> list[Account]:
    return sorted(accounts, key=lambda a: getattr(a, by), reverse=True)


def table(accounts: list[Account], by: str = "ev_expected") -> str:
    h = (f"{'firm':>22} {'plan':>16} {'size':>7} {'T':>7} {'D':>6} {'fee':>6} "
         f"{'P(pass)':>8} {'bound':>7} {'EV':>7} {'EV/$':>6}  flag")
    out = [h, "-" * len(h)]
    for a in rank(accounts, by):
        out.append(
            f"{a.firm:>22} {a.plan:>16} {a.size:>7,} {a.target:>7,.0f} "
            f"{a.drawdown:>6,.0f} {a.fee:>6,.0f} {a.p_pass:>8.1%} "
            f"{a.gross:>7,.0f} {a.ev_expected:>7,.0f} {a.ev_per_dollar:>6.2f}  {a.flag}")
    return "\n".join(out)
