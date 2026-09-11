"""Score a prop-firm evaluation as the barrier option it actually is.

    EV per evaluation = P(pass) x E[extraction] - fee

Derivation, and the two places it can go wrong:

P(pass). For a STATIC floor, optional stopping gives exactly D / (T + D) for any
zero-edge strategy: entries, filters and sizing cannot move it. That is the whole
reason this is an arithmetic screen and not a backtest.

E[extraction]. The firm gifts a drawdown allowance D that is never repaid, so by
the same martingale argument expected withdrawals cannot exceed D, times the profit
split. Payout rules (qualifying days, minimums, caps) claw some of that back, which
`extraction_efficiency` carries.

THE TWO MECHANICS THAT DECIDE EVERYTHING, both published on every firm's site:

  drawdown type   STATIC floors keep the formula exact. A TRAILING floor ratchets
                  upward as you profit, shrinking the room you have left.
  breach basis    Evaluated on LIVE equity, or only on the CLOSING balance? A
                  close-only test is a free option on intraday excursions.

Measured on Lucid (MNQ, 1,685 contiguous sessions, 2026-09-10): trailing + close
scored 1.08x the static baseline, trailing + live scored 0.68x. Those two are
MEASURED. The others are marked EXTRAPOLATED and should not be trusted to rank
close calls -- they exist so a plan is never silently omitted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class Drawdown(str, Enum):
    STATIC = "static"            # floor never moves
    TRAILING_EOD = "trail_eod"   # floor follows the closing balance
    TRAILING_LIVE = "trail_live" # floor follows intraday equity, the worst case


class Breach(str, Enum):
    LIVE = "live"                # touching the floor on unrealised P&L ends it
    CLOSE = "close"              # only the closing balance can breach


# (drawdown, breach) -> (pass multiplier on D/(T+D), extraction efficiency, basis)
#
# Extraction efficiency is the fraction of the martingale bound (split x D) that the
# funded account actually pays out. It is NOT a constant: under a live-breach rule the
# funded account is killed by intraday excursions before the payout rules are satisfied,
# which is why the same firm scores 1.18 on a close-only test and 0.29 on a live one.
# Both of those are measured on Lucid at the size that maximises each.
MECHANICS: dict[tuple["Drawdown", "Breach"], tuple[float, float, str]] = {
    (Drawdown.STATIC, Breach.CLOSE):        (1.25, 0.90, "EXTRAPOLATED"),
    (Drawdown.STATIC, Breach.LIVE):         (1.00, 0.75, "PASS EXACT"),
    (Drawdown.TRAILING_EOD, Breach.CLOSE):  (1.08, 1.18, "MEASURED"),
    (Drawdown.TRAILING_EOD, Breach.LIVE):   (0.68, 0.29, "MEASURED"),
    (Drawdown.TRAILING_LIVE, Breach.CLOSE): (0.85, 0.90, "EXTRAPOLATED"),
    (Drawdown.TRAILING_LIVE, Breach.LIVE):  (0.55, 0.20, "EXTRAPOLATED"),
}



@dataclass(frozen=True)
class Plan:
    firm: str
    plan: str
    size: int
    target: float
    drawdown: float
    dd_type: Drawdown
    breach: Breach
    fee: float
    split: float = 0.90
    fee_discounted: float | None = None
    reset_fee: float | None = None
    max_minis: int | None = None
    payout_min_days: int | None = None
    payout_min_daily: float | None = None
    payout_cap: float | None = None
    consistency: float | None = None
    accounts_allowed: int | None = None
    source: str = ""
    verified: date | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        for f in ("target", "drawdown", "fee"):
            if getattr(self, f) <= 0:
                raise ValueError(f"{self.firm}/{self.plan}: {f} must be positive")
        if not 0 < self.split <= 1:
            raise ValueError(f"{self.firm}/{self.plan}: split must be in (0, 1]")

    @property
    def price(self) -> float:
        """What you would actually pay."""
        return self.fee_discounted if self.fee_discounted is not None else self.fee


@dataclass(frozen=True)
class Score:
    plan: Plan
    pass_static: float      # D/(T+D), the geometry alone
    mech_mult: float
    mech_eff: float
    mech_basis: str
    pass_prob: float
    extraction: float
    ev: float
    ev_per_dollar: float     # EV divided by what you pay -- the ranking key
    verdict: str


def score(plan: Plan, efficiency: float | None = None) -> Score:
    """`efficiency` overrides the mechanics-calibrated value; use it for sensitivity."""
    base = plan.drawdown / (plan.target + plan.drawdown)
    mult, eff, basis = MECHANICS[(plan.dd_type, plan.breach)]
    if efficiency is not None:
        if not 0 < efficiency <= 2:
            raise ValueError("efficiency must be in (0, 2]")
        eff = efficiency
    p = min(base * mult, 1.0)
    extraction = plan.split * plan.drawdown * eff
    ev = p * extraction - plan.price
    per = ev / plan.price
    if ev <= 0:
        verdict = "REJECT"
    elif plan.dd_type is Drawdown.STATIC:
        verdict = "SHORTLIST"
    elif plan.breach is Breach.CLOSE:
        verdict = "SHORTLIST"
    else:
        verdict = "MARGINAL"       # trailing AND live -- the combination that killed Lucid
    return Score(plan, base, mult, eff, basis, p, extraction, ev, per, verdict)


def rank(plans, efficiency: float | None = None) -> list[Score]:
    return sorted((score(p, efficiency) for p in plans),
                  key=lambda s: s.ev_per_dollar, reverse=True)


def table(scores) -> str:
    h = (f"{'firm':>16} {'plan':>14} {'size':>7} {'T':>7} {'D':>6} {'dd':>10} "
         f"{'breach':>6} {'fee':>6} {'P(pass)':>8} {'EV':>7} {'EV/$':>6} "
         f"{'basis':>13} {'verdict':>9}")
    out = [h, "-" * len(h)]
    for s in scores:
        p = s.plan
        out.append(
            f"{p.firm[:16]:>16} {p.plan[:14]:>14} {p.size:>7,} {p.target:>7,.0f} "
            f"{p.drawdown:>6,.0f} {p.dd_type.value:>10} {p.breach.value:>6} "
            f"{p.price:>6,.0f} {s.pass_prob:>8.1%} {s.ev:>7,.0f} "
            f"{s.ev_per_dollar:>6.2f} {s.mech_basis:>13} {s.verdict:>9}")
    return "\n".join(out)
