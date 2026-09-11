"""Several LucidFlex accounts in different phases, tracked together.

The edge is a positive-expectancy lottery ticket: most attempts return nothing and
the mean only appears across many of them. So the unit of management is the
portfolio of accounts, not any single account.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from .account import LucidAccount, Phase, RuleViolation
from .rules import INACTIVITY_DAYS


class Portfolio:
    def __init__(self, accounts: list[LucidAccount] | None = None,
                 fees_paid: float = 0.0):
        self.accounts = accounts or []
        self.fees_paid = float(fees_paid)

    # -------------------------------------------------------------- basics --
    def add(self, account: LucidAccount, fee: float = 0.0) -> LucidAccount:
        if any(a.account_id == account.account_id for a in self.accounts):
            raise ValueError(f"duplicate account_id {account.account_id!r}")
        self.accounts.append(account)
        self.fees_paid += float(fee)
        return account

    def get(self, account_id: str) -> LucidAccount:
        for a in self.accounts:
            if a.account_id == account_id:
                return a
        raise KeyError(account_id)

    @property
    def cash_received(self) -> float:
        return sum(a.cash_received for a in self.accounts)

    @property
    def net(self) -> float:
        return self.cash_received - self.fees_paid

    # -------------------------------------------------------------- alerts --
    def alerts(self, today: date | None = None) -> list[str]:
        """Everything that needs a decision, worst first."""
        today = today or date.today()
        out: list[str] = []
        for a in self.accounts:
            if not a.active:
                continue
            adv = a.payout_advice()
            if adv.eligible and adv.recommendation == "request":
                out.append(f"[{a.account_id}] REQUEST PAYOUT: "
                           f"${adv.amount:,.0f} gross, ${adv.cash_to_trader:,.0f} "
                           f"to you, costs no buffer")
            elif adv.eligible and adv.recommendation == "wait":
                out.append(f"[{a.account_id}] HOLD: eligible for ${adv.amount:,.0f} "
                           f"but requesting now destroys ${adv.mll_jump_cost:,.0f} "
                           f"of buffer. Wait for ${a.spec.trail_balance:,.0f}.")
            if a.buffer < 0.25 * a.spec.mll_amount:
                out.append(f"[{a.account_id}] THIN BUFFER: ${a.buffer:,.0f} left "
                           f"above the MLL of ${a.mll:,.0f}")
            left = a.days_until_deletion(today)
            if left is not None and left <= 7:
                out.append(f"[{a.account_id}] INACTIVITY: {left} day(s) before "
                           f"deletion; needs a session with non-zero P&L")
            if a.payout_pending:
                out.append(f"[{a.account_id}] PAYOUT PENDING: do not trade until "
                           "it settles or the request may be denied")
        return out

    # -------------------------------------------------------------- display --
    def status_table(self) -> str:
        h = (f"{'account':>12} {'size':>8} {'phase':>8} {'balance':>11} {'MLL':>10} "
             f"{'buffer':>9} {'lock':>5} {'qual':>5} {'payouts':>8} {'cash':>9}")
        rows = [h, "-" * len(h)]
        for a in self.accounts:
            rows.append(
                f"{a.account_id:>12} {a.size:>8,} {a.phase.value:>8} "
                f"{a.balance:>11,.0f} {a.mll:>10,.0f} {a.buffer:>9,.0f} "
                f"{('yes' if a.mll_locked else 'no'):>5} "
                f"{(f'{a.qualifying_days}/5' if a.phase is Phase.FUNDED else '-'):>5} "
                f"{a.payouts_taken:>8} {a.cash_received:>9,.0f}")
        rows.append("-" * len(h))
        rows.append(f"{'TOTAL':>12} {'':>8} {'':>8} {'':>11} {'':>10} {'':>9} "
                    f"{'':>5} {'':>5} {'':>8} {self.cash_received:>9,.0f}")
        rows.append(f"fees paid ${self.fees_paid:,.0f}   net ${self.net:,.0f}")
        return "\n".join(rows)

    # ---------------------------------------------------------- persistence --
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"fees_paid": self.fees_paid,
             "accounts": [a.to_dict() for a in self.accounts]}, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Portfolio":
        d = json.loads(Path(path).read_text())
        return cls([LucidAccount.from_dict(a) for a in d["accounts"]],
                   d.get("fees_paid", 0.0))
