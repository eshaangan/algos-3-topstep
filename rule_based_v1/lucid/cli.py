"""Command line front end: python -m lucid.cli --state ~/.lucid/state.json <cmd>"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from .account import LucidAccount, RuleViolation
from .portfolio import Portfolio

DEFAULT_STATE = Path.home() / ".lucid" / "state.json"


def _load(p: Path) -> Portfolio:
    return Portfolio.load(p) if p.exists() else Portfolio()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", type=Path, default=DEFAULT_STATE)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("open", help="register a newly purchased evaluation")
    a.add_argument("account_id"); a.add_argument("--size", type=int, default=100_000)
    a.add_argument("--fee", type=float, default=0.0)

    c = sub.add_parser("close", help="record one session's realised P&L")
    c.add_argument("account_id"); c.add_argument("pnl", type=float)
    c.add_argument("--day", type=date.fromisoformat, default=date.today())

    p = sub.add_parser("payout", help="check or request a payout")
    p.add_argument("account_id"); p.add_argument("--request", action="store_true")
    p.add_argument("--amount", type=float, default=None)
    p.add_argument("--override-wait", action="store_true")

    i = sub.add_parser("intraday", help="both readings of the breach rule")
    i.add_argument("account_id"); i.add_argument("equity", type=float)

    sub.add_parser("status", help="portfolio table and alerts")

    ns = ap.parse_args(argv)
    pf = _load(ns.state)

    try:
        if ns.cmd == "open":
            pf.add(LucidAccount(ns.account_id, ns.size), fee=ns.fee)
            print(f"opened {ns.account_id} ({ns.size:,}), fee ${ns.fee:,.0f}")
        elif ns.cmd == "close":
            r = pf.get(ns.account_id).close_session(ns.day, ns.pnl)
            print(f"{r.day} pnl {r.pnl:+,.2f} -> balance {r.balance:,.2f} "
                  f"MLL {r.mll:,.2f} buffer {r.buffer:,.2f} [{r.phase.value}]"
                  + (f"\n  {r.note}" if r.note else ""))
        elif ns.cmd == "payout":
            acct = pf.get(ns.account_id)
            adv = acct.payout_advice()
            print(f"{adv.recommendation.upper()}: ${adv.amount:,.2f} gross, "
                  f"${adv.cash_to_trader:,.2f} to you")
            for r in adv.reasons:
                print(f"  - {r}")
            if ns.request:
                cash = acct.request_payout(ns.amount, ns.override_wait)
                print(f"requested. ${cash:,.2f} to you; "
                      f"balance {acct.balance:,.2f}, MLL {acct.mll:,.2f}")
        elif ns.cmd == "intraday":
            for k, v in pf.get(ns.account_id).intraday_status(ns.equity).items():
                print(f"  {k:>28}: {v}")
        elif ns.cmd == "status":
            print(pf.status_table())
            al = pf.alerts()
            print("\nalerts:" if al else "\nno alerts")
            for x in al:
                print(f"  * {x}")
    except (RuleViolation, KeyError, ValueError) as exc:
        print(f"refused: {exc}")
        return 1
    finally:
        pf.save(ns.state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
