"""Deadline-aware feasibility audit for event-strategy books.

The ordinary Monte Carlo answers whether an account eventually passes within a
chosen number of event periods. This module answers the stricter question used
for strategy promotion:

    What fraction of complete historical start windows pass within N calendar
    days, while obeying the trailing max-loss floor and consistency rule?

It intentionally uses contiguous calendar windows rather than iid trade
resampling. That preserves clustered losses, irregular event frequency, and
periods with no trade.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from rule_based_v1.diagnostics.portfolio_mc import (
    AccountRules,
    Policy,
    load_book,
    run_path,
)


def constant_micros(micros: int, *, base_micros: int = 2) -> Policy:
    """Return a constant-size policy for a PnL series quoted per ``base_micros``."""
    if micros <= 0:
        raise ValueError("micros must be positive")
    if base_micros <= 0:
        raise ValueError("base_micros must be positive")
    scale = micros / base_micros
    return Policy(
        name=f"constant_{micros}_micros",
        low_scale=scale,
        high_scale=scale,
        rung_balance=0.0,
    )


def _normalise_book(book: pd.DataFrame) -> pd.DataFrame:
    missing = {"day", "pnl"}.difference(book.columns)
    if missing:
        raise ValueError(f"book missing columns {sorted(missing)}")
    out = book[["day", "pnl"]].copy()
    out["day"] = pd.to_datetime(out["day"], errors="raise").dt.tz_localize(None)
    out["pnl"] = pd.to_numeric(out["pnl"], errors="raise")
    return out.groupby("day", as_index=False)["pnl"].sum().sort_values("day").reset_index(drop=True)


def historical_deadline_windows(
    book: pd.DataFrame,
    *,
    policy: Policy,
    deadline_days: int = 28,
    rules: AccountRules = AccountRules(),
) -> pd.DataFrame:
    """Evaluate every complete contiguous calendar window in ``book``.

    A start is included only when the entire deadline interval is observable.
    Events on ``start + deadline_days`` are excluded, making 28 days exactly four
    weeks rather than five weekly observations.
    """
    if deadline_days <= 0:
        raise ValueError("deadline_days must be positive")
    b = _normalise_book(book)
    columns = [
        "window_start",
        "window_end_exclusive",
        "status",
        "events_seen",
        "ending_balance",
    ]
    if b.empty:
        return pd.DataFrame(columns=columns)

    last_observed = b["day"].iloc[-1]
    rows: list[dict] = []
    for start in b["day"]:
        end = start + pd.Timedelta(days=deadline_days)
        if end > last_observed:
            continue
        pnl = b.loc[(b["day"] >= start) & (b["day"] < end), "pnl"].to_numpy(float)
        status, events_seen, ending_balance = run_path(pnl, policy=policy, rules=rules)
        rows.append(
            {
                "window_start": start,
                "window_end_exclusive": end,
                "status": status,
                "events_seen": events_seen,
                "ending_balance": ending_balance,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def audit_size_grid(
    book: pd.DataFrame,
    *,
    micros: Iterable[int],
    deadline_days: int = 28,
    base_micros: int = 2,
    target_pass: float = 0.70,
    max_bust: float = 0.30,
    rules: AccountRules = AccountRules(),
) -> pd.DataFrame:
    """Audit constant position sizes and mark policies meeting both constraints."""
    rows: list[dict] = []
    for size in micros:
        policy = constant_micros(int(size), base_micros=base_micros)
        windows = historical_deadline_windows(
            book,
            policy=policy,
            deadline_days=deadline_days,
            rules=rules,
        )
        n = len(windows)
        if n == 0:
            p_pass = p_bust = p_timeout = float("nan")
        else:
            p_pass = float((windows["status"] == "pass").mean())
            p_bust = float((windows["status"] == "bust").mean())
            p_timeout = float((windows["status"] == "timeout").mean())
        rows.append(
            {
                "micros": int(size),
                "policy": policy.name,
                "n_windows": n,
                "p_pass_by_deadline": p_pass,
                "p_bust_by_deadline": p_bust,
                "p_timeout_by_deadline": p_timeout,
                "meets_target": bool(
                    n > 0
                    and p_pass >= target_pass
                    and p_bust <= max_bust
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_grid(grid: pd.DataFrame, *, max_bust: float) -> dict:
    """Return the best observed policy overall and under the bust constraint."""
    valid = grid.dropna(subset=["p_pass_by_deadline"]).copy()
    if valid.empty:
        return {"best_overall": None, "best_under_bust_limit": None}
    order = ["p_pass_by_deadline", "p_bust_by_deadline", "micros"]
    best = valid.sort_values(order, ascending=[False, True, True]).iloc[0]
    safe = valid[valid["p_bust_by_deadline"] <= max_bust]
    best_safe = (
        safe.sort_values(order, ascending=[False, True, True]).iloc[0]
        if not safe.empty
        else None
    )

    def record(row: pd.Series | None) -> dict | None:
        if row is None:
            return None
        return {
            "micros": int(row["micros"]),
            "n_windows": int(row["n_windows"]),
            "p_pass_by_deadline": float(row["p_pass_by_deadline"]),
            "p_bust_by_deadline": float(row["p_bust_by_deadline"]),
            "p_timeout_by_deadline": float(row["p_timeout_by_deadline"]),
            "meets_target": bool(row["meets_target"]),
        }

    return {
        "best_overall": record(best),
        "best_under_bust_limit": record(best_safe),
    }


def run_deadline_audit(
    book: pd.DataFrame,
    *,
    deadline_days: int,
    min_micros: int,
    max_micros: int,
    base_micros: int,
    target_pass: float,
    max_bust: float,
    split_date: str | None = None,
    rules: AccountRules = AccountRules(),
) -> dict:
    """Run full-sample and optional time-split size-grid audits."""
    if min_micros <= 0 or max_micros < min_micros:
        raise ValueError("invalid micros range")
    b = _normalise_book(book)
    segments: dict[str, pd.DataFrame] = {"full": b}
    if split_date:
        split = pd.Timestamp(split_date)
        segments["development"] = b[b["day"] < split]
        segments["evaluation"] = b[b["day"] >= split]

    result = {
        "deadline_days": deadline_days,
        "target_pass": target_pass,
        "max_bust": max_bust,
        "base_micros": base_micros,
        "rules": asdict(rules),
        "segments": {},
    }
    sizes = range(min_micros, max_micros + 1)
    for name, segment in segments.items():
        grid = audit_size_grid(
            segment,
            micros=sizes,
            deadline_days=deadline_days,
            base_micros=base_micros,
            target_pass=target_pass,
            max_bust=max_bust,
            rules=rules,
        )
        result["segments"][name] = {
            "n_events": int(len(segment)),
            "start": str(segment["day"].min().date()) if len(segment) else None,
            "end": str(segment["day"].max().date()) if len(segment) else None,
            "summary": summarize_grid(grid, max_bust=max_bust),
            "grid": grid.to_dict(orient="records"),
        }
    return result


def print_report(result: dict) -> None:
    print(
        f"deadline={result['deadline_days']} calendar days | "
        f"required P(pass)>={result['target_pass']:.0%} | "
        f"P(bust)<={result['max_bust']:.0%}"
    )
    for name, segment in result["segments"].items():
        print(f"\n[{name}] events={segment['n_events']} {segment['start']}..{segment['end']}")
        summary = segment["summary"]
        for label, key in (
            ("best overall", "best_overall"),
            ("best within bust limit", "best_under_bust_limit"),
        ):
            row = summary[key]
            if row is None:
                print(f"  {label}: no complete windows")
                continue
            print(
                f"  {label}: {row['micros']} micros | "
                f"P(pass)={row['p_pass_by_deadline']:.1%} | "
                f"P(bust)={row['p_bust_by_deadline']:.1%} | "
                f"windows={row['n_windows']} | meets_target={row['meets_target']}"
            )


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    default = f"WK={root / 'data/processed/weekend_hold_strict.csv'}:1.0"
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", action="append", default=None, help="NAME=CSV[:SCALE]")
    ap.add_argument("--deadline-days", type=int, default=28)
    ap.add_argument("--min-micros", type=int, default=1)
    ap.add_argument("--max-micros", type=int, default=20)
    ap.add_argument("--base-micros", type=int, default=2)
    ap.add_argument("--target-pass", type=float, default=0.70)
    ap.add_argument("--max-bust", type=float, default=0.30)
    ap.add_argument("--split-date", default="2025-01-01")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    book = load_book(args.strategy or [default])
    result = run_deadline_audit(
        book,
        deadline_days=args.deadline_days,
        min_micros=args.min_micros,
        max_micros=args.max_micros,
        base_micros=args.base_micros,
        target_pass=args.target_pass,
        max_bust=args.max_bust,
        split_date=args.split_date,
    )
    print_report(result)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
