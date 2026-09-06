"""Feed VPS strategy-lab trade CSVs through the pre-registered validation harness.

Default is report-only. Pass --commit-holdout to append holdout verdicts to
holdout_ledger.jsonl (do that once per strategy after reading the report).

Verdict is holdout-only. Full-sample stats are printed separately for context.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rule_based_v1.validation.harness import (  # noqa: E402
    HoldoutLedger,
    PreRegistration,
    aggregate_stats,
    evaluate,
    monthly_breakdown,
)
from rule_based_v1.validation.vps_csv import load_vps_trades, slice_holdout  # noqa: E402

_VALIDATION_DIR = Path(__file__).resolve().parent
_PREREG_DIR = _VALIDATION_DIR / "vps_prereg"
_DEFAULT_BOOKS = ROOT / "data" / "vps_books"

# (csv stem, prereg yaml stem) — strategy_id lives inside each YAML.
BOOKS = (
    ("orr_reversal", "vps_orr_reversal"),
    ("ifvg_nq_fw3", "vps_ifvg_fw3"),
    ("morningrip_strict", "vps_morning_rip"),
)


def _print_aggregate(title: str, trades: pd.DataFrame) -> None:
    agg = aggregate_stats(trades)
    print(
        f"\n{title}: {agg['n_trades']} trades / {agg['n_days']}d | "
        f"WR={agg['win_rate']:.1%} | PnL=${agg['total_pnl']:,.0f} | "
        f"DD=${agg['max_dd']:,.0f} | Sharpe/trade={agg['sharpe_per_trade']:.2f} | "
        f"exits={agg['exits']}"
    )
    mb = monthly_breakdown(trades)
    for _, r in mb.iterrows():
        flag = "+" if r["pnl"] > 0 else "-"
        print(
            f"    {r['month']}: {int(r['n_trades']):>3} tr  WR={r['win_rate']:.1%}  "
            f"PnL=${r['pnl']:>8,.0f}  DD=${r['max_dd']:>8,.0f}  {flag}"
        )


def _print_verdict(result: dict) -> None:
    dsr = result["dsr"].get("dsr")
    print("\n" + "-" * 78)
    if dsr is not None:
        print(f"  Deflated Sharpe Ratio : {dsr:.3f}")
    else:
        print(f"  Deflated Sharpe Ratio : n/a ({result['dsr'].get('reason')})")
    print(f"  VERDICT               : {result['verdict']}")
    if result["failures"]:
        print("  Gate failures:")
        for fl in result["failures"]:
            print(f"    - {fl}")
    print("-" * 78)


def run_book(
    csv_path: Path,
    prereg_path: Path,
    *,
    commit_holdout: bool,
    ledger: HoldoutLedger,
) -> dict:
    prereg = PreRegistration.load(prereg_path)
    trades = load_vps_trades(csv_path)
    hold = slice_holdout(trades, prereg.holdout["start"], prereg.holdout["end"])

    print("=" * 78)
    print(f"VPS GAUNTLET — {prereg.strategy_id}")
    print(f"  csv          : {csv_path.name}")
    print(f"  holdout      : {prereg.holdout['start']} → {prereg.holdout['end']}")
    print(f"  dsr_n_trials : {prereg.dsr_n_trials}")
    print(f"  config hash  : {prereg.content_hash()[:16]}…")
    print("=" * 78)

    _print_aggregate("FULL-SAMPLE (informational — not the verdict)", trades)

    warns = ledger.warnings_for(prereg)
    for w in warns:
        print(f"\nWARNING: {w}")

    _print_aggregate("HOLDOUT (frozen — the decision is made here)", hold)
    result = evaluate(hold, prereg)
    _print_verdict(result)

    if commit_holdout:
        entry = ledger.record(prereg, result)
        print(f"\nRecorded to ledger: {ledger.path}")
        print(
            f"  {entry['timestamp']}  {entry['verdict']}  "
            f"hash={entry['config_hash'][:16]}…"
        )
    else:
        print("\n(report-only; pass --commit-holdout to write the audit ledger)")

    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run VPS strategy-lab CSVs through the validation harness"
    )
    ap.add_argument(
        "--books-dir",
        type=Path,
        default=_DEFAULT_BOOKS,
        help="Directory containing VPS trade CSVs",
    )
    ap.add_argument(
        "--prereg-dir",
        type=Path,
        default=_PREREG_DIR,
        help="Directory containing per-strategy PreRegistration YAMLs",
    )
    ap.add_argument(
        "--commit-holdout",
        action="store_true",
        help="Append each holdout verdict to the audit ledger (do this once).",
    )
    args = ap.parse_args()

    ledger = HoldoutLedger()
    results: list[dict] = []
    any_missing = False

    for csv_stem, prereg_stem in BOOKS:
        csv_path = args.books_dir / f"{csv_stem}.csv"
        prereg_path = args.prereg_dir / f"{prereg_stem}.yaml"
        if not csv_path.exists():
            print(f"MISSING CSV: {csv_path}", file=sys.stderr)
            any_missing = True
            continue
        if not prereg_path.exists():
            print(f"MISSING PREREG: {prereg_path}", file=sys.stderr)
            any_missing = True
            continue
        results.append(
            run_book(
                csv_path,
                prereg_path,
                commit_holdout=args.commit_holdout,
                ledger=ledger,
            )
        )
        print()

    print("=" * 78)
    print("SUMMARY (holdout verdicts)")
    for r in results:
        print(f"  {r['strategy_id']:<22} {r['verdict']}")
    print("=" * 78)

    if any_missing or not results:
        sys.exit(2)
    sys.exit(0 if all(r["verdict"] == "GO" for r in results) else 1)


if __name__ == "__main__":
    main()
