"""Run the deployed ML model through the pre-registered validation gate.

This is the *adapter*: it turns the LightGBM bundle + v7 features into the
strategy-agnostic ``signals`` table (direction, atr) that harness.simulate
consumes, applying the same session window / threshold / min-ATR filters the live
runner uses. The gate logic, DSR, and ledger all live in harness.py.

Usage
-----
    # Report-only (does NOT touch the holdout ledger):
    python rule_based_v1/validation/validate_model.py

    # Commit the holdout evaluation to the audit ledger (do this ONCE):
    python rule_based_v1/validation/validate_model.py --commit-holdout

    # Override model / data / preregister:
    python rule_based_v1/validation/validate_model.py \
        --model rule_based_v1/models/ml_strategy_mnq_v3.pkl \
        --data  data/processed/mnq_microstructure_5min.parquet
"""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_THIS = Path(__file__).resolve()
ROOT = _THIS.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml_intraday_v3" / "scripts"))

from rule_based_v1.validation.harness import (  # noqa: E402
    PreRegistration, HoldoutLedger, simulate, evaluate, monthly_breakdown, aggregate_stats,
)

DEFAULT_MODEL = ROOT / "rule_based_v1" / "models" / "ml_strategy_mnq_v3.pkl"
DEFAULT_DATA = ROOT / "data" / "processed" / "mnq_microstructure_5min.parquet"


def _in_trading_window(ts: pd.Timestamp) -> bool:
    """Mirror of the live runner's session gate (EST-anchored, afternoon RTH)."""
    h_et = (ts.hour - 5) % 24
    if h_et < 11 or h_et > 15 or h_et == 13:
        return False
    if ts.weekday() == 3:  # Thursday excluded
        return False
    return True


def build_signals(model, feature_cols, bars_feat: pd.DataFrame, prereg: PreRegistration) -> pd.DataFrame:
    """Produce the (direction, atr) signal table for the simulator.

    direction is set to the desired entry (1 long; -1 short only if long_only is
    False and threshold inverted) on bars that pass the live filters, else 0.
    """
    conf = float(prereg.sim["conf"])
    min_atr = float(prereg.sim.get("min_atr_pts", 0.0))
    long_only = bool(prereg.sim.get("long_only", True))

    X = bars_feat[feature_cols]
    valid = X.notna().all(axis=1)
    prob = pd.Series(np.nan, index=bars_feat.index)
    if valid.any():
        prob[valid] = model.predict_proba(X[valid].to_numpy(np.float32))[:, 1]

    atr = bars_feat["atr"]
    in_win = pd.Series([_in_trading_window(ts) for ts in bars_feat.index], index=bars_feat.index)
    tradeable = valid & in_win & atr.notna() & (atr >= min_atr)

    direction = pd.Series(0, index=bars_feat.index, dtype=int)
    direction[tradeable & (prob >= conf)] = 1
    if not long_only:
        direction[tradeable & (prob <= (1 - conf))] = -1

    return pd.DataFrame({"direction": direction, "atr": atr})


def _print_monthly(title: str, trades: pd.DataFrame) -> None:
    agg = aggregate_stats(trades)
    print(f"\n{title}: {agg['n_trades']} trades / {agg['n_days']}d | "
          f"WR={agg['win_rate']:.1%} | PnL=${agg['total_pnl']:,.0f} | "
          f"DD=${agg['max_dd']:,.0f} | Sharpe/trade={agg['sharpe_per_trade']:.2f} | "
          f"exits={agg['exits']}")
    mb = monthly_breakdown(trades)
    for _, r in mb.iterrows():
        flag = "✅" if r["pnl"] > 0 else "❌"
        print(f"    {r['month']}: {int(r['n_trades']):>3} tr  WR={r['win_rate']:.1%}  "
              f"PnL=${r['pnl']:>8,.0f}  DD=${r['max_dd']:>8,.0f}  {flag}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-registered validation of the deployed ML model")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--preregister", type=Path, default=None)
    ap.add_argument("--commit-holdout", action="store_true",
                    help="Append the holdout verdict to the audit ledger (do this once).")
    args = ap.parse_args()

    prereg = (PreRegistration.load(args.preregister) if args.preregister
              else PreRegistration.load())
    ledger = HoldoutLedger()

    with open(args.model, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    from ml_scalper_v7 import build_features  # noqa: E402

    bars = pd.read_parquet(args.data)
    bars.index = pd.to_datetime(bars.index, utc=True)
    feat = build_features(bars)

    signals = build_signals(model, feature_cols, feat, prereg)
    p = prereg.sim_params()

    h_start = pd.Timestamp(prereg.holdout["start"], tz="UTC")
    h_end = pd.Timestamp(prereg.holdout["end"], tz="UTC")

    print("=" * 78)
    print(f"PRE-REGISTERED VALIDATION — {prereg.strategy_id}")
    print(f"  model        : {args.model.name}  (train_end={bundle.get('train_end')})")
    print(f"  holdout      : {prereg.holdout['start']} → {prereg.holdout['end']}")
    print(f"  dsr_n_trials : {prereg.dsr_n_trials}")
    print(f"  config hash  : {prereg.content_hash()[:16]}…")
    print("=" * 78)

    # Development period (pre-holdout) — you ARE allowed to look here.
    dev_mask = (feat.index < h_start)
    dev_trades = simulate(bars[dev_mask], signals[dev_mask], p)
    _print_monthly("DEVELOPMENT (pre-holdout, informational)", dev_trades)

    # Frozen holdout.
    warns = ledger.warnings_for(prereg)
    for w in warns:
        print(f"\n⚠️  {w}")

    hold_mask = (feat.index >= h_start) & (feat.index < h_end)
    hold_trades = simulate(bars[hold_mask], signals[hold_mask], p)
    _print_monthly("HOLDOUT (frozen — the decision is made here)", hold_trades)

    result = evaluate(hold_trades, prereg)

    print("\n" + "-" * 78)
    dsr = result["dsr"].get("dsr")
    print(f"  Deflated Sharpe Ratio : {dsr:.3f}" if dsr is not None
          else f"  Deflated Sharpe Ratio : n/a ({result['dsr'].get('reason')})")
    print(f"  VERDICT               : {result['verdict']}")
    if result["failures"]:
        print("  Gate failures:")
        for fl in result["failures"]:
            print(f"    ✗ {fl}")
    print("-" * 78)

    if args.commit_holdout:
        entry = ledger.record(prereg, result)
        print(f"\nRecorded to ledger: {ledger.path}")
        print(f"  {entry['timestamp']}  {entry['verdict']}  hash={entry['config_hash'][:16]}…")
    else:
        print("\n(report-only; pass --commit-holdout to write the audit ledger)")

    sys.exit(0 if result["verdict"] == "GO" else 1)


if __name__ == "__main__":
    main()
