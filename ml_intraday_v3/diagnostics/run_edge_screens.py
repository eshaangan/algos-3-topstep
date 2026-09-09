#!/usr/bin/env python3
"""
Run the alpha/beta attribution and P/L stability screens over a trade log.

These two screens ask questions the existing DSR/PBO gauntlet does not:

  1. alpha/beta  -- is the P/L just net directional exposure to the instrument,
                    or exposure to realized range? (Quant Guild lecture 96)
  2. stability   -- is the per-trade P/L distribution the same at the end of the
                    sample as at the start? (Quant Guild lecture 77)

Both are retrospective: they run over trade logs already on disk and need no new
market data beyond the bar tape used for the original backtest.

Examples
--------
    # ML scalper v3 OOS trades against the MES tape
    python -m ml_intraday_v3.diagnostics.run_edge_screens \
        --trades ml_intraday_v3/results/ml_scalper_v3_oos_trades.parquet \
        --pnl-col pnl_dollars \
        --bars data/processed/mes_2026_ytd_5m.h5 --bars-key bars_5min \
        --point-value 5.0 \
        --out ml_intraday_v3/results/edge_screens/ml_scalper_v3.json

    # Stability only (trade log has no usable timestamps)
    python -m ml_intraday_v3.diagnostics.run_edge_screens \
        --trades ml_intraday_v3/diagnostics/orb_oos_results.json \
        --json-key trades --pnl-col pnl --no-time \
        --out ml_intraday_v3/results/edge_screens/orb.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

# Allow running as a script from the repository root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from ml_intraday_v3.analysis.alpha_beta import (  # noqa: E402
    daily_pnl_from_trades,
    daily_market_move,
    alpha_beta_screen,
    format_alpha_beta_report,
)
from ml_intraday_v3.analysis.stability import (  # noqa: E402
    stability_screen,
    format_stability_report,
)
from ml_intraday_v3.analysis.edge_stress import (  # noqa: E402
    stress_screen,
    format_stress_report,
)


# Dollars per index point for the micro contracts this repo trades.
POINT_VALUES = {"MES": 5.0, "MNQ": 2.0, "M2K": 5.0, "MCL": 100.0, "MGC": 10.0}


def load_trades(path: str, json_key: Optional[str] = None) -> pd.DataFrame:
    """Load a trade log from parquet, csv, or (optionally nested) json."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".json":
        with open(path) as fh:
            data = json.load(fh)
        if json_key:
            if json_key not in data:
                raise KeyError(
                    f"{json_key!r} not in {path}; top-level keys: {list(data)[:20]}"
                )
            data = data[json_key]
        if not isinstance(data, list):
            raise ValueError(
                f"expected a list of trades; got {type(data).__name__}. "
                f"Use --json-key to point at the trade list."
            )
        return pd.DataFrame(data)
    raise ValueError(f"unsupported trade-log format: {ext}")


def load_bars(path: str, key: Optional[str] = None) -> pd.DataFrame:
    """Load a bar tape and coerce the index to UTC timestamps."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".h5", ".hdf5"):
        if key is None:
            with pd.HDFStore(path, mode="r") as store:
                keys = [k.lstrip("/") for k in store.keys()]
            if len(keys) != 1:
                raise ValueError(
                    f"{path} holds keys {keys}; pass --bars-key to choose one"
                )
            key = keys[0]
        bars = pd.read_hdf(path, key=key)
    elif ext == ".parquet":
        bars = pd.read_parquet(path)
    else:
        raise ValueError(f"unsupported bar format: {ext}")

    if not isinstance(bars.index, pd.DatetimeIndex):
        for cand in ("timestamp", "ts_event", "datetime", "time", "date"):
            if cand in bars.columns:
                bars = bars.set_index(cand)
                break
    bars.index = pd.to_datetime(bars.index, utc=True, errors="coerce")
    bars = bars[bars.index.notna()].sort_index()
    bars.columns = [str(c).lower() for c in bars.columns]
    return bars


def infer_frac_long(trades: pd.DataFrame) -> Optional[float]:
    """Share of trades taken long, across the direction encodings in this repo."""
    for col in ("direction", "side"):
        if col not in trades.columns:
            continue
        s = trades[col]
        if s.dtype == object:
            up = s.astype(str).str.upper()
            longs = up.isin(["LONG", "BUY", "L", "1", "+1"])
            return float(longs.mean())
        num = pd.to_numeric(s, errors="coerce").dropna()
        if len(num):
            return float((num > 0).mean())
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trades", required=True, help="trade log (.parquet/.csv/.json)")
    ap.add_argument("--json-key", default=None,
                    help="key holding the trade list inside a json results file")
    ap.add_argument("--pnl-col", default="pnl", help="per-trade P/L column")
    ap.add_argument("--time-col", default="entry_time", help="trade timestamp column")
    ap.add_argument("--no-time", action="store_true",
                    help="trade log has no timestamps; assume row order is "
                         "chronological and skip the alpha/beta screen")

    ap.add_argument("--bars", default=None, help="bar tape for the benchmark")
    ap.add_argument("--bars-key", default=None, help="HDF5 key inside --bars")
    ap.add_argument("--instrument", default=None, choices=sorted(POINT_VALUES),
                    help="sets --point-value from the contract spec")
    ap.add_argument("--point-value", type=float, default=None,
                    help="dollars per index point")
    ap.add_argument("--traded-days-only", action="store_true",
                    help="exclude flat days from the alpha/beta regression")

    ap.add_argument("--blocks", type=int, default=3, help="stability blocks")
    ap.add_argument("--bins", type=int, default=10, help="stability P/L bins")
    ap.add_argument("--permutations", type=int, default=2000,
                    help="permutation draws for the stability null")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stress-freq", default="Y",
                    help="period alias for the leave-one-out stress (Y/Q/M)")
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args(argv)

    point_value = args.point_value
    if point_value is None and args.instrument:
        point_value = POINT_VALUES[args.instrument]

    trades = load_trades(args.trades, args.json_key)
    if args.pnl_col not in trades.columns:
        print(f"ERROR: --pnl-col {args.pnl_col!r} not in trade log. "
              f"Columns: {list(trades.columns)}", file=sys.stderr)
        return 2

    time_col = None if args.no_time else args.time_col
    label = os.path.basename(args.trades)
    results = {
        "trade_log": args.trades,
        "pnl_col": args.pnl_col,
        "n_trades_raw": int(len(trades)),
        "frac_long": infer_frac_long(trades),
    }

    print(f"\n{'#' * 72}\n# {label}   ({len(trades)} trades)\n{'#' * 72}")

    # ---------------------------------------------------------- stability
    try:
        st = stability_screen(
            trades, pnl_col=args.pnl_col, time_col=time_col,
            n_blocks=args.blocks, n_bins=args.bins,
            n_permutations=args.permutations, random_state=args.seed,
        )
        results["stability"] = st.to_dict()
        print("\n" + format_stability_report(st, "P/L DISTRIBUTION STABILITY"))
    except (ValueError, KeyError) as exc:
        results["stability"] = {"error": str(exc)}
        print(f"\nstability screen skipped: {exc}")

    # ------------------------------------------------------------- stress
    if time_col is None:
        results["stress"] = {"error": "no timestamps; cannot form periods"}
        print("\nstress screen skipped: trade log has no timestamps")
    else:
        try:
            ts = pd.to_datetime(trades[args.time_col], utc=True, errors="coerce")
            ser = pd.Series(
                pd.to_numeric(trades[args.pnl_col], errors="coerce").to_numpy(),
                index=pd.DatetimeIndex(ts),
            ).dropna().sort_index()
            sr = stress_screen(ser, freq=args.stress_freq)
            results["stress"] = sr.to_dict()
            print("\n" + format_stress_report(sr, "EDGE ROBUSTNESS STRESS"))
        except (ValueError, KeyError) as exc:
            results["stress"] = {"error": str(exc)}
            print(f"\nstress screen skipped: {exc}")

    # --------------------------------------------------------- alpha/beta
    if args.no_time:
        results["alpha_beta"] = {"error": "no timestamps; cannot align to a benchmark"}
        print("\nalpha/beta screen skipped: trade log has no timestamps")
    elif not args.bars:
        results["alpha_beta"] = {"error": "no --bars supplied"}
        print("\nalpha/beta screen skipped: pass --bars to supply the benchmark tape")
    else:
        try:
            bars = load_bars(args.bars, args.bars_key)
            daily_pnl = daily_pnl_from_trades(
                trades, pnl_col=args.pnl_col, time_col=args.time_col
            )
            mkt = daily_market_move(bars)
            ab = alpha_beta_screen(
                daily_pnl, mkt,
                include_flat_days=not args.traded_days_only,
                point_value=point_value,
                frac_long=results["frac_long"],
                n_trades=len(trades),
            )
            results["alpha_beta"] = ab.to_dict()
            results["benchmark"] = {"bars": args.bars, "key": args.bars_key,
                                    "point_value": point_value}
            print("\n" + format_alpha_beta_report(ab, "ALPHA / BETA ATTRIBUTION"))
        except (ValueError, KeyError) as exc:
            results["alpha_beta"] = {"error": str(exc)}
            print(f"\nalpha/beta screen skipped: {exc}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2, default=str)
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
