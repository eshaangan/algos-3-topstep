#!/usr/bin/env python3
"""
Large decision-parameter sweep with insight summaries.

Trains once per acceptance window (same as run_decision_threshold_sweep), then replays
backtests across a Cartesian grid of decision (and optional risk) knobs.

Writes:
  - sweep_results.csv: one row per (combo, window)
  - sweep_by_combo.csv: aggregated across windows for each combo
  - insight_summary.json: marginals, top rows, Pareto frontier on combo aggregates
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_intraday_v3.core.instrument import load_instrument_from_execution_spec
from ml_intraday_v3.experiments.run_standalone_topstep_candidate import (
    _derive_cost_mode,
    _fit_promotion_window_artifacts,
    _load_bars,
    _load_yaml,
    _period_bounds,
)
from ml_intraday_v3.backtesting_v3 import run_backtest


def _parse_float_grid(spec: str) -> list[float]:
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def _parse_bool_grid(spec: str) -> list[bool]:
    out: list[bool] = []
    for x in spec.split(","):
        t = x.strip().lower()
        if t in ("true", "1", "yes"):
            out.append(True)
        elif t in ("false", "0", "no"):
            out.append(False)
        elif t:
            raise ValueError(f"Bad bool token: {x!r}")
    return out


def _parse_int_grid(spec: str) -> list[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def _combo_label(meta_t: float, p_long: float, p_short: float, req_meta: bool, max_td: int | None) -> str:
    base = f"m{meta_t}_L{p_long}_S{p_short}_rm{int(req_meta)}"
    if max_td is not None:
        base += f"_mtd{max_td}"
    return base


def _pareto_efficient(df: pd.DataFrame, pnl_col: str, mdd_col: str) -> pd.DataFrame:
    """Maximize pnl, minimize max_drawdown (both columns numeric, lower mdd better)."""
    if df.empty:
        return df
    pnl = df[pnl_col].to_numpy(dtype=float)
    mdd = df[mdd_col].to_numpy(dtype=float)
    mdd = pd.Series(mdd).fillna(1e12).to_numpy()
    n = len(df)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            # j dominates i if j has >= pnl and <= mdd, strictly better in at least one
            better_pnl = pnl[j] >= pnl[i]
            better_mdd = mdd[j] <= mdd[i]
            strict = (pnl[j] > pnl[i]) or (mdd[j] < mdd[i])
            if better_pnl and better_mdd and strict:
                keep[i] = False
                break
    return df.loc[keep].reset_index(drop=True)


def _marginals(df: pd.DataFrame, value_col: str, group_cols: list[str]) -> dict:
    out: dict = {}
    for col in group_cols:
        if col not in df.columns:
            continue
        g = (
            df.groupby(col, dropna=False)[value_col]
            .agg(["count", "mean", "median", "std"])
            .reset_index()
        )
        g.columns = [col, "n", "mean", "median", "std"]
        out[col] = g.to_dict(orient="records")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Large insight sweep (train once, many backtests)")
    parser.add_argument(
        "--acceptance",
        type=str,
        default="ml_intraday_v3/experiments/_ytd_2026_contract_sweep_config.yaml",
    )
    parser.add_argument("--data-path", type=str, default="data/processed/mes_bars_databento_rth.h5")
    parser.add_argument("--hdf-key", type=str, default="bars_5min")
    parser.add_argument(
        "--training-config",
        type=str,
        default="ml_intraday_v3/configs/training_standalone_topstep_recent_decay_dual_meta.yaml",
    )
    parser.add_argument("--labeling-config", type=str, default="ml_intraday_v3/configs/labeling.yaml")
    parser.add_argument(
        "--features-config",
        type=str,
        default="ml_intraday_v3/configs/live_dual_meta_mes_real/features.yaml",
    )
    parser.add_argument(
        "--execution-spec",
        type=str,
        default="ml_intraday_v3/configs/live_dual_meta_mes_real/execution_spec.yaml",
    )
    parser.add_argument(
        "--backtest-config",
        type=str,
        default="ml_intraday_v3/configs/live_dual_meta_mes_real/backtest.yaml",
    )
    parser.add_argument(
        "--risk-config",
        type=str,
        default="ml_intraday_v3/configs/live_dual_meta_mes_real/risk.yaml",
    )
    parser.add_argument(
        "--preset",
        choices=("medium", "large", "custom"),
        default="large",
        help="medium/large set default grids; custom uses only CLI grid strings",
    )
    parser.add_argument("--meta-threshold-grid", type=str, default="")
    parser.add_argument("--primary-long-grid", type=str, default="")
    parser.add_argument("--primary-short-grid", type=str, default="")
    parser.add_argument(
        "--require-meta-grid",
        type=str,
        default="",
        help="Comma-separated bools, e.g. false,true. Empty = false only.",
    )
    parser.add_argument(
        "--max-trades-per-day-grid",
        type=str,
        default="",
        help="Optional comma ints; patches risk intraday_controls.max_trades_per_day per run.",
    )
    parser.add_argument("--min-trades", type=int, default=10)
    parser.add_argument("--max-drawdown", type=float, default=5000.0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ml_intraday_v3/experiments/results/large_insight_sweep",
    )
    args = parser.parse_args()

    if args.preset == "medium":
        meta_g = _parse_float_grid("0.40,0.45,0.50")
        long_g = _parse_float_grid("0.31,0.33,0.38,0.43")
        short_g = _parse_float_grid("0.48,0.53,0.58")
        req_g = [False]
        mtd_g: list[int | None] = [None]
    elif args.preset == "large":
        meta_g = _parse_float_grid("0.35,0.40,0.45,0.50,0.55")
        long_g = _parse_float_grid("0.28,0.31,0.33,0.38,0.43,0.48")
        short_g = _parse_float_grid("0.43,0.48,0.53,0.58,0.63")
        req_g = [False, True]
        mtd_g = [None]
    else:
        meta_g = long_g = short_g = []
        req_g = [False]
        mtd_g = [None]

    if args.meta_threshold_grid.strip():
        meta_g = _parse_float_grid(args.meta_threshold_grid)
    if args.primary_long_grid.strip():
        long_g = _parse_float_grid(args.primary_long_grid)
    if args.primary_short_grid.strip():
        short_g = _parse_float_grid(args.primary_short_grid)
    if args.require_meta_grid.strip():
        req_g = _parse_bool_grid(args.require_meta_grid)
    if args.max_trades_per_day_grid.strip():
        mtd_g = _parse_int_grid(args.max_trades_per_day_grid)
    else:
        mtd_g = [None]

    if not meta_g or not long_g or not short_g:
        raise SystemExit("Define grids via --preset or explicit --*-grid flags")

    acceptance_cfg = _load_yaml(PROJECT_ROOT / args.acceptance)
    training_cfg = _load_yaml(PROJECT_ROOT / args.training_config)
    labeling_cfg = _load_yaml(PROJECT_ROOT / args.labeling_config)
    feature_cfg = _load_yaml(PROJECT_ROOT / args.features_config)
    execution_spec = _load_yaml(PROJECT_ROOT / args.execution_spec)
    base_backtest_cfg = _load_yaml(PROJECT_ROOT / args.backtest_config)
    base_risk_cfg = _load_yaml(PROJECT_ROOT / args.risk_config)

    data_cfg = acceptance_cfg.get("data", {}) or {}
    additional_paths = [PROJECT_ROOT / p for p in data_cfg.get("additional_paths", [])]
    bars = _load_bars([PROJECT_ROOT / args.data_path, *additional_paths], args.hdf_key)

    execution_spec_path = PROJECT_ROOT / args.execution_spec
    instrument_spec = load_instrument_from_execution_spec(execution_spec_path)
    label_schema = {"schema_version": "1.0.0", "cost_mode": _derive_cost_mode(labeling_cfg)}

    bar_size = data_cfg.get("bar_size", "5m")
    schedule_cfg = acceptance_cfg.get("training_window", {}) or {}
    windows = acceptance_cfg.get("windows", []) or []
    if not windows:
        raise SystemExit("acceptance config has no windows")

    window_arts: list[dict] = []
    for window_cfg in windows:
        name = window_cfg["name"]
        train_start, train_end, test_start, test_end = _period_bounds(
            window_cfg=window_cfg,
            default_lookback_days=int(schedule_cfg.get("lookback_days", 180)),
            default_gap_days=int(schedule_cfg.get("gap_days", 0)),
        )
        bars_train = bars[(bars.index >= train_start) & (bars.index <= train_end)].copy()
        bars_test = bars[(bars.index >= test_start) & (bars.index <= test_end)].copy()
        if bars_train.empty or bars_test.empty:
            raise SystemExit(f"Window {name} empty train/test")

        art = _fit_promotion_window_artifacts(
            name=name,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            bars_train=bars_train,
            bars_test=bars_test,
            bar_size=bar_size,
            labeling_cfg=labeling_cfg,
            execution_spec=execution_spec,
            instrument_spec=instrument_spec,
            feature_cfg=feature_cfg,
            training_cfg=training_cfg,
            backtest_cfg=base_backtest_cfg,
        )
        window_arts.append(art)

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for meta_t, p_long, p_short, req_meta, max_td in product(meta_g, long_g, short_g, req_g, mtd_g):
        combo = _combo_label(meta_t, p_long, p_short, req_meta, max_td)
        for art in window_arts:
            local_bt = deepcopy(art["local_backtest_cfg"])
            dec = local_bt.setdefault("decision", {})
            dec["meta_threshold"] = float(meta_t)
            dec["primary_threshold"] = float(min(p_long, p_short))
            dec["primary_threshold_by_side"] = {"long": float(p_long), "short": float(p_short)}
            dec["require_meta_for_trade"] = bool(req_meta)

            risk_eff = deepcopy(base_risk_cfg)
            if max_td is not None:
                intra = risk_eff.setdefault("intraday_controls", {})
                intra["max_trades_per_day"] = int(max_td)

            trades_df, equity_df, backtest_metrics = run_backtest(
                events_df=art["test_events_df"],
                bars_df=art["bars_test"],
                primary_preds_df=art["primary_preds"],
                meta_preds_df=art["meta_preds"],
                execution_spec=execution_spec,
                instrument_spec=instrument_spec,
                label_schema=label_schema,
                risk_cfg=risk_eff,
                backtest_cfg=local_bt,
                bar_size=bar_size,
            )

            tc = int(backtest_metrics.get("trades_count") or 0)
            mdd = float(backtest_metrics.get("max_drawdown_usd") or 0.0)
            pnl = float(backtest_metrics.get("total_pnl_usd") or 0.0)
            ok = tc >= args.min_trades and mdd <= args.max_drawdown

            rows.append(
                {
                    "combo": combo,
                    "window": art["name"],
                    "meta_threshold": meta_t,
                    "primary_long": p_long,
                    "primary_short": p_short,
                    "require_meta_for_trade": req_meta,
                    "max_trades_per_day": max_td if max_td is not None else pd.NA,
                    "trades_count": tc,
                    "total_pnl_usd": pnl,
                    "max_drawdown_usd": mdd,
                    "profit_factor": backtest_metrics.get("profit_factor"),
                    "win_rate": backtest_metrics.get("win_rate"),
                    "avg_trade_usd": backtest_metrics.get("avg_trade_usd"),
                    "passes_constraints": ok,
                }
            )

    results_df = pd.DataFrame(rows)
    results_path = out_dir / "sweep_results.csv"
    results_df.to_csv(results_path, index=False)

    group_keys = [
        "combo",
        "meta_threshold",
        "primary_long",
        "primary_short",
        "require_meta_for_trade",
    ]
    if results_df["max_trades_per_day"].notna().any():
        group_keys.append("max_trades_per_day")

    agg = (
        results_df.groupby(group_keys, dropna=False)
        .agg(
            windows=("window", "nunique"),
            total_pnl_sum=("total_pnl_usd", "sum"),
            mean_pnl_per_window=("total_pnl_usd", "mean"),
            max_drawdown_worst=("max_drawdown_usd", "max"),
            trades_sum=("trades_count", "sum"),
            mean_trades_per_window=("trades_count", "mean"),
        )
        .reset_index()
    )
    agg["pnl_over_mdd"] = agg["total_pnl_sum"] / agg["max_drawdown_worst"].replace(0, float("nan"))
    agg_path = out_dir / "sweep_by_combo.csv"
    agg.to_csv(agg_path, index=False)

    marginal_cols = ["meta_threshold", "primary_long", "primary_short", "require_meta_for_trade"]
    if "max_trades_per_day" in agg.columns and agg["max_trades_per_day"].notna().any():
        marginal_cols.append("max_trades_per_day")

    insight = {
        "run_config": {
            "acceptance": args.acceptance,
            "preset": args.preset,
            "windows": [a["name"] for a in window_arts],
            "grid_sizes": {
                "meta": len(meta_g),
                "long": len(long_g),
                "short": len(short_g),
                "require_meta": len(req_g),
                "max_trades_per_day": len([x for x in mtd_g if x is not None]) or 1,
            },
            "total_backtests": len(results_df),
            "constraints": {"min_trades": args.min_trades, "max_drawdown_usd": args.max_drawdown},
        },
        "marginal_mean_pnl_by_combo_aggregate": _marginals(
            agg, "total_pnl_sum", [c for c in marginal_cols if c in agg.columns]
        ),
        "marginal_mean_pnl_per_window": _marginals(
            results_df, "total_pnl_usd", [c for c in marginal_cols if c in results_df.columns]
        ),
        "top_25_by_total_pnl": agg.nlargest(25, "total_pnl_sum").to_dict(orient="records"),
        "top_25_by_pnl_over_mdd": agg.dropna(subset=["pnl_over_mdd"])
        .sort_values("pnl_over_mdd", ascending=False)
        .head(25)
        .to_dict(orient="records"),
        "passed_constraints_count": int(results_df["passes_constraints"].sum()),
        "top_passed_by_pnl": results_df[results_df["passes_constraints"]]
        .sort_values("total_pnl_usd", ascending=False)
        .head(25)
        .to_dict(orient="records"),
    }

    pareto = _pareto_efficient(agg.copy(), "total_pnl_sum", "max_drawdown_worst")
    pareto_path = out_dir / "pareto_frontier_by_combo.csv"
    pareto.to_csv(pareto_path, index=False)
    insight["pareto_frontier_row_count"] = int(len(pareto))
    insight["pareto_sample"] = pareto.head(20).to_dict(orient="records")

    summary_path = out_dir / "insight_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(insight, f, indent=2, default=str)

    print(f"Wrote {results_path} ({len(results_df)} rows)")
    print(f"Wrote {agg_path} ({len(agg)} combos)")
    print(f"Wrote {pareto_path} ({len(pareto)} Pareto points)")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
