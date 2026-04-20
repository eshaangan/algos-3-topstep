#!/usr/bin/env python3
"""
Constrained decision-threshold sweep: train once per window, re-run backtest only.

Varies ``backtest_cfg["decision"]`` fields (primary / meta thresholds) without
retraining LGBM or meta routes. Use for offline exploration; promote changes only
after full OOS validation.

Example (single YTD window, fast):

  python -m ml_intraday_v3.experiments.run_decision_threshold_sweep \\
    --acceptance ml_intraday_v3/experiments/_ytd_2026_contract_sweep_config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path

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


def _parse_grid(spec: str) -> list[float]:
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Decision threshold sweep (train once per window)")
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
        "--meta-threshold-grid",
        type=str,
        default="0.40,0.45,0.50",
        help="Comma-separated meta_threshold values",
    )
    parser.add_argument(
        "--primary-long-grid",
        type=str,
        default="0.33,0.38,0.43",
        help="Comma-separated primary long thresholds",
    )
    parser.add_argument(
        "--primary-short-grid",
        type=str,
        default="0.43,0.48,0.53",
        help="Comma-separated primary short thresholds",
    )
    parser.add_argument("--min-trades", type=int, default=15)
    parser.add_argument("--max-drawdown", type=float, default=2500.0)
    parser.add_argument("--min-total-pnl", type=float, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ml_intraday_v3/experiments/results/decision_threshold_sweep",
    )
    args = parser.parse_args()

    acceptance_cfg = _load_yaml(PROJECT_ROOT / args.acceptance)
    training_cfg = _load_yaml(PROJECT_ROOT / args.training_config)
    labeling_cfg = _load_yaml(PROJECT_ROOT / args.labeling_config)
    feature_cfg = _load_yaml(PROJECT_ROOT / args.features_config)
    execution_spec = _load_yaml(PROJECT_ROOT / args.execution_spec)
    backtest_cfg = _load_yaml(PROJECT_ROOT / args.backtest_config)
    risk_cfg = _load_yaml(PROJECT_ROOT / args.risk_config)

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
            backtest_cfg=backtest_cfg,
        )
        window_arts.append(art)

    meta_grid = _parse_grid(args.meta_threshold_grid)
    long_grid = _parse_grid(args.primary_long_grid)
    short_grid = _parse_grid(args.primary_short_grid)

    rows: list[dict] = []
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for meta_t, p_long, p_short in product(meta_grid, long_grid, short_grid):
        combo_label = f"m{meta_t}_L{p_long}_S{p_short}"
        for art in window_arts:
            local_bt = deepcopy(art["local_backtest_cfg"])
            dec = local_bt.setdefault("decision", {})
            dec["meta_threshold"] = float(meta_t)
            dec["primary_threshold"] = float(min(p_long, p_short))
            dec["primary_threshold_by_side"] = {"long": float(p_long), "short": float(p_short)}

            trades_df, equity_df, backtest_metrics = run_backtest(
                events_df=art["test_events_df"],
                bars_df=art["bars_test"],
                primary_preds_df=art["primary_preds"],
                meta_preds_df=art["meta_preds"],
                execution_spec=execution_spec,
                instrument_spec=instrument_spec,
                label_schema=label_schema,
                risk_cfg=risk_cfg,
                backtest_cfg=local_bt,
                bar_size=bar_size,
            )

            tc = int(backtest_metrics.get("trades_count") or 0)
            mdd = float(backtest_metrics.get("max_drawdown_usd") or 0.0)
            pnl = float(backtest_metrics.get("total_pnl_usd") or 0.0)
            ok = tc >= args.min_trades and mdd <= args.max_drawdown
            if args.min_total_pnl is not None and pnl < args.min_total_pnl:
                ok = False

            rows.append(
                {
                    "combo": combo_label,
                    "window": art["name"],
                    "meta_threshold": meta_t,
                    "primary_long": p_long,
                    "primary_short": p_short,
                    "trades_count": tc,
                    "total_pnl_usd": pnl,
                    "max_drawdown_usd": mdd,
                    "profit_factor": backtest_metrics.get("profit_factor"),
                    "win_rate": backtest_metrics.get("win_rate"),
                    "passes_constraints": ok,
                }
            )

    results_df = pd.DataFrame(rows)
    csv_path = out_dir / "threshold_sweep_results.csv"
    results_df.to_csv(csv_path, index=False)

    passed = results_df[results_df["passes_constraints"]].sort_values(
        ["total_pnl_usd", "profit_factor"], ascending=[False, False]
    )
    passed_path = out_dir / "threshold_sweep_passed.csv"
    passed.to_csv(passed_path, index=False)

    summary = {
        "total_runs": int(len(results_df)),
        "passed_count": int(passed.shape[0]),
        "constraints": {
            "min_trades": args.min_trades,
            "max_drawdown_usd": args.max_drawdown,
            "min_total_pnl_usd": args.min_total_pnl,
        },
        "top_by_pnl": passed.head(15).to_dict(orient="records"),
    }
    with open(out_dir / "threshold_sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Wrote {csv_path}, {passed_path}, threshold_sweep_summary.json")


if __name__ == "__main__":
    main()
