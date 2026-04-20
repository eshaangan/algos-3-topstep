#!/usr/bin/env python3
"""
Sweep *labeling-time* knobs that can materially change trade frequency and payoff geometry:

- CUSUM threshold (event generation): primary_labeling.cusum.threshold_atr_mult
- Triple-barrier PT/SL multiples (labeling + label_schema stop/target sizing):
    primary_labeling.triple_barrier.pt_multipliers[0]
    primary_labeling.triple_barrier.sl_multipliers[0]

Unlike decision-only sweeps (thresholds/meta), these knobs change the event set and labels,
so we must refit per window per combo.

Default acceptance window is YTD 2026 (Jan–Apr) from:
  ml_intraday_v3/experiments/_ytd_2026_contract_sweep_config.yaml
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

def _parse_int_grid(spec: str) -> list[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def _set_nested(d: dict, path: list[str], value) -> None:
    cur = d
    for k in path[:-1]:
        cur = cur.setdefault(k, {})
    cur[path[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="YTD 2026 CUSUM + barrier sweep (refit per combo)")
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
    parser.add_argument("--cusum-atr-grid", type=str, default="0.8,1.0,1.2")
    parser.add_argument("--pt-grid", type=str, default="2.5,3.0,3.5")
    parser.add_argument("--sl-grid", type=str, default="2.0,2.5,3.0")
    parser.add_argument(
        "--horizon-5m-grid",
        type=str,
        default="12",
        help="Comma-separated int horizons in 5m bars (must be consistent with holding constraints).",
    )
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--max-drawdown", type=float, default=2500.0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ml_intraday_v3/experiments/results/ytd_2026_cusum_barrier_sweep",
    )
    args = parser.parse_args()

    acceptance_cfg = _load_yaml(PROJECT_ROOT / args.acceptance)
    training_cfg = _load_yaml(PROJECT_ROOT / args.training_config)
    labeling_cfg_base = _load_yaml(PROJECT_ROOT / args.labeling_config)
    feature_cfg = _load_yaml(PROJECT_ROOT / args.features_config)
    execution_spec = _load_yaml(PROJECT_ROOT / args.execution_spec)
    backtest_cfg = _load_yaml(PROJECT_ROOT / args.backtest_config)
    risk_cfg = _load_yaml(PROJECT_ROOT / args.risk_config)

    data_cfg = acceptance_cfg.get("data", {}) or {}
    additional_paths = [PROJECT_ROOT / p for p in data_cfg.get("additional_paths", [])]
    bars = _load_bars([PROJECT_ROOT / args.data_path, *additional_paths], args.hdf_key)

    execution_spec_path = PROJECT_ROOT / args.execution_spec
    instrument_spec = load_instrument_from_execution_spec(execution_spec_path)

    bar_size = data_cfg.get("bar_size", "5m")
    schedule_cfg = acceptance_cfg.get("training_window", {}) or {}
    windows = acceptance_cfg.get("windows", []) or []
    if not windows:
        raise SystemExit("acceptance config has no windows")

    cusum_grid = _parse_grid(args.cusum_atr_grid)
    pt_grid = _parse_grid(args.pt_grid)
    sl_grid = _parse_grid(args.sl_grid)
    horizon_grid = _parse_int_grid(args.horizon_5m_grid)

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for cusum_atr, pt_mult, sl_mult, hz5 in product(cusum_grid, pt_grid, sl_grid, horizon_grid):
        combo = f"cus{cusum_atr}_pt{pt_mult}_sl{sl_mult}_hz{hz5}"

        # Patch labeling config for this combo
        labeling_cfg = deepcopy(labeling_cfg_base)
        _set_nested(
            labeling_cfg,
            ["primary_labeling", "cusum", "threshold_atr_mult"],
            float(cusum_atr),
        )
        # Force single PT/SL value (the runners expect lists).
        _set_nested(
            labeling_cfg,
            ["primary_labeling", "triple_barrier", "pt_multipliers"],
            [float(pt_mult)],
        )
        _set_nested(
            labeling_cfg,
            ["primary_labeling", "triple_barrier", "sl_multipliers"],
            [float(sl_mult)],
        )

        # Patch vertical horizon for 5m bars. Keep 1m as-is.
        tb = labeling_cfg.setdefault("primary_labeling", {}).setdefault("triple_barrier", {})
        hz = tb.get("horizon_bars", {}) or {}
        hz = dict(hz)
        hz["5m"] = [int(hz5)]
        tb["horizon_bars"] = hz

        # Ensure backtest exit logic matches horizon (execution spec holding constraints).
        local_execution_spec = deepcopy(execution_spec)
        max_holding = (local_execution_spec.get("holding_constraints", {}) or {}).get("max_holding_bars", {}) or {}
        max_holding = dict(max_holding)
        max_holding["5m"] = int(hz5)
        local_execution_spec.setdefault("holding_constraints", {})["max_holding_bars"] = max_holding

        label_schema = {"schema_version": "1.0.0", "cost_mode": _derive_cost_mode(labeling_cfg)}

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

            trades_df, equity_df, backtest_metrics = run_backtest(
                events_df=art["test_events_df"],
                bars_df=art["bars_test"],
                primary_preds_df=art["primary_preds"],
                meta_preds_df=art["meta_preds"],
                execution_spec=local_execution_spec,
                instrument_spec=instrument_spec,
                label_schema=label_schema,
                risk_cfg=risk_cfg,
                backtest_cfg=art["local_backtest_cfg"],
                bar_size=bar_size,
            )

            tc = int(backtest_metrics.get("trades_count") or 0)
            mdd = float(backtest_metrics.get("max_drawdown_usd") or 0.0)
            pnl = float(backtest_metrics.get("total_pnl_usd") or 0.0)
            ok = (tc >= args.min_trades) and (mdd <= args.max_drawdown)

            rows.append(
                {
                    "combo": combo,
                    "window": name,
                    "cusum_threshold_atr_mult": float(cusum_atr),
                    "pt_mult": float(pt_mult),
                    "sl_mult": float(sl_mult),
                    "horizon_5m_bars": int(hz5),
                    "trades_count": tc,
                    "total_pnl_usd": pnl,
                    "max_drawdown_usd": mdd,
                    "profit_factor": backtest_metrics.get("profit_factor"),
                    "win_rate": backtest_metrics.get("win_rate"),
                    "passes_constraints": ok,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sweep_results.csv", index=False)

    passed = df[df["passes_constraints"]].sort_values(
        ["total_pnl_usd", "profit_factor"], ascending=[False, False]
    )
    passed.to_csv(out_dir / "sweep_passed.csv", index=False)

    summary = {
        "total_runs": int(len(df)),
        "passed_count": int(len(passed)),
        "constraints": {
            "min_trades": args.min_trades,
            "max_drawdown_usd": args.max_drawdown,
        },
        "top_by_pnl": passed.head(20).to_dict(orient="records"),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Wrote {out_dir}/sweep_results.csv, sweep_passed.csv, summary.json")


if __name__ == "__main__":
    main()

