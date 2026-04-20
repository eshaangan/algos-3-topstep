#!/usr/bin/env python3
"""
Compare fixed decision presets across acceptance windows (train once per window each).

Use after a threshold sweep on one slice: score the same presets on multi-year OOS windows
without re-tuning on those windows.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
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

DECISION_PRESETS: dict[str, dict] = {
    "baseline_live_yaml": {
        "primary_long": 0.33,
        "primary_short": 0.53,
        "meta_threshold": 0.45,
        "require_meta_for_trade": False,
    },
    "sweep_winner_L028_S043_reqmeta": {
        "primary_long": 0.28,
        "primary_short": 0.43,
        "meta_threshold": 0.45,
        "require_meta_for_trade": True,
    },
    "sweep_winner_L028_S048_reqmeta": {
        "primary_long": 0.28,
        "primary_short": 0.48,
        "meta_threshold": 0.45,
        "require_meta_for_trade": True,
    },
}


def _apply_decision(local_bt: dict, spec: dict) -> None:
    dec = local_bt.setdefault("decision", {})
    p_long = float(spec["primary_long"])
    p_short = float(spec["primary_short"])
    dec["primary_threshold_by_side"] = {"long": p_long, "short": p_short}
    dec["primary_threshold"] = float(min(p_long, p_short))
    dec["meta_threshold"] = float(spec["meta_threshold"])
    dec["require_meta_for_trade"] = bool(spec["require_meta_for_trade"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Holdout compare: baseline vs sweep presets")
    parser.add_argument(
        "--acceptance",
        type=str,
        default="ml_intraday_v3/experiments/_oos_long_history_annual.yaml",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/mes_bars_databento_rth.h5",
        help="Primary HDF (annual YAML usually has no additional_paths)",
    )
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
        "--output-dir",
        type=str,
        default="ml_intraday_v3/experiments/results/holdout_threshold_compare_annual",
    )
    args = parser.parse_args()

    acceptance_cfg = _load_yaml(PROJECT_ROOT / args.acceptance)
    training_cfg = _load_yaml(PROJECT_ROOT / args.training_config)
    labeling_cfg = _load_yaml(PROJECT_ROOT / args.labeling_config)
    feature_cfg = _load_yaml(PROJECT_ROOT / args.features_config)
    execution_spec = _load_yaml(PROJECT_ROOT / args.execution_spec)
    base_backtest_cfg = _load_yaml(PROJECT_ROOT / args.backtest_config)
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

    rows: list[dict] = []
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
            raise SystemExit(f"Window {name} empty train/test bars")

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

        for preset_name, spec in DECISION_PRESETS.items():
            local_bt = deepcopy(art["local_backtest_cfg"])
            _apply_decision(local_bt, spec)

            trades_df, equity_df, bt_metrics = run_backtest(
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

            row = {
                "preset": preset_name,
                "window": name,
                "train_start": str(train_start),
                "train_end": str(train_end),
                "test_start": str(test_start),
                "test_end": str(test_end),
                "primary_long": spec["primary_long"],
                "primary_short": spec["primary_short"],
                "meta_threshold": spec["meta_threshold"],
                "require_meta_for_trade": spec["require_meta_for_trade"],
            }
            for k, v in bt_metrics.items():
                row[f"metric_{k}"] = v
            rows.append(row)

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = out_dir / "holdout_compare_results.csv"
    df.to_csv(csv_path, index=False)

    summary: dict = {
        "acceptance": args.acceptance,
        "data_path": args.data_path,
        "presets": list(DECISION_PRESETS.keys()),
        "windows": [w["name"] for w in windows],
    }

    agg = (
        df.groupby("preset", dropna=False)
        .agg(
            total_pnl_sum=("metric_total_pnl_usd", "sum"),
            mean_pnl_per_window=("metric_total_pnl_usd", "mean"),
            worst_max_dd=("metric_max_drawdown_usd", "max"),
            total_trades=("metric_trades_count", "sum"),
            sum_skipped=("metric_skipped_count", "sum"),
            sum_mtm_liq=("metric_mtm_liquidations", "sum"),
        )
        .reset_index()
    )
    summary["aggregate_by_preset"] = agg.to_dict(orient="records")
    summary["per_row"] = df.to_dict(orient="records")

    json_path = out_dir / "holdout_compare_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print("\nAggregate by preset (total_pnl sum over windows, worst-window max DD):")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
