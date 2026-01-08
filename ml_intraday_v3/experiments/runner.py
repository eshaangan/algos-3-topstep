"""
Experiment runner for V3 pipeline.
"""

from __future__ import annotations

import itertools
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import yaml

from ml_intraday_v3.run_manifest import hash_content
from ml_intraday_v3.backtesting_v3 import run_backtest
from ml_intraday_v3.training import train_on_splits
from ml_intraday_v3.core.instrument import (
    InstrumentSpec,
    load_instrument_from_execution_spec,
    validate_risk_config_no_instrument_economics,
)

from .aggregation import aggregate_split_metrics
from .diagnostics import compute_pbo, compute_dsr

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _enumerate_grid(grid: dict) -> List[Dict]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = []
    for key in keys:
        val = grid[key]
        if isinstance(val, list):
            values.append(val)
        else:
            values.append([val])
    combos = []
    for combo in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, combo)})
    return combos


def _resolve_training_dir(
    run_dir: Path, bar_size: str, cv_kind: str, training_dir_arg: str | None
) -> Path:
    if training_dir_arg:
        base = Path(training_dir_arg)
        if (base / f"bar_size={bar_size}").exists():
            return base / f"bar_size={bar_size}" / "training" / cv_kind
        if base.name == cv_kind:
            return base
        if base.name == "training":
            return base / cv_kind
        if "bar_size=" in base.parts:
            return base
    return run_dir / f"bar_size={bar_size}" / "training" / cv_kind


def _load_splits(bar_dir: Path, cv_kind: str) -> Tuple[List[Dict], str, str]:
    cv_path = bar_dir / "cv_splits.json"
    if not cv_path.exists():
        raise FileNotFoundError(f"cv_splits.json not found: {cv_path}")
    with open(cv_path, "r") as f:
        cv_data = json.load(f)
    if cv_kind == "purged_kfold":
        return cv_data.get("purged_kfold", []), "fold", "fold"
    if cv_kind == "cpcv":
        return cv_data.get("cpcv", []), "path_id", "path"
    raise ValueError(f"Unsupported cv_kind: {cv_kind}")


def _load_bar_sizes(run_dir: Path) -> List[str]:
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        return manifest.get("bar_sizes", ["1m", "5m"])
    bar_size_dirs = [
        d.name
        for d in run_dir.iterdir()
        if d.is_dir() and d.name.startswith("bar_size=")
    ]
    return [d.replace("bar_size=", "") for d in bar_size_dirs]


def _run_backtests_for_variant(
    run_dir: Path,
    bar_size: str,
    cv_kind: str,
    training_dir: Path,
    backtest_cfg: dict,
    execution_spec: dict,
    instrument_spec: InstrumentSpec,
    risk_cfg: dict,
    variant_id: str,
    output_root: Path,
) -> Tuple[List[Dict], Dict]:
    bar_dir = run_dir / f"bar_size={bar_size}"
    events_path = bar_dir / "events.parquet"
    bars_path = bar_dir / "bars.parquet"
    label_schema_path = bar_dir / "label_schema.json"
    if not events_path.exists() or not bars_path.exists() or not label_schema_path.exists():
        raise FileNotFoundError(
            f"Missing events/bars for {bar_size}: {events_path}, {bars_path}"
        )
    events_df = pd.read_parquet(events_path)
    bars_df = pd.read_parquet(bars_path)
    with open(label_schema_path, "r") as f:
        label_schema = json.load(f)

    splits, split_id_key, prefix = _load_splits(bar_dir, cv_kind)
    output_dir = (
        output_root
        / f"variant_{variant_id}"
        / f"bar_size={bar_size}"
        / cv_kind
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    split_metrics = []
    returns = []
    daily_returns = []
    holding_minutes = []
    cost_modes = set()
    cost_mode_policy = None
    for split in splits:
        split_id = split.get(split_id_key)
        split_dir = training_dir / f"{prefix}_{split_id}"
        primary_path = split_dir / "preds.parquet"
        meta_path = split_dir / "meta_preds.parquet"
        if not primary_path.exists():
            raise FileNotFoundError(f"Missing preds: {primary_path}")
        primary_preds = pd.read_parquet(primary_path)

        meta_preds = None
        if backtest_cfg.get("decision", {}).get("use_meta", False):
            if not meta_path.exists():
                raise FileNotFoundError(f"Missing meta preds: {meta_path}")
            meta_preds = pd.read_parquet(meta_path)

        test_ids = split.get("test_event_ids", [])
        test_events = events_df[events_df["event_id"].isin(test_ids)].copy()
        test_events = test_events.sort_values("t0")

        split_out_dir = output_dir / f"{prefix}_{split_id}"
        split_out_dir.mkdir(parents=True, exist_ok=True)

        trades_df, equity_df, metrics = run_backtest(
            events_df=test_events,
            bars_df=bars_df,
            primary_preds_df=primary_preds,
            meta_preds_df=meta_preds,
            execution_spec=execution_spec,
            instrument_spec=instrument_spec,
            label_schema=label_schema,
            risk_cfg=risk_cfg,
            backtest_cfg=backtest_cfg,
            bar_size=bar_size,
        )

        if backtest_cfg.get("outputs", {}).get("write_trade_log", True):
            trades_df.to_parquet(split_out_dir / "trades.parquet")
        if backtest_cfg.get("outputs", {}).get("write_equity_curve", True):
            equity_df.to_parquet(split_out_dir / "equity.parquet")

        with open(split_out_dir / "backtest_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        split_metrics.append({"split_id": split_id, **metrics})

        if "pnl_usd" in trades_df.columns:
            executed = trades_df[trades_df["executed"]].copy()
            returns.extend(
                executed["pnl_usd"].dropna().astype(float).tolist()
            )
            if "exit_ts" in executed.columns:
                exit_ts = pd.to_datetime(
                    executed["exit_ts"], utc=True, errors="coerce"
                )
                daily = (
                    executed.assign(_exit_date=exit_ts.dt.floor("D"))
                    .dropna(subset=["_exit_date"])
                    .groupby("_exit_date")["pnl_usd"]
                    .sum()
                )
                daily_returns.extend(daily.astype(float).tolist())
            if "entry_ts" in executed.columns and "exit_ts" in executed.columns:
                entry_ts = pd.to_datetime(
                    executed["entry_ts"], utc=True, errors="coerce"
                )
                exit_ts = pd.to_datetime(
                    executed["exit_ts"], utc=True, errors="coerce"
                )
                deltas = (
                    (exit_ts - entry_ts).dt.total_seconds() / 60.0
                )
                holding_minutes.extend(
                    deltas.dropna().astype(float).tolist()
                )
            if "cost_mode" in executed.columns:
                cost_modes.update(
                    executed["cost_mode"].dropna().unique().tolist()
                )
        if metrics.get("cost_mode_policy") and cost_mode_policy is None:
            cost_mode_policy = metrics.get("cost_mode_policy")

    avg_holding = (
        float(sum(holding_minutes) / len(holding_minutes))
        if holding_minutes
        else None
    )
    cost_mode = None
    if cost_modes:
        cost_mode = cost_modes.pop() if len(cost_modes) == 1 else "mixed"

    return split_metrics, {
        "returns_per_trade": returns,
        "returns_per_day": daily_returns,
        "n_trades": len(returns),
        "avg_holding_minutes": avg_holding,
        "cost_mode": cost_mode,
        "cost_mode_policy": cost_mode_policy,
    }


def run_experiments(run_dir: Path | str, grid_config_path: Path | str) -> dict:
    """
    Run experiment grid with optional retraining and backtests.
    """
    run_dir = Path(run_dir)
    grid_config_path = Path(grid_config_path)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    if not grid_config_path.exists():
        raise FileNotFoundError(f"Grid config not found: {grid_config_path}")

    grid_cfg = _load_yaml(grid_config_path)
    cv_kind = grid_cfg.get("cv_kind", "purged_kfold")
    grid = grid_cfg.get("grid", {})
    diagnostics_cfg = grid_cfg.get("diagnostics", {})
    selection_metric = diagnostics_cfg.get("selection_metric", "total_pnl_usd")

    training_cfg_path = Path(
        grid_cfg.get("training_config", "ml_intraday_v3/configs/training.yaml")
    )
    backtest_cfg_path = Path(
        grid_cfg.get("backtest_config", "ml_intraday_v3/configs/backtest.yaml")
    )
    execution_spec_path = Path(
        grid_cfg.get("execution_spec", "ml_intraday_v3/configs/execution_spec.yaml")
    )
    risk_cfg_path = Path(
        grid_cfg.get("risk_config", "ml_intraday_v3/configs/risk.yaml")
    )
    base_training_cfg = _load_yaml(training_cfg_path)
    base_backtest_cfg = _load_yaml(backtest_cfg_path)
    execution_spec = _load_yaml(execution_spec_path)
    risk_cfg = _load_yaml(risk_cfg_path)
    validate_risk_config_no_instrument_economics(risk_cfg)
    instrument_spec = load_instrument_from_execution_spec(execution_spec_path)

    bar_sizes = _load_bar_sizes(run_dir)
    variants = _enumerate_grid(grid)

    exp_root = run_dir / "experiments"
    exp_root.mkdir(parents=True, exist_ok=True)

    exp_snapshot = {
        "grid_config_path": str(grid_config_path),
        "grid_config": grid_cfg,
        "training_config_path": str(training_cfg_path),
        "training_config_hash": hash_content(base_training_cfg),
        "backtest_config_path": str(backtest_cfg_path),
        "backtest_config_hash": hash_content(base_backtest_cfg),
        "execution_spec_hash": hash_content(execution_spec),
        "risk_config_hash": hash_content(risk_cfg),
        "cv_kind": cv_kind,
        "bar_sizes": bar_sizes,
    }
    exp_id = hash_content(exp_snapshot)[:12]
    exp_dir = exp_root / f"exp_{exp_id}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "config_snapshot.json", "w") as f:
        json.dump(exp_snapshot, f, indent=2)

    results_rows = []
    split_rows = []
    returns_by_variant = {}

    for variant in variants:
        variant_id = hash_content(variant)[:12]
        primary_threshold = variant.get("primary_threshold")
        meta_threshold = variant.get("meta_threshold")
        model_c = variant.get("model_C")

        training_cfg = deepcopy(base_training_cfg)
        model_cfg = training_cfg.get("model", {}) or {}
        model_kind = model_cfg.get("kind", "logreg")

        retrain_needed = False
        if model_kind == "logreg":
            base_c = (
                model_cfg.get("params", {})
                .get("C", 1.0)
            )
            if model_c is not None and float(model_c) != float(base_c):
                training_cfg.setdefault("model", {}).setdefault("params", {})[
                    "C"
                ] = float(model_c)
                retrain_needed = True

        for bar_size in bar_sizes:
            existing_training_dir = _resolve_training_dir(
                run_dir, bar_size, cv_kind, None
            )
            need_train = retrain_needed or (not existing_training_dir.exists())
            if need_train:
                training_dir = (
                    exp_dir
                    / "training"
                    / f"variant_{variant_id}"
                    / f"bar_size={bar_size}"
                    / cv_kind
                )
                train_on_splits(
                    run_dir=run_dir,
                    bar_size=bar_size,
                    training_config=training_cfg,
                    cv_kind=cv_kind,
                    training_dir_override=training_dir,
                )
            else:
                training_dir = existing_training_dir

            backtest_cfg = deepcopy(base_backtest_cfg)
            if primary_threshold is not None:
                backtest_cfg.setdefault("decision", {})[
                    "primary_threshold"
                ] = float(primary_threshold)
            if meta_threshold is not None:
                backtest_cfg.setdefault("decision", {})[
                    "meta_threshold"
                ] = float(meta_threshold)

            split_metrics, returns_meta = _run_backtests_for_variant(
                run_dir=run_dir,
                bar_size=bar_size,
                cv_kind=cv_kind,
                training_dir=training_dir,
                backtest_cfg=backtest_cfg,
                execution_spec=execution_spec,
                instrument_spec=instrument_spec,
                risk_cfg=risk_cfg,
                variant_id=variant_id,
                output_root=exp_dir / "backtests",
            )

            for row in split_metrics:
                split_rows.append(
                    {
                        "variant_id": variant_id,
                        "bar_size": bar_size,
                        "cv_kind": cv_kind,
                        **row,
                    }
                )

            agg = aggregate_split_metrics(split_metrics)
            agg["variant_id"] = variant_id
            agg["bar_size"] = bar_size
            agg["cv_kind"] = cv_kind
            agg["primary_threshold"] = primary_threshold
            agg["meta_threshold"] = meta_threshold
            agg["model_C"] = model_c
            agg["selection_metric"] = selection_metric
            if selection_metric in agg:
                agg["selection_value"] = agg[selection_metric]

            results_rows.append(agg)

            returns_by_variant[(variant_id, bar_size)] = returns_meta

    results_df = pd.DataFrame(results_rows)
    results_df.to_parquet(exp_dir / "results.parquet")

    split_df = pd.DataFrame(split_rows)

    pbo_payload = {"cv_kind": cv_kind, "metric": selection_metric, "bar_sizes": {}}
    if cv_kind != "cpcv":
        pbo_payload["not_applicable"] = True
        pbo_payload["reason"] = "cv_kind_not_cpcv"
    elif diagnostics_cfg.get("compute_pbo", True):
        if selection_metric not in split_df.columns:
            pbo_payload["reason"] = "selection_metric_missing"
        else:
            for bar_size in bar_sizes:
                perf_df = split_df[
                    (split_df["bar_size"] == bar_size)
                    & (split_df[selection_metric].notna())
                ][["variant_id", "split_id", selection_metric]].rename(
                    columns={selection_metric: "metric"}
                )
                perf_df["metric"] = perf_df["metric"].astype(float)
                pbo_payload["bar_sizes"][bar_size] = compute_pbo(
                    perf_df, metric_col="metric"
                )
    else:
        pbo_payload["disabled"] = True

    with open(exp_dir / "pbo.json", "w") as f:
        json.dump(pbo_payload, f, indent=2)

    dsr_payload = {
        "metric": "pnl_usd",
        "n_trials": int(diagnostics_cfg.get("n_trials", len(variants))),
        "target_sharpe": float(diagnostics_cfg.get("target_sharpe", 0.0)),
        "return_series": {
            "per_trade": "pnl_usd",
            "per_day": "daily_pnl_usd",
        },
        "return_series_notes": (
            "per_trade uses executed trade pnl_usd; per_day aggregates "
            "pnl_usd by exit date (UTC)."
        ),
        "bar_sizes": {},
    }
    if diagnostics_cfg.get("compute_dsr", True):
        for bar_size in bar_sizes:
            by_variant = {}
            for variant in variants:
                variant_id = hash_content(variant)[:12]
                returns_meta = returns_by_variant.get(
                    (variant_id, bar_size), {}
                )
                returns_trade = returns_meta.get("returns_per_trade", [])
                returns_day = returns_meta.get("returns_per_day", [])
                by_variant[variant_id] = {
                    "per_trade": compute_dsr(
                        returns_trade,
                        n_trials=dsr_payload["n_trials"],
                        target_sharpe=dsr_payload["target_sharpe"],
                    ),
                    "per_day": compute_dsr(
                        returns_day,
                        n_trials=dsr_payload["n_trials"],
                        target_sharpe=dsr_payload["target_sharpe"],
                    )
                    if returns_day
                    else {"dsr": None, "reason": "no_daily_returns"},
                    "n_trades": returns_meta.get("n_trades"),
                    "avg_holding_minutes": returns_meta.get("avg_holding_minutes"),
                    "cost_mode": returns_meta.get("cost_mode"),
                    "cost_mode_policy": returns_meta.get("cost_mode_policy"),
                }

            best_variant = None
            if not results_df.empty and selection_metric in results_df.columns:
                subset = results_df[results_df["bar_size"] == bar_size]
                if not subset.empty:
                    best_variant = subset.sort_values(
                        selection_metric, ascending=False
                    )["variant_id"].iloc[0]

            dsr_payload["bar_sizes"][bar_size] = {
                "by_variant": by_variant,
                "best_variant": best_variant,
            }
    else:
        dsr_payload["disabled"] = True

    with open(exp_dir / "dsr.json", "w") as f:
        json.dump(dsr_payload, f, indent=2)

    _write_leaderboards(exp_root)

    return {
        "exp_id": exp_id,
        "exp_dir": exp_dir,
        "n_variants": len(variants),
    }


def _write_leaderboards(exp_root: Path) -> None:
    exp_dirs = sorted([p for p in exp_root.iterdir() if p.is_dir() and p.name.startswith("exp_")])
    if not exp_dirs:
        return

    frames = []
    for exp_dir in exp_dirs:
        results_path = exp_dir / "results.parquet"
        if results_path.exists():
            df = pd.read_parquet(results_path)
            df["exp_id"] = exp_dir.name.replace("exp_", "")
            frames.append(df)

    if not frames:
        return

    all_results = pd.concat(frames, ignore_index=True)
    for bar_size, subset in all_results.groupby("bar_size"):
        metric = "selection_value" if "selection_value" in subset.columns else "total_pnl_usd"
        leaderboard = subset.sort_values(metric, ascending=False).reset_index(drop=True)
        leaderboard_path = exp_root / f"leaderboard_bar_size={bar_size}.parquet"
        leaderboard.to_parquet(leaderboard_path)
