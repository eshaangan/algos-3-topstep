#!/usr/bin/env python3
"""
Run Topstep pipeline: train -> backtest -> Monte Carlo pass-rate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from analysis.monte_carlo_combine import simulate_combine
from core.risk_presets import get_risk_preset
from models.nn_inference import artifact_compatibility_issues


def _needs_retrain(config_path: Path) -> bool:
    if not config_path.exists():
        return True
    try:
        cfg = json.loads(config_path.read_text())
    except Exception:
        return True
    issues = artifact_compatibility_issues(cfg, strict_versions=True)
    if issues:
        print(f"Artifact incompatibility: {issues}")
        return True
    return False


def _run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Topstep pipeline: train, backtest, Monte Carlo")
    parser.add_argument("--data-path", default="/mnt/data/es_bars_2010_2025.h5")
    parser.add_argument("--dataset-key", default="bars_5min")
    parser.add_argument("--artifact-dir", default="models/nn_saved")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--max-bars", type=int, default=None)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--out-trades-csv", default="analysis/notebook_backtest_trades_50k.csv")
    parser.add_argument("--preset", default="TOPSTEP_50K")
    parser.add_argument("--mc-runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rth-only", action="store_true", default=False)
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("RISK_PRESET_NAME", args.preset)

    config_path = Path(args.artifact_dir) / f"fold_{args.fold}" / "config.json"
    if args.force_retrain or _needs_retrain(config_path):
        train_cmd = [
            sys.executable,
            "models/nn_train.py",
            "--data-path",
            args.data_path,
            "--dataset-key",
            args.dataset_key,
            "--output-dir",
            args.artifact_dir,
        ]
        if args.fast:
            train_cmd.append("--fast")
        if args.max_bars is not None:
            train_cmd += ["--max-bars", str(args.max_bars)]
        _run(train_cmd)

    backtest_cmd = [
        sys.executable,
        "backtesting/backtest.py",
        "--strategy",
        "nn",
        "--data-path",
        args.data_path,
        "--dataset-key",
        args.dataset_key,
        "--model-dir",
        args.artifact_dir,
        "--fold",
        str(args.fold),
        "--save-trades",
        args.out_trades_csv,
    ]
    if args.fast:
        backtest_cmd.append("--fast")
    if args.max_bars is not None:
        backtest_cmd += ["--max-bars", str(args.max_bars)]
    if args.max_days is not None:
        backtest_cmd += ["--max-days", str(args.max_days)]
    if args.rth_only:
        backtest_cmd.append("--rth-only")
    _run(backtest_cmd)

    trades = pd.read_csv(args.out_trades_csv)
    preset = get_risk_preset(args.preset)
    runs = args.mc_runs if args.mc_runs is not None else (5000 if args.fast else 20000)
    result = simulate_combine(
        trades,
        starting_balance=preset.risk_config.starting_balance,
        profit_target=preset.profit_target,
        daily_loss_limit=preset.risk_config.max_daily_loss,
        trailing_drawdown=preset.risk_config.trailing_drawdown,
        runs=runs,
        seed=args.seed,
        max_days=252,
        consistency_limit=preset.consistency_limit,
    )

    payload = {
        "artifact_dir": args.artifact_dir,
        "trades_csv": args.out_trades_csv,
        "preset": args.preset,
        "monte_carlo": result,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
