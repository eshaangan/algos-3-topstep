#!/usr/bin/env python3
"""
Run training/backtest permutations and summarize metrics for Topstep combine.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
import sys

import importlib.util
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_mc_path = Path(__file__).parent / "monte_carlo_combine.py"
_spec = importlib.util.spec_from_file_location("monte_carlo_combine", _mc_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load monte_carlo_combine from {_mc_path}")
_mc_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mc_module)
simulate_combine = _mc_module.simulate_combine


LABEL_RUNS = {
    "cusum": Path("runs/regularized_cusum_24h_20260114"),
    "trend": Path("runs/regularized_24h_20260114"),
}

CONFIGS = {
    "baseline": {
        "long": Path("ml_intraday_v3/configs/training.yaml"),
        "short": Path("ml_intraday_v3/configs/training_short.yaml"),
    },
    "regularized": {
        "long": Path("ml_intraday_v3/configs/training_regularized.yaml"),
        "short": Path("ml_intraday_v3/configs/training_short_regularized.yaml"),
    },
}

OUTPUT_BASE = Path("runs/permutation_grid")
BAR_SIZES = "5m"
CV_KIND = "purged_kfold"

TOPSTEP_RULES = {
    "starting_balance": 50000.0,
    "profit_target": 3000.0,
    "daily_loss_limit": 1000.0,
    "trailing_drawdown": 2500.0,
    # Topstep consistency: best day <= 50% of total profit (ratio rule).
    "consistency_max_day_fraction": 0.5,
    "runs": 10_000,
    "seed": 42,
    "max_days": 252,
}


def run_cmd(args: list[str]) -> None:
    print("\n>>", " ".join(args))
    subprocess.run(args, check=True)


def training_summary_path(base_dir: Path) -> Path:
    return base_dir / f"bar_size={BAR_SIZES}" / "training" / CV_KIND / "summary.json"


def ensure_training(run_dir: Path, config_path: Path, out_dir: Path, force: bool) -> None:
    summary_path = training_summary_path(out_dir)
    if summary_path.exists() and not force:
        print(f"[skip] training exists: {summary_path}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            "python",
            "-m",
            "ml_intraday_v3.cli",
            "build-train",
            "--run-dir",
            str(run_dir),
            "--training-config",
            str(config_path),
            "--training-dir",
            str(out_dir),
            "--cv-kind",
            CV_KIND,
            "--bar-sizes",
            BAR_SIZES,
        ]
    )


def backtest_dir(run_dir: Path) -> Path:
    return run_dir / f"bar_size={BAR_SIZES}" / "backtests" / CV_KIND


def copy_backtest(run_dir: Path, name: str, force: bool) -> Path:
    src = backtest_dir(run_dir)
    dst = src.parent / f"{src.name}__{name}"
    if dst.exists():
        if not force:
            print(f"[skip] backtest copy exists: {dst}")
            return dst
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def run_backtest(
    run_dir: Path,
    training_long: Path,
    training_short: Path | None,
    name: str,
    force: bool,
) -> Path:
    dst = backtest_dir(run_dir).parent / f"{CV_KIND}__{name}"
    if dst.exists() and not force:
        print(f"[skip] backtest already copied: {dst}")
        return dst

    args = [
        "python",
        "-m",
        "ml_intraday_v3.cli",
        "build-backtest",
        "--run-dir",
        str(run_dir),
        "--training-dir",
        str(training_long),
        "--cv-kind",
        CV_KIND,
        "--bar-sizes",
        BAR_SIZES,
    ]
    if training_short is not None:
        args += ["--secondary-training-dir", str(training_short)]
    run_cmd(args)
    return copy_backtest(run_dir, name, force=True)


def _load_trades(backtest_path: Path) -> pd.DataFrame:
    trades = []
    for trades_path in sorted(backtest_path.glob("fold_*/trades.parquet")):
        trades.append(pd.read_parquet(trades_path))
    if not trades:
        return pd.DataFrame()
    return pd.concat(trades, ignore_index=True)


def _daily_pnl(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    executed = trades[trades["executed"]].copy()
    if executed.empty:
        return pd.Series(dtype=float)
    ts = pd.to_datetime(executed["exit_ts"], utc=True, errors="coerce")
    executed = executed.loc[ts.notna()].copy()
    executed["exit_ts"] = ts[ts.notna()]

    # Topstep daily reset: 17:00 America/Chicago.
    exit_ct = executed["exit_ts"].dt.tz_convert("America/Chicago")
    cutoff = exit_ct.dt.normalize() + pd.Timedelta(hours=17)
    session_day = exit_ct.dt.normalize()
    session_day = session_day.where(exit_ct >= cutoff, session_day - pd.Timedelta(days=1))
    executed["session_day"] = session_day.dt.date

    return executed.groupby("session_day")["pnl_usd"].sum()


def _sharpe_from_daily(daily_pnl: pd.Series) -> float | None:
    if daily_pnl is None or len(daily_pnl) < 2:
        return None
    mean = daily_pnl.mean()
    std = daily_pnl.std(ddof=0)
    if std == 0 or math.isnan(std):
        return None
    return float(mean / std * math.sqrt(252))


def summarize_backtest(backtest_path: Path) -> dict:
    summary_path = backtest_path / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json: {summary_path}")
    summary = json.load(open(summary_path))

    metrics_by_split = summary.get("metrics_by_split", [])
    pnl_by_split = [m.get("total_pnl_usd", 0.0) for m in metrics_by_split]
    pnl_mean = float(np.mean(pnl_by_split)) if pnl_by_split else 0.0
    pnl_std = float(np.std(pnl_by_split, ddof=0)) if pnl_by_split else 0.0
    cv = None
    if pnl_mean != 0:
        cv = float(pnl_std / pnl_mean)

    max_drawdown = None
    if metrics_by_split:
        dd_values = [m.get("max_drawdown_usd") for m in metrics_by_split]
        dd_values = [d for d in dd_values if d is not None]
        max_drawdown = float(max(dd_values)) if dd_values else None

    trades = _load_trades(backtest_path)
    executed = trades[trades["executed"]].copy() if not trades.empty else trades
    win_rate = None
    profit_factor = None
    total_pnl = float(executed["pnl_usd"].sum()) if not executed.empty else 0.0
    if not executed.empty:
        wins = executed[executed["pnl_usd"] > 0]
        losses = executed[executed["pnl_usd"] < 0]
        win_rate = float(len(wins) / len(executed)) if len(executed) else None
        if len(wins) and len(losses):
            profit_factor = float(wins["pnl_usd"].sum() / abs(losses["pnl_usd"].sum()))

    daily_pnl = _daily_pnl(trades)
    daily_loss_violations = int((daily_pnl <= -TOPSTEP_RULES["daily_loss_limit"]).sum()) if not daily_pnl.empty else 0
    daily_loss_violation_pct = float(daily_loss_violations / len(daily_pnl)) if len(daily_pnl) else None
    sharpe = _sharpe_from_daily(daily_pnl)

    mc_result = None
    if not executed.empty:
        mc_trades = executed[["pnl_usd", "exit_ts"]].copy()
        mc_trades = mc_trades.rename(columns={"pnl_usd": "pnl", "exit_ts": "entry_time"})
        mc_trades["entry_time"] = pd.to_datetime(mc_trades["entry_time"], utc=True, errors="coerce")
        mc_trades = mc_trades.dropna(subset=["entry_time"])
        mc_result = simulate_combine(
            mc_trades,
            starting_balance=TOPSTEP_RULES["starting_balance"],
            profit_target=TOPSTEP_RULES["profit_target"],
            daily_loss_limit=TOPSTEP_RULES["daily_loss_limit"],
            trailing_drawdown=TOPSTEP_RULES["trailing_drawdown"],
            runs=TOPSTEP_RULES["runs"],
            seed=TOPSTEP_RULES["seed"],
            max_days=TOPSTEP_RULES["max_days"],
            consistency_limit=None,
            consistency_max_day_fraction=TOPSTEP_RULES["consistency_max_day_fraction"],
        )

    return {
        "total_pnl_usd": total_pnl,
        "max_drawdown_usd": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "trades": int(len(executed)) if executed is not None else 0,
        "daily_loss_violations": daily_loss_violations,
        "daily_loss_violation_pct": daily_loss_violation_pct,
        "sharpe": sharpe,
        "cv": cv,
        "mc_pass_rate": mc_result["pass_rate"] if mc_result else None,
        "mc_pass_within_10d": (mc_result or {}).get("pass_within_days", {}).get("10"),
        "mc_pass_within_15d": (mc_result or {}).get("pass_within_days", {}).get("15"),
        "mc_pass_within_20d": (mc_result or {}).get("pass_within_days", {}).get("20"),
        "mc_days_p50": (mc_result or {}).get("days_to_pass", {}).get("p50"),
        "mc_days_p95": (mc_result or {}).get("days_to_pass", {}).get("p95"),
        "mc_result": mc_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run permutation grid")
    parser.add_argument("--force", action="store_true", help="Re-run training/backtests even if outputs exist")
    args = parser.parse_args()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    scenarios = []
    for label_name, run_dir in LABEL_RUNS.items():
        for config_name, cfg in CONFIGS.items():
            scenarios.append(
                {
                    "label": label_name,
                    "run_dir": run_dir,
                    "config": config_name,
                    "long_cfg": cfg["long"],
                    "short_cfg": cfg["short"],
                }
            )

    results = []

    for scenario in scenarios:
        label = scenario["label"]
        config = scenario["config"]
        run_dir = scenario["run_dir"]

        long_dir = OUTPUT_BASE / f"{label}_{config}_long"
        short_dir = OUTPUT_BASE / f"{label}_{config}_short"

        print(f"\n=== Training: {label} / {config} ===")
        ensure_training(run_dir, scenario["long_cfg"], long_dir, args.force)
        ensure_training(run_dir, scenario["short_cfg"], short_dir, args.force)

        for mode in ["single", "dual"]:
            print(f"\n=== Backtest: {label} / {config} / {mode} ===")
            backtest_name = f"{label}_{config}_{mode}"
            backtest_path = run_backtest(
                run_dir,
                training_long=long_dir,
                training_short=short_dir if mode == "dual" else None,
                name=backtest_name,
                force=args.force,
            )
            metrics = summarize_backtest(backtest_path)
            metrics.update(
                {
                    "label": label,
                    "config": config,
                    "mode": mode,
                    "backtest_path": str(backtest_path),
                }
            )
            results.append(metrics)

    out_json = Path("analysis/permutation_grid_results.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "scenarios": results,
                "topstep_rules": TOPSTEP_RULES,
            },
            f,
            indent=2,
        )

    df = pd.DataFrame(results)
    df = df.sort_values(["label", "config", "mode"]).reset_index(drop=True)
    out_md = Path("analysis/permutation_grid_results.md")
    with open(out_md, "w") as f:
        f.write("# Permutation Grid Results\n\n")
        cols = df.columns.tolist()
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for _, row in df.iterrows():
            values = []
            for col in cols:
                val = row[col]
                if isinstance(val, float):
                    values.append(f"{val:.6f}")
                else:
                    values.append(str(val))
            f.write("| " + " | ".join(values) + " |\n")
        f.write("\n")

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
