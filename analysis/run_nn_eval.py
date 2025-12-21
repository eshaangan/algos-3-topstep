"""
Quick end-to-end NN evaluation on the provided H5 data.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backtesting.backtest import run_backtest_nn
from core.selection import bars_per_day
from core.risk_presets import RISK_PRESET_NAME, get_risk_config
from data.clean_bars import clean_bars
from models.nn_inference import load_nn_bundle, predict_scores_for_bars, validate_nn_config

RISK_CONFIG = get_risk_config(RISK_PRESET_NAME)


def _load_bars(h5_path: str) -> pd.DataFrame:
    with pd.HDFStore(h5_path, "r") as store:
        bars = store["bars_5min"].copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    bars = clean_bars(bars, tick_size=RISK_CONFIG.tick_size, verbose=False)
    assert bars["timestamp"].is_monotonic_increasing, "Bars must be time-ordered"
    return bars


def _ensure_model(
    data_path: str,
    model_dir: str,
    *,
    fold: int = 0,
    force_train: bool = False,
) -> None:
    model_path = Path(model_dir) / f"fold_{fold}" / "config.json"
    if model_path.exists() and not force_train:
        try:
            cfg = json.loads(model_path.read_text())
            validate_nn_config(cfg.get("nn_config", {}))
            return
        except Exception as exc:
            print(f"Model config invalid ({exc}); retraining.")
    cmd = [
        sys.executable,
        str(project_root / "models" / "nn_train.py"),
        "--data-path",
        data_path,
        "--output-dir",
        model_dir,
    ]
    subprocess.run(cmd, check=True)


def _score_report(scores: pd.Series) -> Dict[str, float]:
    percentiles = [50, 90, 95, 97, 98, 99, 99.5]
    if scores.empty:
        return {}
    return {f"p{p}": float(np.nanpercentile(scores, p)) for p in percentiles}


def _trade_distribution(trades: pd.DataFrame) -> Dict[str, int]:
    if trades.empty:
        return {}
    trades["day"] = pd.to_datetime(trades["entry_time"], utc=True).dt.date
    counts = trades.groupby("day")["entry_time"].count()
    return counts.value_counts().sort_index().to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NN training + backtest evaluation")
    parser.add_argument("--data-path", default="/mnt/data/es_bars_2010_2025.h5")
    parser.add_argument("--model-dir", default="models/nn_saved")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()

    _ensure_model(args.data_path, args.model_dir, fold=args.fold, force_train=args.force_train)

    bundle = load_nn_bundle(args.model_dir, fold=args.fold)
    nn_cfg = bundle.config.get("nn_config", {})
    risk_cfg = bundle.config.get("risk_config", {})
    session_start = RISK_CONFIG.session_start
    session_end = RISK_CONFIG.session_end
    if isinstance(risk_cfg.get("session_start"), str):
        session_start = pd.Timestamp(risk_cfg["session_start"]).time()
    if isinstance(risk_cfg.get("session_end"), str):
        session_end = pd.Timestamp(risk_cfg["session_end"]).time()

    bars = _load_bars(args.data_path)
    prob_df = predict_scores_for_bars(bars, bundle)

    score_series = prob_df["score"].dropna()
    score_stats = _score_report(score_series)

    results = run_backtest_nn(
        bars,
        prob_df,
        score_threshold=float(nn_cfg["score_threshold"]),
        max_trades_per_day=int(nn_cfg["max_trades_per_day"]),
        min_bars_between_trades=int(nn_cfg["min_bars_between_trades"]),
        enable_long=bool(nn_cfg["enable_long"]),
        enable_short=bool(nn_cfg["enable_short"]),
        horizon_bars=int(nn_cfg["horizon_bars"]),
        execution_mode=str(nn_cfg["execution_mode"]),
        exit_price_mode=str(nn_cfg["exit_price_mode"]),
        session_mode=str(nn_cfg["session_mode"]),
        deadline_time=nn_cfg.get("deadline_time"),
        deadline_relax_factor=float(nn_cfg.get("deadline_relax_factor", 0.98)),
        bar_minutes=int(nn_cfg["bar_minutes"]),
        session_start=session_start,
        session_end=session_end,
        stop_loss_ticks=int(nn_cfg["stop_loss_ticks"]),
        target_multiplier=float(nn_cfg["target_multiplier"]),
        max_hold_bars=int(nn_cfg["max_hold_bars"]),
        tick_size=float(nn_cfg["tick_size"]),
        tick_value=float(nn_cfg["tick_value"]),
    )

    trades_df = pd.DataFrame(results.get("trades", []))
    dist = _trade_distribution(trades_df)

    print("\n" + "=" * 60)
    print("NN EVALUATION SUMMARY")
    print("=" * 60)
    print("\nScore percentiles:")
    print(json.dumps(score_stats, indent=2))
    print("\nBacktest summary:")
    print(json.dumps(results["summary"], indent=2))
    print("\nDaily stats:")
    print(json.dumps(results["daily_stats"], indent=2))
    print("\nTrades/day distribution (count -> days):")
    print(json.dumps(dist, indent=2))

    avg_trades = results["daily_stats"].get("avg_trades_per_day", 0.0)
    if avg_trades < 1.0:
        print("\nWARNING: avg trades/day below 1.0")
        print(
            f"score_threshold={nn_cfg['score_threshold']:.4f} "
            f"score_quantile={nn_cfg['score_quantile']:.4f}"
        )
        bars_day = bars_per_day(
            session_mode=str(nn_cfg.get("session_mode", "RTH")),
            session_start=session_start,
            session_end=session_end,
            bar_minutes=int(nn_cfg.get("bar_minutes", 5)),
        )
        print(f"bars_per_day={bars_day} target_trades_per_day={nn_cfg.get('target_trades_per_day')}")


if __name__ == "__main__":
    main()
