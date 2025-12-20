"""
Quick pipeline test: prepare data (if needed), train on a subset, backtest.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import time
from pathlib import Path
import sys

import joblib
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.backtest import run_backtest
from core.simple_config import RISK_CONFIG, TRAINING_CONFIG
from data.prepare_dataset import prepare_training_data
from models.train import load_bars, train_models


def _maybe_prepare_data(data_path: str) -> None:
    if Path(data_path).exists():
        return
    prepare_training_data(output_path=data_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end pipeline test")
    parser.add_argument("--data-path", default="data/processed/mes_bars.h5")
    parser.add_argument("--bars", type=int, default=5000)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--save-trades", default=None)
    args = parser.parse_args()

    _maybe_prepare_data(args.data_path)

    bars = load_bars(args.data_path)

    if args.bars:
        bars = bars.tail(args.bars).reset_index(drop=True)

    results = train_models(bars)
    policy = results.get("policy", {})
    backtest = run_backtest(
        bars,
        results["long_model"],
        results["short_model"],
        results["feature_cols"],
        save_trades_path=args.save_trades,
        min_probability_long=policy.get("min_probability_long"),
        min_probability_short=policy.get("min_probability_short"),
        enable_long=policy.get("enable_long"),
        enable_short=policy.get("enable_short"),
    )

    print("\nQuick pipeline results:")
    print(json.dumps(backtest["summary"], indent=2))
    if results.get("backtest_metrics"):
        print("\nSplit backtest summaries (trade-identical evaluation):")
        summaries = {}
        for split_name, split_res in results["backtest_metrics"].items():
            summaries[split_name] = split_res.get("summary", {})
        print(json.dumps({"policy": policy, "splits": summaries}, indent=2))

    if args.save_models:
        output_dir = Path("models/saved")
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(results["long_model"], output_dir / "model_long.joblib")
        joblib.dump(results["short_model"], output_dir / "model_short.joblib")

        # Convert config to dict and serialize time objects to strings
        risk_dict = asdict(RISK_CONFIG)
        training_dict = asdict(TRAINING_CONFIG)
        
        # Convert time objects to strings for JSON serialization
        if "session_start" in risk_dict and isinstance(risk_dict["session_start"], time):
            risk_dict["session_start"] = risk_dict["session_start"].isoformat()
        if "session_end" in risk_dict and isinstance(risk_dict["session_end"], time):
            risk_dict["session_end"] = risk_dict["session_end"].isoformat()
        
        metadata = {
            "config": {"risk": risk_dict, "training": training_dict},
            "feature_cols": results["feature_cols"],
            "metrics": results["metrics"],
        }
        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved models to {output_dir}")


if __name__ == "__main__":
    main()
