"""
Monte Carlo harness to stress-test strategy robustness.

Runs a single backtest to collect realised trades, then bootstraps those
outcomes across many trials to profile equity and drawdown distributions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backtester import Backtester as EMABacktester, build_context as build_ema_context
from backtester_composite import Backtester as CompositeBacktester, build_context as build_composite_context
from config import ProjectConfig, load_config
from utils.monte_carlo import MonteCarloSimulator


def resolve_mode(requested: str, config: ProjectConfig) -> str:
    if requested != "auto":
        return requested
    return "composite" if getattr(config.strategy, "mode", "") == "composite_v1" else "ema"


def run_backtest(cfg: ProjectConfig, data_path: str | None, mode: str):
    if mode == "composite":
        context = build_composite_context(cfg, data_path)
        return CompositeBacktester(context).run()
    context = build_ema_context(cfg, data_path)
    return EMABacktester(context).run()


def build_baseline_payload(stats, initial_equity: float) -> dict:
    ending_equity = stats.equity_curve[-1] if stats.equity_curve else initial_equity
    return {
        "total_trades": stats.total_trades,
        "win_rate": stats.win_rate,
        "profit_factor": stats.profit_factor,
        "avg_r_multiple": stats.avg_r_multiple,
        "expectancy": stats.expectancy,
        "max_drawdown": stats.max_drawdown,
        "max_daily_drawdown": stats.max_daily_drawdown,
        "ending_equity": ending_equity,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo simulator for the Topstep strategies.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML.")
    parser.add_argument("--data", type=str, default=None, help="Optional CSV override.")
    parser.add_argument(
        "--mode",
        choices=["auto", "ema", "composite"],
        default="auto",
        help="Force a strategy mode. 'auto' follows StrategyConfig.mode.",
    )
    parser.add_argument("--runs", type=int, default=500, help="Number of Monte Carlo trials.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Trades to draw per trial (defaults to the realised trade count).",
    )
    parser.add_argument("--random-seed", type=int, default=13, help="Seed for reproducible sampling.")
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/monte_carlo.json",
        help="Where to save the Monte Carlo report (JSON).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    mode = resolve_mode(args.mode, cfg)
    stats = run_backtest(cfg, args.data, mode)

    simulator = MonteCarloSimulator(stats.trades, cfg.backtest.initial_equity, seed=args.random_seed)
    mc_result = simulator.run(runs=args.runs, sample_size=args.sample_size)
    summary = mc_result["summary"]

    baseline = build_baseline_payload(stats, cfg.backtest.initial_equity)
    eq = summary["ending_equity"]
    dd = summary["max_drawdown"]
    win = summary["win_rate"]

    print(
        f"[BASELINE] mode={mode} trades={stats.total_trades} "
        f"PF={stats.profit_factor:.2f} win={stats.win_rate:.2%} MDD={stats.max_drawdown:.0f}"
    )
    print(
        f"[MC] equity p05={eq['p05']:.0f} p50={eq['p50']:.0f} p95={eq['p95']:.0f} "
        f"(runs={mc_result['runs']} sample={mc_result['sample_size']})"
    )
    print(
        f"[MC] max DD p05={dd['p05']:.0f} p50={dd['p50']:.0f} p95={dd['p95']:.0f} | "
        f"win rate p05={win['p05']:.2%} p50={win['p50']:.2%} p95={win['p95']:.2%}"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": args.config,
        "data": args.data,
        "mode": mode,
        "baseline": baseline,
        "monte_carlo": mc_result,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nMonte Carlo report saved to {output_path}")


if __name__ == "__main__":
    main()

