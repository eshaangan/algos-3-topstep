"""Walk-forward validation for rule-based trading system.

Rolls through data in train_months/test_months windows.
Since rules are parameter-based (not trained), this validates
that performance is consistent across different market regimes.

Usage:
    python scripts/run_walkforward.py
    python scripts/run_walkforward.py --train-months 6 --test-months 1
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_backtest import build_rules, load_config
from engine.signal_aggregator import SignalAggregator
from engine.risk_manager import RiskManager
from engine.backtest_engine import BacktestEngine
from utils.data_loader import load_bars


def main():
    parser = argparse.ArgumentParser(description="Walk-forward validation")
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--config-dir", type=str)
    parser.add_argument("--output", type=str, default="walkforward_results.json")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    config_dir = Path(args.config_dir) if args.config_dir else Path(__file__).parent.parent / "configs"
    configs = load_config(config_dir)

    data_path = args.data_path or configs["backtest"]["data"]["path"]
    bars = load_bars(data_path)
    logger.info(f"Loaded {len(bars)} total bars")

    # Generate test windows
    start_date = bars.index[0]
    end_date = bars.index[-1]

    windows = []
    current = start_date + pd.DateOffset(months=args.train_months)
    while current + pd.DateOffset(months=args.test_months) <= end_date:
        test_start = current
        test_end = current + pd.DateOffset(months=args.test_months)
        # For rules, "training" is just the lookback for indicators
        # We use train window only to ensure enough warmup
        train_start = current - pd.DateOffset(months=args.train_months)
        windows.append({
            "train_start": train_start,
            "test_start": test_start,
            "test_end": test_end,
        })
        current += pd.DateOffset(months=args.step_months)

    logger.info(f"Generated {len(windows)} walk-forward windows")

    # Run each window
    results = []
    for i, window in enumerate(windows):
        test_start = window["test_start"]
        test_end = window["test_end"]

        logger.info(f"Window {i+1}/{len(windows)}: test {test_start.date()} to {test_end.date()}")

        # Filter bars for this window (include warmup)
        warmup_start = test_start - pd.DateOffset(days=30)  # 30 days warmup
        window_bars = bars[(bars.index >= warmup_start) & (bars.index < test_end)]

        if len(window_bars) < 100:
            logger.warning(f"Skipping window {i+1}: only {len(window_bars)} bars")
            continue

        # Build fresh system
        rules = build_rules(configs["rules"])
        aggregator = SignalAggregator(
            primary_rule=rules["primary"],
            filter_rules=rules["filters"],
            confirmation_rules=rules["confirmations"],
        )
        risk_cfg = configs["risk"]
        risk_manager = RiskManager(
            contracts=risk_cfg["position"]["contracts"],
            point_value=risk_cfg["position"]["point_value"],
            tick_size=risk_cfg["position"]["tick_size"],
            tick_value=risk_cfg["position"]["tick_value"],
            max_daily_loss=risk_cfg["daily_limits"]["max_daily_loss"],
            per_trade_max_loss=risk_cfg["daily_limits"]["per_trade_max_loss"],
            max_consecutive_losses=risk_cfg["circuit_breaker"]["max_consecutive_losses"],
            cooldown_bars=risk_cfg["circuit_breaker"]["cooldown_bars"],
        )
        exit_cfg = configs["rules"]["exit_strategy"]
        bt_cfg = configs["backtest"]
        engine = BacktestEngine(
            aggregator=aggregator,
            risk_manager=risk_manager,
            commission_per_side=bt_cfg["costs"]["commission_per_side"],
            slippage_ticks=bt_cfg["costs"]["slippage_ticks"],
            profit_target_atr=exit_cfg["profit_target_atr"],
            stop_loss_atr=exit_cfg["stop_loss_atr"],
            time_stop_bars=exit_cfg["time_stop_bars"],
            trailing_activation_atr=exit_cfg["trailing_activation_atr"],
            trailing_distance_atr=exit_cfg["trailing_distance_atr"],
        )

        result = engine.run(window_bars)
        summary = result.summary()

        # Only count trades in the test period
        test_trades = [
            t for t in result.trades
            if window_bars.index[t.entry_bar] >= test_start
        ]
        test_pnl = sum(t.pnl for t in test_trades)

        window_result = {
            "window": i + 1,
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "total_trades": len(test_trades),
            "test_pnl": test_pnl,
            **summary,
        }
        results.append(window_result)

        side = "LONG" if test_pnl > 0 else "SHORT" if test_pnl < 0 else "FLAT"
        logger.info(
            f"  Trades: {len(test_trades)}, P&L: ${test_pnl:.2f}, "
            f"WR: {summary['win_rate']:.0%}, Sharpe: {summary['sharpe_ratio']:.2f}"
        )

    # Summary
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("=" * 80)
    print(f"{'Window':<10} {'Period':<25} {'Trades':<8} {'P&L':>10} {'WR':>8} {'Sharpe':>8} {'PF':>8}")
    print("-" * 80)

    total_pnl = 0
    positive_windows = 0
    for r in results:
        total_pnl += r["test_pnl"]
        if r["test_pnl"] > 0:
            positive_windows += 1
        print(
            f"{r['window']:<10} {r['test_start']} - {r['test_end']:<12} "
            f"{r['total_trades']:<8} ${r['test_pnl']:>9.2f} "
            f"{r['win_rate']:>7.0%} {r['sharpe_ratio']:>7.2f} {r['profit_factor']:>7.2f}"
        )

    print("-" * 80)
    print(f"Total P&L: ${total_pnl:.2f}")
    print(f"Positive windows: {positive_windows}/{len(results)}")
    if results:
        avg_sharpe = sum(r["sharpe_ratio"] for r in results) / len(results)
        print(f"Avg Sharpe: {avg_sharpe:.2f}")

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
