"""Parameter robustness testing via systematic sweep.

Tests each parameter at ±20% to verify the strategy remains profitable.

Usage:
    python scripts/run_parameter_sweep.py
    python scripts/run_parameter_sweep.py --variation 0.20 --output sweep_results.json
"""

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_backtest import build_rules, load_config
from engine.signal_aggregator import SignalAggregator
from engine.risk_manager import RiskManager
from engine.backtest_engine import BacktestEngine
from utils.data_loader import load_bars


# Parameters to sweep with their config paths
SWEEP_PARAMS = [
    ("ema_trend.fast_period", int),
    ("ema_trend.slow_period", int),
    ("ema_trend.min_spread_atr_ratio", float),
    ("volume_breakout.min_ratio", float),
    ("volume_breakout.max_ratio", float),
    ("mean_reversion.long_bb_threshold", float),
    ("mean_reversion.rsi_long_threshold", float),
    ("rejection_pattern.min_wick_body_ratio", float),
    ("exit_strategy.profit_target_atr", float),
    ("exit_strategy.stop_loss_atr", float),
    ("exit_strategy.time_stop_bars", int),
    ("exit_strategy.trailing_activation_atr", float),
    ("exit_strategy.trailing_distance_atr", float),
]


def get_nested(d: dict, path: str):
    keys = path.split(".")
    for k in keys:
        d = d[k]
    return d


def set_nested(d: dict, path: str, value):
    keys = path.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def run_single_backtest(configs: dict, bars, starting_equity: float = 50000.0) -> dict:
    """Run a single backtest and return summary."""
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
    result = engine.run(bars, starting_equity=starting_equity)
    return result.summary()


def main():
    parser = argparse.ArgumentParser(description="Parameter robustness sweep")
    parser.add_argument("--variation", type=float, default=0.20, help="Variation fraction (default 0.20 = ±20%%)")
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--config-dir", type=str)
    parser.add_argument("--output", type=str, default="parameter_sweep_results.json")
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
    logger.info(f"Loaded {len(bars)} bars")

    # Baseline
    logger.info("Running baseline...")
    baseline = run_single_backtest(configs, bars)
    logger.info(f"Baseline: P&L=${baseline['total_pnl']:.2f}, Sharpe={baseline['sharpe_ratio']:.2f}")

    # Sweep
    results = {"baseline": baseline, "sweeps": []}

    for param_path, param_type in SWEEP_PARAMS:
        base_value = get_nested(configs["rules"], param_path)
        low_value = param_type(base_value * (1.0 - args.variation))
        high_value = param_type(base_value * (1.0 + args.variation))

        # Ensure integer params stay sane
        if param_type == int:
            low_value = max(1, low_value)

        for variant_name, variant_value in [("low", low_value), ("high", high_value)]:
            modified = copy.deepcopy(configs)
            set_nested(modified["rules"], param_path, variant_value)

            logger.info(f"Testing {param_path} = {variant_value} ({variant_name})")
            try:
                summary = run_single_backtest(modified, bars)
                result = {
                    "parameter": param_path,
                    "variant": variant_name,
                    "base_value": base_value,
                    "test_value": variant_value,
                    **summary,
                }
                results["sweeps"].append(result)

                status = "PASS" if summary["total_pnl"] > 0 else "FAIL"
                logger.info(
                    f"  [{status}] P&L=${summary['total_pnl']:.2f}, "
                    f"Sharpe={summary['sharpe_ratio']:.2f}, Trades={summary['num_trades']}"
                )
            except Exception as e:
                logger.error(f"  ERROR: {e}")
                results["sweeps"].append({
                    "parameter": param_path,
                    "variant": variant_name,
                    "base_value": base_value,
                    "test_value": variant_value,
                    "error": str(e),
                })

    # Summary table
    print("\n" + "=" * 100)
    print("PARAMETER SWEEP RESULTS")
    print("=" * 100)
    print(f"{'Parameter':<40} {'Variant':<6} {'Value':>10} {'P&L':>10} {'Sharpe':>8} {'Trades':>8}")
    print("-" * 100)

    all_profitable = True
    all_sharpe_positive = True
    for r in results["sweeps"]:
        if "error" in r:
            print(f"{r['parameter']:<40} {r['variant']:<6} {r['test_value']:>10} ERROR: {r['error']}")
            all_profitable = False
            continue

        status = "+" if r["total_pnl"] > 0 else "-"
        print(
            f"{r['parameter']:<40} {r['variant']:<6} {r['test_value']:>10.4g} "
            f"${r['total_pnl']:>9.2f} {r['sharpe_ratio']:>7.2f} {r['num_trades']:>8}"
        )
        if r["total_pnl"] <= 0:
            all_profitable = False
        if r["sharpe_ratio"] <= 0:
            all_sharpe_positive = False

    print("-" * 100)
    print(f"All variants profitable: {'YES' if all_profitable else 'NO'}")
    print(f"All variants Sharpe > 0: {'YES' if all_sharpe_positive else 'NO'}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
