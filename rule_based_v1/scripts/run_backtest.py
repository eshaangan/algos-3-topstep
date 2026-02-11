"""CLI backtest runner for rule-based trading system.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --start 2025-01-01 --end 2025-06-30
    python scripts/run_backtest.py --no-volume-filter --no-lunch-filter
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.ema_trend import EMATrendRule
from rules.time_of_day import TimeOfDayRule
from rules.volume_breakout import VolumeBreakoutRule
from rules.mean_reversion import MeanReversionRule
from rules.rejection_pattern import RejectionPatternRule
from engine.signal_aggregator import SignalAggregator
from engine.risk_manager import RiskManager
from engine.backtest_engine import BacktestEngine
from utils.data_loader import load_bars


def load_config(config_dir: Path) -> dict:
    """Load all config files."""
    configs = {}
    for name in ["rules", "risk", "backtest"]:
        path = config_dir / f"{name}.yaml"
        with open(path) as f:
            configs[name] = yaml.safe_load(f)
    return configs


def build_rules(rules_cfg: dict, overrides: dict = None) -> dict:
    """Instantiate rules from config."""
    overrides = overrides or {}

    ema_cfg = rules_cfg["ema_trend"]
    primary = EMATrendRule(
        fast_period=ema_cfg["fast_period"],
        slow_period=ema_cfg["slow_period"],
        min_spread_atr_ratio=ema_cfg["min_spread_atr_ratio"],
        slope_lookback=ema_cfg["slope_lookback"],
        atr_period=ema_cfg["atr_period"],
    )

    filters = []
    tod_cfg = rules_cfg["time_of_day"]
    filters.append(TimeOfDayRule(
        session_start=tod_cfg["session_start"],
        session_end=tod_cfg["session_end"],
        lunch_filter_enabled=tod_cfg.get("lunch_filter_enabled", False),
        lunch_start=tod_cfg.get("lunch_start", "12:00"),
        lunch_end=tod_cfg.get("lunch_end", "13:00"),
    ))

    if not overrides.get("no_volume_filter"):
        vol_cfg = rules_cfg["volume_breakout"]
        filters.append(VolumeBreakoutRule(
            lookback=vol_cfg["lookback"],
            min_ratio=vol_cfg["min_ratio"],
            max_ratio=vol_cfg["max_ratio"],
        ))

    confirmations = []
    mr_cfg = rules_cfg["mean_reversion"]
    confirmations.append(MeanReversionRule(
        bb_period=mr_cfg["bb_period"],
        bb_std=mr_cfg["bb_std"],
        long_bb_threshold=mr_cfg["long_bb_threshold"],
        short_bb_threshold=mr_cfg["short_bb_threshold"],
        rsi_period=mr_cfg["rsi_period"],
        rsi_long_threshold=mr_cfg["rsi_long_threshold"],
        rsi_short_threshold=mr_cfg["rsi_short_threshold"],
    ))

    rej_cfg = rules_cfg["rejection_pattern"]
    confirmations.append(RejectionPatternRule(
        min_wick_body_ratio=rej_cfg["min_wick_body_ratio"],
    ))

    return {
        "primary": primary,
        "filters": filters,
        "confirmations": confirmations,
    }


def main():
    parser = argparse.ArgumentParser(description="Rule-based backtest runner")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--data-path", type=str, help="Path to HDF5 data file")
    parser.add_argument("--config-dir", type=str, default=None)
    parser.add_argument("--no-volume-filter", action="store_true")
    parser.add_argument("--no-lunch-filter", action="store_true")
    parser.add_argument("--min-confirmations", type=int, default=1)
    parser.add_argument("--starting-equity", type=float, default=50000.0)
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Load configs
    config_dir = Path(args.config_dir) if args.config_dir else Path(__file__).parent.parent / "configs"
    configs = load_config(config_dir)

    # Override lunch filter
    if args.no_lunch_filter:
        configs["rules"]["time_of_day"]["lunch_filter_enabled"] = False

    # Build rules
    rules = build_rules(configs["rules"], {"no_volume_filter": args.no_volume_filter})

    # Build aggregator
    aggregator = SignalAggregator(
        primary_rule=rules["primary"],
        filter_rules=rules["filters"],
        confirmation_rules=rules["confirmations"],
        min_confirmations=args.min_confirmations,
    )

    # Build risk manager
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
        flatten_minutes_before_close=risk_cfg["session"]["flatten_minutes_before_close"],
        drawdown_buffer=risk_cfg["drawdown"]["buffer_from_max"],
    )

    # Build backtest engine
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

    # Load data
    data_path = args.data_path or bt_cfg["data"]["path"]
    start = args.start or bt_cfg["dates"]["start"]
    end = args.end or bt_cfg["dates"]["end"]

    logger.info(f"Loading data from {data_path} ({start} to {end})")
    bars = load_bars(data_path, start_date=start, end_date=end)
    logger.info(f"Loaded {len(bars)} bars")

    # Run backtest
    logger.info("Running backtest...")
    result = engine.run(bars, starting_equity=args.starting_equity)

    # Print results
    summary = result.summary()
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Period:              {start} to {end}")
    print(f"Total bars:          {len(bars)}")
    print(f"Total trades:        {summary['num_trades']}")
    print(f"Win rate:            {summary['win_rate']:.1%}")
    print(f"Profit factor:       {summary['profit_factor']:.2f}")
    print(f"Total P&L:           ${summary['total_pnl']:.2f}")
    print(f"Avg trade P&L:       ${summary['avg_trade_pnl']:.2f}")
    print(f"Sharpe ratio (ann.): {summary['sharpe_ratio']:.2f}")
    print(f"Max drawdown:        ${summary['max_drawdown']:.2f}")
    print(f"Max consec. losses:  {summary['max_consecutive_losses']}")
    print("=" * 60)

    # Exit reason breakdown
    if result.trades:
        reasons = {}
        for t in result.trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        print("\nExit reasons:")
        for reason, count in sorted(reasons.items()):
            print(f"  {reason}: {count}")

    # Daily P&L stats
    if not result.daily_pnl.empty:
        active_days = result.daily_pnl[result.daily_pnl != 0]
        if len(active_days) > 0:
            print(f"\nActive trading days: {len(active_days)}")
            print(f"Avg daily P&L:       ${active_days.mean():.2f}")
            print(f"Best day:            ${active_days.max():.2f}")
            print(f"Worst day:           ${active_days.min():.2f}")
            positive_days = (active_days > 0).sum()
            print(f"Positive days:       {positive_days}/{len(active_days)} ({positive_days/len(active_days):.0%})")
            if summary['num_trades'] > 0:
                days_in_period = (bars.index[-1] - bars.index[0]).days
                print(f"Avg trades/day:      {summary['num_trades'] / max(1, len(active_days)):.1f}")

    # Save to JSON
    if args.output:
        output = {
            "config": {
                "start": start,
                "end": end,
                "data_path": str(data_path),
                "min_confirmations": args.min_confirmations,
                "volume_filter": not args.no_volume_filter,
            },
            "summary": summary,
            "trades": [
                {
                    "entry_bar": t.entry_bar,
                    "exit_bar": t.exit_bar,
                    "direction": t.direction,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "pnl": t.pnl,
                    "exit_reason": t.exit_reason,
                }
                for t in result.trades
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
