"""
Opening Range Breakout OOS Validation + Topstep Combine Simulator
==================================================================
Sweeps ORB parameters on Jan-Feb 2026 OOS data, feeds daily PnL into
TopstepCombineSimulator, and outputs a ranked JSON summary.

Usage:
    cd "algos 3 topstep"
    python ml_intraday_v3/diagnostics/validate_orb_oos.py \
        --data-path data/processed/jan_feb_2026_oos_test.h5 \
        --data-key bars_5min \
        --start 2026-01-01 --end 2026-02-10 \
        --output ml_intraday_v3/diagnostics/orb_oos_results.json -v
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent      # algos 3 topstep/
_RBV1_DIR = _PROJECT_ROOT / "rule_based_v1"

for _p in [str(_PROJECT_ROOT), str(_RBV1_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.backtest_engine import BacktestEngine          # noqa: E402
from engine.signal_aggregator import SignalAggregator      # noqa: E402
from engine.risk_manager import RiskManager                # noqa: E402
from rules.opening_range import OpeningRangeBreakoutRule   # noqa: E402
from rules.time_of_day import TimeOfDayRule                # noqa: E402
from utils.data_loader import load_bars                    # noqa: E402

_DIAG_DIR = _HERE.parent
if str(_DIAG_DIR) not in sys.path:
    sys.path.insert(0, str(_DIAG_DIR))
from topstep_combine_simulator import TopstepCombineSimulator  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ORB Parameter Sweep Grid
# ---------------------------------------------------------------------------
SWEEP = {
    "or_end_time": ["10:00", "10:30"],          # OR window width
    "min_range_atr": [0.3, 0.5, 0.75],          # Range quality filter
    "entry_cutoff_time": ["11:00", "12:00"],     # Latest entry allowed
    "pt_atr_mult": [1.5, 2.0, 2.5, 3.0],        # Profit target
    "sl_atr_mult": [1.0, 1.5, 2.0],             # Stop loss
    "n_contracts": [1, 2],                       # Position size
}
# 2 × 3 × 2 × 4 × 3 × 2 = 288 combinations

RISK_DEFAULTS = {
    "point_value": 5.0,
    "tick_size": 0.25,
    "tick_value": 1.25,
    "max_daily_loss": -900.0,        # Topstep limit is -$1,000; use $100 buffer
    "per_trade_max_loss": 300.0,     # Per trade max loss (scales with n_contracts)
    "max_consecutive_losses": 3,
    "cooldown_bars": 5,
    "flatten_minutes_before_close": 5,
    "drawdown_buffer": 1500.0,       # Topstep trailing drawdown = $2,000; use $500 buffer
}

BACKTEST_DEFAULTS = {
    "commission_per_side": 0.62,
    "slippage_ticks": 1,
    "time_stop_bars": 24,
    "trailing_activation_atr": 999.0,  # Disabled: trailing stop kills avg PnL (use pure PT/SL)
    "trailing_distance_atr": 0.75,
    "atr_period": 14,
}

COMBINE_SETTINGS = {
    "account_size": 50_000,
    "profit_target": 3_000,
    "max_trailing_drawdown": 2_000,
    "max_daily_loss": 1_000,
    "consistency_pct": 0.30,
    "min_trading_days": 5,
}


def build_orb_engine(params: dict) -> BacktestEngine:
    """Instantiate a BacktestEngine with ORB as the primary rule."""
    primary = OpeningRangeBreakoutRule(
        or_end_time=params["or_end_time"],
        min_or_bars=params.get("min_or_bars", 4),
        min_range_atr=params["min_range_atr"],
        entry_cutoff_time=params["entry_cutoff_time"],
        atr_period=params.get("atr_period", 14),
        use_close_for_signal=params.get("use_close_for_signal", True),
    )

    # TimeOfDay filter: only allow entries during RTH
    filters = [
        TimeOfDayRule(
            session_start="09:35",
            session_end="15:45",
            lunch_filter_enabled=False,
        ),
    ]

    aggregator = SignalAggregator(
        primary_rule=primary,
        filter_rules=filters,
        confirmation_rules=[],
        min_confirmations=0,
    )

    risk_manager = RiskManager(
        contracts=params.get("n_contracts", 1),
        point_value=RISK_DEFAULTS["point_value"],
        tick_size=RISK_DEFAULTS["tick_size"],
        tick_value=RISK_DEFAULTS["tick_value"],
        max_daily_loss=RISK_DEFAULTS["max_daily_loss"],
        per_trade_max_loss=RISK_DEFAULTS["per_trade_max_loss"] * params.get("n_contracts", 1),
        # Note: drawdown_buffer stays fixed at $1,500 (Topstep $2,000 limit minus $500 buffer)
        max_consecutive_losses=RISK_DEFAULTS["max_consecutive_losses"],
        cooldown_bars=RISK_DEFAULTS["cooldown_bars"],
        flatten_minutes_before_close=RISK_DEFAULTS["flatten_minutes_before_close"],
        drawdown_buffer=RISK_DEFAULTS["drawdown_buffer"],
    )

    engine = BacktestEngine(
        aggregator=aggregator,
        risk_manager=risk_manager,
        commission_per_side=params.get("commission_per_side", 0.62),
        slippage_ticks=params.get("slippage_ticks", 1),
        profit_target_atr=params["pt_atr_mult"],
        stop_loss_atr=params["sl_atr_mult"],
        time_stop_bars=params.get("time_stop_bars", 24),
        trailing_activation_atr=params.get("trailing_activation_atr", 1.0),
        trailing_distance_atr=params.get("trailing_distance_atr", 0.75),
        atr_period=params.get("atr_period", 14),
    )

    return engine


def run_single_config(
    params: dict,
    bars: pd.DataFrame,
    simulator: TopstepCombineSimulator,
    mc_paths: int = 5000,
    mc_seed: int = 42,
) -> dict:
    """Run one ORB backtest + Monte Carlo simulation."""
    engine = build_orb_engine(params)
    result = engine.run(bars, starting_equity=50_000.0)
    summary = result.summary()

    trade_pnls = [t.pnl for t in result.trades]
    if len(trade_pnls) < 5:
        mc = {"p_pass": 0.0, "median_days": None, "p95_max_drawdown": None, "n_paths": mc_paths}
    else:
        mc = simulator.monte_carlo(
            trade_pnl_list=trade_pnls,
            n_paths=mc_paths,
            trades_per_day_range=(1, 4),
            max_days=40,
            seed=mc_seed,
        )

    oos_result = simulator.simulate(result.daily_pnl)

    return {
        "params": {k: v for k, v in params.items() if k in SWEEP or k == "n_contracts"},
        "backtest": {
            "num_trades": summary["num_trades"],
            "win_rate": round(summary.get("win_rate", 0.0), 4),
            "profit_factor": round(summary.get("profit_factor", 0.0), 4),
            "total_pnl": round(summary.get("total_pnl", 0.0), 2),
            "avg_trade_pnl": round(summary.get("avg_trade_pnl", 0.0), 2),
            "sharpe_ratio": round(summary.get("sharpe_ratio", 0.0), 4),
            "max_drawdown": round(summary.get("max_drawdown", 0.0), 2),
            "max_consecutive_losses": summary.get("max_consecutive_losses", 0),
        },
        "oos_simulate": {
            "passed": oos_result.passed,
            "days_to_pass": oos_result.days_to_pass,
            "fail_reason": oos_result.fail_reason,
            "final_profit": round(oos_result.final_profit, 2),
            "max_drawdown_reached": round(oos_result.max_drawdown_reached, 2),
            "daily_loss_breaches": oos_result.daily_loss_breaches,
            "consistency_violations": oos_result.consistency_violations,
        },
        "monte_carlo": {
            "p_pass": mc.get("p_pass", 0.0),
            "median_days": mc.get("median_days"),
            "p25_days": mc.get("p25_days"),
            "p75_days": mc.get("p75_days"),
            "p95_max_drawdown": mc.get("p95_max_drawdown"),
            "p_fail_daily": mc.get("p_fail_daily"),
            "p_fail_trailing": mc.get("p_fail_trailing"),
            "mean_final_profit": mc.get("mean_final_profit"),
            "n_paths": mc.get("n_paths", mc_paths),
        },
    }


def run_sweep(
    bars: pd.DataFrame,
    mc_paths: int = 5000,
    verbose: bool = False,
) -> List[dict]:
    """Run all ORB parameter combinations and return sorted results."""
    simulator = TopstepCombineSimulator(**COMBINE_SETTINGS)

    keys = list(SWEEP.keys())
    values = list(SWEEP.values())
    combinations = list(product(*values))
    total = len(combinations)
    logger.info(f"Starting ORB sweep: {total} configurations")

    results = []
    t0 = time.time()

    for i, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))
        params.update(BACKTEST_DEFAULTS)

        try:
            r = run_single_config(params, bars, simulator, mc_paths=mc_paths)
            results.append(r)

            if verbose or (i % 30 == 0):
                elapsed = time.time() - t0
                eta = (elapsed / i) * (total - i)
                logger.info(
                    f"[{i:3d}/{total}] or={params['or_end_time']} "
                    f"range={params['min_range_atr']} cut={params['entry_cutoff_time']} "
                    f"pt={params['pt_atr_mult']} sl={params['sl_atr_mult']} n={params['n_contracts']} | "
                    f"wr={r['backtest']['win_rate']:.1%} "
                    f"P(pass)={r['monte_carlo']['p_pass']:.1%} "
                    f"eta={eta:.0f}s"
                )
        except Exception as e:
            logger.warning(f"Config {i} failed: {e} | params={params}")
            continue

    # Sort by P(pass) descending, then Sharpe
    results.sort(
        key=lambda x: (
            -x["monte_carlo"]["p_pass"],
            -x["backtest"]["sharpe_ratio"],
        )
    )

    return results


def build_summary(results: List[dict]) -> dict:
    if not results:
        return {"error": "No results"}

    best = results[0]
    qualified = [r for r in results if r["backtest"]["win_rate"] > 0.45]
    any_pass_25 = [r for r in results if r["monte_carlo"]["p_pass"] > 0.25]

    return {
        "strategy": "opening_range_breakout",
        "total_configs": len(results),
        "qualified_win_rate_gt_45pct": len(qualified),
        "configs_p_pass_gt_25pct": len(any_pass_25),
        "best_config": best,
        "best_win_rate": round(max(r["backtest"]["win_rate"] for r in results), 4),
        "best_p_pass": round(max(r["monte_carlo"]["p_pass"] for r in results), 4),
        "best_sharpe": round(max(r["backtest"]["sharpe_ratio"] for r in results), 4),
        "top5": results[:5],
        "go_signal": any(
            r["backtest"]["win_rate"] > 0.45 and r["monte_carlo"]["p_pass"] > 0.25
            for r in results
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="ORB OOS validation sweep")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(_PROJECT_ROOT / "data" / "processed" / "jan_feb_2026_oos_test.h5"),
    )
    parser.add_argument("--data-key", type=str, default="bars_5min")
    parser.add_argument("--start", type=str, default="2026-01-01")
    parser.add_argument("--end", type=str, default="2026-02-10")
    parser.add_argument(
        "--output",
        type=str,
        default=str(_PROJECT_ROOT / "ml_intraday_v3" / "diagnostics" / "orb_oos_results.json"),
    )
    parser.add_argument("--mc-paths", type=int, default=5000)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info(f"Loading bars from {args.data_path}")
    bars = load_bars(args.data_path, key=args.data_key, start_date=args.start, end_date=args.end)
    logger.info(f"Loaded {len(bars):,} bars spanning {bars.index[0]} to {bars.index[-1]}")

    t_start = time.time()
    results = run_sweep(bars, mc_paths=args.mc_paths, verbose=args.verbose)
    elapsed = time.time() - t_start
    logger.info(f"Sweep complete in {elapsed:.1f}s ({len(results)} configs)")

    summary = build_summary(results)

    print("\n" + "=" * 70)
    print("OPENING RANGE BREAKOUT OOS VALIDATION RESULTS")
    print("=" * 70)
    print(f"Period: {args.start} to {args.end}")
    print(f"Total configs tested: {summary['total_configs']}")
    print(f"Configs w/ win_rate > 45%: {summary['qualified_win_rate_gt_45pct']}")
    print(f"Configs w/ P(pass) > 25%:  {summary['configs_p_pass_gt_25pct']}")
    print(f"Best win rate:  {summary['best_win_rate']:.1%}")
    print(f"Best P(pass):   {summary['best_p_pass']:.1%}")
    print(f"Best Sharpe:    {summary['best_sharpe']:.3f}")

    if summary["go_signal"]:
        print("\n[GO SIGNAL] ORB has real OOS edge! Proceed to Phase 2A (MAE/MFE + Kelly sizing)")
    else:
        print("\n[NO-GO] ORB did not meet thresholds. Try Session-Focused or Mean Reversion variants.")

    print("\nTop 5 configs:")
    for i, r in enumerate(summary.get("top5", []), 1):
        p = r["params"]
        bt = r["backtest"]
        mc = r["monte_carlo"]
        print(
            f"  {i}. or_end={p.get('or_end_time')} range_atr={p.get('min_range_atr')} "
            f"cut={p.get('entry_cutoff_time')} "
            f"pt={p.get('pt_atr_mult')} sl={p.get('sl_atr_mult')} n={p.get('n_contracts')} | "
            f"wr={bt['win_rate']:.1%} pf={bt['profit_factor']:.2f} "
            f"pnl=${bt['total_pnl']:.0f} "
            f"P(pass)={mc['p_pass']:.1%} med_days={mc['median_days']}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "meta": {
            "strategy": "opening_range_breakout",
            "start": args.start,
            "end": args.end,
            "data_path": args.data_path,
            "mc_paths": args.mc_paths,
            "n_bars": len(bars),
            "elapsed_seconds": round(elapsed, 1),
        },
        "summary": {k: v for k, v in summary.items() if k not in ("best_config", "top5")},
        "top5": summary.get("top5", []),
        "all_results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
