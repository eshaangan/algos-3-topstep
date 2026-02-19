"""
Rule-Based OOS Validation + Topstep Combine Simulator
======================================================
Runs a parameter sweep over rule_based_v1 configurations on Jan-Feb 2026 OOS data,
feeds daily PnL into TopstepCombineSimulator, and outputs a ranked JSON summary.

Usage:
    cd "algos 3 topstep"
    python ml_intraday_v3/diagnostics/validate_rule_based_oos.py \
        --data-path data/processed/mes_bars_databento_rth.h5 \
        --start 2026-01-01 --end 2026-02-14 \
        --output ml_intraday_v3/diagnostics/rule_based_oos_results.json

Requirements:
    - rule_based_v1/ must be importable from project root
    - HDF5 data file must exist at --data-path (or default path)

Parameter sweep: 3 x 3 x 4 x 3 x 2 = 216 combinations (runs locally in ~5 min).
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
# Path setup: add project root and rule_based_v1 to sys.path
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent      # algos 3 topstep/
_RBV1_DIR = _PROJECT_ROOT / "rule_based_v1"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_RBV1_DIR) not in sys.path:
    sys.path.insert(0, str(_RBV1_DIR))

# Import rule_based_v1 components
from engine.backtest_engine import BacktestEngine          # noqa: E402
from engine.signal_aggregator import SignalAggregator      # noqa: E402
from engine.risk_manager import RiskManager                # noqa: E402
from rules.ema_trend import EMATrendRule                   # noqa: E402
from rules.time_of_day import TimeOfDayRule                # noqa: E402
from rules.volume_breakout import VolumeBreakoutRule       # noqa: E402
from rules.mean_reversion import MeanReversionRule         # noqa: E402
from rules.rejection_pattern import RejectionPatternRule   # noqa: E402
from utils.data_loader import load_bars                    # noqa: E402

# Import our simulator
_DIAG_DIR = _HERE.parent
if str(_DIAG_DIR) not in sys.path:
    sys.path.insert(0, str(_DIAG_DIR))
from topstep_combine_simulator import TopstepCombineSimulator, simulate_from_trades  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sweep grid (plan spec)
# ---------------------------------------------------------------------------
SWEEP = {
    "ema_fast": [8, 13, 21],
    "ema_slow": [34, 55, 89],
    "pt_atr_mult": [1.5, 2.0, 2.5, 3.0],
    "sl_atr_mult": [1.0, 1.5, 2.0],
    "n_contracts": [1, 2],
}

# Fixed risk settings for combine (Topstep 50k)
RISK_DEFAULTS = {
    "point_value": 5.0,
    "tick_size": 0.25,
    "tick_value": 1.25,
    "max_daily_loss": -800.0,    # conservative combine buffer (real limit $1k)
    "per_trade_max_loss": 200.0,
    "max_consecutive_losses": 3,
    "cooldown_bars": 5,
    "flatten_minutes_before_close": 5,
    "drawdown_buffer": 500.0,
}

BACKTEST_DEFAULTS = {
    "commission_per_side": 0.62,
    "slippage_ticks": 1,
    "time_stop_bars": 24,
    "trailing_activation_atr": 1.0,
    "trailing_distance_atr": 0.75,
    "session_start": "09:35",
    "session_end": "15:45",
    "min_spread_atr_ratio": 0.3,
    "slope_lookback": 3,
    "atr_period": 14,
    "bb_period": 20,
    "bb_std": 2.0,
    "rsi_period": 14,
    "rsi_long_threshold": 35.0,
    "rsi_short_threshold": 65.0,
    "long_bb_threshold": 0.3,
    "short_bb_threshold": 0.7,
    "min_wick_body_ratio": 1.5,
    "volume_lookback": 20,
    "volume_min_ratio": 1.2,
    "volume_max_ratio": 2.0,
}

COMBINE_SETTINGS = {
    "account_size": 50_000,
    "profit_target": 3_000,
    "max_trailing_drawdown": 2_000,
    "max_daily_loss": 1_000,
    "consistency_pct": 0.30,
    "min_trading_days": 5,
}


def build_engine(params: dict) -> BacktestEngine:
    """Instantiate a BacktestEngine from the given parameter dict."""
    primary = EMATrendRule(
        fast_period=params["ema_fast"],
        slow_period=params["ema_slow"],
        min_spread_atr_ratio=params.get("min_spread_atr_ratio", 0.3),
        slope_lookback=params.get("slope_lookback", 3),
        atr_period=params.get("atr_period", 14),
    )

    filters = [
        TimeOfDayRule(
            session_start=params.get("session_start", "09:35"),
            session_end=params.get("session_end", "15:45"),
            lunch_filter_enabled=False,
        ),
        VolumeBreakoutRule(
            lookback=params.get("volume_lookback", 20),
            min_ratio=params.get("volume_min_ratio", 1.2),
            max_ratio=params.get("volume_max_ratio", 2.0),
        ),
    ]

    confirmations = [
        MeanReversionRule(
            bb_period=params.get("bb_period", 20),
            bb_std=params.get("bb_std", 2.0),
            long_bb_threshold=params.get("long_bb_threshold", 0.3),
            short_bb_threshold=params.get("short_bb_threshold", 0.7),
            rsi_period=params.get("rsi_period", 14),
            rsi_long_threshold=params.get("rsi_long_threshold", 35.0),
            rsi_short_threshold=params.get("rsi_short_threshold", 65.0),
        ),
        RejectionPatternRule(
            min_wick_body_ratio=params.get("min_wick_body_ratio", 1.5),
        ),
    ]

    aggregator = SignalAggregator(
        primary_rule=primary,
        filter_rules=filters,
        confirmation_rules=confirmations,
        min_confirmations=1,
    )

    risk_manager = RiskManager(
        contracts=params.get("n_contracts", 1),
        point_value=RISK_DEFAULTS["point_value"],
        tick_size=RISK_DEFAULTS["tick_size"],
        tick_value=RISK_DEFAULTS["tick_value"],
        max_daily_loss=RISK_DEFAULTS["max_daily_loss"],
        per_trade_max_loss=RISK_DEFAULTS["per_trade_max_loss"] * params.get("n_contracts", 1),
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
    )

    return engine


def run_single_config(
    params: dict,
    bars: pd.DataFrame,
    simulator: TopstepCombineSimulator,
    mc_paths: int = 5000,
    mc_seed: int = 42,
) -> dict:
    """Run one backtest + Monte Carlo simulation. Returns metrics dict."""
    engine = build_engine(params)
    result = engine.run(bars, starting_equity=50_000.0)
    summary = result.summary()

    # Monte Carlo from actual trade PnL distribution
    trade_pnls = [t.pnl for t in result.trades]
    if len(trade_pnls) < 5:
        mc = {"p_pass": 0.0, "median_days": None, "p95_max_drawdown": None, "n_paths": mc_paths}
    else:
        # Scale PnL by n_contracts (already scaled in engine via point_value, but trades may be 1-contract)
        mc = simulator.monte_carlo(
            trade_pnl_list=trade_pnls,
            n_paths=mc_paths,
            trades_per_day_range=(2, 8),
            max_days=40,
            seed=mc_seed,
        )

    # Actual OOS deterministic simulation
    oos_result = simulator.simulate(result.daily_pnl)

    return {
        "params": params,
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
    """Run all 216 parameter combinations and return sorted results."""
    simulator = TopstepCombineSimulator(**COMBINE_SETTINGS)

    # Build all parameter combinations
    keys = list(SWEEP.keys())
    values = list(SWEEP.values())
    combinations = list(product(*values))
    total = len(combinations)
    logger.info(f"Starting sweep: {total} configurations")

    results = []
    t0 = time.time()

    for i, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))
        params.update(BACKTEST_DEFAULTS)

        try:
            r = run_single_config(params, bars, simulator, mc_paths=mc_paths)
            results.append(r)

            if verbose or (i % 20 == 0):
                elapsed = time.time() - t0
                eta = (elapsed / i) * (total - i)
                logger.info(
                    f"[{i:3d}/{total}] ema={params['ema_fast']}/{params['ema_slow']} "
                    f"pt={params['pt_atr_mult']} sl={params['sl_atr_mult']} "
                    f"n={params['n_contracts']} | "
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
    """Build a high-level summary from sweep results."""
    if not results:
        return {"error": "No results"}

    best = results[0]
    qualified = [r for r in results if r["backtest"]["win_rate"] > 0.45]
    any_pass_25 = [r for r in results if r["monte_carlo"]["p_pass"] > 0.25]

    return {
        "total_configs": len(results),
        "qualified_win_rate_gt_45pct": len(qualified),
        "configs_p_pass_gt_25pct": len(any_pass_25),
        "best_config": best,
        "best_win_rate": round(
            max(r["backtest"]["win_rate"] for r in results), 4
        ),
        "best_p_pass": round(
            max(r["monte_carlo"]["p_pass"] for r in results), 4
        ),
        "best_sharpe": round(
            max(r["backtest"]["sharpe_ratio"] for r in results), 4
        ),
        "top5": results[:5],
        "go_signal": (
            any(r["backtest"]["win_rate"] > 0.45 and r["monte_carlo"]["p_pass"] > 0.25
                for r in results)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Rule-based OOS validation sweep")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(_PROJECT_ROOT / "data" / "processed" / "jan_feb_2026_oos_test.h5"),
        help="Path to HDF5 data file",
    )
    parser.add_argument("--data-key", type=str, default="bars_5min", help="HDF5 key")
    parser.add_argument("--start", type=str, default="2026-01-01", help="OOS start date")
    parser.add_argument("--end", type=str, default="2026-02-10", help="OOS end date")
    parser.add_argument(
        "--output",
        type=str,
        default=str(_PROJECT_ROOT / "ml_intraday_v3" / "diagnostics" / "rule_based_oos_results.json"),
        help="Output JSON path",
    )
    parser.add_argument("--mc-paths", type=int, default=5000, help="Monte Carlo paths per config")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Load data
    logger.info(f"Loading bars from {args.data_path} ({args.start} to {args.end})")
    try:
        bars = load_bars(args.data_path, key=args.data_key, start_date=args.start, end_date=args.end)
    except FileNotFoundError as e:
        # Try alternative common paths
        alt_paths = [
            _PROJECT_ROOT / "data" / "processed" / "jan_feb_2026_oos_test.h5",
            _PROJECT_ROOT / "data" / "processed" / "mes_bars_databento_rth.h5",
            _PROJECT_ROOT / "ml_intraday_v3" / "data" / "mes_bars_databento_rth.h5",
        ]
        for alt in alt_paths:
            if alt.exists():
                logger.info(f"Using alternative path: {alt}")
                bars = load_bars(str(alt), key=args.data_key, start_date=args.start, end_date=args.end)
                break
        else:
            logger.error(f"Data file not found. Tried: {args.data_path} and {[str(p) for p in alt_paths]}")
            raise e

    logger.info(f"Loaded {len(bars):,} bars spanning {bars.index[0]} to {bars.index[-1]}")

    # Run sweep
    t_start = time.time()
    results = run_sweep(bars, mc_paths=args.mc_paths, verbose=args.verbose)
    elapsed = time.time() - t_start
    logger.info(f"Sweep complete in {elapsed:.1f}s ({len(results)} configs)")

    # Build summary
    summary = build_summary(results)

    # Print key results
    print("\n" + "=" * 70)
    print("RULE-BASED OOS VALIDATION RESULTS")
    print("=" * 70)
    print(f"Period: {args.start} to {args.end}")
    print(f"Total configs tested: {summary['total_configs']}")
    print(f"Configs w/ win_rate > 45%: {summary['qualified_win_rate_gt_45pct']}")
    print(f"Configs w/ P(pass) > 25%: {summary['configs_p_pass_gt_25pct']}")
    print(f"Best win rate:  {summary['best_win_rate']:.1%}")
    print(f"Best P(pass):   {summary['best_p_pass']:.1%}")
    print(f"Best Sharpe:    {summary['best_sharpe']:.3f}")

    if summary["go_signal"]:
        print("\n[GO SIGNAL] Edge found! Proceed to Phase C (MAE/MFE) and Phase D (meta-labeling)")
    else:
        print("\n[NO-GO] No config met win_rate > 45% AND P(pass) > 25%")
        print("  -> Fallback: test RTH open-only (09:30-10:30) or 1-min bars")

    print("\nTop 5 configs:")
    for i, r in enumerate(summary.get("top5", []), 1):
        p = r["params"]
        bt = r["backtest"]
        mc = r["monte_carlo"]
        print(
            f"  {i}. ema={p['ema_fast']}/{p['ema_slow']} "
            f"pt={p['pt_atr_mult']} sl={p['sl_atr_mult']} n={p['n_contracts']} | "
            f"wr={bt['win_rate']:.1%} sharpe={bt['sharpe_ratio']:.2f} "
            f"P(pass)={mc['p_pass']:.1%} med_days={mc['median_days']}"
        )

    # Save full results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "meta": {
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
