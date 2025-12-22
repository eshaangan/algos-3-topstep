"""
Run Monte Carlo simulation with ATR<=p35 & conf>=0.06 filter (baseline).

This script:
1. Runs backtest with no confidence filter to get all trades
2. Filters trades post-hoc using ATR<=p35 & conf>=0.06
3. Saves filtered trades to CSV
4. Runs Monte Carlo combine simulation
5. Reports pass-rate and failure reasons
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Setup paths
project_root = Path(__file__).resolve().parent.parent
os.chdir(project_root)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.environ.setdefault("RISK_PRESET_NAME", "TOPSTEP_50K")

from core.risk_presets import get_risk_preset
from data.clean_bars import clean_bars
from features.engineer import add_features
from models.nn_inference import load_nn_bundle, predict_scores_for_bars
from backtesting.backtest import run_backtest_nn
from analysis.monte_carlo_combine import simulate_combine


def enrich_trades_with_features(trades_df, bars_df, prob_df, bars_with_features):
    """Enrich trades with volatility features and model outputs."""
    trades_enriched = []

    for idx, trade in trades_df.iterrows():
        entry_time = pd.to_datetime(trade["entry_time"], utc=True)
        entry_idx = bars_df[bars_df["timestamp"] == entry_time].index

        if len(entry_idx) == 0 or entry_idx[0] == 0:
            continue

        signal_idx = entry_idx[0] - 1
        signal_bar = bars_with_features.iloc[signal_idx]
        prob_row = prob_df.iloc[signal_idx] if signal_idx < len(prob_df) else None

        if prob_row is None:
            continue

        trade_dict = trade.to_dict()
        trade_dict.update({
            "atr_ticks": signal_bar.get("atr_ticks", None),
            "vol_percentile": signal_bar.get("vol_percentile", None),
            "confidence": prob_row.get("confidence", 0.0),
        })

        trades_enriched.append(trade_dict)

    return pd.DataFrame(trades_enriched)


def main():
    # Configuration
    data_path = "data/processed/es_bars_2010_2025.h5"
    dataset_key = "bars_5min"
    model_dir = "models/nn_saved"
    fold = 0
    fast_max_bars = 250_000

    print("=" * 100)
    print("MONTE CARLO SIMULATION - BASELINE WITH ATR<=p35 & CONF>=0.06 FILTER")
    print("=" * 100)

    # Load data
    print(f"\nLoading data from {data_path}...")
    with pd.HDFStore(data_path, "r") as store:
        bars = store[dataset_key].copy()

    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values("timestamp").reset_index(drop=True)

    preset = get_risk_preset("TOPSTEP_50K")
    bars = clean_bars(bars, tick_size=preset.risk_config.tick_size, verbose=False)
    bars = bars.tail(fast_max_bars).reset_index(drop=True)

    print(f"Loaded {len(bars):,} bars")
    print(f"Date range: {bars['timestamp'].min()} to {bars['timestamp'].max()}")

    # Add features
    print("\nComputing features...")
    bars_with_features = add_features(bars, verbose=False)

    # Load model
    print(f"\nLoading model from {model_dir}...")
    bundle = load_nn_bundle(model_dir, fold=fold)
    nn_cfg = bundle.config.get("nn_config", {})

    # Compute probabilities
    print("Computing model probabilities...")
    prob_df = predict_scores_for_bars(bars, bundle)

    # Run backtest (no confidence filter to get all trades)
    print("\nRunning backtest (confidence_min=0.0)...")
    risk_cfg = preset.risk_config

    results = run_backtest_nn(
        bars,
        prob_df,
        score_threshold=float(nn_cfg["score_threshold"]),
        selection_mode=str(nn_cfg.get("selection_mode", "global_threshold")),
        day_percentile_floor=float(nn_cfg.get("day_percentile_floor", 0.90)),
        global_floor_score=float(nn_cfg.get("global_floor_score", nn_cfg["score_threshold"])),
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
        session_start=risk_cfg.session_start,
        session_end=risk_cfg.session_end,
        stop_loss_ticks=int(nn_cfg["stop_loss_ticks"]),
        target_multiplier=float(nn_cfg["target_multiplier"]),
        catastrophic_stop_ticks=int(nn_cfg.get("catastrophic_stop_ticks", int(nn_cfg["threshold_ticks"]) * 4)),
        max_hold_bars=int(nn_cfg["max_hold_bars"]),
        tick_size=float(nn_cfg["tick_size"]),
        tick_value=float(nn_cfg["tick_value"]),
        save_trades_path=None,
    )

    trades_df = pd.DataFrame(results["trades"])
    print(f"Generated {len(trades_df)} total trades")

    # Enrich with features
    print("\nEnriching trades with features...")
    trades_enriched = enrich_trades_with_features(trades_df, bars, prob_df, bars_with_features)
    print(f"Enriched {len(trades_enriched)} trades")

    # Apply filter: ATR<=p35 & conf>=0.06
    print("\nApplying filter: ATR<=p35 & conf>=0.06...")
    atr_p35 = trades_enriched["atr_ticks"].quantile(0.35)
    conf_min = 0.06

    filtered_trades = trades_enriched[
        (trades_enriched["atr_ticks"] <= atr_p35) &
        (trades_enriched["confidence"] >= conf_min)
    ].copy()

    print(f"Filtered trades: {len(filtered_trades)}")
    print(f"Filter removed: {len(trades_enriched) - len(filtered_trades)} trades ({(1 - len(filtered_trades)/len(trades_enriched))*100:.1f}%)")

    # Analyze filtered trades
    print("\n" + "=" * 100)
    print("FILTERED BACKTEST STATISTICS")
    print("=" * 100)

    catastop = filtered_trades[filtered_trades["reason"] == "CATASTOP"]
    timeexit = filtered_trades[filtered_trades["reason"] == "TIME_EXIT"]
    winners = filtered_trades[filtered_trades["pnl"] > 0]
    losers = filtered_trades[filtered_trades["pnl"] <= 0]

    catastop_rate = len(catastop) / len(filtered_trades) * 100 if len(filtered_trades) > 0 else 0
    win_rate = len(winners) / len(filtered_trades) * 100 if len(filtered_trades) > 0 else 0

    gross_wins = winners["pnl"].sum() if len(winners) > 0 else 0
    gross_losses = abs(losers["pnl"].sum()) if len(losers) > 0 else 0
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    print(f"\nTrades: {len(filtered_trades)}")
    print(f"CATASTOP: {len(catastop)} ({catastop_rate:.1f}%)")
    print(f"TIME_EXIT: {len(timeexit)} ({len(timeexit)/len(filtered_trades)*100:.1f}%)")
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Profit factor: {pf:.2f}")
    print(f"Avg PnL: ${filtered_trades['pnl'].mean():.2f}")
    print(f"Net PnL: ${filtered_trades['pnl'].sum():.2f}")

    # Save filtered trades
    output_path = "analysis/trades_filtered_baseline.csv"
    filtered_trades.to_csv(output_path, index=False)
    print(f"\nSaved filtered trades to: {output_path}")

    # Run Monte Carlo simulation
    print("\n" + "=" * 100)
    print("MONTE CARLO SIMULATION (10,000 runs)")
    print("=" * 100)

    mc_results = simulate_combine(
        filtered_trades,
        starting_balance=risk_cfg.starting_balance,
        profit_target=5000.0,  # Topstep 50K target
        daily_loss_limit=risk_cfg.max_daily_loss,
        trailing_drawdown=risk_cfg.trailing_drawdown,
        runs=10_000,
        seed=42,
        max_days=252,  # ~1 year trading
    )

    print(f"\nRuns: {mc_results['runs']}")
    print(f"Pass rate: {mc_results['pass_rate']*100:.2f}%")

    print(f"\nDays to pass (for successful runs):")
    if mc_results['days_to_pass']:
        print(f"  p05: {mc_results['days_to_pass'].get('p05', 0):.1f} days")
        print(f"  Median: {mc_results['days_to_pass'].get('p50', 0):.1f} days")
        print(f"  p95: {mc_results['days_to_pass'].get('p95', 0):.1f} days")
        print(f"  Mean: {mc_results['days_to_pass'].get('mean', 0):.1f} days")
    else:
        print("  No successful passes")

    print(f"\nFail reasons:")
    total_fails = sum(mc_results['fail_reasons'].values())
    for reason, count in mc_results['fail_reasons'].items():
        pct = count / total_fails * 100 if total_fails > 0 else 0
        print(f"  {reason}: {count} ({pct:.1f}%)")

    print(f"\nMax drawdown distribution:")
    if mc_results['max_drawdown']:
        print(f"  p05: ${mc_results['max_drawdown'].get('p05', 0):.2f}")
        print(f"  Median: ${mc_results['max_drawdown'].get('p50', 0):.2f}")
        print(f"  p95: ${mc_results['max_drawdown'].get('p95', 0):.2f}")
        print(f"  Mean: ${mc_results['max_drawdown'].get('mean', 0):.2f}")

    # Save results
    results_path = "analysis/monte_carlo_filtered_baseline.json"
    with open(results_path, "w") as f:
        json.dump(mc_results, f, indent=2)
    print(f"\nSaved Monte Carlo results to: {results_path}")

    print("\n" + "=" * 100)
    print("BASELINE COMPLETE")
    print("=" * 100)
    print(f"\nFilter: ATR<=p35 & conf>=0.06")
    print(f"Trades: {len(filtered_trades)}")
    print(f"CATASTOP rate: {catastop_rate:.1f}%")
    print(f"Profit factor: {pf:.2f}")
    print(f"Pass rate: {mc_results['pass_rate']*100:.2f}%")
    print(f"Median days to pass: {mc_results['days_to_pass'].get('p50', 0):.1f}")
    print("\nNext: Implement dynamic catastrophic stop sizing")
    print("=" * 100)


if __name__ == "__main__":
    main()
