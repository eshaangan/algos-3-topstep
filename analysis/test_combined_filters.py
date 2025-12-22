"""
Test combined filters to achieve <10% CATASTOP target.

Based on the quality metrics exploration, this script tests:
1. More aggressive ATR-only filters (p25, p30, p35, p40)
2. Combined ATR + confidence filters
3. Combined ATR + score filters
4. Regime-adaptive approaches
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


def enrich_trades(trades_df, bars_df, prob_df, bars_with_features):
    """Enrich trades with model outputs and volatility features."""
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
            "p_long": prob_row.get("long_prob", 0.5),
            "p_short": prob_row.get("short_prob", 0.5),
            "score": prob_row.get("score", 0.5),
            "confidence": prob_row.get("confidence", 0.0),
            "atr_ticks": signal_bar.get("atr_ticks", None),
            "vol_20": signal_bar.get("vol_20", None),
            "vol_percentile": signal_bar.get("vol_percentile", None),
            "bb_width": signal_bar.get("bb_width", None),
        })

        trades_enriched.append(trade_dict)

    return pd.DataFrame(trades_enriched)


def test_filter(trades_df, filter_name, filter_func):
    """Test a filter and return performance metrics."""
    filtered = filter_func(trades_df)

    if len(filtered) == 0:
        return None

    catastop = filtered[filtered["reason"] == "CATASTOP"]
    winners = filtered[filtered["pnl"] > 0]
    losers = filtered[filtered["pnl"] <= 0]

    catastop_rate = len(catastop) / len(filtered) * 100
    win_rate = len(winners) / len(filtered) * 100
    avg_pnl = filtered["pnl"].mean()

    gross_wins = winners["pnl"].sum() if len(winners) > 0 else 0
    gross_losses = abs(losers["pnl"].sum()) if len(losers) > 0 else 0
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    return {
        "filter": filter_name,
        "trades": len(filtered),
        "catastop_count": len(catastop),
        "catastop_rate": catastop_rate,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "profit_factor": pf,
        "net_pnl": filtered["pnl"].sum(),
        "meets_target": catastop_rate < 10.0,
    }


def main():
    # Configuration
    data_path = "data/processed/es_bars_2010_2025.h5"
    dataset_key = "bars_5min"
    model_dir = "models/nn_saved"
    fold = 0
    fast_max_bars = 250_000

    print("=" * 100)
    print("COMBINED FILTER TESTING FOR <10% CATASTOP TARGET")
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

    # Run backtest
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
    print(f"Generated {len(trades_df)} trades")

    # Enrich trades
    print("\nEnriching trades with features...")
    trades = enrich_trades(trades_df, bars, prob_df, bars_with_features)
    print(f"Enriched {len(trades)} trades")

    # Baseline
    baseline = test_filter(trades, "BASELINE (no filter)", lambda df: df)

    print("\n" + "=" * 100)
    print("BASELINE PERFORMANCE")
    print("=" * 100)
    print(f"Trades: {baseline['trades']}")
    print(f"CATASTOP: {baseline['catastop_count']} ({baseline['catastop_rate']:.1f}%)")
    print(f"Win rate: {baseline['win_rate']:.1f}%")
    print(f"Avg PnL: ${baseline['avg_pnl']:.2f}")
    print(f"Profit factor: {baseline['profit_factor']:.2f}")
    print(f"Net PnL: ${baseline['net_pnl']:.2f}")

    # Test filters
    filter_results = [baseline]

    # 1. Aggressive ATR-only filters
    print("\n" + "=" * 100)
    print("ATR-ONLY FILTERS (more aggressive)")
    print("=" * 100)

    atr_percentiles = [25, 30, 35, 40, 45, 50]
    for p in atr_percentiles:
        thresh = trades["atr_ticks"].quantile(p / 100.0)
        result = test_filter(
            trades,
            f"ATR <= p{p} ({thresh:.1f} ticks)",
            lambda df, t=thresh: df[df["atr_ticks"] <= t],
        )
        if result:
            filter_results.append(result)

    print(f"{'Filter':<30} {'Trades':<8} {'CATASTOP':<10} {'Rate%':<8} {'WinRate%':<10} {'AvgPnL':<10} {'PF':<8} {'Target':<8}")
    print("-" * 100)
    for res in filter_results[-len(atr_percentiles):]:
        target_str = "✓" if res["meets_target"] else ""
        print(
            f"{res['filter']:<30} "
            f"{res['trades']:<8} "
            f"{res['catastop_count']:<10} "
            f"{res['catastop_rate']:<8.1f} "
            f"{res['win_rate']:<10.1f} "
            f"${res['avg_pnl']:<9.2f} "
            f"{res['profit_factor']:<8.2f} "
            f"{target_str:<8}"
        )

    # 2. Combined ATR + Confidence
    print("\n" + "=" * 100)
    print("COMBINED ATR + CONFIDENCE FILTERS")
    print("=" * 100)

    atr_thresh_p40 = trades["atr_ticks"].quantile(0.40)
    conf_thresholds = [0.05, 0.06, 0.07, 0.08]

    for conf_min in conf_thresholds:
        result = test_filter(
            trades,
            f"ATR<=p40 & conf>={conf_min:.2f}",
            lambda df, a=atr_thresh_p40, c=conf_min: df[(df["atr_ticks"] <= a) & (df["confidence"] >= c)],
        )
        if result:
            filter_results.append(result)

    print(f"{'Filter':<30} {'Trades':<8} {'CATASTOP':<10} {'Rate%':<8} {'WinRate%':<10} {'AvgPnL':<10} {'PF':<8} {'Target':<8}")
    print("-" * 100)
    for res in filter_results[-len(conf_thresholds):]:
        target_str = "✓" if res["meets_target"] else ""
        print(
            f"{res['filter']:<30} "
            f"{res['trades']:<8} "
            f"{res['catastop_count']:<10} "
            f"{res['catastop_rate']:<8.1f} "
            f"{res['win_rate']:<10.1f} "
            f"${res['avg_pnl']:<9.2f} "
            f"{res['profit_factor']:<8.2f} "
            f"{target_str:<8}"
        )

    # 3. Vol percentile filters
    print("\n" + "=" * 100)
    print("VOL_PERCENTILE FILTERS")
    print("=" * 100)

    vol_pct_thresholds = [0.3, 0.4, 0.5, 0.6]

    for vol_thresh in vol_pct_thresholds:
        result = test_filter(
            trades,
            f"vol_pct <= {vol_thresh:.1f}",
            lambda df, v=vol_thresh: df[df["vol_percentile"] <= v],
        )
        if result:
            filter_results.append(result)

    print(f"{'Filter':<30} {'Trades':<8} {'CATASTOP':<10} {'Rate%':<8} {'WinRate%':<10} {'AvgPnL':<10} {'PF':<8} {'Target':<8}")
    print("-" * 100)
    for res in filter_results[-len(vol_pct_thresholds):]:
        target_str = "✓" if res["meets_target"] else ""
        print(
            f"{res['filter']:<30} "
            f"{res['trades']:<8} "
            f"{res['catastop_count']:<10} "
            f"{res['catastop_rate']:<8.1f} "
            f"{res['win_rate']:<10.1f} "
            f"${res['avg_pnl']:<9.2f} "
            f"{res['profit_factor']:<8.2f} "
            f"{target_str:<8}"
        )

    # 4. Multi-factor filters (best combination)
    print("\n" + "=" * 100)
    print("MULTI-FACTOR FILTERS (combining best metrics)")
    print("=" * 100)

    # Test various combinations
    combos = [
        ("ATR<=p25 & vol_pct<=0.5", lambda df: df[(df["atr_ticks"] <= df["atr_ticks"].quantile(0.25)) & (df["vol_percentile"] <= 0.5)]),
        ("ATR<=p30 & vol_pct<=0.5", lambda df: df[(df["atr_ticks"] <= df["atr_ticks"].quantile(0.30)) & (df["vol_percentile"] <= 0.5)]),
        ("ATR<=p35 & conf>=0.06", lambda df: df[(df["atr_ticks"] <= df["atr_ticks"].quantile(0.35)) & (df["confidence"] >= 0.06)]),
        ("ATR<=p40 & vol_pct<=0.4", lambda df: df[(df["atr_ticks"] <= df["atr_ticks"].quantile(0.40)) & (df["vol_percentile"] <= 0.4)]),
    ]

    for name, func in combos:
        result = test_filter(trades, name, func)
        if result:
            filter_results.append(result)

    print(f"{'Filter':<30} {'Trades':<8} {'CATASTOP':<10} {'Rate%':<8} {'WinRate%':<10} {'AvgPnL':<10} {'PF':<8} {'Target':<8}")
    print("-" * 100)
    for res in filter_results[-len(combos):]:
        target_str = "✓" if res["meets_target"] else ""
        print(
            f"{res['filter']:<30} "
            f"{res['trades']:<8} "
            f"{res['catastop_count']:<10} "
            f"{res['catastop_rate']:<8.1f} "
            f"{res['win_rate']:<10.1f} "
            f"${res['avg_pnl']:<9.2f} "
            f"{res['profit_factor']:<8.2f} "
            f"{target_str:<8}"
        )

    # Summary: Best filters
    print("\n" + "=" * 100)
    print("BEST FILTERS (ranked by CATASTOP rate, min 100 trades)")
    print("=" * 100)

    qualifying_filters = [r for r in filter_results if r["trades"] >= 100]
    qualifying_filters.sort(key=lambda x: x["catastop_rate"])

    print(f"{'Filter':<30} {'Trades':<8} {'CATASTOP%':<12} {'WinRate%':<12} {'AvgPnL':<12} {'PF':<10} {'NetPnL':<12} {'Target':<8}")
    print("-" * 100)

    for res in qualifying_filters[:10]:  # Top 10
        target_str = "✓ MEETS" if res["meets_target"] else ""
        pf_str = f"{res['profit_factor']:.2f}" if res['profit_factor'] != float("inf") else "inf"
        print(
            f"{res['filter']:<30} "
            f"{res['trades']:<8} "
            f"{res['catastop_rate']:<12.1f} "
            f"{res['win_rate']:<12.1f} "
            f"${res['avg_pnl']:<11.2f} "
            f"{pf_str:<10} "
            f"${res['net_pnl']:<11.2f} "
            f"{target_str:<8}"
        )

    # Identify best candidate
    meets_target = [r for r in qualifying_filters if r["meets_target"]]
    if meets_target:
        best = meets_target[0]
        print("\n" + "=" * 100)
        print("✓ RECOMMENDED FILTER (meets <10% CATASTOP target)")
        print("=" * 100)
        print(f"\nFilter: {best['filter']}")
        print(f"Trades: {best['trades']}")
        print(f"CATASTOP rate: {best['catastop_rate']:.1f}%")
        print(f"Win rate: {best['win_rate']:.1f}%")
        print(f"Avg PnL: ${best['avg_pnl']:.2f}")
        print(f"Profit factor: {best['profit_factor']:.2f}")
        print(f"Net PnL: ${best['net_pnl']:.2f}")
    else:
        print("\n⚠️  No filter achieved <10% CATASTOP with >=100 trades.")
        print("Best available filter:")
        if qualifying_filters:
            best = qualifying_filters[0]
            print(f"\nFilter: {best['filter']}")
            print(f"Trades: {best['trades']}")
            print(f"CATASTOP rate: {best['catastop_rate']:.1f}% (target: <10%)")
            print(f"Win rate: {best['win_rate']:.1f}%")
            print(f"Avg PnL: ${best['avg_pnl']:.2f}")
            print(f"Profit factor: {best['profit_factor']:.2f}")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
